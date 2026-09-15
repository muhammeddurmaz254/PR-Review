"""Access to Bitbucket Cloud: pull requests, their diffs, the tree at a commit, and PR comments.

This is the only way the project reaches a repository. Everything it does is a
read, except one thing: creating or updating a general comment on a pull request
(`post_comment`, `update_comment`), which only `prdetect.cli.comment --post` calls.
It never approves, merges, declines or changes code.

A pull request costs two requests: its diff, and one archive of its source commit
for the whole tree. Fetching the files one by one would cost a request per file,
which the hourly API limit turns into hours on a repository of any size. Responses
are cached on disk under the commit they belong to, so fetching an unchanged pull
request again asks for nothing but the pull request list.
"""
from __future__ import annotations

import base64
import io
import json
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from prdetect import paths, settings

DEFAULT_WORKSPACE = "muhammeddurmazytuce"
API = "https://api.bitbucket.org/2.0"
WEB = "https://bitbucket.org"


class Bitbucket:
    def __init__(self, email: str, token: str, cache_dir: Path | None = paths.BITBUCKET_CACHE,
                 api: str = API, web: str = WEB):
        self._auth = "Basic " + base64.b64encode(f"{email}:{token}".encode()).decode()
        self.cache_dir = cache_dir
        self.api = api
        self.web = web

    @classmethod
    def from_env(cls, env_file: Path | None = None,
                 cache_dir: Path | None = paths.BITBUCKET_CACHE) -> "Bitbucket":
        """Credentials from the environment, else from the project's `.env`."""
        values = settings.load(env_file)
        missing = [key for key in ("BITBUCKET_CLOUD_EMAIL", "BITBUCKET_CLOUD_API_TOKEN") if not values.get(key)]
        if missing:
            raise SystemExit(f"missing {', '.join(missing)} (environment or {env_file or paths.ENV_FILE})")
        return cls(values["BITBUCKET_CLOUD_EMAIL"], values["BITBUCKET_CLOUD_API_TOKEN"], cache_dir)

    # -- transport ---------------------------------------------------------------

    def _request(self, method: str, url: str, body: dict | None = None) -> bytes:
        """One request, retried when Bitbucket asks to wait.

        A rate limit (429) is retried for every method, since the request was not
        processed. A server error is retried only for GET and PUT: a POST that
        failed half-way may already have created its comment.
        """
        retryable = {429} | ({500, 502, 503, 504} if method in ("GET", "PUT") else set())
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Authorization": self._auth}
        if data is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(6):
            request = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code in retryable and attempt < 5:
                    time.sleep(int(error.headers.get("Retry-After") or 2 ** (attempt + 1)))
                    continue
                detail = error.read().decode(errors="replace")[:300]
                raise SystemExit(f"{method} {url}: HTTP {error.code} {detail}") from error
        raise SystemExit(f"{method} {url}: gave up after retries")

    def _get(self, url: str) -> bytes:
        return self._request("GET", url)

    def _pages(self, url: str) -> list[dict]:
        values = []
        while url:
            page = json.loads(self._get(url))
            values.extend(page.get("values", []))
            url = page.get("next")
        return values

    def _cached(self, key: str, fetch) -> bytes:
        if self.cache_dir is None:
            return fetch()
        path = self.cache_dir / key
        if path.exists():
            return path.read_bytes()
        data = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    # -- pull requests -----------------------------------------------------------

    def pull_requests(self, workspace: str, repo: str, state: str = "OPEN") -> list[dict]:
        """Every pull request in `state`, oldest id first. Never cached: the list changes."""
        query = urllib.parse.urlencode({"state": state, "pagelen": 50})
        pulls = self._pages(f"{self.api}/repositories/{workspace}/{repo}/pullrequests?{query}")
        return sorted(pulls, key=lambda pr: pr["id"])

    def pull_request(self, workspace: str, repo: str, pull_id: int) -> dict:
        """One pull request as it is now. Never cached: its source branch can move."""
        return json.loads(self._get(f"{self.api}/repositories/{workspace}/{repo}/pullrequests/{pull_id}"))

    def diff(self, workspace: str, repo: str, pull: dict) -> str:
        """The pull request's diff, cached under its source and destination commits."""
        source = pull["source"]["commit"]["hash"]
        destination = pull["destination"]["commit"]["hash"]
        key = f"{workspace}/{repo}/diff/{pull['id']}-{source}-{destination}.diff"
        url = f"{self.api}/repositories/{workspace}/{repo}/pullrequests/{pull['id']}/diff"
        return self._cached(key, lambda: self._get(url)).decode("utf-8", errors="replace")

    def tree(self, workspace: str, repo: str, commit: str) -> dict[str, str]:
        """Every text file at `commit`, by path. Binary files are left out."""
        key = f"{workspace}/{repo}/tree/{commit}.json"

        def fetch() -> bytes:
            archive = self._get(f"{self.web}/{workspace}/{repo}/get/{commit}.tar.gz")
            files = {}
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    path = member.name.split("/", 1)[1] if "/" in member.name else member.name
                    try:
                        files[path] = tar.extractfile(member).read().decode("utf-8")
                    except UnicodeDecodeError:
                        continue
            return json.dumps(files, ensure_ascii=False, sort_keys=True).encode()

        return json.loads(self._cached(key, fetch))

    # -- pull request comments: the only writes ----------------------------------

    def _comments_url(self, workspace: str, repo: str, pull_id: int) -> str:
        return f"{self.api}/repositories/{workspace}/{repo}/pullrequests/{pull_id}/comments"

    def comments(self, workspace: str, repo: str, pull_id: int) -> list[dict]:
        """Every comment on a pull request, general and inline. Never cached."""
        return self._pages(self._comments_url(workspace, repo, pull_id) + "?pagelen=100")

    def post_comment(self, workspace: str, repo: str, pull_id: int, raw: str) -> dict:
        """A new general comment -- not attached to any file or line -- in markdown."""
        return json.loads(self._request("POST", self._comments_url(workspace, repo, pull_id),
                                        {"content": {"raw": raw}}))

    def update_comment(self, workspace: str, repo: str, pull_id: int, comment_id: int, raw: str) -> dict:
        return json.loads(self._request("PUT", f"{self._comments_url(workspace, repo, pull_id)}/{comment_id}",
                                        {"content": {"raw": raw}}))
