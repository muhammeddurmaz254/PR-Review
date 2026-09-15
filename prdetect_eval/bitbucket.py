"""Read-only access to Bitbucket Cloud: pull requests, their diffs, the tree at a commit.

This is the only way the harness reaches a repository. It has a GET and nothing
else, so no stage can comment on, approve or change a pull request by accident.

A pull request costs two requests: its diff, and one archive of the source commit
for the whole tree the verifier and the facts read. Fetching files one by one is
sixty requests a pull request on the wide repository, which the hourly API limit
turns into hours. Responses are cached on disk under the commit they belong to,
so a second fetch of an unchanged pull request asks for nothing.
"""
from __future__ import annotations

import base64
import io
import json
import os
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE.parent / ".env"
CACHE_DIR = HERE / ".bitbucket_cache"
DEFAULT_WORKSPACE = "muhammeddurmazytuce"
API = "https://api.bitbucket.org/2.0"
WEB = "https://bitbucket.org"


def _read_env(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class Bitbucket:
    def __init__(self, email: str, token: str, cache_dir: Path | None = CACHE_DIR):
        self._auth = "Basic " + base64.b64encode(f"{email}:{token}".encode()).decode()
        self.cache_dir = cache_dir

    @classmethod
    def from_env(cls, env_file: Path = ENV_FILE, cache_dir: Path | None = CACHE_DIR) -> "Bitbucket":
        """Credentials from the environment, else from the project's `.env`."""
        values = {**_read_env(env_file), **os.environ}
        missing = [key for key in ("BITBUCKET_CLOUD_EMAIL", "BITBUCKET_CLOUD_API_TOKEN") if not values.get(key)]
        if missing:
            raise SystemExit(f"missing {', '.join(missing)} (environment or {env_file})")
        return cls(values["BITBUCKET_CLOUD_EMAIL"], values["BITBUCKET_CLOUD_API_TOKEN"], cache_dir)

    # -- transport: GET only ---------------------------------------------------

    def _get(self, url: str) -> bytes:
        for attempt in range(6):
            request = urllib.request.Request(url, method="GET", headers={"Authorization": self._auth})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(int(error.headers.get("Retry-After") or 2 ** (attempt + 1)))
                    continue
                detail = error.read().decode(errors="replace")[:300]
                raise SystemExit(f"GET {url}: HTTP {error.code} {detail}") from error
        raise SystemExit(f"GET {url}: gave up after retries")

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
        url, pulls = f"{API}/repositories/{workspace}/{repo}/pullrequests?{query}", []
        while url:
            page = json.loads(self._get(url))
            pulls.extend(page.get("values", []))
            url = page.get("next")
        return sorted(pulls, key=lambda pr: pr["id"])

    def diff(self, workspace: str, repo: str, pull: dict) -> str:
        """The pull request's diff, cached under its source and destination commits."""
        source = pull["source"]["commit"]["hash"]
        destination = pull["destination"]["commit"]["hash"]
        key = f"{workspace}/{repo}/diff/{pull['id']}-{source}-{destination}.diff"
        url = f"{API}/repositories/{workspace}/{repo}/pullrequests/{pull['id']}/diff"
        return self._cached(key, lambda: self._get(url)).decode("utf-8", errors="replace")

    def tree(self, workspace: str, repo: str, commit: str) -> dict[str, str]:
        """Every text file at `commit`, by path. Binary files are left out."""
        key = f"{workspace}/{repo}/tree/{commit}.json"

        def fetch() -> bytes:
            archive = self._get(f"{WEB}/{workspace}/{repo}/get/{commit}.tar.gz")
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
