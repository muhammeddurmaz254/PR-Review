"""The step file: one readable record of what a run found and what it missed.

Aggregates tell you a detector scored sixty percent. They never tell you which
six of ten, whether the four misses share a shape, or whether a hit landed on
the defect or merely somewhere in the same function. This file carries the rows
that answer that, and the summary is computed from the same rows so the two can
never disagree.

Two localization numbers are reported side by side on purpose. ``region`` asks
whether the report points at the code that contains the defect, which is what a
reviewer needs. ``IoU`` asks how tightly, and is averaged only over findings
that were located -- folding misses into it would turn a detection failure into
a localization number.
"""
from __future__ import annotations

from typing import Sequence


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    if not rows:
        return ["_(yok)_", ""]
    widths = [len(str(h)) for h in headers]
    text = [[str(cell) for cell in row] for row in rows]
    for row in text:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    out = [
        "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    out += ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in text]
    return out + [""]


def _span(pair) -> str:
    if not pair:
        return "-"
    start, end = pair
    return str(start) if start == end else f"{start}-{end}"


def render(run_id: str, manifest: dict, result: dict, notes: Sequence[str] = ()) -> str:
    rows = result["per_finding"]
    loc = result["localization"]
    alarms = result["false_alarm_rows"]
    cases = manifest.get("cases", 0)

    lines = [
        f"# Adım `{run_id}`",
        "",
        f"Veri seti **{manifest.get('corpus', '?')}** · vaka **{cases}** · "
        f"bulgu **{loc['findings']}** · model **{manifest.get('model') or '-'}** · "
        f"prompt **{manifest.get('prompt_version') or '-'}**",
        "",
    ]
    if notes:
        lines += [f"> {note}" for note in notes] + [""]

    lines += ["## Özet", "", "Doğruluk, her boyut için ayrı ayrı.", ""]
    lines += _table(
        ["boyut", "doğru", "toplam", "oran"],
        [
            ["Tür", sum(r["type_correct"] for r in rows), loc["findings"], _pct(loc["type_accuracy"])],
            ["Dosya", sum(r["file_correct"] for r in rows), loc["findings"], _pct(loc["file_accuracy"])],
            ["Satır (bölge içinde)", loc["located"], loc["findings"], _pct(loc["region_accuracy"])],
        ],
    )
    lines += [
        "IoU yalnız bulunan bulgular üzerinden, aralık etiketleri için ayrı:",
        "",
    ]
    lines += _table(
        ["ölçüm", "değer", "ne demek"],
        [
            ["IoU (bölge)", f"{loc['mean_iou_region']:.3f}",
             "tahmin aralığı ile kusuru içeren fonksiyon örtüşmesi"],
            ["IoU (odak)", f"{loc['mean_iou_focus']:.3f}",
             "tahmin aralığı ile kusurun dar hâli örtüşmesi"],
            ["Aralık etiketi", f"{loc['range_labels']} bulgu",
             f"bölge doğruluğu {_pct(loc['range_region_accuracy'])}"],
        ],
    )

    special = []
    if any(r["cross_file"] for r in rows):
        n = sum(1 for r in rows if r["cross_file"])
        special.append(["Çapraz dosya", n, _pct(loc["cross_file_accuracy"])])
    if any(r["pure_deletion"] for r in rows):
        n = sum(1 for r in rows if r["pure_deletion"])
        special.append(["Saf silme", n, _pct(loc["pure_deletion_accuracy"])])
    if special:
        lines += ["Zor alt kümeler:", ""]
        lines += _table(["alt küme", "bulgu", "bölge doğruluğu"], special)

    lines += ["## Bulgu bazında", "", "Her satır bir ground-truth bulgusu.", ""]
    lines += _table(
        ["vaka", "beklenen tür", "beklenen yer", "tahmin türü", "tahmin yeri",
         "tür", "dosya", "bölge", "IoU-b", "IoU-o"],
        [[
            row["case_id"][:26],
            row["expected_type"],
            f"{row['expected_file']}:{_span(row['expected_region'])}",
            row["predicted_type"] or "-",
            f"{row['predicted_file']}:{_span(row['predicted_lines'])}" if row["predicted_file"] else "-",
            "✓" if row["type_correct"] else "✗",
            "✓" if row["file_correct"] else "✗",
            "✓" if row["inside_region"] else "✗",
            f"{row['iou_region']:.2f}",
            "-" if row["iou_focus"] is None else f"{row['iou_focus']:.2f}",
        ] for row in rows],
    )

    lines += ["### Kaçırılan bulgular", ""]
    missed = [row for row in rows if not row["inside_region"]]
    if missed:
        lines += _table(
            ["vaka", "tür", "yer", "başlık"],
            [[row["case_id"][:26], row["expected_type"],
              f"{row['expected_file']}:{_span(row['expected_region'])}", row["title"][:64]]
             for row in missed],
        )
    else:
        lines += ["_Yok._", ""]

    lines += [
        "## Yanlış alarmlar",
        "",
        f"Hiçbir etiketle eşleşmeyen rapor: **{len(alarms)}** "
        f"(PR başına {len(alarms) / cases:.2f})." if cases else "",
        "",
    ]
    lines += _table(
        ["vaka", "kusurlu mu", "tür", "yer", "başlık"],
        [[row["case_id"][:26], "evet" if row["is_defective"] else "temiz", row["type"],
          f"{row['file']}:{_span(row['lines'])}", row["title"][:56]] for row in alarms],
    )

    lines += ["## Tür bazında", ""]
    by_type = result.get("by_type", {})
    lines += _table(
        ["tür", "TP", "FN", "recall"],
        [[name, card["tp"], card["fn"], _pct(card["recall"])] for name, card in sorted(by_type.items())],
    )
    return "\n".join(lines) + "\n"
