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

The pull-request verdict leads, because it is the question the product answers:
a bot that misses a line is worse than one that finds it, but a bot that walks
past the whole pull request is the failure a reviewer notices. It is reported as
balanced accuracy, which is 0.5 for any strategy that ignores the input on any
class balance -- unlike plain accuracy, which a corpus of mostly-clean cases
hands out for free.
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


def render(run_id: str, manifest: dict, result: dict, notes: Sequence[str] = (),
           baselines: dict[str, dict] | None = None) -> str:
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

    flagging = result["pr_level_in_scope"]
    caught = flagging["tp"] + flagging["fn"]
    quiet = flagging["tn"] + flagging["fp"]
    lines += [
        "## PR seviyesi — \"bu PR'a yorum yazılmalı mı\"",
        "",
        "Ürünün asıl sorduğu soru bu. Bir satırı kaçırmak bir şey, PR'ın tamamını "
        "sessiz geçmek başka bir şey. **Dengeli doğruluk** sınıf dengesinden bağımsızdır: "
        "girdiyi hiç okumayan her strateji için 0.50'dir.",
        "",
    ]
    lines += _table(
        ["ölçüm", "değer", "ne demek"],
        [
            ["Yakalanan kusurlu PR", f"{flagging['tp']}/{caught}", _pct(flagging["recall"]) + " (recall)"],
            ["Sessiz geçilen temiz PR", f"{flagging['tn']}/{quiet}",
             _pct(flagging["specificity"]) + " (specificity)"],
            ["**Dengeli doğruluk**", f"**{_pct(flagging['balanced_accuracy'])}**",
             "0.50 = girdiyi yok saymak"],
            ["Ham doğruluk", _pct(flagging["accuracy"]), "sınıf dengesine bağlı, karşılaştırılamaz"],
        ],
    )
    if baselines:
        honest = {name: card for name, card in baselines.items() if not card.get("leaky")}
        if honest:
            reference = max(honest.items(), key=lambda item: item[1]["pr_level"]["balanced_accuracy"])
            lines += [
                f"Karşılaştırma noktası — en iyi dürüst taban çizgisi "
                f"`{reference[0]}`: dengeli doğruluk "
                f"**{_pct(reference[1]['pr_level']['balanced_accuracy'])}**.",
                "",
            ]

    lines += [
        "## Bulgu seviyesi",
        "",
        "Yer bulmak ile tür adlandırmak ayrı yetenekler ve ayrı ayrı bozuluyorlar, "
        "bu yüzden ayrı ölçülüyor. Tür doğruluğu **yeri bulunan** bulgular üzerinden.",
        "",
    ]
    located = [r for r in rows if r["inside_region"]]
    lines += _table(
        ["boyut", "doğru", "toplam", "oran"],
        [
            ["Dosya", sum(r["file_correct"] for r in rows), loc["findings"], _pct(loc["file_accuracy"])],
            ["Satır (bölge içinde)", loc["located"], loc["findings"], _pct(loc["region_accuracy"])],
            ["Tür (bulunanlar içinde)", sum(r["type_correct"] for r in located),
             loc["located"], _pct(loc["type_accuracy"])],
            ["**Her ikisi birden**", sum(r["inside_region"] and r["type_correct"] for r in rows),
             loc["findings"], _pct(loc["strict_accuracy"])],
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

    mislabelled = [row for row in located if not row["type_correct"]]
    if mislabelled:
        lines += [
            "### Yeri doğru, türü yanlış",
            "",
            "Bunlar kaçırılmış bulgular değil: kusur bulundu, kategorisi yanlış adlandırıldı.",
            "",
        ]
        lines += _table(
            ["vaka", "beklenen tür", "dediği tür", "yer", "başlık"],
            [[row["case_id"][:26], row["expected_type"], row["predicted_type"] or "-",
              f"{row['expected_file']}:{_span(row['predicted_lines'])}", row["predicted_title"][:48]]
             for row in mislabelled],
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
