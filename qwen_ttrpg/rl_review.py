"""Create a private side-by-side review of source-grounded agent evaluations."""

import argparse
import html
import json
from pathlib import Path

from .agent_rl import new_private, write

DISPLAY_NAMES = {"base": "Base model", "sft": "Supervised baseline",
                 "rl": "Supervised + RL", "sft140": "Extra supervised control"}


def render(paths, output, *, notes=()):
    reports = [json.loads(Path(path).read_text()) for path in paths]
    if not reports or any(r.get("status") != "complete" for r in reports):
        raise ValueError("Use completed candidate evaluations.")
    labels = [r["candidate"] for r in reports]
    if len(labels) != len(set(labels)):
        raise ValueError("Candidate labels must be unique.")
    first = reports[0]
    for report in reports[1:]:
        if report["dataset"] != first["dataset"] or report["settings"] != first["settings"]:
            raise ValueError("Compare the same data snapshot and generation settings.")
    identity = lambda row: (row["id"], row["generation_seed"])
    grouped = [{identity(row): row for row in r["cases"]} for r in reports]
    if any(set(rows) != set(grouped[0]) for rows in grouped[1:]):
        raise ValueError("Candidate case/seed sets differ.")
    out = new_private(output)
    esc = lambda value: html.escape(str(value))
    sections = []
    annotations = {}
    for note in notes:
        key = (note["candidate"], note["id"], note["seed"])
        annotations.setdefault(key, []).append(note)
    for index, key in enumerate(grouped[0]):
        reference = grouped[0][key]
        sections.append(f'<article id="case-{index}"><h2>{index+1}. {esc(reference["family"])} · seed {key[1]}</h2><p>{esc(json.loads(reference["prompt"][1]["content"])["task"])}</p><details><summary>Initial prompt, source world, and checker target</summary><pre>{esc(json.dumps({k:reference.get(k) for k in ["prompt","source_context","expected","acceptable_sources"]},indent=2))}</pre></details><div class="answers">')
        for report, rows in zip(reports, grouped):
            row = rows[key]
            grade = row["assessment"]
            final = {k:v for k,v in (row["final"] or {}).items()
                     if k in {"action", "value", "allowed", "sources", "missing", "question", "reason"}
                     and v is not None and v != "" and v != []}
            sections.append(f'<section><h3>{esc(DISPLAY_NAMES.get(report["candidate"],report["candidate"]))}</h3><p>{"Pass" if grade["success"] else "Fail"} · {grade["tool_calls"]} calls · {grade["seconds"]:.2f}s</p><pre>{esc(json.dumps(final or None,indent=2))}</pre><details><summary>Decisions, tool results, and grading</summary><pre>{esc(json.dumps({"assessment":grade,"trace":row["trace"]},indent=2))}</pre></details></section>')
            matched = annotations.get((report["candidate"], *key), [])
            if matched:
                annotation = '<h4>Source review</h4>' + ''.join(
                    '<p><strong>' + esc(note.get("reviewer", "Reviewer")) + '</strong>: '
                    + esc(note["finding"]) + '</p>' for note in matched
                )
                sections[-1] = sections[-1].removesuffix('</section>') + annotation + '</section>'
        sections.append('</div></article>')
    summary = {r["candidate"]:r["summary"] for r in reports}
    header = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Evidence policy review</title><style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:24px;background:#f5efe4;color:#35291e}article{border-top:1px solid #c8b99e;padding:20px 0}.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:18px}section{padding:16px;background:#fffaf1;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}nav{display:flex;flex-wrap:wrap;gap:12px}</style><h1>Evidence policy comparison</h1><p>Original procedural scenarios. Pass/fail covers the structured conclusion and its evidence. It does not certify the free-form prose. Candidate identities are visible in this technical review.</p>'
    header += '<div style="overflow-x:auto"><table style="width:100%;text-align:left;border-spacing:12px"><thead><tr><th>Candidate</th><th>Task passes</th><th>Correct conclusion fields</th><th>Grounded citations</th><th>Mean tool calls</th><th>Median time</th></tr></thead><tbody>'
    for label, stats in summary.items():
        overall = stats.get("overall", {})
        header += '<tr>' + ''.join('<td>' + esc(value) + '</td>' for value in [
            DISPLAY_NAMES.get(label,label), f'{overall.get("successes", "—")}/{overall.get("cases", "—")}',
            overall.get("correct_conclusions", "—"), overall.get("grounded_citations", "—"),
            round(overall["mean_tool_calls"], 2) if "mean_tool_calls" in overall else "—",
            f'{overall["median_seconds"]:.2f}s' if "median_seconds" in overall else "—",
        ]) + '</tr>'
    header += '</tbody></table></div><details><summary>Aggregate results by scenario family</summary><pre>' + esc(json.dumps(summary,indent=2)) + '</pre></details><nav>'
    header += ''.join(f'<a href="#case-{i}">{i+1}</a>' for i in range(len(grouped[0]))) + '</nav>'
    (out/'review.html').write_text(header+''.join(sections))
    write(out/'report.json',{"status":"complete","cases":[{"id":k[0],"seed":k[1]} for k in grouped[0]],"summary":summary})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',nargs='+')
    parser.add_argument('--output',required=True)
    parser.add_argument('--notes', action='append', default=[], help='Optional private review JSON with a findings list')
    args=parser.parse_args()
    notes = [note for path in args.notes for note in json.loads(Path(path).read_text())["findings"]]
    print(json.dumps(render(args.reports,args.output,notes=notes),indent=2))


if __name__=='__main__':
    main()
