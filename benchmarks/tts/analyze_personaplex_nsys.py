#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU idle / launch-overhead analysis for the PersonaPlex nsys reports.

For every ``.nsys-rep`` under a directory: export to SQLite, merge the
kernel (+memcpy/memset) activity intervals into a busy timeline, and report
the GPU busy fraction, kernel launch counts, and the top kernels by total
time. This is the RFC #7389 §6 "GPU idle fraction and launch overhead per
tick" evidence computed from the capture window that
``profile_personaplex_nsys.py`` bracketed around a paced load run.

Typical: python analyze_personaplex_nsys.py --reports /root/pplex_nsys
"""

import argparse
import json
import sqlite3
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

BUSY_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET")


def _string_ids(conn: sqlite3.Connection) -> dict[int, str]:
    try:
        rows = conn.execute("SELECT id, value FROM StringIds").fetchall()
        return {i: v for i, v in rows}
    except sqlite3.OperationalError:
        return {}


def _merged_busy(intervals: list[tuple[int, int]]) -> tuple[int, int]:
    """Union length of the intervals plus the span they cover."""
    if not intervals:
        return 0, 0
    total = 0
    begin, end = intervals[0]
    for start, stop in intervals[1:]:
        if start > end:
            total += end - begin
            begin, end = start, stop
        else:
            end = max(end, stop)
    total += end - begin
    return total, end - min(s for s, _ in intervals)


def analyze_report(rep: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / (rep.stem + ".sqlite")
        completed = subprocess.run(
            ["nsys", "export", "--type", "sqlite", "-o", str(db), "--force-overwrite", "true", str(rep)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode != 0 or not db.is_file():
            return {"report": rep.name, "error": completed.stderr[-500:] or "export failed"}
        conn = sqlite3.connect(db)
        strings = _string_ids(conn)
        intervals: list[tuple[int, int]] = []
        launches = 0
        by_name: dict[str, list[int]] = defaultdict(list)
        for table in BUSY_TABLES:
            try:
                rows = conn.execute(f"SELECT start, end, shortName FROM {table}").fetchall()
            except sqlite3.OperationalError:
                continue
            launches += len(rows)
            for start, end, name_id in rows:
                intervals.append((int(start), int(end)))
                by_name[strings.get(int(name_id), f"id:{name_id}")].append(int(end) - int(start))
        conn.close()

    busy_ns, span_ns = _merged_busy(sorted(intervals))
    top = sorted(((name, len(durs), sum(durs)) for name, durs in by_name.items()), key=lambda r: -r[2])[:12]
    return {
        "report": rep.name,
        "capture_span_s": span_ns / 1e9,
        "gpu_busy_s": busy_ns / 1e9,
        "gpu_idle_fraction": 1.0 - (busy_ns / span_ns) if span_ns else None,
        "gpu_launches": launches,
        "launches_per_s": launches / (span_ns / 1e9) if span_ns else None,
        "top_kernels": [{"name": name[:120], "count": count, "total_ms": total / 1e6} for name, count, total in top],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, default=Path("/root/pplex_nsys"))
    args = p.parse_args()
    results = [analyze_report(rep) for rep in sorted(args.reports.rglob("*.nsys-rep"))]
    for result in results:
        if "error" in result:
            print(f"{result['report']}: ERROR {result['error']}")
            continue
        print(
            f"{result['report']}: span={result['capture_span_s']:.1f}s busy={result['gpu_busy_s']:.1f}s "
            f"idle={result['gpu_idle_fraction']:.1%} launches={result['gpu_launches']} "
            f"({result['launches_per_s']:.0f}/s)"
        )
    out = args.reports / "nsys_analysis.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"details: {out}")


if __name__ == "__main__":
    main()
