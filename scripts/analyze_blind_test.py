#!/usr/bin/env python3
"""Analyze blind evaluation responses and compute statistical metrics.

Calculates LoRA vs Base win counts, ties, win rates among decisive choices,
exact two-sided binomial p-values (H0: p=0.5), and 95% Wilson confidence intervals.
Outputs summary table and writes outputs/blind_test_summary.csv.
"""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def compute_stats(lora_wins: int, base_wins: int) -> Tuple[float, float, float, float]:
    """Compute decisive win rate, two-sided binomial p-value vs 0.5, and 95% Wilson CI.

    Args:
        lora_wins: Count of decisive LoRA preferences.
        base_wins: Count of decisive Base preferences.

    Returns:
        (win_rate, p_value, ci_low, ci_high)
    """
    n_decisive = lora_wins + base_wins
    if n_decisive == 0:
        return 0.0, 1.0, 0.0, 1.0

    win_rate = lora_wins / n_decisive

    try:
        from scipy.stats import binomtest

        res = binomtest(k=lora_wins, n=n_decisive, p=0.5, alternative="two-sided")
        p_val = float(res.pvalue)
        ci = res.proportion_ci(confidence_level=0.95, method="wilson")
        ci_low = float(ci.low)
        ci_high = float(ci.high)
    except Exception:
        # Exact closed-form Wilson score interval formula fallback
        import math

        z = 1.959963984540054
        denom = 1.0 + (z * z) / n_decisive
        center = (win_rate + (z * z) / (2.0 * n_decisive)) / denom
        radicand = (win_rate * (1.0 - win_rate) / n_decisive) + (z * z) / (4.0 * n_decisive * n_decisive)
        margin = (z / denom) * math.sqrt(max(0.0, radicand))
        ci_low = max(0.0, center - margin)
        ci_high = min(1.0, center + margin)

        # Fallback two-sided binomial p-value using combination sum
        from math import comb

        p_k = sum(comb(n_decisive, i) * (0.5**n_decisive) for i in range(n_decisive + 1) if comb(n_decisive, i) <= comb(n_decisive, lora_wins))
        p_val = min(1.0, float(p_k))

    return win_rate, p_val, ci_low, ci_high


def load_responses(csv_path: Union[str, Path]) -> List[Dict[str, str]]:
    """Load responses from CSV file."""
    p = Path(csv_path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def analyze_responses(
    responses: List[Dict[str, str]],
    out_csv: Optional[Union[str, Path]] = "outputs/blind_test_summary.csv",
) -> List[Dict[str, Any]]:
    """Analyze blind test responses per scene and overall.

    Args:
        responses: List of response dicts with keys: rater_id, scene, winner.
        out_csv: Destination path for summary CSV.

    Returns:
        List of summary dicts for each scene and overall.
    """
    if not responses:
        print("No responses found to analyze.")
        return []

    # Grouping
    raters_all = set()
    rater_counts = defaultdict(int)

    scene_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "raters": set(),
        "total": 0,
        "lora_wins": 0,
        "base_wins": 0,
        "ties": 0,
    })

    for r in responses:
        rid = r.get("rater_id", "").strip()
        sc = r.get("scene", "").strip() or "unknown"
        w = r.get("winner", "").strip().lower()

        raters_all.add(rid)
        rater_counts[rid] += 1

        d = scene_data[sc]
        d["raters"].add(rid)
        d["total"] += 1

        if w == "lora":
            d["lora_wins"] += 1
        elif w == "base":
            d["base_wins"] += 1
        else:
            d["ties"] += 1

    summary_rows = []

    # Per-scene stats
    for sc in sorted(scene_data.keys()):
        d = scene_data[sc]
        w_rate, p_val, ci_l, ci_u = compute_stats(d["lora_wins"], d["base_wins"])
        summary_rows.append({
            "scene": sc,
            "raters": len(d["raters"]),
            "answers": d["total"],
            "lora_wins": d["lora_wins"],
            "base_wins": d["base_wins"],
            "ties": d["ties"],
            "win_rate": round(w_rate, 4),
            "p_value": round(p_val, 4),
            "ci_lower": round(ci_l, 4),
            "ci_upper": round(ci_u, 4),
        })

    # Overall stats
    tot_answers = sum(d["total"] for d in scene_data.values())
    tot_lora = sum(d["lora_wins"] for d in scene_data.values())
    tot_base = sum(d["base_wins"] for d in scene_data.values())
    tot_ties = sum(d["ties"] for d in scene_data.values())
    ov_rate, ov_p, ov_cil, ov_ciu = compute_stats(tot_lora, tot_base)

    summary_rows.append({
        "scene": "OVERALL",
        "raters": len(raters_all),
        "answers": tot_answers,
        "lora_wins": tot_lora,
        "base_wins": tot_base,
        "ties": tot_ties,
        "win_rate": round(ov_rate, 4),
        "p_value": round(ov_p, 4),
        "ci_lower": round(ov_cil, 4),
        "ci_upper": round(ov_ciu, 4),
    })

    # 1. Print Main Table
    print("\n" + "=" * 105)
    print("Blind Test Analysis Summary")
    print("=" * 105)
    print(
        f"{'Scene':<16} {'Raters':<8} {'Answers':<9} {'LoRA Wins':<11} {'Base Wins':<11} {'Ties':<8} "
        f"{'Win Rate':<10} {'p-value':<9} {'95% Wilson CI':<16}"
    )
    print("-" * 105)

    for row in summary_rows:
        ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        print(
            f"{row['scene']:<16} {row['raters']:<8} {row['answers']:<9} {row['lora_wins']:<11} "
            f"{row['base_wins']:<11} {row['ties']:<8} {row['win_rate']:<10.3f} {row['p_value']:<9.4f} "
            f"{ci_str:<16}"
        )

    print("=" * 105)

    # 2. Print Per-Rater Counts
    print("\nPer-Rater Answer Counts:")
    for rid, cnt in sorted(rater_counts.items(), key=lambda x: -x[1]):
        short_id = rid[:8] + "..." if len(rid) > 12 else rid
        print(f"  Rater {short_id}: {cnt} answers")

    print("\n⚠️ Note: The sample is small. Treat p-values and win rates as preliminary exploratory evidence.")

    # 3. Write summary CSV
    if out_csv:
        out_p = Path(out_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "scene",
                "raters",
                "answers",
                "lora_wins",
                "base_wins",
                "ties",
                "win_rate",
                "p_value",
                "ci_lower",
                "ci_upper",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary written to: {out_p}")

    return summary_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze blind evaluation responses.")
    parser.add_argument("--responses", type=str, default="results/blind_test_responses.csv", help="Responses CSV path")
    parser.add_argument("--out", type=str, default="outputs/blind_test_summary.csv", help="Destination summary CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    responses = load_responses(args.responses)
    analyze_responses(responses, out_csv=args.out)


if __name__ == "__main__":
    main()
