import json
import os


def load_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_comparison():
    baseline_path = os.getenv("BASELINE_METRICS", "baseline/metrics.json")
    current_path = os.getenv("CURRENT_METRICS", "metrics.json")

    baseline = load_metrics(baseline_path)
    current = load_metrics(current_path)

    if not current:
        print("Error: Current metrics not found at", current_path)
        return

    print(
        "| Metric | Baseline (main) | Current (PR) | Delta (Δ) |\n| --- | --- | --- | --- |"
    )

    if not baseline:
        print("| **Note** | *Baseline not found* | - | - |")
        for k, v in current.items():
            print(f"| {k} | - | {v:.4f} | - |")
        return

    all_keys = set(baseline.keys()).union(set(current.keys()))

    for k in sorted(all_keys):
        b_val = baseline.get(k, 0)
        c_val = current.get(k, 0)
        delta = c_val - b_val
        delta_str = f"{delta:+.4f}" if delta != 0 else "--"
        print(f"| {k} | {b_val:.4f} | {c_val:.4f} | {delta_str} |")


if __name__ == "__main__":
    run_comparison()
