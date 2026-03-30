import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from tqdm import tqdm

COCOEVAL_SCRIPT = "evaluation/cocoeval.py"
PYTHON_BIN = "python"  # If necessary, it can be changed to an absolute path.

metric_patterns = {
    "Bleu_1": r"Bleu_1:\s*([0-9.]+)",
    "Bleu_2": r"Bleu_2:\s*([0-9.]+)",
    "Bleu_3": r"Bleu_3:\s*([0-9.]+)",
    "Bleu_4": r"Bleu_4:\s*([0-9.]+)",
    "METEOR": r"METEOR:\s*([0-9.]+)",
    "ROUGE_L": r"ROUGE_L:\s*([0-9.]+)",
    "CIDEr": r"CIDEr:\s*([0-9.]+)",
    "SPICE": r"SPICE:\s*([0-9.]+)",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_groups(data):
    order = []
    for item in data:
        if item["image_name"] not in order:
            order.append(item["image_name"])

    group_size = len(order)
    groups = []

    for i in range(0, len(data), group_size):
        groups.append(data[i:i + group_size])

    return groups


def run_cocoeval(json_path):
    cmd = [
        PYTHON_BIN,
        COCOEVAL_SCRIPT,
        "--result_file_path",
        json_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )

    return result.stdout


def parse_metrics(output_text):
    metrics = {}
    for name, pattern in metric_patterns.items():
        match = re.search(pattern, output_text)
        if match:
            metrics[name] = float(match.group(1))
        else:
            raise ValueError(f"No metrics found in the output {name}")

    return metrics


def average_metrics(metrics_list):
    avg = defaultdict(float)
    for m in metrics_list:
        for k, v in m.items():
            avg[k] += v

    for k in avg:
        avg[k] /= len(metrics_list)

    return dict(avg)


def main(total_json_path):
    data = load_json(total_json_path)
    groups = split_groups(data)

    all_group_metrics = []

    temp_files = []

    try:
        for idx, group in enumerate(
            tqdm(groups, desc="Eval", unit="group")
        ):
            if len(group) in (6, 61):
                continue
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=f"_group_{idx}.json",
                delete=False,
                encoding="utf-8",
            ) as f:
                json.dump(group, f, ensure_ascii=False, indent=2)
                temp_json_path = f.name
                temp_files.append(temp_json_path)

            output = run_cocoeval(temp_json_path)
            metrics = parse_metrics(output)
            all_group_metrics.append(metrics)

        avg_metrics = average_metrics(all_group_metrics)

        print("\n========== Metrics ==========")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.4f}")

    finally:
        # Clean up temp files
        for path in temp_files:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python group_eval_and_average.py total.json")
        sys.exit(1)

    main(sys.argv[1])
