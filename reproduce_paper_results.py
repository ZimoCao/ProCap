import subprocess
import itertools
import sys
import time
import os
import re
import pandas as pd

TRAIN_SCRIPT_PATH = "scripts/train_procap.sh"
EVAL_SCRIPT_PATH = "scripts/eval_procap.sh"
GPUS = "1,2,3"
TASK = "all"
MODEL_TYPES = [
    "TinyLlama/TinyLlama-1.1B-step-50K-105b",
    "facebook/opt-2.7b",
    "openlm-research/open_llama_3b",
    "lmsys/vicuna-7b-v1.5",
    "allura-forge/Llama-3.3-8B-Instruct",
]
DATASETS = [
    "coco",
    "nocaps",
    "whoops"
]
SCENE_FLAGS = [
    "--unseen_scene",
    "--seen_scene",
    "--newsetting",
]
STATES = [
    "refinement",
    "mask",
    "scene_qformer",
    "proj_qformer"
]

def run_train():
    total_jobs = len(MODEL_TYPES)
    print(f"Total configurations to run: {total_jobs}")
    print("=" * 60)

    for index, model_type in enumerate(MODEL_TYPES, 1):
        print(f"[{index}/{total_jobs}] Starting Task:")
        print(f"  Model:   {model_type}")
        print("-" * 60)

        cmd = [
            "bash", TRAIN_SCRIPT_PATH,
            "--gpus", GPUS,
            "--model_type", model_type
        ]

        try:
            subprocess.run(cmd, check=True)
            
            print(f"\n[SUCCESS] Completed: {model_type}")
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Failed execution for: {model_type}")
            print(f"Exit code: {e.returncode}")
        
        print("=" * 60)
        time.sleep(2)

def run_evaluation():
    combinations = list(itertools.product(MODEL_TYPES, DATASETS, SCENE_FLAGS))

    total_jobs = len(combinations)
    print(f"Total configurations to run: {total_jobs}")
    print("=" * 60)

    for index, (model_type, dataset, scene_flag) in enumerate(combinations, 1):
        print(f"[{index}/{total_jobs}] Starting Task:")
        print(f"  Model:   {model_type}")
        print(f"  Dataset: {dataset}")
        print(f"  Scene:   {scene_flag}")
        print("-" * 60)

        cmd = [
            "bash", EVAL_SCRIPT_PATH,
            "--gpus", GPUS,
            "--model_type", model_type,
            "--dataset", dataset,
            "--task", TASK,
            scene_flag 
        ]

        try:
            subprocess.run(cmd, check=True)
            
            print(f"\n[SUCCESS] Completed: {model_type} | {dataset} | {TASK} | {scene_flag}")
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Failed execution for: {model_type} | {dataset} | {TASK} | {scene_flag}")
            print(f"Exit code: {e.returncode}")
        
        print("=" * 60)
        time.sleep(2)

def run_ablation():
    model_type = ["openlm-research/open_llama_3b"]
    combinations = list(itertools.product(model_type, STATES))

    total_jobs = len(combinations)
    print(f"Total configurations to run: {total_jobs}")
    print("=" * 60)

    for index, (model_type, disable_state) in enumerate(combinations, 1):
        print(f"[{index}/{total_jobs}] Starting Task:")
        print(f"  Model:   {model_type}")
        print(f"  Disable: {disable_state}")
        print("-" * 60)

        cmd = [
            "bash", TRAIN_SCRIPT_PATH,
            "--gpus", GPUS,
            "--model_type", model_type
        ]

        cmd.append(f"--no_{disable_state}")

        try:
            subprocess.run(cmd, check=True)
            
            print(f"\n[SUCCESS] Completed: {model_type} | {disable_state}")
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Failed execution for: {model_type} | {disable_state}")
            print(f"Exit code: {e.returncode}")
        
        print("=" * 60)
        time.sleep(2)
 
    model_type = ["openlm-research/open_llama_3b"]
    combinations = list(itertools.product(model_type, STATES, DATASETS, SCENE_FLAGS))

    total_jobs = len(combinations)
    print(f"Total configurations to run: {total_jobs}")
    print("=" * 60)

    for index, (model_type, disable_state, dataset, scene_flag) in enumerate(combinations, 1):
        print(f"[{index}/{total_jobs}] Starting Task:")
        print(f"  Model:   {model_type}")
        print(f"  Disable: {disable_state}")
        print(f"  Dataset: {dataset}")
        print(f"  Scene:   {scene_flag}")
        print("-" * 60)

        cmd = [
            "bash", EVAL_SCRIPT_PATH,
            "--gpus", GPUS,
            "--model_type", model_type,
            "--dataset", dataset,
            "--task", "projection" if disable_state == "scene_qformer" else ("scene" if disable_state == "proj_qformer" else TASK),
            scene_flag 
        ]

        cmd.append(f"--no_{disable_state}")

        try:
            subprocess.run(cmd, check=True)
            
            print(f"\n[SUCCESS] Completed: {model_type} | {disable_state} | {dataset} | {scene_flag}")
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Failed execution for: {model_type} | {disable_state} | {dataset} | {scene_flag}")
            print(f"Exit code: {e.returncode}")
        
        print("=" * 60)
        time.sleep(2)

def run_metric_computation():
    json_file_folder_path = '/path/to/your/directory/results/'
    output_excel = 'metrics.xlsx'
    
    rows = []

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
    
    json_files = []

    for root, _, files in os.walk(json_file_folder_path):
        for fname in files:
            if fname.endswith(".json"):
                json_files.append(os.path.join(root, fname))

    json_files.sort()

    for json_file_path in json_files:
        print(f"Processing: {json_file_path}")

        cmd = [
            "python", "group_eval_and_average.py",
            json_file_path
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            for line in process.stdout:
                print(line, end="") 
                output_lines.append(line)

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            output = "".join(output_lines)

        except subprocess.CalledProcessError:
            print(f"Failed: {json_file_path}")
            continue
        
        row = {
            "file": json_file_path
        }

        for metric, pattern in metric_patterns.items():
            match = re.search(pattern, output)
            row[metric] = float(match.group(1)) if match else None

        rows.append(row)
        
        # print(f"Processed: {json_file_path}")
        # print("=" * 60)
        # input("Press Enter to continue processing...")
    
    df = pd.DataFrame(rows)
    df.to_excel(output_excel, index=False)
    print(f"\nSaved to {output_excel}")

if __name__ == "__main__":
    run_train()
    run_evaluation()
    run_ablation()
    run_metric_computation()
