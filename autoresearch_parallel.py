#!/usr/bin/env python3
"""
AutoResearch Parallel — Run 4 GRPO experiments simultaneously on 4 GPUs.
Each iteration proposes 4 mutations, tests all in parallel, keeps the best.
Uses real eval every 3 iterations (since we test 4x per iteration = 12 experiments per eval).
"""
import os, sys, json, time, re, shutil, subprocess, copy, threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Vertex AI imports
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GRequest
import requests as http_requests

CREDENTIALS_PATH = "/root/vertex_credentials.json"
PROJECT = "ondemand-421015"
REGION = "us-east5"
MODEL = "claude-opus-4-6"

MUTATION_SYSTEM_PROMPT = """You are an AI research assistant optimizing GRPO hyperparameters for a 0.8B parameter function-calling model evaluated across 19 BFCL categories.

Your job: propose 4 DIFFERENT modifications to a JSON config file, each changing ONE parameter. Return all 4 as separate configs.

Parameters you can modify:
- lr: learning rate (current best found: 3e-6)
- beta: KL penalty coefficient
- num_generations: completions per prompt for GRPO
- batch_size: prompts per batch
- grad_accum: gradient accumulation steps
- max_new_tokens: max generation length
- temperature: sampling temperature
- weight_decay: AdamW weight decay
- warmup_steps: LR warmup steps
- max_steps: max training steps
- max_samples: dataset samples to use
- top_p: nucleus sampling threshold
- format_bonus: bonus for bracket format output (keep LOW ~0.1 to preserve irrelevance detection)
- reward_threshold: min reward to compute gradient
- max_grad_norm: gradient clipping norm

Rules:
1. Each config changes ONLY ONE parameter from the current best
2. Make 4 DIVERSE changes — don't test similar values
3. IMPORTANT: Keep format_bonus LOW (0.05-0.2) — high values destroy irrelevance detection
4. The model is evaluated on 19 categories including irrelevance, multi-turn, memory
5. Lower learning rates (3e-6) have worked better than high ones (8e-6)

Return in this EXACT format:

CHANGE_1: [description]
```json
{config 1}
```

CHANGE_2: [description]
```json
{config 2}
```

CHANGE_3: [description]
```json
{config 3}
```

CHANGE_4: [description]
```json
{config 4}
```"""


def get_credentials():
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(GRequest())
    return creds


def call_claude(creds, system_prompt, user_prompt, max_tokens=4096):
    if creds.expired:
        creds.refresh(GRequest())
    url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/publishers/anthropic/models/{MODEL}:rawPredict"
    r = http_requests.post(url,
        headers={"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"},
        json={"anthropic_version": "vertex-2023-10-16", "messages": [{"role": "user", "content": user_prompt}], "system": system_prompt, "max_tokens": max_tokens},
        timeout=300)
    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return ""


def run_training(gpu_id, model_path, dataset_path, output_dir, config, train_minutes=2):
    """Run a single GRPO training experiment on a specific GPU."""
    # Write config
    config_path = os.path.join(output_dir, "grpo_config.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Copy train script
    train_script = "/root/autoresearch_v14/scripts/train_grpo.py"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, train_script,
        "--model", model_path,
        "--dataset", dataset_path,
        "--output-dir", output_dir,
        "--max-minutes", str(train_minutes),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=train_minutes*60+120)
        # Read metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            return metrics.get("final_reward_avg20", 0.0), metrics
        return 0.0, {}
    except Exception as e:
        print(f"  [GPU {gpu_id}] Training error: {e}")
        return 0.0, {}


def run_eval(gpu_id, model_path, test_data_path, output_path):
    """Run eval on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [sys.executable, "/tmp/eval_fixed_scorer.py", model_path, test_data_path, output_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        if os.path.exists(output_path):
            with open(output_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"  [EVAL GPU {gpu_id}] Error: {e}")
    return None


def parse_4_configs(response, base_config):
    """Parse Claude's response into 4 configs."""
    configs = []
    descriptions = []

    for i in range(1, 5):
        marker = f"CHANGE_{i}:"
        if marker in response:
            desc_start = response.index(marker) + len(marker)
            desc_end = response.find("```json", desc_start)
            if desc_end > desc_start:
                desc = response[desc_start:desc_end].strip()
                json_start = desc_end + len("```json")
                json_end = response.find("```", json_start)
                if json_end > json_start:
                    try:
                        config = json.loads(response[json_start:json_end].strip())
                        configs.append(config)
                        descriptions.append(desc)
                    except json.JSONDecodeError:
                        pass

    # Fill remaining with base config if less than 4 parsed
    while len(configs) < 4:
        configs.append(copy.deepcopy(base_config))
        descriptions.append("fallback (no change)")

    return configs[:4], descriptions[:4]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AutoResearch Parallel — 4 GPUs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--work-dir", default="/root/autoresearch_parallel")
    parser.add_argument("--train-minutes", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated GPU IDs")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir = work_dir / "results"
    results_dir.mkdir(exist_ok=True)
    checkpoints_dir = work_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    creds = get_credentials()

    # Default config (from AutoResearch v3 best so far)
    best_config = {
        "lr": 3e-6,
        "beta": 0.01,
        "num_generations": 16,
        "batch_size": 2,
        "grad_accum": 4,
        "max_new_tokens": 512,
        "temperature": 0.6,
        "weight_decay": 0.005,
        "warmup_steps": 10,
        "max_steps": 450,
        "max_samples": 3000,
        "top_p": 0.9,
        "format_bonus": 0.1,
        "reward_threshold": 0.3,
        "max_grad_norm": 1.5,
        "max_prompt_tokens": 3072
    }

    # Load or initialize history
    history_path = work_dir / "experiment_history.json"
    history = []
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    best_train_reward = 0.0
    best_eval_score = 0.0
    eval_interval = 3  # Run eval every 3 iterations (= 12 experiments)

    print("=" * 70)
    print("  AUTORESEARCH PARALLEL — 4 GPUs")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Train data: {args.train_data}")
    print(f"  Test data: {args.test_data}")
    print(f"  GPUs: {gpus}")
    print(f"  Train minutes: {args.train_minutes} per experiment")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Experiments per iteration: {len(gpus)}")
    print(f"  Total experiments: {args.max_iterations * len(gpus)}")
    print("=" * 70)
    sys.stdout.flush()

    t_start = time.time()

    for iteration in range(args.max_iterations):
        print(f'\n{"="*70}')
        print(f'  ITERATION {iteration} | Best train: {best_train_reward:.4f} | Best eval: {best_eval_score:.4f}')
        print(f'  Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Elapsed: {(time.time()-t_start)/60:.0f} min')
        print(f'{"="*70}')
        sys.stdout.flush()

        if iteration == 0:
            # Baseline — run same config on all 4 GPUs, take average
            print("  [BASELINE] Running default config on all GPUs...")
            configs = [copy.deepcopy(best_config)] * len(gpus)
            descriptions = ["baseline"] * len(gpus)
        else:
            # Ask Claude for 4 mutations
            print("  [MUTATE] Asking Claude for 4 mutations...")
            sys.stdout.flush()

            history_text = ""
            for h in history[-20:]:
                kept = "KEPT" if h.get("kept") else "DISC"
                history_text += f"\n  Iter {h['iteration']}: reward={h.get('train_reward',0):.4f} eval={h.get('eval_score','N/A')} [{kept}] {h.get('change','?')[:80]}"

            user_prompt = (
                f"Iteration: {iteration}\n"
                f"Best train reward: {best_train_reward:.4f}\n"
                f"Best eval score: {best_eval_score:.4f}\n\n"
                f"EXPERIMENT HISTORY:\n{history_text}\n\n"
                "CURRENT BEST CONFIG:\n```json\n"
                + json.dumps(best_config, indent=2) +
                "\n```\n\n"
                "Propose 4 DIFFERENT single-parameter modifications. Be creative — try values we haven't tested."
            )

            try:
                response = call_claude(creds, MUTATION_SYSTEM_PROMPT, user_prompt, max_tokens=4000)
                configs, descriptions = parse_4_configs(response, best_config)
                for i, (c, d) in enumerate(zip(configs, descriptions)):
                    print(f"  [GPU {gpus[i]}] {d[:80]}")
            except Exception as e:
                print(f"  [MUTATE] Error: {e}")
                configs = [copy.deepcopy(best_config)] * len(gpus)
                descriptions = ["fallback"] * len(gpus)

        # Run all 4 experiments in parallel
        print(f"\n  [TRAIN] Running {len(gpus)} experiments in parallel ({args.train_minutes} min each)...")
        sys.stdout.flush()

        results = [None] * len(gpus)
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = {}
            for i, gpu_id in enumerate(gpus):
                out_dir = str(checkpoints_dir / f"iter_{iteration:04d}_gpu{gpu_id}")
                future = executor.submit(
                    run_training, gpu_id, args.model, args.train_data, out_dir, configs[i], args.train_minutes
                )
                futures[future] = i

            for future in as_completed(futures):
                i = futures[future]
                reward, metrics = future.result()
                results[i] = {"reward": reward, "metrics": metrics, "config": configs[i], "description": descriptions[i], "gpu": gpus[i]}
                print(f"  [GPU {gpus[i]}] reward={reward:.4f} | {descriptions[i][:60]}")

        # Find best from this iteration
        best_this = max(results, key=lambda r: r["reward"])
        print(f"\n  [BEST THIS ITER] GPU {best_this['gpu']}: reward={best_this['reward']:.4f} | {best_this['description'][:60]}")

        # Run eval every N iterations
        eval_score = None
        if iteration % eval_interval == 0:
            best_model_path = str(checkpoints_dir / f"iter_{iteration:04d}_gpu{best_this['gpu']}")
            eval_output = str(results_dir / f"eval_iter_{iteration:04d}.json")
            print(f"  [EVAL] Running 190-sample eval on best config...")
            sys.stdout.flush()
            eval_data = run_eval(gpus[0], best_model_path, args.test_data, eval_output)
            if eval_data:
                eval_score = eval_data.get("overall_accuracy", 0.0)
                correct = eval_data.get("correct", 0)
                total = eval_data.get("total", 0)
                print(f"  [EVAL] Score: {eval_score:.4f} ({correct}/{total})")
                cat_scores = eval_data.get("category_scores", {})
                for cat in sorted(cat_scores.keys()):
                    print(f"    {cat}: {cat_scores[cat]:.3f}")

        # Ratchet decision
        if eval_score is not None:
            score = eval_score
            source = "EVAL"
        else:
            score = best_this["reward"]
            source = "TRAIN"

        improved = score > (best_eval_score if eval_score is not None else best_train_reward)

        if improved:
            if eval_score is not None:
                best_eval_score = eval_score
            best_train_reward = best_this["reward"]
            best_config = best_this["config"]
            print(f"\n  [RESULT] IMPROVEMENT! {source}={score:.4f} | {best_this['description']}")
        else:
            print(f"\n  [RESULT] No improvement ({source}={score:.4f})")

        # Save all results to history
        for r in results:
            history.append({
                "iteration": iteration,
                "change": r["description"],
                "train_reward": r["reward"],
                "eval_score": eval_score if r == best_this else None,
                "kept": r == best_this and improved,
                "gpu": r["gpu"],
                "config": r["config"],
                "timestamp": datetime.now().isoformat(),
            })

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save best config
        with open(work_dir / "best_config.json", "w") as f:
            json.dump(best_config, f, indent=2)

        sys.stdout.flush()

    elapsed = time.time() - t_start
    print(f'\n{"="*70}')
    print(f'  AUTORESEARCH PARALLEL COMPLETE')
    print(f'  Time: {elapsed/60:.0f} min ({elapsed/3600:.1f} hours)')
    print(f'  Total experiments: {args.max_iterations * len(gpus)}')
    print(f'  Best train reward: {best_train_reward:.4f}')
    print(f'  Best eval score: {best_eval_score:.4f}')
    print(f'  Best config saved to: {work_dir / "best_config.json"}')
    print(f'{"="*70}')

    print("\nBest config:")
    print(json.dumps(best_config, indent=2))


if __name__ == "__main__":
    main()
