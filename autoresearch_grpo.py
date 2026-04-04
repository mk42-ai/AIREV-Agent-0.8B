#!/usr/bin/env python3
"""
AutoResearch GRPO — Karpathy-style automated research loop for GRPO hyperparameter search.

Architecture:
  1. A FROZEN eval.py that runs 100 BFCL test cases and returns accuracy
  2. An EDITABLE train_grpo.py (shortened GRPO that runs for 5 minutes)
  3. A Claude mutation agent (via Vertex AI) that reads the code, proposes changes,
     and decides what to modify next based on eval results
  4. Git-based checkpoint management: keep improvements, discard regressions

The mutation agent can modify:
  - Reward function weights
  - Learning rate, beta, temperature
  - Curriculum phase timing
  - Number of generations
  - Diversity penalty thresholds
  - Thinking reward components
  - Any other training hyperparameter

Vertex AI credentials:
  - Project: ondemand-421015
  - Location: us-east5
  - Model: claude-opus-4-6
  - SA: ondevdemand@ondemand-421015.iam.gserviceaccount.com

Usage:
  python autoresearch_grpo.py \\
    --model /root/checkpoints/sft_v14/best \\
    --train-data /root/datasets/bfcl_90k_train.jsonl \\
    --test-data /root/datasets/bfcl_100_test.jsonl \\
    --work-dir /root/autoresearch_v14 \\
    --max-iterations 50
"""
import os
import sys
import json
import time
import shutil
import subprocess
import argparse
import hashlib
from datetime import datetime
from pathlib import Path

# ===========================================================================
# Vertex AI / Claude API Setup
# ===========================================================================
def get_claude_client():
    """Initialize Vertex AI credentials."""
    import json as _json
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    creds = _json.load(open("/root/vertex_credentials.json"))
    credentials = service_account.Credentials.from_service_account_info(
        creds, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    return credentials

def call_claude(client, system_prompt, user_prompt, max_tokens=4096):
    """Call Claude via Vertex AI REST API."""
    import requests as _req
    from google.auth.transport.requests import Request as _R
    if client.expired:
        client.refresh(_R())
    url = "https://us-east5-aiplatform.googleapis.com/v1/projects/ondemand-421015/locations/us-east5/publishers/anthropic/models/claude-opus-4-6:rawPredict"
    r = _req.post(url, headers={"Authorization": "Bearer " + client.token, "Content-Type": "application/json"},
        json={"anthropic_version": "vertex-2023-10-16", "messages": [{"role": "user", "content": user_prompt}], "system": system_prompt, "max_tokens": max_tokens}, timeout=300)
    if r.status_code != 200:
        raise Exception("HTTP " + str(r.status_code) + ": " + r.text[:200])
    data = r.json()
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return str(data.get("content", ""))

# ===========================================================================
# Frozen Eval Script (written to disk, never modified)
# ===========================================================================
EVAL_SCRIPT = r'''
#!/usr/bin/env python3
"""Frozen eval script — BFCL official format evaluation."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")

import json, sys, argparse, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BFCL_SYSTEM_PROMPT = "You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\nIf none of the functions can be used, point it out and refuse to answer.\nIf the given question lacks the parameters required by the function, also point it out.\n\nYou SHOULD NOT:\n- Generate any content that is not a function call or a reasoning process.\n\nYou should ONLY interact with the presented APIs using following format:\n\n[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)...]\n\nYou SHOULD NOT:\n1. Use any APIs that are not provided to you.\n2. Generate any content that is not a function call.\n3. Generate excessive content."

try:
    from bfcl_eval.model_handler.utils import default_decode_ast_prompting
    HAS_BFCL = True
    print("[INFO] BFCL decoder available")
except ImportError:
    HAS_BFCL = False
    print("[WARN] BFCL decoder not available, using regex scoring")


def format_functions(functions):
    """Format function schemas for the system prompt."""
    parts = []
    for func in functions:
        parts.append(json.dumps(func))
    return "\n\nHere is a list of functions in JSON format that you can invoke:\n" + "\n".join(parts)


def build_prompt(question_turns, functions, tokenizer):
    """Build a prompt from BFCL test data format."""
    turns = question_turns[0] if isinstance(question_turns[0], list) else question_turns
    system_msg = BFCL_SYSTEM_PROMPT + format_functions(functions)
    messages = [{"role": "system", "content": system_msg}]
    for turn in turns:
        messages.append(turn)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not prompt.endswith("\n"):
        prompt += "\n"
    prompt += "<think>\n"
    return prompt


def extract_answer(text):
    """Extract function call from model output, stripping thinking."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        text = text.split("<think>")[0].strip()
    return text.strip()


def simple_score(prediction, ground_truth):
    """Score prediction against ground truth."""
    pred = prediction.strip()

    # Try BFCL decoder
    if HAS_BFCL:
        try:
            decoded = default_decode_ast_prompting(pred, {})
            if decoded and len(decoded) > 0:
                if len(decoded) == len(ground_truth):
                    gt_names = set()
                    for gt_call in ground_truth:
                        if isinstance(gt_call, dict):
                            gt_names.update(gt_call.keys())
                    pred_names = set()
                    for p_call in decoded:
                        if isinstance(p_call, tuple) and len(p_call) >= 1:
                            pred_names.add(p_call[0])
                        elif isinstance(p_call, dict):
                            pred_names.update(p_call.keys())
                    if gt_names == pred_names:
                        return 1.0
                    return 0.5
                return 0.25
        except Exception:
            pass

    # Fallback: regex matching
    if pred.startswith("[") and "]" in pred:
        func_calls = re.findall(r'(\w+)\s*\(', pred)
        gt_names = []
        for gt_call in ground_truth:
            if isinstance(gt_call, dict):
                gt_names.extend(gt_call.keys())
        if set(func_calls) == set(gt_names) and len(func_calls) == len(gt_names):
            return 1.0
        elif len(func_calls) > 0:
            overlap = len(set(func_calls) & set(gt_names))
            return overlap / max(len(gt_names), 1) * 0.8

    return 0.0


def evaluate(model_path, test_data_path, max_samples=100):
    """Run BFCL evaluation."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    samples = []
    with open(test_data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    samples = samples[:max_samples]

    scores = []
    for idx, sample in enumerate(samples):
        if "question" in sample and "function" in sample:
            prompt_text = build_prompt(sample["question"], sample["function"], tokenizer)
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
            gold = sample.get("ground_truth", [])
        elif "messages" in sample:
            msgs = sample["messages"]
            system_text = ""
            user_text = ""
            gold_str = ""
            for m in msgs:
                if m["role"] == "system":
                    system_text = m["content"]
                elif m["role"] == "user":
                    user_text = m["content"]
                elif m["role"] == "assistant":
                    gold_str = m["content"]
            prompt_text = (
                "<|im_start|>system\n" + BFCL_SYSTEM_PROMPT + "\n" + system_text + "<|im_end|>\n"
                "<|im_start|>user\n" + user_text + "<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
            )
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
            gold = gold_str
        else:
            print(f"  Sample {idx}: unknown format, skipping")
            continue

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.01,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        answer = extract_answer(response)

        if isinstance(gold, str):
            score = 1.0 if answer.strip() == gold.strip() else 0.0
            if score == 0.0 and gold.strip() in answer:
                score = 0.5
        elif isinstance(gold, list):
            score = simple_score(answer, gold)
        else:
            score = 0.0

        scores.append(score)
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(samples)}] acc={sum(scores)/len(scores):.3f} | last={answer[:80]}")

    accuracy = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  FINAL ACCURACY: {accuracy:.4f} ({sum(s >= 0.5 for s in scores)}/{len(scores)} correct)")

    return {
        "accuracy": accuracy,
        "total": len(scores),
        "correct": sum(s >= 0.5 for s in scores),
        "perfect": sum(s >= 1.0 for s in scores),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results = evaluate(args.model, args.test_data, args.max_samples)
    print(json.dumps(results, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

'''


# ===========================================================================
# AutoResearch Loop
# ===========================================================================
MUTATION_SYSTEM_PROMPT = """You are an AI research assistant optimizing GRPO (Group Relative Policy Optimization) hyperparameters for a 0.8B parameter function-calling model.

Your job: modify a JSON config file to improve training reward. The config controls these hyperparameters:

- lr: learning rate (default 3e-7)
- beta: KL penalty coefficient (default 0.01)
- num_generations: completions per prompt for GRPO (default 4)
- batch_size: prompts per batch (default 2)
- grad_accum: gradient accumulation steps (default 2)
- max_new_tokens: max generation length (default 512)
- temperature: sampling temperature (default 0.7)
- weight_decay: AdamW weight decay (default 0.01)
- warmup_steps: LR warmup steps (default 10)
- max_steps: max training steps (default 200)
- max_samples: dataset samples to use (default 2000)
- top_p: nucleus sampling threshold (default 0.95)
- format_bonus: bonus for bracket format [func()] (default 0.1)
- reward_threshold: min reward to compute gradient (default 0.3)
- max_grad_norm: gradient clipping norm (default 1.0)

Rules:
1. Change ONLY ONE parameter at a time (scientific method)
2. Explain WHY you think this change will help based on the history
3. Return the COMPLETE JSON config (all parameters, even unchanged ones)
4. Keep values in reasonable ranges
5. The model is 0.8B params on a single H100 — don't set batch_size too high

Return your response in this EXACT format:

CHANGE_DESCRIPTION: [one line describing what you changed and why]

```json
{complete config JSON}
```"""


class AutoResearchLoop:
    """Karpathy-style automated research loop for GRPO tuning."""

    def __init__(self, args):
        self.args = args
        self.work_dir = Path(args.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.scripts_dir = self.work_dir / "scripts"
        self.results_dir = self.work_dir / "results"
        self.checkpoints_dir = self.work_dir / "checkpoints"
        self.scripts_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

        # Write frozen eval script
        self.eval_script_path = self.scripts_dir / "eval.py"
        with open(self.eval_script_path, "w") as f:
            f.write(EVAL_SCRIPT)

        # Initialize the editable training script from grpo_v14
        self.train_script_path = self.scripts_dir / "train_grpo.py"
        if not self.train_script_path.exists():
            # Copy the base GRPO v14 script as starting point
            base_grpo = Path("/tmp/grpo_v14_autoresearch.py")
            if base_grpo.exists():
                shutil.copy(base_grpo, self.train_script_path)
            else:
                print("ERROR: /tmp/grpo_v14_autoresearch.py not found. Run GRPO v14 script creation first.")
                sys.exit(1)

        # Git init for version control
        self._git_init()

        # Experiment history
        self.history_path = self.work_dir / "experiment_history.json"
        if self.history_path.exists():
            with open(self.history_path) as f:
                self.history = json.load(f)
        else:
            self.history = []

        self.best_accuracy = 0.0
        if self.history:
            self.best_accuracy = max(h.get("accuracy", 0) for h in self.history)

        # Claude client
        self.client = get_claude_client()

    def _git_init(self):
        """Initialize git repo in work_dir for tracking experiments."""
        git_dir = self.work_dir / ".git"
        if not git_dir.exists():
            subprocess.run(["git", "init"], cwd=self.work_dir, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=self.work_dir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial AutoResearch setup"],
                cwd=self.work_dir, capture_output=True,
            )

    def _git_commit(self, message):
        """Commit current state."""
        subprocess.run(["git", "add", "."], cwd=self.work_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.work_dir, capture_output=True,
        )

    def _git_revert(self):
        """Revert to last commit (discard failed experiment)."""
        subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=self.work_dir, capture_output=True,
        )

    
    def _save_history(self):
        """Save experiment history to JSON file."""
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def run_training(self, iteration):
        """Run the 5-minute GRPO training and return the output checkpoint path."""
        ckpt_dir = self.checkpoints_dir / f"iter_{iteration:04d}"
        ckpt_dir.mkdir(exist_ok=True)

        cmd = [
            sys.executable, str(self.train_script_path),
            "--model", self.args.model,
            "--dataset", self.args.train_data,
            "--output-dir", str(ckpt_dir),
            "--max-minutes", str(self.args.train_minutes),
        ]

        print(f"\n  [TRAIN] Running 5-min GRPO training (iter {iteration})...")
        print(f"  [TRAIN] CMD: {' '.join(cmd)}")
        sys.stdout.flush()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"
        t0 = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=self.args.train_minutes * 60 + 120,  # Extra buffer
        )
        elapsed = time.time() - t0

        # Print training output
        if result.stdout:
            # Show last 20 lines of output
            lines = result.stdout.strip().split("\n")
            for line in lines[-20:]:
                print(f"    {line}")

        if result.returncode != 0:
            print(f"  [TRAIN] FAILED (exit code {result.returncode})")
            if result.stderr:
                print(f"  [TRAIN] STDERR: {result.stderr[-500:]}")
            return None, elapsed

        # Find the best/final checkpoint
        final_path = ckpt_dir / "final"
        if final_path.exists():
            return str(final_path), elapsed

        # Fallback: find latest checkpoint
        ckpts = sorted(ckpt_dir.glob("checkpoint-*"))
        if ckpts:
            return str(ckpts[-1]), elapsed

        # Fallback: model saved directly in ckpt_dir
        if (ckpt_dir / 'model.safetensors').exists() or (ckpt_dir / 'pytorch_model.bin').exists():
            return str(ckpt_dir), elapsed

        return None, elapsed

    def run_eval(self, model_path, iteration):
        """Run frozen eval and return results."""
        output_path = self.results_dir / f"eval_iter_{iteration:04d}.json"

        cmd = [
            sys.executable, str(self.eval_script_path),
            "--model", model_path,
            "--test-data", self.args.test_data,
            "--max-samples", "100",
            "--output", str(output_path),
        ]

        print(f"\n  [EVAL] Running 100-sample BFCL eval...")
        sys.stdout.flush()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"
        t0 = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for eval
        )
        elapsed = time.time() - t0

        if result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines[-5:]:
                print(f"    {line}")

        if result.returncode != 0:
            print(f"  [EVAL] FAILED (exit code {result.returncode})")
            if result.stderr:
                print(f"  [EVAL] STDERR: {result.stderr[-500:]}")
            return None, elapsed

        if output_path.exists():
            with open(output_path) as f:
                return json.load(f), elapsed

        return None, elapsed

    def propose_mutation(self, iteration):
        """Ask Claude to modify the config JSON."""
        config_path = self.work_dir / 'grpo_config.json'
        with open(config_path) as f:
            current_config = f.read()

        history_text = 'No previous experiments yet.' if not self.history else ''
        for h in self.history[-15:]:
            history_text += (
                f'\nIteration {h["iteration"]}: '
                f'reward={h.get("accuracy", 0):.4f} | '
                f'change: {h.get("change_description", "baseline")}'
            )
            if h.get('kept', False):
                history_text += ' [KEPT]'
            else:
                history_text += ' [DISCARDED]'

        user_prompt = (
            f'Current iteration: {iteration}\n'
            f'Current best reward: {self.best_accuracy:.4f}\n\n'
            f'EXPERIMENT HISTORY:\n{history_text}\n\n'
            'CURRENT CONFIG:\n```json\n'
            + current_config +
            '\n```\n\n'
            'Propose ONE targeted modification to improve training reward.'
        )

        print(f'\n  [MUTATE] Asking Claude for config modification...')
        sys.stdout.flush()
        response = call_claude(self.client, MUTATION_SYSTEM_PROMPT, user_prompt, max_tokens=2000)

        change_desc = 'unknown change'
        for line in response.split('\n'):
            if line.startswith('CHANGE_DESCRIPTION:'):
                change_desc = line.split(':', 1)[1].strip()
                break

        # Extract JSON from response
        json_str = None
        for marker in ['```json', '```']:
            if marker in response:
                start = response.find(marker) + len(marker)
                if marker == '```':
                    start = response.find('\n', start) + 1
                end = response.find('```', start)
                if end > start:
                    json_str = response[start:end].strip()
                    break

        if json_str:
            try:
                new_config = json.loads(json_str)
                with open(config_path, 'w') as f:
                    json.dump(new_config, f, indent=4)
                print(f'  [MUTATE] Change: {change_desc}')
                return change_desc, new_config
            except json.JSONDecodeError as e:
                print(f'  [MUTATE] Invalid JSON: {e}')
                return change_desc, None
        return change_desc, None

    def run_iteration(self, iteration):
        print(f'\n{"="*70}')
        print(f'  AUTORESEARCH ITERATION {iteration}')
        print(f'  Current best reward: {self.best_accuracy:.4f}')
        print(f'  Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'{"="*70}')
        sys.stdout.flush()

        change_desc = 'baseline (no changes)'
        if iteration > 0:
            try:
                change_desc, new_config = self.propose_mutation(iteration)
                if new_config is None:
                    print('  [MUTATE] Failed. Skipping iteration.')
                    return
            except Exception as e:
                print(f'  [MUTATE] Error: {e}')
                return

        model_path, train_time = self.run_training(iteration)
        if not model_path:
            print('  [RESULT] Training failed. Reverting.')
            self._git_revert()
            self.history.append({'iteration': iteration, 'change_description': change_desc, 'accuracy': 0.0, 'kept': False, 'error': 'train_failed', 'timestamp': datetime.now().isoformat()})
            self._save_history()
            return

        # Read training reward from metrics.json
        metrics_path = os.path.join(model_path, 'metrics.json')
        train_reward = 0.0
        if os.path.exists(metrics_path):
            with open(metrics_path) as mf:
                metrics = json.load(mf)
            train_reward = metrics.get('final_reward_avg20', 0.0)
            print(f'  [TRAIN REWARD] {train_reward:.4f} | Steps: {metrics.get("total_steps", 0)}')
        else:
            print(f'  [METRICS] No metrics.json at {metrics_path}')

        # Every 12 iterations (~1 hour), run REAL eval as source of truth
        eval_score = None
        eval_interval = 12
        if iteration % eval_interval == 0 or iteration == 0:
            print(f'  [EVAL] Running real 100-sample eval (iteration {iteration})...')
            sys.stdout.flush()
            eval_cmd = [
                sys.executable, '/tmp/eval_fixed_scorer.py',
                model_path,
                self.args.test_data,
                str(self.results_dir / f'eval_iter_{iteration:04d}.json'),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "1"
            try:
                eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, env=env, timeout=600)
                if eval_result.returncode == 0:
                    eval_output_path = self.results_dir / f'eval_iter_{iteration:04d}.json'
                    if eval_output_path.exists():
                        with open(eval_output_path) as ef:
                            eval_data = json.load(ef)
                        eval_score = eval_data.get('overall_accuracy', 0.0)
                        # Show per-category
                        print(f'  [EVAL] Score: {eval_score:.4f} ({eval_data.get("correct",0)}/{eval_data.get("total",0)})')
                        cat_scores = eval_data.get('category_scores', {})
                        for cat in sorted(cat_scores.keys()):
                            print(f'    {cat}: {cat_scores[cat]:.3f}')
                else:
                    print(f'  [EVAL] Failed: {eval_result.stderr[-200:]}')
            except Exception as e:
                print(f'  [EVAL] Error: {e}')

        # Use eval score when available, otherwise training reward
        if eval_score is not None:
            reward = eval_score
            reward_source = 'EVAL'
        else:
            reward = train_reward
            reward_source = 'TRAIN'

        print(f'  [RATCHET] Using {reward_source} score: {reward:.4f} (best: {self.best_accuracy:.4f})')

        improved = reward > self.best_accuracy
        entry = {'iteration': iteration, 'change_description': change_desc, 'accuracy': reward, 'train_reward': train_reward, 'eval_score': eval_score, 'reward_source': reward_source, 'train_time_sec': train_time, 'model_path': model_path, 'kept': improved, 'timestamp': datetime.now().isoformat()}

        if improved:
            self.best_accuracy = reward
            self._git_commit(f'Iter {iteration}: {reward_source}={reward:.4f} (+) | {change_desc}')
            print(f'\n  [RESULT] IMPROVEMENT! {reward_source}={reward:.4f} > previous best')
        else:
            self._git_revert()
            print(f'\n  [RESULT] No improvement ({reward_source}={reward:.4f} <= {self.best_accuracy:.4f})')
            print(f'  [RESULT] Discarding: {change_desc}')

        self.history.append(entry)
        self._save_history()

    def run(self):
        """Main AutoResearch loop."""
        print("=" * 70)
        print("  AUTORESEARCH GRPO — Karpathy-Style Automated Research")
        print("=" * 70)
        print(f"  Base model:     {self.args.model}")
        print(f"  Train data:     {self.args.train_data}")
        print(f"  Test data:      {self.args.test_data}")
        print(f"  Work dir:       {self.args.work_dir}")
        print(f"  Train minutes:  {self.args.train_minutes} per iteration")
        print(f"  Max iterations: {self.args.max_iterations}")
        print(f"  Mutation agent: Claude Opus 4.6 via Vertex AI")
        print(f"  Previous runs:  {len(self.history)}")
        print(f"  Best accuracy:  {self.best_accuracy:.4f}")
        print("=" * 70)
        sys.stdout.flush()

        start_iter = len(self.history)
        t_total = time.time()

        for i in range(start_iter, start_iter + self.args.max_iterations):
            try:
                self.run_iteration(i)
            except KeyboardInterrupt:
                print("\n\n  INTERRUPTED by user. Saving state...")
                self._save_history()
                break
            except Exception as e:
                print(f"\n  ERROR in iteration {i}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next iteration
                self.history.append({
                    "iteration": i,
                    "change_description": "error",
                    "accuracy": 0.0,
                    "kept": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                self._save_history()
                self._git_revert()

        total_time = (time.time() - t_total) / 3600
        print(f"\n{'='*70}")
        print(f"  AUTORESEARCH COMPLETE")
        print(f"  Total time:     {total_time:.1f} hours")
        print(f"  Iterations:     {len(self.history)}")
        print(f"  Best accuracy:  {self.best_accuracy:.4f}")
        print(f"  History saved:  {self.history_path}")
        print(f"{'='*70}")

        # Print leaderboard
        print(f"\n  TOP 5 EXPERIMENTS:")
        sorted_history = sorted(self.history, key=lambda h: h.get("accuracy", 0), reverse=True)
        for rank, h in enumerate(sorted_history[:5], 1):
            status = "KEPT" if h.get("kept") else "discarded"
            print(f"    #{rank}: acc={h.get('accuracy', 0):.4f} | {h.get('change_description', '?')} [{status}]")


def main():
    parser = argparse.ArgumentParser(description="AutoResearch GRPO — Automated hyperparameter search")
    parser.add_argument("--model", type=str, required=True,
                        help="Base model path (SFT v14 best checkpoint)")
    parser.add_argument("--train-data", type=str, default="/root/datasets/bfcl_90k_train.jsonl",
                        help="GRPO training data (original 90K)")
    parser.add_argument("--test-data", type=str, default="/root/datasets/bfcl_100_test.jsonl",
                        help="BFCL test data (100 samples)")
    parser.add_argument("--work-dir", type=str, default="/root/autoresearch_v14",
                        help="Working directory for experiments")
    parser.add_argument("--train-minutes", type=int, default=5,
                        help="Minutes to train per iteration")
    parser.add_argument("--max-iterations", type=int, default=50,
                        help="Maximum number of research iterations")
    args = parser.parse_args()

    loop = AutoResearchLoop(args)
    loop.run()


if __name__ == "__main__":
    main()
