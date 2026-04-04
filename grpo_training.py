#!/usr/bin/env python3
"""
GRPO v14 Full Run — Long training on BFCL-format data.
Uses the Opus 50K dataset (messages format) with proper extraction.
Runs for hours, not minutes. Progressive curriculum.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ["WANDB_DISABLED"] = "true"

import sys, json, argparse, time, re, random, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

BFCL_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out and refuse to answer.
If the given question lacks the parameters required by the function, also point it out.

You SHOULD NOT:
- Generate any content that is not a function call or a reasoning process.

You should ONLY interact with the presented APIs using following format:

[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]

You SHOULD NOT:
1. Use any APIs that are not provided to you.
2. Generate any content that is not a function call.
3. Generate excessive content.

Here is a list of functions in JSON format that you can invoke.
"""


def load_dataset(path, max_samples=None):
    """Load dataset - handles both messages format and BFCL format."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                # Convert to unified format
                if "messages" in sample:
                    # Opus 50K format
                    msgs = sample["messages"]
                    user_text = ""
                    gold = ""
                    system_text = ""
                    for m in msgs:
                        if m["role"] == "system":
                            system_text = m["content"]
                        elif m["role"] == "user":
                            user_text = m["content"]
                        elif m["role"] == "assistant":
                            content = m["content"]
                            if "</think>" in content:
                                gold = content.split("</think>")[-1].strip()
                            else:
                                gold = content.strip()
                    data.append({
                        "system": system_text,
                        "user": user_text,
                        "gold": gold,
                        "category": sample.get("category", "unknown"),
                    })
                elif "question" in sample and "function" in sample:
                    # BFCL format
                    turns = sample["question"][0] if isinstance(sample["question"][0], list) else sample["question"]
                    user_text = turns[0]["content"] if turns else ""
                    funcs = sample["function"]
                    system_text = BFCL_SYSTEM_PROMPT + "\n".join(json.dumps(f) for f in funcs)
                    # Ground truth
                    gt = sample.get("ground_truth", [])
                    gold_parts = []
                    for gt_call in gt:
                        if isinstance(gt_call, dict):
                            for fname, fparams in gt_call.items():
                                params_str = ", ".join(f"{k}={repr(v[0]) if isinstance(v, list) and v else repr(v)}" for k, v in fparams.items())
                                gold_parts.append(f"{fname}({params_str})")
                    gold = "[" + ", ".join(gold_parts) + "]" if gold_parts else ""
                    data.append({
                        "system": system_text,
                        "user": user_text,
                        "gold": gold,
                        "category": sample.get("id", "unknown").rsplit("_", 1)[0],
                    })
                if max_samples and len(data) >= max_samples:
                    break
    return data


def extract_func_call_bracket(text):
    """Extract function calls from bracket format [func(params)]."""
    calls = []
    m = re.search(r'\[(.+)\]', text, re.DOTALL)
    if not m:
        return calls
    inner = m.group(1)
    parts = re.split(r'\)\s*,\s*(?=[a-zA-Z_])', inner)
    for p in parts:
        p = p.strip()
        if not p.endswith(')'):
            p += ')'
        mm = re.match(r'([a-zA-Z_][\w.]*)\s*\((.*)\)', p, re.DOTALL)
        if mm:
            calls.append({"name": mm.group(1), "args_str": mm.group(2)})
    return calls


def compute_reward(generated, gold, category=""):
    """Compute reward comparing generated output to gold."""
    # Irrelevance: model should NOT call functions
    if not gold or gold.strip() == "" or gold.strip() == "[]":
        # No function should be called
        gen_calls = extract_func_call_bracket(generated)
        if len(gen_calls) == 0:
            # Check if model explicitly refuses
            refusal_patterns = ["no function", "none of the", "cannot", "not possible", "don't match", "no suitable"]
            if any(p in generated.lower() for p in refusal_patterns):
                return 1.0
            return 0.7  # Didn't call anything but didn't explicitly refuse
        return 0.0  # Called a function when shouldn't have

    gen_calls = extract_func_call_bracket(generated)
    gold_calls = extract_func_call_bracket(gold)

    if not gold_calls:
        return 0.5
    if not gen_calls:
        return 0.05

    # Score each gold call
    score = 0.0
    for gc in gold_calls:
        best = 0.0
        for g in gen_calls:
            s = 0.0
            if g["name"] == gc["name"]:
                s += 0.4  # Name match
                # Compare args
                if g["args_str"].strip() == gc["args_str"].strip():
                    s += 0.6  # Exact match
                else:
                    gen_vals = set(re.findall(r'[\w.]+', g["args_str"]))
                    gold_vals = set(re.findall(r'[\w.]+', gc["args_str"]))
                    if gold_vals:
                        overlap = len(gen_vals & gold_vals) / len(gold_vals)
                        s += 0.6 * overlap
            best = max(best, s)
        score += best
    return score / len(gold_calls)


def main():
    parser = argparse.ArgumentParser(description="GRPO v14 Full Run")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-hours", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()
    deadline = start_time + args.max_hours * 3600

    print(f"{'='*70}")
    print(f"  GRPO v14 FULL RUN")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {args.output_dir}")
    print(f"  Max hours: {args.max_hours}")
    print(f"  LR: {args.lr} | Beta: {args.beta} | BS: {args.batch_size}")
    print(f"  Generations: {args.num_generations} | Temp: {args.temperature}")
    print(f"{'='*70}")

    # Load model on GPU 0 for training
    num_gpus = torch.cuda.device_count()
    print(f"\n[INFO] Loading model from {args.model}")
    print(f"[INFO] GPUs available: {num_gpus}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda:0")
    model.config.use_cache = False

    # Load copies on other GPUs for parallel generation
    gen_models = [model]  # GPU 0 is the main model
    if num_gpus > 1:
        print(f"[INFO] Loading {num_gpus-1} generation copies on GPUs 1-{num_gpus-1}...")
        for gpu_id in range(1, num_gpus):
            gen_model = AutoModelForCausalLM.from_pretrained(
                args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
            ).to(f"cuda:{gpu_id}")
            gen_model.eval()
            gen_models.append(gen_model)
        print(f"[INFO] {len(gen_models)} models loaded for parallel generation")

    # Load dataset
    print(f"[INFO] Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)
    random.shuffle(dataset)
    print(f"[INFO] Loaded {len(dataset)} samples")

    # Count categories
    from collections import Counter
    cats = Counter(d["category"] for d in dataset)
    print(f"[INFO] Categories: {dict(cats.most_common(10))}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    num_steps = len(dataset) // args.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=num_steps)

    model.train()
    step = 0
    epoch = 0
    best_reward = 0.0
    reward_history = []
    category_rewards = {}

    while time.time() < deadline:
        epoch += 1
        random.shuffle(dataset)
        print(f"\n{'='*70}")
        print(f"  Epoch {epoch} | Best reward: {best_reward:.4f} | Steps: {step}")
        print(f"  Time: {(time.time()-start_time)/60:.1f} min elapsed, {(deadline-time.time())/60:.1f} min remaining")
        print(f"{'='*70}")

        for batch_start in range(0, len(dataset) - args.batch_size + 1, args.batch_size):
            if time.time() > deadline:
                print("[INFO] Time limit reached")
                break

            batch = dataset[batch_start:batch_start + args.batch_size]
            optimizer.zero_grad()
            batch_rewards = []

            for sample in batch:
                system_text = sample["system"]
                user_text = sample["user"]
                gold = sample["gold"]
                category = sample["category"]

                if system_text:
                    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
                else:
                    messages = [{"role": "system", "content": BFCL_SYSTEM_PROMPT}, {"role": "user", "content": user_text}]

                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if not prompt.endswith("\n"):
                    prompt += "\n"
                prompt += "<think>\n"

                input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).input_ids.to(model.device)
                prompt_len = input_ids.shape[1]

                # Generate multiple completions across GPUs
                rewards = []
                gens_per_gpu = args.num_generations // len(gen_models)
                remainder = args.num_generations % len(gen_models)
                all_outputs = []

                def gen_on_gpu(gm, gpu_id, n_gens):
                    ids = input_ids.to(f"cuda:{gpu_id}")
                    with torch.no_grad():
                        out = gm.generate(
                            ids.expand(n_gens, -1),
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            do_sample=True,
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    return out.cpu()

                if len(gen_models) == 1:
                    with torch.no_grad():
                        all_outputs = model.generate(
                            input_ids.expand(args.num_generations, -1),
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            do_sample=True,
                            top_p=0.95,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                else:
                    import concurrent.futures
                    futures = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gen_models)) as ex:
                        for gpu_id, gm in enumerate(gen_models):
                            n = gens_per_gpu + (1 if gpu_id < remainder else 0)
                            if n > 0:
                                futures.append(ex.submit(gen_on_gpu, gm, gpu_id, n))
                    gpu_outputs = [f.result() for f in futures]
                    # Pad to same length before concatenating
                    max_len = max(o.shape[1] for o in gpu_outputs)
                    padded = []
                    for o in gpu_outputs:
                        if o.shape[1] < max_len:
                            pad = torch.full((o.shape[0], max_len - o.shape[1]), tokenizer.pad_token_id, dtype=o.dtype)
                            o = torch.cat([o, pad], dim=1)
                        padded.append(o)
                    all_outputs = torch.cat(padded, dim=0).to("cuda:0")

                outputs = all_outputs
                for i in range(args.num_generations):
                    gen_ids = outputs[i][prompt_len:]
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    if "</think>" in gen_text:
                        output_part = gen_text.split("</think>")[-1].strip()
                    else:
                        output_part = gen_text.strip()

                    r = compute_reward(output_part, gold, category)
                    # Format bonus
                    if re.search(r'\[.*\(.*\).*\]', output_part, re.DOTALL):
                        r += 0.1
                    rewards.append(min(r, 1.0))

                # GRPO advantages
                rewards_t = torch.tensor(rewards, device=model.device)
                mean_r = rewards_t.mean()
                std_r = rewards_t.std() + 1e-8
                advantages = (rewards_t - mean_r) / std_r

                # Train on best completion
                best_idx = rewards_t.argmax().item()
                if rewards[best_idx] > 0.2:
                    gen_ids_best = outputs[best_idx][prompt_len:].to("cuda:0")
                    full_best = outputs[best_idx:best_idx + 1].to("cuda:0")
                    logits_train = model(full_best).logits[:, prompt_len - 1:-1, :]
                    lp_train = F.log_softmax(logits_train, dim=-1)
                    token_lp_train = lp_train.gather(2, gen_ids_best.unsqueeze(0).unsqueeze(-1)).squeeze(-1)

                    pg_loss = -(advantages[best_idx] * token_lp_train.mean())
                    with torch.no_grad():
                        ref_logits = model(full_best).logits[:, prompt_len - 1:-1, :]
                        ref_lp = F.log_softmax(ref_logits, dim=-1)
                        ref_token_lp = ref_lp.gather(2, gen_ids_best.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                    kl = torch.mean(token_lp_train - ref_token_lp[:, :token_lp_train.shape[1]])

                    loss = (pg_loss + args.beta * kl) / args.batch_size
                    loss.backward()

                batch_rewards.extend(rewards)

                # Track per-category
                if category not in category_rewards:
                    category_rewards[category] = []
                category_rewards[category].append(max(rewards))

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Sync weights to generation copies every 10 steps
            if len(gen_models) > 1 and step % 10 == 0:
                state = model.state_dict()
                for gm in gen_models[1:]:
                    gm.load_state_dict({k: v.to(gm.device) for k, v in state.items()})

            step += 1
            avg_r = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
            reward_history.append(avg_r)
            if avg_r > best_reward:
                best_reward = avg_r

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                avg20 = sum(reward_history[-20:]) / len(reward_history[-20:])
                print(f"  S{step} | r={avg_r:.3f} (avg20={avg20:.3f}) | best={best_reward:.3f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed/60:.0f}min")

            if step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"  [SAVED] {save_path}")

    # Final save
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  GRPO v14 FULL RUN COMPLETE in {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"  Total steps: {step}")
    print(f"  Best reward: {best_reward:.4f}")
    print(f"  Final avg20: {sum(reward_history[-20:])/len(reward_history[-20:]) if reward_history else 0:.4f}")
    print(f"{'='*70}")

    # Per-category summary
    print(f"\nPer-category avg reward:")
    for cat in sorted(category_rewards.keys()):
        vals = category_rewards[cat]
        print(f"  {cat}: {sum(vals)/len(vals):.3f} ({len(vals)} samples)")

    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n[INFO] Final model saved to {final_path}")

    # Metrics
    metrics = {
        "total_steps": step,
        "epochs": epoch,
        "elapsed_hours": elapsed / 3600,
        "best_reward": best_reward,
        "final_avg20": sum(reward_history[-20:]) / len(reward_history[-20:]) if reward_history else 0,
        "category_rewards": {k: sum(v)/len(v) for k, v in category_rewards.items()},
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
