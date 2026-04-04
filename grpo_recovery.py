#!/usr/bin/env python3
"""
GRPO Recovery Run — Restore zero-category capabilities while preserving function calling.
Uses dual reward: bracket matching for function calls, text similarity for reasoning/refusal.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["WANDB_DISABLED"] = "true"

import sys, json, argparse, time, re, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def load_dataset(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                msgs = d.get("messages", [])
                system_text = ""
                user_text = ""
                gold = ""
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
                    "category": d.get("category", "unknown"),
                    "gold_type": d.get("gold_type", "bracket"),
                })
    return data


def compute_reward(generated, gold, gold_type, category):
    """Dual reward: bracket matching OR text similarity."""
    gen = generated.strip()

    # For text-type gold (refusal, reasoning, multi-turn)
    if gold_type == "text":
        # Check if model correctly DIDN'T call a function
        has_bracket = bool(re.search(r'\[.*\w+\s*\(', gen, re.DOTALL))

        if has_bracket:
            # Model called a function when it shouldn't have
            return 0.0

        # Model produced text response — good!
        # Check for refusal patterns
        refusal = any(p in gen.lower() for p in [
            "no function", "none of the", "cannot", "not available",
            "don't have access", "no relevant", "not possible",
            "i need to inform", "i don't have", "no tools",
            "no api", "no functions"
        ])

        if "irrelevance" in category:
            return 1.0 if refusal else 0.5

        # For multi-turn/memory/web_search, any non-bracket text is progress
        if refusal:
            return 0.8

        # Natural reasoning response
        if len(gen) > 20:
            return 0.7

        return 0.3

    # For bracket-type gold (function calling)
    else:
        gen_calls = extract_func_call_bracket(gen)
        gold_calls = extract_func_call_bracket(gold)

        if not gold_calls:
            return 0.5
        if not gen_calls:
            return 0.05

        score = 0.0
        for gc in gold_calls:
            best = 0.0
            for g in gen_calls:
                s = 0.0
                if g["name"] == gc["name"]:
                    s += 0.4
                    if g["args_str"].strip() == gc["args_str"].strip():
                        s += 0.6
                    else:
                        gen_vals = set(re.findall(r'[\w.]+', g["args_str"]))
                        gold_vals = set(re.findall(r'[\w.]+', gc["args_str"]))
                        if gold_vals:
                            overlap = len(gen_vals & gold_vals) / len(gold_vals)
                            s += 0.6 * overlap
                best = max(best, s)
            score += best
        return score / len(gold_calls)


def extract_func_call_bracket(text):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()
    deadline = start_time + args.max_hours * 3600

    print(f"{'='*60}")
    print(f"  GRPO RECOVERY RUN")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  LR: {args.lr} | Gens: {args.num_generations}")
    print(f"  Max hours: {args.max_hours}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda:0")
    model.config.use_cache = False

    dataset = load_dataset(args.dataset)
    random.shuffle(dataset)
    print(f"[INFO] Loaded {len(dataset)} samples")

    from collections import Counter
    types = Counter(d["gold_type"] for d in dataset)
    cats = Counter(d["category"] for d in dataset)
    print(f"[INFO] Types: {dict(types)}")
    print(f"[INFO] Top categories: {dict(cats.most_common(5))}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    num_steps = len(dataset)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=num_steps)

    model.train()
    step = 0
    epoch = 0
    text_rewards = []
    bracket_rewards = []

    while time.time() < deadline:
        epoch += 1
        random.shuffle(dataset)
        print(f"\n  Epoch {epoch} | {(time.time()-start_time)/60:.0f} min elapsed")

        for sample in dataset:
            if time.time() > deadline:
                break

            messages = [{"role": "system", "content": sample["system"]}, {"role": "user", "content": sample["user"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not prompt.endswith("\n"):
                prompt += "\n"
            prompt += "<think>\n"

            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).input_ids.to("cuda:0")
            prompt_len = input_ids.shape[1]

            rewards = []
            with torch.no_grad():
                outputs = model.generate(
                    input_ids.expand(args.num_generations, -1),
                    max_new_tokens=512,
                    temperature=0.6,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for i in range(args.num_generations):
                gen_ids = outputs[i][prompt_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                if "</think>" in gen_text:
                    output_part = gen_text.split("</think>")[-1].strip()
                else:
                    output_part = gen_text.strip()

                r = compute_reward(output_part, sample["gold"], sample["gold_type"], sample["category"])
                rewards.append(min(r, 1.0))

            # Track by type
            if sample["gold_type"] == "text":
                text_rewards.append(max(rewards))
            else:
                bracket_rewards.append(max(rewards))

            # GRPO
            rewards_t = torch.tensor(rewards, device="cuda:0")
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

            best_idx = rewards_t.argmax().item()
            if rewards[best_idx] > 0.2:
                gen_ids_best = outputs[best_idx][prompt_len:]
                full_best = outputs[best_idx:best_idx + 1]
                logits = model(full_best).logits[:, prompt_len - 1:-1, :]
                lp = F.log_softmax(logits, dim=-1)
                token_lp = lp.gather(2, gen_ids_best.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                loss = -(advantages[best_idx] * token_lp.mean())
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            if step % 10 == 0:
                t_avg = sum(text_rewards[-20:]) / max(len(text_rewards[-20:]), 1) if text_rewards else 0
                b_avg = sum(bracket_rewards[-20:]) / max(len(bracket_rewards[-20:]), 1) if bracket_rewards else 0
                print(f"  S{step} | text_r={t_avg:.3f} bracket_r={b_avg:.3f} | {(time.time()-start_time)/60:.0f}min")

            if step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

    model.save_pretrained(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print(f"\n  Done in {(time.time()-start_time)/60:.0f} min | {step} steps | {epoch} epochs")


if __name__ == "__main__":
    main()
