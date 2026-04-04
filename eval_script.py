
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

