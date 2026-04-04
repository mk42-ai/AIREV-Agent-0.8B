#!/usr/bin/env python3
"""Fixed eval with proper BFCL ground truth comparison."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "4")

import json, sys, re, ast
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BFCL_SYSTEM_PROMPT = "You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\nIf none of the functions can be used, point it out and refuse to answer.\nIf the given question lacks the parameters required by the function, also point it out.\n\nYou SHOULD NOT:\n- Generate any content that is not a function call or a reasoning process.\n\nYou should ONLY interact with the presented APIs using following format:\n\n[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)...]\n\nYou SHOULD NOT:\n1. Use any APIs that are not provided to you.\n2. Generate any content that is not a function call.\n3. Generate excessive content."


def format_functions(functions):
    parts = []
    for func in functions:
        parts.append(json.dumps(func))
    return "\n\nHere is a list of functions in JSON format that you can invoke:\n" + "\n".join(parts)


def build_prompt(question_turns, functions, tokenizer):
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
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        text = text.split("<think>")[0].strip()
    return text.strip()


def parse_model_calls(text):
    """Parse model output like [func(a=1, b='x'), func2(c=3)] into structured form."""
    calls = []
    m = re.search(r'\[(.+)\]', text, re.DOTALL)
    if not m:
        return calls
    inner = m.group(1)
    # Split on ), followed by comma and function name
    parts = re.split(r'\)\s*,\s*(?=[a-zA-Z_])', inner)
    for p in parts:
        p = p.strip()
        if not p.endswith(')'):
            p += ')'
        mm = re.match(r'([a-zA-Z_][\w.]*)\s*\((.*)\)', p, re.DOTALL)
        if mm:
            func_name = mm.group(1)
            args_str = mm.group(2).strip()
            # Parse arguments
            params = {}
            if args_str:
                # Try to parse key=value pairs
                for arg_match in re.finditer(r'(\w+)\s*=\s*', args_str):
                    key = arg_match.group(1)
                    start = arg_match.end()
                    # Find the value — handle strings, numbers, lists, bools
                    rest = args_str[start:]
                    # Try to extract value up to next , key= or end
                    val_match = re.match(r'((?:"[^"]*"|\'[^\']*\'|\[[^\]]*\]|[^,])+)', rest)
                    if val_match:
                        val_str = val_match.group(1).strip()
                        # Clean up the value
                        val_str = val_str.rstrip(',').strip()
                        params[key] = val_str
            calls.append({"name": func_name, "params": params})
    return calls


def normalize_value(val):
    """Normalize a value for comparison."""
    if isinstance(val, str):
        val = val.strip().strip('"').strip("'").strip()
    val_str = str(val).lower().strip()
    # Normalize booleans
    if val_str in ('true', '1', 'yes'):
        return 'true'
    if val_str in ('false', '0', 'no'):
        return 'false'
    # Normalize numbers
    try:
        num = float(val_str)
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        pass
    return val_str


def score_call(pred_call, gt_call_dict):
    """Score a single predicted call against a ground truth call.
    gt_call_dict is like {'func_name': {'param1': [val1, val2_alt], 'param2': [val]}}
    """
    gt_name = list(gt_call_dict.keys())[0]
    gt_params = gt_call_dict[gt_name]

    # Check function name
    if pred_call["name"] != gt_name:
        return 0.0

    # Name matches — base score 0.4
    score = 0.4

    if not gt_params:
        return 1.0  # No params to check, name match is enough

    # Check each ground truth parameter
    param_scores = []
    for param_name, acceptable_values in gt_params.items():
        if param_name in pred_call["params"]:
            pred_val = normalize_value(pred_call["params"][param_name])
            # acceptable_values is a list of acceptable values
            matched = False
            for av in acceptable_values:
                if normalize_value(av) == pred_val:
                    matched = True
                    break
                # Also try substring match for complex values
                if str(normalize_value(av)) in pred_val or pred_val in str(normalize_value(av)):
                    matched = True
                    break
            param_scores.append(1.0 if matched else 0.3)
        else:
            param_scores.append(0.0)  # Missing required param

    if param_scores:
        param_score = sum(param_scores) / len(param_scores)
        score += 0.6 * param_score
    else:
        score += 0.6  # No params to check

    return score


def score_prediction(prediction, ground_truth):
    """Score a prediction against BFCL ground truth.
    ground_truth is a list like [{'func': {'p1': [v1]}}, {'func2': {'p2': [v2]}}]
    """
    pred = prediction.strip()

    # Handle irrelevance (no function should be called)
    if not ground_truth or ground_truth == []:
        # Model should NOT call any function
        has_bracket_call = bool(re.search(r'\[.*\w+\s*\(', pred, re.DOTALL))
        if has_bracket_call:
            return 0.0  # Called a function when shouldn't have
        else:
            return 1.0  # Correctly refused

    # Parse model output
    pred_calls = parse_model_calls(pred)

    if not pred_calls:
        return 0.0  # No function calls found but should have called

    # Match predicted calls to ground truth calls
    # Use greedy matching — for each GT call, find best matching pred call
    gt_scores = []
    used_pred = set()

    for gt_call in ground_truth:
        best_score = 0.0
        best_idx = -1
        for idx, pc in enumerate(pred_calls):
            if idx in used_pred:
                continue
            s = score_call(pc, gt_call)
            if s > best_score:
                best_score = s
                best_idx = idx
        if best_idx >= 0:
            used_pred.add(best_idx)
        gt_scores.append(best_score)

    if not gt_scores:
        return 0.0

    # Average score across all expected calls
    avg_score = sum(gt_scores) / len(gt_scores)

    # Penalty for extra calls (mild)
    extra_calls = len(pred_calls) - len(ground_truth)
    if extra_calls > 0:
        avg_score *= max(0.7, 1.0 - 0.05 * extra_calls)

    return avg_score


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/root/checkpoints/grpo_v14_optimal/checkpoint-1200"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "/root/datasets/bfcl_100_test.jsonl"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "/root/eval_fixed_results.json"

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    samples = []
    with open(test_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    results = []
    category_scores = {}

    for idx, sample in enumerate(samples):
        sample_id = sample.get("id", f"sample_{idx}")
        category = sample.get("category", sample_id.rsplit("_", 1)[0] if "_" in sample_id else "unknown")

        # Handle both BFCL format (question/function) and messages format
        if "question" in sample and "function" in sample:
            prompt_text = build_prompt(sample["question"], sample["function"], tokenizer)
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
            gold = sample.get("ground_truth", [])
            turns = sample["question"][0] if isinstance(sample["question"][0], list) else sample["question"]
            user_q = turns[0]["content"][:150] if turns else ""
        elif "messages" in sample:
            msgs = sample["messages"]
            system_text = ""
            user_q = ""
            gold_text = ""
            for m in msgs:
                if m["role"] == "system":
                    system_text = m["content"]
                elif m["role"] == "user":
                    user_q = m["content"][:150]
                elif m["role"] == "assistant":
                    content = m["content"]
                    if "</think>" in content:
                        gold_text = content.split("</think>")[-1].strip()
                    else:
                        gold_text = content.strip()
            # Build prompt from messages
            chat_msgs = []
            for m in msgs:
                if m["role"] != "assistant":
                    chat_msgs.append(m)
            prompt_text = tokenizer.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
            if not prompt_text.endswith("\n"):
                prompt_text += "\n"
            prompt_text += "<think>\n"
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
            # Parse gold_text into structured format for scoring
            gold_calls = parse_model_calls(gold_text)
            gold = []
            for gc in gold_calls:
                gold.append({gc["name"]: {k: [v] for k, v in gc["params"].items()}})
            if not gold_calls and gold_text:
                gold = []  # Irrelevance / refusal
        else:
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
        score = score_prediction(answer, gold)

        gt_func_names = []
        for gt_call in gold:
            if isinstance(gt_call, dict):
                gt_func_names.extend(gt_call.keys())

        entry = {
            "id": sample_id,
            "category": category,
            "score": score,
            "user_question": user_q,
            "expected_functions": gt_func_names,
            "expected_gt": str(gold)[:200],
            "model_output": answer[:300],
        }
        results.append(entry)

        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)

        if (idx + 1) % 10 == 0:
            acc = sum(s["score"] for s in results) / len(results)
            print(f"  [{idx+1}/{len(samples)}] running_acc={acc:.3f}")

    overall_acc = sum(r["score"] for r in results) / len(results)
    print(f"\n{'='*70}")
    print(f"RESULTS (Fixed Scorer)")
    print(f"{'='*70}")
    print(f"Overall: {overall_acc:.4f} ({sum(1 for r in results if r['score']>=0.5)}/{len(results)} correct)")
    print(f"Perfect (>=0.9): {sum(1 for r in results if r['score']>=0.9)}/{len(results)}")

    print(f"\nPer-category:")
    for cat in sorted(category_scores.keys()):
        scores = category_scores[cat]
        acc = sum(scores) / len(scores)
        perfect = sum(1 for s in scores if s >= 0.9)
        print(f"  {cat:20s}: {acc:.3f} ({perfect}/{len(scores)} perfect)")

    failures = [r for r in results if r["score"] < 0.5]
    print(f"\nFailures (<0.5): {len(failures)}/{len(results)}")
    for r in failures[:5]:
        print(f"  [{r['id']}] score={r['score']:.2f} | got: {r['model_output'][:80]}")

    output_data = {
        "overall_accuracy": overall_acc,
        "total": len(results),
        "correct": sum(1 for r in results if r["score"] >= 0.5),
        "perfect": sum(1 for r in results if r["score"] >= 0.9),
        "category_scores": {k: sum(v)/len(v) for k, v in category_scores.items()},
        "category_counts": {k: len(v) for k, v in category_scores.items()},
        "failures": failures,
        "all_results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
