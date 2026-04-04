#!/usr/bin/env python3
"""
Generate ground truth for real BFCL V4 test data using Claude Opus 4.6.
Takes the official test questions (no ground_truth) and has Claude produce
correct function calls in BFCL bracket format with <think> reasoning.
Output: JSONL with messages format ready for GRPO training.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json, time, random, sys, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GRequest

# Config
CREDENTIALS_PATH = "/root/vertex_credentials.json"
PROJECT = "ondemand-421015"
REGION = "us-east5"
MODEL = "claude-opus-4-6"
MAX_RETRIES = 5
WORKERS = 50

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
3. Generate excessive content."""

# Auth
_creds = None
_creds_lock = threading.Lock()

def get_token():
    global _creds
    with _creds_lock:
        if _creds is None:
            _creds = service_account.Credentials.from_service_account_file(
                CREDENTIALS_PATH,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if _creds.expired or not _creds.token:
            _creds.refresh(GRequest())
        return _creds.token


def call_claude(system_prompt, user_prompt, retries=MAX_RETRIES):
    url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/publishers/anthropic/models/{MODEL}:rawPredict"
    body = {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    for attempt in range(retries):
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {get_token()}", "Content-Type": "application/json"},
                json=body,
                timeout=120,
            )
            if r.status_code == 429 or r.status_code >= 500:
                wait = 2 ** attempt + random.uniform(0, 1)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            for block in data.get("content", []):
                if block.get("type") == "text":
                    return block["text"]
            return ""
        except requests.exceptions.Timeout:
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [ERROR] {e}")
                return None
    return None


def format_functions(functions):
    parts = []
    for func in functions:
        parts.append(json.dumps(func))
    return "\n\nHere is a list of functions in JSON format that you can invoke:\n" + "\n".join(parts)


def process_sample(sample, idx, total):
    sample_id = sample.get("id", f"unknown_{idx}")
    question = sample.get("question", [])
    functions = sample.get("function", [])

    # Get the user question
    turns = question[0] if question and isinstance(question[0], list) else question
    if not turns:
        return None

    user_content = turns[0].get("content", "") if isinstance(turns[0], dict) else str(turns[0])

    # Build system prompt with functions
    system = BFCL_SYSTEM_PROMPT + format_functions(functions)

    # Ask Claude for the answer
    prompt = f"""Given the functions available to you, answer this user request by calling the appropriate function(s).

IMPORTANT RULES:
1. First reason step-by-step inside <think>...</think> tags about which function(s) to call and with what parameters
2. Then output ONLY the function call(s) in bracket format: [func_name(param1=value1, param2=value2)]
3. If multiple functions need to be called, put them all in one bracket: [func1(params), func2(params)]
4. If NO function matches the request, say "No function matches the user's request." (no brackets)
5. Use exact parameter names and appropriate types from the function schemas
6. String values should be in quotes, numbers without quotes, booleans as true/false

User request: {user_content}"""

    response = call_claude(system, prompt)
    if response is None:
        return None

    # Extract category from ID
    category = sample_id.rsplit("_", 1)[0] if "_" in sample_id else "unknown"

    # Build training format
    result = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"<think>\n{response.split('</think>')[0].replace('<think>', '').strip()}\n</think>\n{response.split('</think>')[-1].strip() if '</think>' in response else response.strip()}"},
        ],
        "category": category,
        "id": sample_id,
    }

    if (idx + 1) % 50 == 0:
        print(f"  [{idx+1}/{total}] {sample_id} done")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/root/datasets/bfcl_v4_full_test.jsonl")
    parser.add_argument("--output", default="/root/datasets/bfcl_v4_ground_truth.jsonl")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load test data
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} BFCL test samples")

    # Resume support
    done_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done_ids.add(d.get("id", ""))
        print(f"Resuming: {len(done_ids)} already done")
        samples = [s for s in samples if s.get("id", "") not in done_ids]
        print(f"Remaining: {len(samples)}")

    total = len(samples)
    print(f"Generating ground truth with {args.workers} workers...")
    print(f"Output: {args.output}")

    mode = "a" if args.resume else "w"
    completed = 0
    errors = 0
    lock = threading.Lock()

    with open(args.output, mode) as out:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_sample, s, i, total): i for i, s in enumerate(samples)}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    with lock:
                        out.write(json.dumps(result) + "\n")
                        completed += 1
                        if completed % 100 == 0:
                            out.flush()
                            print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%) | errors={errors}")
                else:
                    errors += 1

    print(f"\nDone! {completed} ground truth samples generated, {errors} errors")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
