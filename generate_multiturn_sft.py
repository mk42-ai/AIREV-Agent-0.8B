#!/usr/bin/env python3
"""
Generate proper SFT data for multi-turn, memory, web_search categories.
Uses the ACTUAL BFCL function schemas from the pre-compiled docs.
Sends to Claude Opus with the real function schemas so it generates
correct [func(params)] bracket format calls.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json, time, random, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GRequest

CREDENTIALS_PATH = "/root/vertex_credentials.json"
PROJECT = "ondemand-421015"
REGION = "us-east5"
MODEL = "claude-opus-4-6"

_creds = None
_creds_lock = threading.Lock()

def get_token():
    global _creds
    with _creds_lock:
        if _creds is None:
            _creds = service_account.Credentials.from_service_account_file(
                CREDENTIALS_PATH, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if _creds.expired or not _creds.token:
            _creds.refresh(GRequest())
        return _creds.token


def call_claude(system, user, retries=5):
    url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/publishers/anthropic/models/{MODEL}:rawPredict"
    body = {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": 8000,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    for attempt in range(retries):
        try:
            r = requests.post(url,
                headers={"Authorization": f"Bearer {get_token()}", "Content-Type": "application/json"},
                json=body, timeout=180)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt + random.uniform(0, 1))
                continue
            r.raise_for_status()
            for block in r.json().get("content", []):
                if block.get("type") == "text":
                    return block["text"]
            return ""
        except:
            time.sleep(2 ** attempt)
    return None


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


def load_func_docs():
    """Load all pre-compiled function docs from BFCL."""
    from bfcl_eval.constants.eval_config import MULTI_TURN_FUNC_DOC_PATH
    docs = {}
    for fname in os.listdir(MULTI_TURN_FUNC_DOC_PATH):
        if fname.endswith('.json'):
            class_name = fname.replace('.json', '')
            funcs = []
            with open(os.path.join(MULTI_TURN_FUNC_DOC_PATH, fname)) as f:
                for line in f:
                    if line.strip():
                        funcs.append(json.loads(line))
            docs[class_name] = funcs
    return docs


# Map involved_classes to func doc files
CLASS_TO_DOC = {
    "GorillaFileSystem": "gorilla_file_system",
    "TwitterAPI": "posting_api",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
    "WebSearchAPI": "web_search",
    "MemoryAPI": "memory_kv",
}


def process_sample(sample, func_docs, idx, total):
    """Process a BFCL multi-turn sample — send to Claude with real function schemas."""
    sample_id = sample.get("id", f"unknown_{idx}")
    involved_classes = sample.get("involved_classes", [])
    question_turns = sample.get("question", [])

    # Gather function schemas for all involved classes
    all_funcs = []
    for cls in involved_classes:
        doc_name = CLASS_TO_DOC.get(cls, cls.lower())
        if doc_name in func_docs:
            all_funcs.extend(func_docs[doc_name])

    if not all_funcs:
        return None

    # Build system prompt with actual function schemas
    func_json = "\n".join(json.dumps(f) for f in all_funcs)
    system_prompt = BFCL_SYSTEM_PROMPT + func_json

    # Build multi-turn conversation
    messages = [{"role": "system", "content": system_prompt}]
    assistant_responses = []

    for turn_idx, turn_list in enumerate(question_turns):
        if isinstance(turn_list, list):
            for turn in turn_list:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                messages.append({"role": role, "content": content})
        else:
            messages.append({"role": "user", "content": str(turn_list)})

    # Send ALL turns to Claude at once, ask it to respond to each
    turns_text = ""
    for turn_idx, turn_list in enumerate(question_turns):
        if isinstance(turn_list, list):
            for turn in turn_list:
                turns_text += f"\nTurn {turn_idx + 1} [{turn.get('role', 'user')}]: {turn.get('content', '')}\n"

    user_prompt = f"""You are given a multi-turn conversation. For EACH user turn, provide a response.

Available functions are in the system prompt above.

{turns_text}

For EACH turn, respond with:
1. <think> reasoning about which function(s) to call </think>
2. The function call(s) in bracket format: [func_name(param=value)]

If a turn asks for something no function can do, respond with:
<think>reasoning</think>
No function matches the user's request.

Format your response as:

TURN 1:
<think>reasoning</think>
[function_call(params)]

TURN 2:
<think>reasoning</think>
[function_call(params)]

etc."""

    response = call_claude(system_prompt, user_prompt)
    if not response:
        return None

    # Parse into multi-turn messages format
    result_messages = [{"role": "system", "content": system_prompt}]

    # Split response by TURN markers
    import re
    turn_splits = re.split(r'TURN \d+:', response)
    turn_splits = [t.strip() for t in turn_splits if t.strip()]

    for turn_idx, turn_list in enumerate(question_turns):
        # Add user message
        if isinstance(turn_list, list):
            for turn in turn_list:
                result_messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})

        # Add assistant response
        if turn_idx < len(turn_splits):
            result_messages.append({"role": "assistant", "content": turn_splits[turn_idx].strip()})

    # Determine category from id
    category = sample_id.rsplit("_", 1)[0] if "_" in sample_id else "unknown"
    # Clean up category
    for cat in ["multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param", "memory", "web_search"]:
        if cat in sample_id:
            category = cat
            break

    if (idx + 1) % 20 == 0:
        print(f"  [{idx+1}/{total}] {sample_id} done")

    return {
        "messages": result_messages,
        "category": category,
        "id": sample_id,
    }


def main():
    output = "/root/datasets/sft_multiturn_proper.jsonl"
    workers = 100

    # Load function docs
    func_docs = load_func_docs()
    print(f"Loaded function docs: {list(func_docs.keys())}")

    # Load ALL BFCL multi-turn, memory, web_search samples
    bfcl_dir = "/usr/local/lib/python3.10/dist-packages/bfcl_eval/data"
    samples = []
    for cat in ["multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param", "memory", "web_search"]:
        path = os.path.join(bfcl_dir, f"BFCL_v4_{cat}.json")
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        try:
                            samples.append(json.loads(line))
                        except:
                            pass
    print(f"Loaded {len(samples)} BFCL samples for multi-turn/memory/web_search")

    from collections import Counter
    cats = Counter()
    for s in samples:
        sid = s.get("id", "")
        for cat in ["multi_turn_base", "multi_turn_long_context", "multi_turn_miss_func", "multi_turn_miss_param", "memory", "web_search"]:
            if cat in sid:
                cats[cat] += 1
                break
    for cat, cnt in cats.most_common():
        print(f"  {cat}: {cnt}")

    total = len(samples)
    print(f"\nGenerating with {workers} workers...")

    completed = 0
    errors = 0
    lock = threading.Lock()

    with open(output, "w") as out:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_sample, s, func_docs, i, total): i for i, s in enumerate(samples)}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    with lock:
                        out.write(json.dumps(result) + "\n")
                        completed += 1
                        if completed % 50 == 0:
                            out.flush()
                            print(f"  Progress: {completed}/{total} ({completed/total*100:.1f}%)")
                else:
                    errors += 1

    print(f"\nDone! {completed} samples generated, {errors} errors")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
