#!/usr/bin/env python3
"""Run official BFCL eval using transformers directly."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, sys, time, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model

MODEL_PATH = "/root/checkpoints/sft_multiturn/best"
TEMPERATURE = 0.6
MAX_NEW_TOKENS = 512

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, tokenizer

def format_prompt(messages, tokenizer):
    prompt = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n<think>\n"
    return prompt

def generate_response(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
            do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    return response

CLASS_TO_DOC = {
    "GorillaFileSystem": "gorilla_file_system", "TwitterAPI": "posting_api",
    "MathAPI": "math_api", "MessageAPI": "message_api", "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot", "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control", "WebSearchAPI": "web_search",
    "MemoryAPI": "memory_kv",
}

def main():
    model, tokenizer = load_model()
    bfcl_dir = "/usr/local/lib/python3.10/dist-packages/bfcl_eval/data"
    func_doc_dir = os.path.join(bfcl_dir, "multi_turn_func_doc")
    output_dir = Path("/root/bfcl_results")
    output_dir.mkdir(exist_ok=True)

    categories = [
        "simple_python", "simple_java", "simple_javascript",
        "multiple", "parallel", "parallel_multiple",
        "irrelevance", "live_simple", "live_multiple",
        "live_parallel", "live_parallel_multiple",
        "live_irrelevance", "live_relevance",
        "multi_turn_base", "multi_turn_long_context",
        "multi_turn_miss_func", "multi_turn_miss_param",
        "memory", "web_search", "format_sensitivity",
    ]

    total_samples = 0
    total_time = time.time()

    for cat in categories:
        fname = f"BFCL_v4_{cat}.json"
        path = os.path.join(bfcl_dir, fname)
        if not os.path.exists(path):
            print(f"  [{cat}] Not found")
            continue

        samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except:
                        continue

        print(f"\n{'='*60}")
        print(f"  {cat}: {len(samples)} samples")
        print(f"{'='*60}")

        results = []
        t0 = time.time()

        for idx, sample in enumerate(samples):
            sample_id = sample.get("id", f"{cat}_{idx}")

            if "function" in sample:
                # Standard categories
                question = sample["question"]
                function = sample["function"]
                turns = question[0] if isinstance(question[0], list) else question
                messages = system_prompt_pre_processing_chat_model(turns, function, [])
                prompt = format_prompt(messages, tokenizer)
                response = generate_response(model, tokenizer, prompt)
                results.append({"id": sample_id, "result": response})

            elif "involved_classes" in sample:
                # Multi-turn / memory / web_search
                all_funcs = []
                for cls in sample.get("involved_classes", []):
                    doc_name = CLASS_TO_DOC.get(cls, cls.lower())
                    doc_path = os.path.join(func_doc_dir, f"{doc_name}.json")
                    if os.path.exists(doc_path):
                        with open(doc_path) as df:
                            for dline in df:
                                if dline.strip():
                                    all_funcs.append(json.loads(dline))

                question_turns = sample["question"]
                turn_responses = []
                for turn_list in question_turns:
                    if isinstance(turn_list, list):
                        user_content = turn_list[0].get("content", "") if turn_list else ""
                    else:
                        user_content = str(turn_list)
                    messages = system_prompt_pre_processing_chat_model(
                        [{"role": "user", "content": user_content}], all_funcs, []
                    )
                    prompt = format_prompt(messages, tokenizer)
                    resp = generate_response(model, tokenizer, prompt)
                    turn_responses.append(resp)

                results.append({"id": sample_id, "result": turn_responses})
            else:
                results.append({"id": sample_id, "result": ""})

            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(samples)}] {time.time()-t0:.0f}s")

        out_path = output_dir / f"{cat}_result.json"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        total_samples += len(samples)
        print(f"  Done: {len(samples)} in {time.time()-t0:.0f}s")

    print(f"\nCOMPLETE: {total_samples} samples in {(time.time()-total_time)/60:.1f} min")
    print(f"Results: {output_dir}")

if __name__ == "__main__":
    main()
