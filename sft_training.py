#!/usr/bin/env python3
"""Quick SFT on irrelevance data to reinforce refusal behavior."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["WANDB_DISABLED"] = "true"

import json, argparse, time, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load data
    print(f"Loading dataset from {args.dataset}")
    data = []
    with open(args.dataset) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples")

    # Prepare training examples
    examples = []
    for d in data:
        msgs = d.get("messages", [])
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        examples.append(text)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(examples) // args.batch_size) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

    start = time.time()
    step = 0
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        random.shuffle(examples)
        epoch_loss = 0
        epoch_steps = 0

        for i in range(0, len(examples) - args.batch_size + 1, args.batch_size):
            batch_texts = examples[i:i + args.batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss / 1  # no grad accum needed for small dataset

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            if step % 20 == 0:
                avg = epoch_loss / epoch_steps
                elapsed = time.time() - start
                print(f"  E{epoch} S{step}/{total_steps} | loss={loss.item():.4f} avg={avg:.4f} | {elapsed/60:.1f}min")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n  Epoch {epoch} complete | avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.output_dir, "best")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  Saved best to {save_path}")

    # Final save
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min | best_loss={best_loss:.4f}")
    print(f"Final saved to {final_path}")

if __name__ == "__main__":
    main()
