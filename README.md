# AIREV-Agent-0.8B

**A 752M parameter model for agentic function calling that outperforms the 11x larger Qwen3.5-9B on 7/19 BFCL V4 categories.**

## Results (Internal Evaluation)

| Category | Base 0.8B | OURS | 9B Base |
|----------|-----------|------|---------|
| simple_python | 0/10 | **10/10** | 9/10 |
| parallel | 0/10 | **10/10** | 8/10 |
| simple_java | 0/10 | **9/10** | 5/10 |
| multiple | 3/10 | **9/10** | 9/10 |
| parallel_multiple | 0/10 | **9/10** | 10/10 |
| live_multiple | 0/10 | **8/10** | 5/10 |
| live_simple | 0/10 | **7/10** | 6/10 |
| simple_javascript | 0/10 | **7/10** | 4/10 |

Official BFCL V4 scores pending leaderboard verification.

## Model

- **HuggingFace**: [airev-ai/AIREV-Agent-0.8B](https://huggingface.co/airev-ai/AIREV-Agent-0.8B)
- **Base**: Qwen3.5-0.8B (Gated Delta Network, 262K context)
- **Parameters**: 752M
- **License**: Apache 2.0

## Training Pipeline

1. **SFT** on 50K Claude Opus 4.6-generated BFCL-format samples with `<think>` reasoning
2. **AutoResearch** - Karpathy-style automated hyperparameter discovery (240+ experiments on 4 GPUs)
3. **GRPO** with AutoResearch-optimized config on 43K clean training samples
4. **Targeted SFT** on multi-turn, memory, and web search categories

## AutoResearch

The key innovation: an automated research loop where Claude Opus proposes hyperparameter changes, trains 5-minute GRPO experiments, and keeps improvements via a git-based ratchet.

### Key Discoveries
- **Learning rate**: Optimal for easy categories (lr=8e-6) differs from optimal for full benchmark (lr=2e-6)
- **Format bonus**: 0.7 is optimal for function calling but destroys irrelevance detection; 0.1 preserves both
- **Hyperparameter interactions**: `num_generations=8` failed at low LR but succeeded at high LR
- **Eval-based ratchet**: Using real eval scores instead of training reward prevents overfitting to proxy metrics

### Best Config (AutoResearch-discovered)
```json
{
    "lr": 2e-6,
    "beta": 0.01,
    "num_generations": 24,
    "batch_size": 2,
    "grad_accum": 4,
    "temperature": 0.6,
    "format_bonus": 0.1,
    "max_grad_norm": 1.5,
    "top_p": 0.9,
    "warmup_steps": 60,
    "max_steps": 260
}
```

## Repository Structure

```
autoresearch_grpo.py          # Single-GPU AutoResearch loop
autoresearch_parallel.py      # 4-GPU parallel AutoResearch
grpo_training.py              # Full GRPO training script
grpo_train_script.py          # GRPO script for AutoResearch experiments
grpo_recovery.py              # Dual-reward GRPO for capability recovery
sft_training.py               # SFT training script
eval_scorer.py                # Evaluation scorer with parameter matching
eval_detailed.py              # Detailed per-category evaluation
run_bfcl_eval.py              # Official BFCL V4 evaluation runner
generate_ground_truth.py      # Generate ground truth via Claude Opus
generate_multiturn_sft.py     # Generate multi-turn SFT data with real BFCL schemas
best_grpo_config.json         # AutoResearch-discovered optimal config
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("airev-ai/AIREV-Agent-0.8B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("airev-ai/AIREV-Agent-0.8B", trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are an expert in composing functions..."},
    {"role": "user", "content": "What's the weather in Dubai?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt += "\n<think>\n"  # Activate thinking tokens

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=512, temperature=0.6)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Citation

```bibtex
@article{khalid2026airev,
  title={AIREV-Agent-0.8B: Agentic AI on the Edge},
  author={Khalid, Muhammed},
  year={2026},
  note={AIREV, On-Demand.io}
}
```

## Acknowledgments

- **UAE leadership** for the innovation ecosystem
- **Dr. Andrew Jackson** for mentorship during 9-month incubation
- **Core42 AI Cloud** for H100 compute infrastructure
- **Qualcomm** and **Intel** for edge deployment support
