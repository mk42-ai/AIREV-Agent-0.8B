# === FROZEN HEADER === DO NOT MODIFY ANYTHING ABOVE THIS LINE ===
# This script is called with: --model MODEL --dataset DATASET --output-dir DIR --max-minutes MINS
# These 4 CLI arguments MUST always be accepted by argparse.
# CUDA_VISIBLE_DEVICES is set externally. Do NOT set it in this script.
# Do NOT use GPU 7 (index 7) — it is dead and will crash NCCL.
# The argparse section at the bottom MUST always have: --model, --dataset, --output-dir, --max-minutes
# Internal defaults: max_samples=2000, logging_steps=2, save_steps=50
# === END FROZEN HEADER ===
import os, sys, json, argparse, time, re, random, math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
os.environ.setdefault("WANDB_DISABLED","true")

BFCL_SYSTEM_PROMPT="""You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out and refuse to answer.
If the given question lacks the parameters required by the function, also point it out.

You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke.\n"""

def load_dataset(path, max_samples=2000):
    data=[]
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
            if len(data)>=max_samples:
                break
    return data

def extract_func_call_bracket(text):
    calls=[]
    m=re.search(r'\[(.+)\]',text,re.DOTALL)
    if not m:
        return calls
    inner=m.group(1)
    parts=re.split(r'\)\s*,\s*(?=[a-zA-Z_])',inner)
    for p in parts:
        p=p.strip()
        if not p.endswith(')'):
            p+=')'
        mm=re.match(r'([a-zA-Z_][\w.]*)\s*\((.*)\)',p,re.DOTALL)
        if mm:
            calls.append({"name":mm.group(1),"args_str":mm.group(2)})
    return calls

def compute_reward(generated, gold):
    gen_calls=extract_func_call_bracket(generated)
    gold_calls=extract_func_call_bracket(gold)
    if not gold_calls:
        return 0.5
    if not gen_calls:
        return 0.05
    score=0.0
    for gc in gold_calls:
        best=0.0
        for g in gen_calls:
            s=0.0
            if g["name"]==gc["name"]:
                s+=0.5
                if g["args_str"].strip()==gc["args_str"].strip():
                    s+=0.5
                else:
                    gen_vals=set(re.findall(r'[\w.]+',g["args_str"]))
                    gold_vals=set(re.findall(r'[\w.]+',gc["args_str"]))
                    if gold_vals:
                        overlap=len(gen_vals&gold_vals)/len(gold_vals)
                        s+=0.5*overlap
            best=max(best,s)
        score+=best
    return score/len(gold_calls)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=str,required=True)
    parser.add_argument("--dataset",type=str,required=True)
    parser.add_argument("--output-dir",type=str,required=True)
    parser.add_argument("--max-minutes",type=int,required=True)
    args=parser.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    start_time=time.time()
    deadline=start_time+args.max_minutes*60-120
    print(f"[INFO] Loading model from {args.model}")
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map="auto")
    model.config.use_cache=False
    print(f"[INFO] Loading dataset from {args.dataset}")
    dataset=load_dataset(args.dataset,max_samples=2000)
    random.shuffle(dataset)
    print(f"[INFO] Loaded {len(dataset)} samples")
    # Load hyperparams from config JSON (AutoResearch modifies this file)
    config_path=os.path.join(args.output_dir,"grpo_config.json")
    if not os.path.exists(config_path):
        config_path=os.path.join(os.path.dirname(args.output_dir),"grpo_config.json")
    if not os.path.exists(config_path):
        config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","grpo_config.json")
    if os.path.exists(config_path):
        with open(config_path) as cf:
            config=json.load(cf)
        print(f"[INFO] Loaded config from {config_path}")
    else:
        config={}
        print("[INFO] No config found, using defaults")
    lr=config.get("lr",3e-7)
    beta=config.get("beta",0.01)
    num_generations=config.get("num_generations",4)
    batch_size=config.get("batch_size",2)
    grad_accum=config.get("grad_accum",2)
    max_new_tokens=config.get("max_new_tokens",512)
    temperature=config.get("temperature",0.7)
    weight_decay=config.get("weight_decay",0.01)
    warmup_steps=config.get("warmup_steps",10)
    max_steps=config.get("max_steps",200)
    top_p=config.get("top_p",0.95)
    format_bonus=config.get("format_bonus",0.1)
    reward_threshold=config.get("reward_threshold",0.3)
    max_grad_norm=config.get("max_grad_norm",1.0)
    print(f"[CONFIG] lr={lr} beta={beta} gens={num_generations} bs={batch_size} temp={temperature} fmt_bonus={format_bonus}")
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    num_steps=min(len(dataset)//batch_size,max_steps)
    scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=num_steps)
    model.train()
    step=0
    best_reward=0.0
    all_rewards=[]
    save_count=0
    for batch_start in range(0,len(dataset)-batch_size+1,batch_size):
        if time.time()>deadline:
            print("[INFO] Time limit approaching, stopping training")
            break
        if step>=num_steps:
            break
        batch=dataset[batch_start:batch_start+batch_size]
        batch_loss=0.0
        optimizer.zero_grad()
        for sample in batch:
            # Extract from messages format (Opus 50K data)
            msgs=sample.get("messages",[])
            user_text=""
            gold=""
            system_text=""
            for msg in msgs:
                if msg["role"]=="system":
                    system_text=msg["content"]
                elif msg["role"]=="user":
                    user_text=msg["content"]
                elif msg["role"]=="assistant":
                    content=msg["content"]
                    if "</think>" in content:
                        gold=content.split("</think>")[-1].strip()
                    else:
                        gold=content.strip()
            # Use system prompt from training data (already has functions)
            if system_text:
                messages=[{"role":"system","content":system_text},{"role":"user","content":user_text}]
            else:
                messages=[{"role":"system","content":BFCL_SYSTEM_PROMPT},{"role":"user","content":user_text}]
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            if not prompt.endswith("\n"):
                prompt+="\n"
            prompt+="<think>\n"
            input_ids=tokenizer(prompt,return_tensors="pt",truncation=True,max_length=3072).input_ids.to(model.device)
            prompt_len=input_ids.shape[1]
            # Generate multiple completions
            rewards=[]
            all_logprobs=[]
            all_ref_logprobs=[]
            with torch.no_grad():
                outputs=model.generate(input_ids.expand(num_generations,-1),max_new_tokens=max_new_tokens,temperature=temperature,do_sample=True,top_p=top_p,pad_token_id=tokenizer.pad_token_id)
            for i in range(num_generations):
                gen_ids=outputs[i][prompt_len:]
                gen_text=tokenizer.decode(gen_ids,skip_special_tokens=True)
                if "</think>" in gen_text:
                    output_part=gen_text.split("</think>")[-1].strip()
                else:
                    output_part=gen_text.strip()
                r=compute_reward(output_part,gold)
                # Format bonus
                if re.search(r'\[.*\(.*\).*\]',output_part,re.DOTALL):
                    r+=format_bonus
                rewards.append(min(r,1.0))
                # Compute log probs
                full_ids=outputs[i:i+1]
                with torch.no_grad():
                    logits=model(full_ids).logits[:,prompt_len-1:-1,:]
                    lp=F.log_softmax(logits,dim=-1)
                    token_lp=lp.gather(2,gen_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                    all_logprobs.append(token_lp)
                    all_ref_logprobs.append(token_lp.detach())
            # GRPO: compute advantages
            rewards_t=torch.tensor(rewards,device=model.device)
            mean_r=rewards_t.mean()
            std_r=rewards_t.std()+1e-8
            advantages=(rewards_t-mean_r)/std_r
            # Policy gradient loss on best completions
            best_idx=rewards_t.argmax().item()
            if rewards[best_idx]>reward_threshold:
                gen_ids_best=outputs[best_idx][prompt_len:]
                full_best=outputs[best_idx:best_idx+1]
                logits_train=model(full_best).logits[:,prompt_len-1:-1,:]
                lp_train=F.log_softmax(logits_train,dim=-1)
                token_lp_train=lp_train.gather(2,gen_ids_best.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                pg_loss=-(advantages[best_idx]*token_lp_train.mean())
                # KL penalty
                with torch.no_grad():
                    ref_lp=all_ref_logprobs[best_idx]
                kl=torch.mean(token_lp_train-ref_lp[:,:token_lp_train.shape[1]])
                loss=(pg_loss+beta*kl)/(batch_size*grad_accum)
                loss.backward()
                batch_loss+=loss.item()
        if (step+1)%grad_accum==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        step+=1
        avg_r=sum(rewards)/len(rewards) if rewards else 0
        if step%2==0:
            elapsed=time.time()-start_time
            print(f"[Step {step}/{num_steps}] loss={batch_loss:.4f} avg_reward={avg_r:.3f} time={elapsed:.0f}s")
        all_rewards.append(avg_r)
        if avg_r>best_reward:
            best_reward=avg_r
        if step%50==0 or step==num_steps:
            save_path=os.path.join(args.output_dir,f"checkpoint-{step}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"[INFO] Saved checkpoint to {save_path}")
            save_count+=1
    # Final save
    print(f"[INFO] Training complete. Best avg reward: {best_reward:.3f}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] Final model saved to {args.output_dir}")
    # Write metrics for AutoResearch ratchet
    
    avg20=sum(all_rewards[-20:])/max(len(all_rewards[-20:]),1) if all_rewards else 0
    avg_all=sum(all_rewards)/max(len(all_rewards),1) if all_rewards else 0
    metrics={"total_steps":step,"elapsed_minutes":(time.time()-start_time)/60,"final_reward_avg20":avg20,"avg_reward_all":avg_all,"best_reward":best_reward,"final_think_avg20":0,"final_value_avg20":0,"final_format_avg20":0}
    with open(os.path.join(args.output_dir,"metrics.json"),"w") as mf:
        json.dump(metrics,mf,indent=2)
    print(f"[INFO] Metrics saved: avg20={avg20:.4f} avg_all={avg_all:.4f} best={best_reward:.4f} steps={step}")

if __name__=="__main__":
    main()