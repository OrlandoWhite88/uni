#!/usr/bin/env python3
"""
Simple test: TRAINING memory scaling to detect O(n) vs O(n²) attention.

BUG FOUND: Unsloth checks for "gpt_oss" (underscore) but model name has "gpt-oss" (hyphen).
Fix: Set UNSLOTH_MODEL_NAME with underscore before importing unsloth.
"""

import os

# ============================================================================
# FIX: Force Unsloth to enable Flex Attention for GPT-OSS
# Bug: /usr/local/lib/python3.12/dist-packages/unsloth_zoo/temporary_patches/gpt_oss.py:741
#      checks for "gpt_oss" (underscore) but model name has "gpt-oss" (hyphen)
# ============================================================================
os.environ["UNSLOTH_MODEL_NAME"] = "gpt_oss"  # Force match with underscore
os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "1"  # Enable flex attention

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

import gc
import torch

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def main():
    print("=" * 70)
    print(" TRAINING MEMORY SCALING TEST (with Flex Attention fix)")
    print("=" * 70)
    
    print(f"\nEnvironment variables set:")
    print(f"  UNSLOTH_MODEL_NAME = {os.environ.get('UNSLOTH_MODEL_NAME', 'NOT SET')}")
    print(f"  UNSLOTH_ENABLE_FLEX_ATTENTION = {os.environ.get('UNSLOTH_ENABLE_FLEX_ATTENTION', 'NOT SET')}")
    
    from unsloth import FastLanguageModel
    
    model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    
    print(f"\nLoading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=131072,  # 128k to test up to 100k
        load_in_4bit=True,
    )
    
    # Apply LoRA exactly like in training
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    model.train()
    
    # Check what attention implementation we got
    attn_impl = getattr(model.config, '_attn_implementation', 'NOT SET')
    print(f"\nModel config._attn_implementation: {attn_impl}")
    
    # Check for flex attention in modules
    for name, module in model.named_modules():
        module_name = type(module).__name__
        if 'flex' in module_name.lower() or 'flash' in module_name.lower():
            print(f"  ✓ Found efficient attention: {name} -> {module_name}")
            break
    else:
        # Check the actual attention class
        for name, module in model.named_modules():
            if 'self_attn' in name and not any(x in name for x in ['proj', 'norm']):
                print(f"  Attention module: {name} -> {type(module).__name__}")
                break
    
    # Test increasing sequence lengths
    test_lengths = [4096, 8192, 16384, 32768, 45000, 55000, 65536, 75000, 85000, 100000]
    results = []
    
    print("\n" + "=" * 70)
    print(" TRAINING: forward + backward pass")
    print("=" * 70)
    
    for seq_len in test_lengths:
        cleanup()
        torch.cuda.reset_peak_memory_stats()
        
        input_ids = torch.randint(100, 10000, (1, seq_len), device="cuda")
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        try:
            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward - this is where O(n²) would explode
            loss.backward()
            
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            results.append((seq_len, peak_gb))
            print(f"  seq_len={seq_len:5d}: {peak_gb:6.2f} GB")
            
            model.zero_grad(set_to_none=True)
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"  seq_len={seq_len:5d}: 💥 OOM")
            results.append((seq_len, float('inf')))
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
        
        del input_ids, attention_mask, labels
        cleanup()
    
    # Analyze
    print("\n" + "=" * 70)
    print(" SCALING ANALYSIS")
    print("=" * 70)
    
    for i in range(1, len(results)):
        if results[i][1] != float('inf') and results[i-1][1] != float('inf'):
            len_ratio = results[i][0] / results[i-1][0]
            mem_ratio = results[i][1] / results[i-1][1]
            
            if mem_ratio > 3.5:
                verdict = "⚠️ O(n²) - EAGER ATTENTION!"
            elif mem_ratio > 2.5:
                verdict = "~ Suspicious"
            else:
                verdict = "✓ O(n) - Flash/SDPA working"
            
            print(f"  {results[i-1][0]:5d} -> {results[i][0]:5d}: "
                  f"len x{len_ratio:.1f}, mem x{mem_ratio:.2f}  {verdict}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
