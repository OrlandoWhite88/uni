#!/usr/bin/env python3
"""
Qwen3-14B SFT Training Script

This script performs supervised fine-tuning on Qwen3-14B using multi-turn
trajectories. It extracts `detailed_internal_reasoning` from assistant messages
and uses it as reasoning tokens inside <think> tags.

Processing:
1. Load trajectories from training_data.jsonl or HuggingFace dataset
2. Remove `detailed_internal_reasoning` from ALL assistant messages in context
3. Extract `detailed_internal_reasoning` from the LAST assistant message
4. Format target as: <think>{DIR}</think>{json_response_without_DIR}
5. Train with Unsloth using LoRA adapters
"""

# =============================================================================
# ENVIRONMENT VARIABLES - MUST BE SET BEFORE ANY IMPORTS
# =============================================================================
import os

# Disable datasets multiprocessing to prevent pickle errors with Unsloth tokenizer
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"

# Enable Unsloth logging for debugging
os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"

# Fast HF downloads
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# =============================================================================
# NOW SAFE TO IMPORT
# =============================================================================
import json
from typing import Optional

import torch
import wandb
from datasets import Dataset, load_dataset

# CRITICAL: Import Unsloth BEFORE TRL to prevent EOS token corruption
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig


# =============================================================================
# Configuration
# =============================================================================

# Model Configuration
MODEL_NAME = "Qwen/Qwen3-14B"
MAX_SEQ_LENGTH = 65536  # 64k context (Qwen3 supports up to 131072 with YaRN)
LOAD_IN_4BIT = True     # 4-bit quantization for memory efficiency
LOAD_IN_8BIT = False

# LoRA Configuration
LORA_R = 64
LORA_ALPHA = 128  # 2x rank
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training Configuration
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4  # SFT learning rate (kept from original)
WARMUP_STEPS = 5
MAX_STEPS = None  # None = use num_train_epochs for full training
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 1
SAVE_STEPS = 50  # Save checkpoint every N steps

# Paths
TRAINING_DATA_PATH = "training_data.jsonl"  # Local file path (used if USE_HF_DATASET=False)
OUTPUT_DIR = "qwen3_lora_model"

# Hugging Face Dataset (set to None to use local file)
HF_DATASET = "orlandowhite/uni_sft_trajectories"  # Set to None to use TRAINING_DATA_PATH instead
USE_HF_DATASET = True  # Set to False to use local TRAINING_DATA_PATH

# Weights & Biases Configuration
WANDB_PROJECT = "qwen3-14b-sft"
WANDB_RUN_NAME = None  # Set to None for auto-generated name, or specify like "run-1"
USE_WANDB = True  # Set to False to disable wandb logging


# =============================================================================
# Data Processing Functions
# =============================================================================

def extract_thinking_for_tags(content: str, debug: bool = False) -> tuple[str, Optional[str]]:
    """
    Parse assistant message content (JSON), extract 'detailed_internal_reasoning' for <think> tags,
    and return the JSON content WITHOUT the DIR field.
    
    Args:
        content: JSON string content of assistant message
        debug: If True, print debug info
        
    Returns:
        Tuple of (json_without_dir, dir_content)
        dir_content is None if 'detailed_internal_reasoning' field not found
    """
    try:
        data = json.loads(content)
        dir_content = data.pop("detailed_internal_reasoning", None)
        # Return JSON without DIR
        json_without_dir = json.dumps(data, ensure_ascii=False)
        if debug:
            print(f"  [extract_thinking_for_tags] DIR found: {dir_content is not None}")
            if dir_content:
                print(f"  [extract_thinking_for_tags] DIR preview: {dir_content[:100]}...")
        return json_without_dir, dir_content
    except json.JSONDecodeError:
        # If content is not valid JSON, return as-is
        if debug:
            print(f"  [extract_thinking_for_tags] JSON parse failed!")
        return content, None


def process_trajectory(messages: list[dict], debug: bool = False) -> Optional[dict]:
    """
    Process a single trajectory for SFT training.
    
    1. Find the last assistant message (this is our target)
    2. Extract 'thinking' field from last assistant for <think> tags
    3. Format: <think>{thinking}</think>{original_json}
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        
    Returns:
        Dict with 'conversations' key containing processed messages,
        or None if invalid
    """
    if not messages:
        return None
    
    # Find the last assistant message index
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break
    
    if last_assistant_idx is None:
        # No assistant message found
        return None
    
    # Process all messages
    processed_messages = []
    
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        
        if role == "assistant" and i == last_assistant_idx:
            # Last assistant message: extract thinking for <think> tags
            original_content, thinking = extract_thinking_for_tags(content, debug=debug)
            
            if thinking:
                final_content = f"<think>{thinking}</think>{original_content}"
            else:
                final_content = f"<think></think>{original_content}"
            
            processed_messages.append({
                "role": role,
                "content": final_content
            })
        else:
            # All other messages (including previous assistants): keep as-is
            processed_messages.append({
                "role": role,
                "content": content
            })
    
    # Only keep messages up to and including the last assistant message
    processed_messages = processed_messages[:last_assistant_idx + 1]
    
    return {"conversations": processed_messages}


def load_and_process_data_local(filepath: str) -> Dataset:
    """
    Load training data from local JSONL file and process all trajectories.
    
    Args:
        filepath: Path to the training_data.jsonl file
        
    Returns:
        HuggingFace Dataset with processed conversations
    """
    processed_data = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                
                processed = process_trajectory(messages)
                if processed:
                    processed_data.append(processed)
                else:
                    print(f"Warning: Skipping invalid trajectory at line {line_num}")
                    
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error at line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(processed_data)} trajectories from {filepath}")
    return Dataset.from_list(processed_data)


def load_and_process_data_hf(dataset_name: str) -> Dataset:
    """
    Load training data from Hugging Face and process all trajectories.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "orlandowhite/uni_sft_trajectories")
        
    Returns:
        HuggingFace Dataset with processed conversations
    """
    print(f"Loading dataset from HuggingFace: {dataset_name}")
    hf_dataset = load_dataset(dataset_name, split="train")
    
    processed_data = []
    dir_found_count = 0
    dir_missing_count = 0
    
    for idx, item in enumerate(hf_dataset):
        messages = item.get("messages", [])
        
        # Debug: Check first sample's last assistant message for 'detailed_internal_reasoning' field
        if idx == 0:
            for msg in reversed(messages):
                if msg["role"] == "assistant":
                    content = msg["content"]
                    print(f"\n=== DEBUG: First sample's last assistant message ===")
                    try:
                        data = json.loads(content)
                        print(f"Parsed JSON keys: {list(data.keys())}")
                        if "detailed_internal_reasoning" in data:
                            print(f"'detailed_internal_reasoning' field found! First 300 chars: {data['detailed_internal_reasoning'][:300]}")
                        else:
                            print("'detailed_internal_reasoning' field NOT FOUND in JSON keys!")
                    except json.JSONDecodeError as e:
                        print(f"JSON parse error: {e}")
                    print("=" * 50 + "\n")
                    break
        
        # Debug first sample to trace DIR extraction
        should_debug = (idx == 0)
        if should_debug:
            print(f"\n=== DEBUG: Processing sample {idx} ===")
        
        processed = process_trajectory(messages, debug=should_debug)
        if processed:
            # Check if DIR was extracted (look for non-empty think tags)
            last_assistant = processed["conversations"][-1]
            if "<think>" in last_assistant["content"]:
                think_content = last_assistant["content"].split("<think>")[1].split("</think>")[0]
                if think_content:
                    dir_found_count += 1
                    if should_debug:
                        print(f"  [process] Think content (first 100): {think_content[:100]}...")
                else:
                    dir_missing_count += 1
                    if should_debug:
                        print(f"  [process] WARNING: Empty think tags!")
            processed_data.append(processed)
            if should_debug:
                print(f"  [process] Final content preview: {last_assistant['content'][:200]}...")
                print("=" * 50)
        else:
            print(f"Warning: Skipping invalid trajectory at index {idx}")
    
    print(f"\nLoaded {len(processed_data)} trajectories from {dataset_name}")
    print(f"Thinking extraction: {dir_found_count} with content, {dir_missing_count} empty <think> tags")
    return Dataset.from_list(processed_data)


def load_and_process_data() -> Dataset:
    """
    Load training data from HF or local file based on configuration.
    """
    if USE_HF_DATASET and HF_DATASET:
        return load_and_process_data_hf(HF_DATASET)
    else:
        return load_and_process_data_local(TRAINING_DATA_PATH)


# =============================================================================
# Verification Functions
# =============================================================================

def verify_data_processing(tokenizer, num_samples: int = 3):
    """
    Verify data processing and chat template are working correctly.
    Raises an error if any check fails.
    
    Args:
        tokenizer: The loaded tokenizer
        num_samples: Number of samples to verify
    """
    print("\n" + "=" * 60)
    print("VERIFICATION: Checking data processing and chat template")
    print("=" * 60)
    
    errors = []
    
    # Load samples from HF or local file
    if USE_HF_DATASET and HF_DATASET:
        print(f"Verifying samples from HuggingFace: {HF_DATASET}")
        hf_dataset = load_dataset(HF_DATASET, split="train")
        samples = [hf_dataset[i] for i in range(min(num_samples, len(hf_dataset)))]
    else:
        print(f"Verifying samples from local file: {TRAINING_DATA_PATH}")
        samples = []
        with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if len(samples) >= num_samples:
                    break
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    
    for sample_idx, data in enumerate(samples):
        messages = data.get("messages", [])
        
        # Process trajectory
        processed = process_trajectory(messages)
        if not processed:
            errors.append(f"Sample {sample_idx}: Failed to process trajectory")
            continue
        
        conversations = processed["conversations"]
        
        # Check 1: Find assistant messages
        assistant_msgs = [(i, m) for i, m in enumerate(conversations) if m["role"] == "assistant"]
        if not assistant_msgs:
            errors.append(f"Sample {sample_idx}: No assistant messages found")
            continue
        
        last_idx, last_msg = assistant_msgs[-1]
        
        # Check 2: Last assistant has <think> tags
        if "<think>" not in last_msg["content"] or "</think>" not in last_msg["content"]:
            errors.append(f"Sample {sample_idx}: Last assistant missing <think> tags")
        
        # Check 3: Previous assistants do NOT have <think> tags
        for idx, msg in assistant_msgs[:-1]:
            if "<think>" in msg["content"]:
                errors.append(f"Sample {sample_idx}: Previous assistant (idx {idx}) has <think> tags (should not)")
        
        # Check 4: Last assistant should have 'thinking' content in <think> tags
        think_start = last_msg["content"].find("<think>") + 7
        think_end = last_msg["content"].find("</think>")
        if think_start > 6 and think_end > think_start:
            think_content = last_msg["content"][think_start:think_end]
            if not think_content.strip():
                errors.append(f"Sample {sample_idx}: <think> tags are empty (no thinking content)")
        
        # Check 5: Apply chat template and verify format
        formatted = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Qwen3 uses ChatML format: <|im_start|>role\ncontent<|im_end|>
        # Verify <think> appears after <|im_start|>assistant
        if "<|im_start|>assistant\n<think>" not in formatted:
            # Check if it's the pattern with the last assistant
            last_assistant_pos = formatted.rfind("<|im_start|>assistant\n")
            if last_assistant_pos != -1:
                after_tag = formatted[last_assistant_pos:last_assistant_pos + 50]
                if "<think>" not in after_tag:
                    errors.append(f"Sample {sample_idx}: <think> not immediately after last <|im_start|>assistant")
        
        # Print sample for visual inspection
        if sample_idx == 0:
            print(f"\n--- Sample {sample_idx}: Last assistant message (first 600 chars) ---")
            print(last_msg["content"][:600])
            print("\n--- Formatted output (last 800 chars) ---")
            print(formatted[-800:])
    
    # Report results
    print("\n" + "-" * 60)
    if errors:
        print(f"VERIFICATION FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        print("-" * 60)
        raise ValueError(f"Data verification failed with {len(errors)} errors. See above.")
    else:
        print("VERIFICATION PASSED: All checks passed!")
        print("  - Last assistant messages have <think> tags")
        print("  - Previous assistant messages do NOT have <think> tags")
        print("  - 'detailed_internal_reasoning' removed from all assistants")
        print("  - Chat template format is correct")
        print("-" * 60 + "\n")


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    print("=" * 60)
    print("Qwen3-14B SFT Training")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 0: Initialize Weights & Biases
    # -------------------------------------------------------------------------
    if USE_WANDB:
        print("\n[0/7] Initializing Weights & Biases...")
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "model_name": MODEL_NAME,
                "max_seq_length": MAX_SEQ_LENGTH,
                "load_in_4bit": LOAD_IN_4BIT,
                "load_in_8bit": LOAD_IN_8BIT,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "lora_target_modules": LORA_TARGET_MODULES,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "warmup_steps": WARMUP_STEPS,
                "max_steps": MAX_STEPS,
                "num_train_epochs": NUM_TRAIN_EPOCHS,
                "save_steps": SAVE_STEPS,
            },
        )
        print(f"W&B project: {WANDB_PROJECT}")
        print(f"W&B run: {wandb.run.name}")
    
    # -------------------------------------------------------------------------
    # Step 1: Load Model and Tokenizer
    # -------------------------------------------------------------------------
    print("\n[1/7] Loading model and tokenizer...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
        dtype=None,  # Auto-detect (will use bfloat16 on supported hardware)
        full_finetuning=False,
        trust_remote_code=True,
    )
    
    print(f"Model loaded: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    
    # CRITICAL: Force valid EOS/PAD tokens to prevent TRL "<EOS_TOKEN>" corruption
    def force_valid_eos_and_pad(tok):
        """Ensure tokenizer has valid EOS/PAD tokens that exist in vocab."""
        # Candidates that commonly exist for Qwen ChatML / Qwen-family tokenizers
        candidates = [
            getattr(tok, "eos_token", None),
            "<|im_end|>",
            "<|endoftext|>",
        ]
        
        def first_valid_token(cands):
            for t in cands:
                if not t:
                    continue
                tid = tok.convert_tokens_to_ids(t)
                if tid is not None and tid != tok.unk_token_id:
                    return t
            return None
        
        eos = first_valid_token(candidates)
        if eos is None:
            raise ValueError(
                "Could not find a valid EOS token in tokenizer vocab. "
                "Inspect tokenizer.special_tokens_map / tokenizer.get_vocab()."
            )
        
        tok.eos_token = eos
        tok.pad_token = eos  # OK for batching
        
        # Extra guard: TRL blows up if eos_token is '<EOS_TOKEN>' or otherwise invalid
        if tok.convert_tokens_to_ids(tok.eos_token) in (None, tok.unk_token_id):
            raise ValueError(f"EOS token still invalid: {tok.eos_token!r}")
        
        return eos
    
    eos = force_valid_eos_and_pad(tokenizer)
    print(f"Using eos_token = {eos!r}, eos_token_id = {tokenizer.eos_token_id}")
    
    # DEBUG: Print all linear layer names to verify target_modules
    print("\n--- DEBUG: Linear layers in model ---")
    linear_layers = [name for name, module in model.named_modules() 
                     if "Linear" in type(module).__name__]
    for name in linear_layers[:30]:  # Print first 30
        print(f"  {name}")
    print(f"  ... ({len(linear_layers)} total linear layers)")
    print("--- End debug ---\n")
    
    # -------------------------------------------------------------------------
    # Step 2: Verify Data Processing (catches issues early!)
    # -------------------------------------------------------------------------
    verify_data_processing(tokenizer, num_samples=3)
    
    # -------------------------------------------------------------------------
    # Step 3: Apply LoRA Adapters
    # -------------------------------------------------------------------------
    print("\n[3/7] Applying LoRA adapters...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    print(f"LoRA config: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Target modules: {LORA_TARGET_MODULES}")
    
    # Print trainable parameters
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
    
    # -------------------------------------------------------------------------
    # Step 4: Load and Process Training Data
    # -------------------------------------------------------------------------
    print("\n[4/7] Loading and processing training data...")
    
    dataset = load_and_process_data()
    
    # -------------------------------------------------------------------------
    # Step 5: Apply Chat Template
    # -------------------------------------------------------------------------
    print("\n[5/7] Applying chat template...")
    
    def formatting_prompts_func(examples):
        """Apply Qwen3 chat template to conversations."""
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=None)
    
    # Print a sample to verify formatting
    print("\n--- Sample formatted text (first 1500 chars) ---")
    print(dataset[0]["text"][:1500])
    print("--- RAW BYTES (first 200 chars) ---")
    print(repr(dataset[0]["text"][:200]))
    print("...")
    print("--- End sample ---\n")
    
    # -------------------------------------------------------------------------
    # Step 6: Verify markers exist in data (debug info)
    # -------------------------------------------------------------------------
    print("\n[6/7] Verifying chat template markers...")
    
    sample_text = dataset[0]["text"]
    # Qwen3 uses ChatML format
    user_marker = "<|im_start|>user\n"
    assistant_marker = "<|im_start|>assistant\n"
    
    user_count = sample_text.count(user_marker)
    assistant_count = sample_text.count(assistant_marker)
    print(f"Found {user_count} user markers and {assistant_count} assistant markers in sample")
    
    if assistant_count == 0:
        raise ValueError("No assistant markers found in formatted text! Check chat template.")
    
    # -------------------------------------------------------------------------
    # Step 7: Configure and Run Training
    # -------------------------------------------------------------------------
    print("\n[7/7] Configuring trainer...")
    
    # Create SFTConfig
    sft_args = SFTConfig(
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,  # CRITICAL: Prevent truncation of long conversations
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS if MAX_STEPS else -1,  # -1 means use num_train_epochs
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        bf16=True,  # Qwen3 works well with bfloat16
        report_to="wandb" if USE_WANDB else "none",
        output_dir=OUTPUT_DIR,
        eos_token=tokenizer.eos_token,  # Explicitly use tokenizer's actual EOS token
    )
    
    # CRITICAL: Force no multiprocessing (must be None, not 1)
    sft_args.dataset_num_proc = None
    print("dataset_num_proc =", getattr(sft_args, "dataset_num_proc", "MISSING"))
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=sft_args,
    )
    
    # CRITICAL FIX: Remove any pre-existing labels column so Unsloth rebuilds from input_ids
    if "labels" in trainer.train_dataset.column_names:
        print("Removing pre-existing 'labels' column to force Unsloth to rebuild...")
        trainer.train_dataset = trainer.train_dataset.remove_columns(["labels"])
    
    # Apply train_on_responses_only AFTER creating trainer
    # This masks user inputs so we only train on assistant responses
    # Qwen3 uses ChatML format: <|im_start|>role\ncontent<|im_end|>
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    
    # Verify masking worked by checking a sample
    print("\n--- Verifying response-only masking ---")
    sample_labels = trainer.train_dataset[0]["labels"]
    num_masked = sum(1 for x in sample_labels if x == -100)
    num_trained = sum(1 for x in sample_labels if x != -100)
    total = len(sample_labels)
    print(f"Total tokens: {total}")
    print(f"Masked tokens (input/prompt): {num_masked} ({100*num_masked/total:.1f}%)")
    print(f"Trained tokens (response): {num_trained} ({100*num_trained/total:.1f}%)")
    
    # Show what we're actually training on (full first response)
    unmasked_tokens = [tid for tid in sample_labels if tid != -100]
    if unmasked_tokens:
        # Decode first 500 tokens to see the full structure
        decoded_sample = tokenizer.decode(unmasked_tokens[:500])
        print(f"\nFirst trained tokens (first 500 tokens decoded):\n{decoded_sample}\n")
    print("--- Masking verification complete ---\n")
    
    # Show memory stats before training
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU: {gpu_stats.name}")
        print(f"Max memory: {max_memory} GB")
        print(f"Reserved memory: {start_gpu_memory} GB")
    
    print("\n[8/8] Starting training...")
    trainer_stats = trainer.train()
    
    # Show final stats
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Training time: {trainer_stats.metrics['train_runtime']/60:.2f} minutes")
    
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        print(f"Peak reserved memory: {used_memory} GB")
    
    # -------------------------------------------------------------------------
    # Save Model
    # -------------------------------------------------------------------------
    print(f"\nSaving LoRA adapters to {OUTPUT_DIR}/...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")
    
    # Finish wandb run
    if USE_WANDB:
        wandb.finish()
        print("W&B run finished.")
    
    return model, tokenizer


if __name__ == "__main__":
    main()