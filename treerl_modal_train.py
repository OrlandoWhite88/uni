#!/usr/bin/env python3
"""
TreeRL GRPO Training on Modal

Runs the TreeRL training pipeline on Modal's serverless GPUs.
This abstracts away local GPU infrastructure entirely.

Usage:
    # Deploy and run training
    modal run treerl_modal_train.py --chapter 84 --num-rulings 20 --epochs 3

    # Run with specific GPU
    modal run treerl_modal_train.py --gpu h100 --chapter 84
    
    # Deploy as a persistent app (can trigger via API)
    modal deploy treerl_modal_train.py
"""

import modal
import os
from pathlib import Path

# ============================================================================
# MODAL CONFIGURATION
# ============================================================================

# Create the Modal app
app = modal.App("treerl-grpo-training")

# Persistent volume for checkpoints, samples, and model artifacts
# This persists across runs and can be accessed by other Modal functions
volume = modal.Volume.from_name("treerl-training-data", create_if_missing=True)

# GPU options mapping
GPU_OPTIONS = {
    "a100": "A100",
    "a100-40gb": "A100",  
    "a100-80gb": "A100-80GB",
    "h100": "H100",
    "h200": "H200",
    "h200x4": "H200:4",  # 4x H200 for 128k context
    "l4": "L4",
    "a10g": "A10G",
    "t4": "T4",
}

# =============================================================================
# IMAGE CONFIGURATION - Pinned to working H200 setup (2026-01-09)
# =============================================================================
# Key versions from verified H200 install:
#   - Python: 3.10.12
#   - torch: 2.9.0+cu128  
#   - CUDA: 12.8
#   - triton: 3.5.0
#   - mamba-ssm: 2.2.6.post3 (required for Nemotron-H)
#   - causal-conv1d: 1.5.3.post1 (required for Nemotron-H)

def build_image():
    """Build Modal image with Unsloth, vLLM, and project dependencies."""
    
    return (
        # Use CUDA 12.8 base to match working H200 setup
        modal.Image.from_registry(
            "nvidia/cuda:12.8.0-devel-ubuntu22.04",
            add_python="3.10",
        )
        # System dependencies
        .apt_install(
            "git",
            "curl",
            "build-essential",
            "g++",              # C++ compiler for CUDA extensions
            "ninja-build",      # Fast builds for mamba-ssm/causal-conv1d
            "python3.10-dev",
        )
        # Set compiler to g++ (not clang++)
        .env({
            "CC": "gcc",
            "CXX": "g++",
        })
        # PyTorch 2.9.0 with CUDA 12.8 - exact working version
        .pip_install(
            "torch==2.9.0",
            "torchvision==0.24.0",
            "triton==3.5.0",
            extra_index_url="https://download.pytorch.org/whl/cu128",
        )
        # Build tools for CUDA extensions
        .pip_install(
            "ninja",       # Fast parallel builds
            "packaging",   # Required by causal-conv1d setup
            "wheel",
        )
        # Mamba/Causal-Conv1d - REQUIRED for Nemotron-H architecture
        # Built from source with CUDA support
        .pip_install(
            "causal-conv1d==1.5.3.post1",
            "mamba-ssm==2.2.6.post3",
        )
        # Unsloth - pinned to exact working commits
        .run_commands(
            "pip install --no-deps 'unsloth @ git+https://github.com/unslothai/unsloth.git@010775fbdebecf3f413002e593161393c72c0a09'",
            "pip install --no-deps 'unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo.git@c315ec1b0782a43893f34ed1dc264de9f2600236'",
        )
        # Training stack - exact versions from working setup
        .pip_install(
            "transformers==4.56.2",
            "tokenizers==0.22.2",
            "trl==0.22.2",
            "peft>=0.11.0",
            "accelerate>=0.30.0",
            "bitsandbytes==0.49.1",
            "xformers",
        )
        # vLLM for fast inference during rollouts
        .pip_install("vllm>=0.8.0")
        # Training utilities
        .pip_install(
            "datasets",
            "sentencepiece",
            "protobuf",
            "einops",
            "safetensors",
        )
        # Utilities and API dependencies (from requirements.txt)
        .pip_install(
            "requests",
            "openai",
            "setuptools",  # Often needed for package builds
            "rank-bm25",   # BM25 search for cross rulings
            "pandas",      # Data processing
            "openpyxl",    # Excel file support
            "fastapi",     # API framework
            "uvicorn",     # ASGI server
            "pydantic",    # Data validation
            "python-multipart",  # Form data handling
            "google-auth",       # Google Cloud auth
            "google-genai",      # Gemini API
            "groq",              # Groq API
        )
        # Set environment variables
        .env({
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "TREERL_BEAM_SIZE": "4",
            "DISABLE_CROSS_RULING_INJECTION": "true",
        })
    )


image = build_image()

# ============================================================================
# MODAL VOLUME PATHS
# ============================================================================

# Volume mount point inside Modal container
VOLUME_PATH = "/data"

# Directories within the volume
CHECKPOINTS_DIR = f"{VOLUME_PATH}/checkpoints"
SAMPLES_DIR = f"{VOLUME_PATH}/samples"
ADAPTER_SYNC_DIR = f"{VOLUME_PATH}/adapter_sync"
LOGS_DIR = f"{VOLUME_PATH}/logs"
RULINGS_FILE = f"{VOLUME_PATH}/cross_rulings_dataset.json"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=300,  # 5 minutes for upload
)
def upload_rulings(rulings_data: list):
    """Upload rulings dataset to Modal volume."""
    import json
    
    os.makedirs(os.path.dirname(RULINGS_FILE), exist_ok=True)
    with open(RULINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(rulings_data, f, ensure_ascii=False, indent=2)
    
    volume.commit()
    return f"Uploaded {len(rulings_data)} rulings to {RULINGS_FILE}"


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=60,
)
def list_checkpoints():
    """List available checkpoints in the volume."""
    checkpoints = []
    
    if os.path.exists(CHECKPOINTS_DIR):
        for item in os.listdir(CHECKPOINTS_DIR):
            item_path = os.path.join(CHECKPOINTS_DIR, item)
            if os.path.isdir(item_path):
                checkpoints.append(item)
    
    return sorted(checkpoints)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=300,
)
def download_checkpoint(checkpoint_name: str) -> bytes:
    """Download a checkpoint as a tar archive."""
    import tarfile
    import io
    
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")
    
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        tar.add(checkpoint_path, arcname=checkpoint_name)
    
    return buffer.getvalue()


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

@app.function(
    image=image,
    gpu="H200:4",  # 4x H200 GPUs for 128k context training
    volumes={VOLUME_PATH: volume},
    timeout=86400,  # 24 hours max
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=60.0,
    ),
    # Memory for large models with multi-GPU
    memory=65536,  # 64GB system RAM
)
def train_treerl(
    # Model settings
    base_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    sft_adapter: str = "orlandowhite/nemotron3_nano_sft",
    max_seq_length: int = 128000,  # 128k context
    train_max_seq_length: int = 65536,  # 64k per sample (distributed)
    load_in_4bit: bool = True,
    
    # Training settings
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    
    # Data settings
    chapter: str = "84",
    num_rulings_per_epoch: int = 20,
    
    # TreeRL settings
    beam_size: int = 4,
    max_questions: int = 3,
    
    # vLLM settings
    vllm_gpu_memory_utilization: float = 0.90,
    vllm_max_model_len: int = 128000,  # 128k context
    vllm_tensor_parallel_size: int = 4,  # Distribute across 4 GPUs
    
    # Resume from checkpoint
    resume_from: str = None,
):
    """
    Main TreeRL GRPO training function running on Modal GPU.
    
    This implements the full training loop:
    1. Load/merge SFT adapter
    2. Per epoch:
       a. Start vLLM server for rollouts
       b. Collect samples from all rulings
       c. Stop vLLM, train with Unsloth
       d. Save checkpoint
    """
    import sys
    import json
    import time
    import logging
    import gc
    import random
    from datetime import datetime
    
    import torch
    
    # Setup logging
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger("treerl_modal")
    
    logger.info("=" * 70)
    logger.info("TREERL GRPO TRAINING ON MODAL")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Base model: {base_model}")
    logger.info(f"SFT adapter: {sft_adapter}")
    logger.info(f"Chapter: {chapter}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Rulings per epoch: {num_rulings_per_epoch}")
    
    # Create directories
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(ADAPTER_SYNC_DIR, exist_ok=True)
    
    # Load rulings
    if not os.path.exists(RULINGS_FILE):
        raise FileNotFoundError(
            f"Rulings file not found at {RULINGS_FILE}. "
            "Please upload it first using upload_rulings()"
        )
    
    with open(RULINGS_FILE, 'r', encoding='utf-8') as f:
        all_rulings = json.load(f)
    
    chapter_rulings = [r for r in all_rulings if r.get("hts_code", "").startswith(chapter)]
    logger.info(f"Loaded {len(chapter_rulings)} rulings for chapter {chapter}")
    
    if not chapter_rulings:
        raise ValueError(f"No rulings found for chapter {chapter}")
    
    # =============================================
    # PHASE 0: PRE-MERGE SFT ADAPTER
    # =============================================
    
    def merge_sft_adapter():
        """Merge SFT adapter for vLLM serving."""
        if not sft_adapter:
            return None
        
        merged_path = os.path.join(ADAPTER_SYNC_DIR, "sft_merged")
        
        # Check if already merged
        if os.path.exists(os.path.join(merged_path, "config.json")):
            logger.info(f"SFT merged model already exists: {merged_path}")
            return merged_path
        
        logger.info(f"Merging SFT adapter: {sft_adapter}")
        
        from unsloth import FastLanguageModel
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_adapter,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
        )
        
        model.save_pretrained_merged(
            merged_path,
            tokenizer,
            save_method="merged_16bit",
        )
        
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"SFT merged model saved: {merged_path}")
        volume.commit()  # Persist to volume
        
        return merged_path
    
    # =============================================
    # PHASE 1-5: TRAINING LOOP
    # =============================================
    
    def run_vllm_rollouts(model_path, epoch_rulings):
        """Run vLLM inference for rollouts (inline, not as server)."""
        import subprocess
        import requests
        import threading
        
        logger.info(f"Starting vLLM server for rollouts...")
        
        # Start vLLM server
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", "127.0.0.1",
            "--port", "8000",
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(vllm_gpu_memory_utilization),
            "--max-model-len", str(vllm_max_model_len),
            "--tensor-parallel-size", str(vllm_tensor_parallel_size),
            "--disable-log-requests",
        ]
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        
        # Wait for server to be ready
        def wait_for_server(timeout=300):
            import time
            start = time.time()
            while time.time() - start < timeout:
                try:
                    resp = requests.get("http://127.0.0.1:8000/health", timeout=5)
                    if resp.status_code == 200:
                        return True
                except:
                    pass
                time.sleep(2)
            return False
        
        if not wait_for_server():
            proc.terminate()
            raise RuntimeError("vLLM server failed to start")
        
        logger.info("vLLM server ready")
        
        # Collect samples via API calls
        samples = []
        
        for ruling_idx, ruling in enumerate(epoch_rulings):
            product_desc = ruling.get("short_product_description", "")[:50]
            logger.info(f"  Ruling {ruling_idx+1}/{len(epoch_rulings)}: {product_desc}...")
            
            # Run rollout (simplified - in production, import your rollout code)
            try:
                ruling_samples = run_single_rollout(ruling, model_path)
                samples.extend(ruling_samples)
                logger.info(f"    → Collected {len(ruling_samples)} samples")
            except Exception as e:
                logger.error(f"    → Rollout error: {e}")
        
        # Stop vLLM server
        proc.terminate()
        proc.wait()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return samples
    
    def run_single_rollout(ruling, model_path):
        """Run rollout for a single ruling using vLLM API."""
        # Simplified placeholder - in full implementation, 
        # import and use your existing rollout code
        import requests
        import json
        
        product_desc = ruling.get("short_product_description", "")
        gold_code = ruling.get("hts_code", "")
        
        # Make API call to local vLLM server
        messages = [
            {"role": "system", "content": "You are an HTS classification expert."},
            {"role": "user", "content": f"Classify this product: {product_desc}"},
        ]
        
        try:
            resp = requests.post(
                "http://127.0.0.1:8000/v1/chat/completions",
                json={
                    "model": model_path,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                },
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            
            output = result["choices"][0]["message"]["content"]
            
            # Return sample (simplified structure)
            return [{
                "messages": messages + [{"role": "assistant", "content": output}],
                "step_rewards": [{"step": 0, "R": 0.5}],  # Placeholder
                "gold_code": gold_code,
                "leaf_reward": 0.5,  # Placeholder - compute actual reward
            }]
            
        except Exception as e:
            logger.error(f"Rollout API error: {e}")
            return []
    
    def train_on_samples(samples, adapter_path=None):
        """Train on collected samples using Unsloth."""
        from unsloth import FastLanguageModel
        from torch.optim import AdamW
        import torch.nn.functional as F
        
        logger.info(f"Training on {len(samples)} samples...")
        
        # Load model
        load_adapter = adapter_path or sft_adapter
        
        if load_adapter:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=load_adapter,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                trust_remote_code=True,
            )
            FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                trust_remote_code=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Setup optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        
        model.train()
        
        # Training loop (simplified)
        total_loss = 0.0
        num_samples = 0
        optimizer.zero_grad()
        accumulated_steps = 0
        
        for sample_idx, sample in enumerate(samples):
            messages = sample.get("messages", [])
            if not messages:
                continue
            
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=train_max_seq_length,
                    padding=False,
                )
                
                input_ids = inputs["input_ids"].cuda()
                attention_mask = inputs["attention_mask"].cuda()
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    return_dict=True,
                )
                
                loss = outputs.loss
                leaf_reward = sample.get("leaf_reward", 1.0)
                weighted_loss = loss * leaf_reward
                
                scaled_loss = weighted_loss / gradient_accumulation_steps
                scaled_loss.backward()
                
                total_loss += loss.item()
                num_samples += 1
                accumulated_steps += 1
                
                del input_ids, attention_mask, loss, scaled_loss
                
            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM on sample {sample_idx}, skipping")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                accumulated_steps = 0
                continue
            
            if accumulated_steps >= gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accumulated_steps = 0
        
        # Final optimizer step
        if accumulated_steps > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Save adapter
        timestamp = int(time.time())
        new_adapter_path = os.path.join(ADAPTER_SYNC_DIR, f"adapter_{timestamp}")
        os.makedirs(new_adapter_path, exist_ok=True)
        model.save_pretrained(new_adapter_path)
        tokenizer.save_pretrained(new_adapter_path)
        
        # Merge for vLLM
        merged_path = os.path.join(ADAPTER_SYNC_DIR, f"merged_{timestamp}")
        model.save_pretrained_merged(
            merged_path,
            tokenizer,
            save_method="merged_16bit",
        )
        
        avg_loss = total_loss / max(num_samples, 1)
        
        # Cleanup
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
        volume.commit()  # Persist checkpoint to volume
        
        return new_adapter_path, merged_path, {"avg_loss": avg_loss, "num_samples": num_samples}
    
    # =============================================
    # MAIN TRAINING LOOP
    # =============================================
    
    logger.info("\n--- Phase 0: Pre-merging SFT adapter ---")
    current_vllm_model = merge_sft_adapter()
    current_adapter_path = sft_adapter if sft_adapter else None
    
    all_metrics = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logger.info(f"{'=' * 70}")
        
        # Sample rulings for this epoch
        epoch_rulings = random.sample(
            chapter_rulings,
            min(num_rulings_per_epoch, len(chapter_rulings))
        )
        
        # Phase 1-2: vLLM rollouts
        logger.info(f"\n--- Phases 1-2: vLLM rollouts for {len(epoch_rulings)} rulings ---")
        
        all_epoch_samples = run_vllm_rollouts(
            current_vllm_model or base_model,
            epoch_rulings
        )
        
        if not all_epoch_samples:
            logger.warning(f"No samples collected for epoch {epoch + 1}, skipping training")
            continue
        
        # Save samples for debugging
        samples_file = os.path.join(SAMPLES_DIR, f"epoch{epoch+1}_samples.json")
        with open(samples_file, 'w') as f:
            json.dump(all_epoch_samples, f, indent=2, default=str)
        
        # Phase 3-4: Train with Unsloth
        logger.info(f"\n--- Phases 3-4: Training on {len(all_epoch_samples)} samples ---")
        
        new_adapter_path, merged_model_path, train_metrics = train_on_samples(
            all_epoch_samples,
            adapter_path=current_adapter_path,
        )
        
        # Update paths for next epoch
        current_adapter_path = new_adapter_path
        current_vllm_model = merged_model_path
        
        # Record metrics
        epoch_time = time.time() - epoch_start
        all_metrics.append({
            "epoch": epoch + 1,
            "num_samples": train_metrics.get("num_samples", 0),
            "avg_loss": train_metrics.get("avg_loss", 0),
            "time_seconds": epoch_time,
            "adapter_path": new_adapter_path,
        })
        
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Samples: {train_metrics.get('num_samples', 0)}")
        logger.info(f"  Avg Loss: {train_metrics.get('avg_loss', 0):.4f}")
        logger.info(f"  Time: {epoch_time:.1f}s")
        logger.info(f"{'=' * 50}")
        
        # Save epoch checkpoint
        import shutil
        checkpoint_dir = os.path.join(CHECKPOINTS_DIR, f"epoch-{epoch + 1}")
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        shutil.copytree(current_adapter_path, checkpoint_dir)
        
        volume.commit()  # Persist after each epoch
    
    # Save final artifacts
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    
    if current_adapter_path:
        final_dir = os.path.join(CHECKPOINTS_DIR, "final")
        import shutil
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.copytree(current_adapter_path, final_dir)
    
    # Save metrics
    metrics_file = os.path.join(CHECKPOINTS_DIR, "training_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    volume.commit()
    
    return {
        "status": "complete",
        "epochs": num_epochs,
        "final_adapter": current_adapter_path,
        "metrics": all_metrics,
    }


# ============================================================================
# ALTERNATIVE: VLLM-BASED INFERENCE CLASS (for more control)
# ============================================================================

@app.cls(
    image=image,
    gpu="H200:4",  # 4x H200 for inference
    volumes={VOLUME_PATH: volume},
    timeout=7200,  # 2 hours
    container_idle_timeout=300,  # Keep warm for 5 minutes
)
class VLLMInference:
    """
    Stateful vLLM inference class for batch rollouts.
    
    The model is loaded once in __enter__ and kept warm for multiple calls.
    This is more efficient than restarting the server each epoch.
    """
    
    model_path: str = modal.parameter()
    
    @modal.enter()
    def setup(self):
        """Load vLLM model once when container starts."""
        from vllm import LLM, SamplingParams
        
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=128000,  # 128k context
            tensor_parallel_size=4,  # Distribute across 4 GPUs
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
        )
    
    @modal.method()
    def generate(self, prompts: list[str]) -> list[str]:
        """Generate completions for a batch of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    @modal.method()
    def chat(self, messages_batch: list[list[dict]]) -> list[str]:
        """Generate chat completions for a batch of message lists."""
        # Apply chat template to each conversation
        prompts = []
        for messages in messages_batch:
            # Use vLLM's chat template if available
            prompt = self._format_chat(messages)
            prompts.append(prompt)
        
        return self.generate(prompts)
    
    def _format_chat(self, messages: list[dict]) -> str:
        """Format messages as a chat prompt."""
        # Simple formatting - in production use tokenizer.apply_chat_template
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|{role}|>\n{content}\n"
        formatted += "<|assistant|>\n"
        return formatted


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main(
    # Model settings
    base_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    sft_adapter: str = "orlandowhite/nemotron3_nano_sft",
    
    # Training settings
    epochs: int = 3,
    chapter: str = "84",
    num_rulings: int = 20,
    
    # GPU settings  
    gpu: str = "h100",
    
    # Data
    rulings_file: str = None,
):
    """
    Run TreeRL GRPO training on Modal.
    
    Example:
        modal run treerl_modal_train.py --chapter 84 --epochs 3 --gpu h100
    """
    import json
    
    print("=" * 70)
    print("TreeRL GRPO Training on Modal")
    print("=" * 70)
    print(f"GPU: {gpu.upper()}")
    print(f"Base model: {base_model}")
    print(f"SFT adapter: {sft_adapter}")
    print(f"Chapter: {chapter}")
    print(f"Epochs: {epochs}")
    print(f"Rulings per epoch: {num_rulings}")
    
    # Upload rulings file if provided
    if rulings_file and os.path.exists(rulings_file):
        print(f"\nUploading rulings from {rulings_file}...")
        with open(rulings_file, 'r') as f:
            rulings_data = json.load(f)
        result = upload_rulings.remote(rulings_data)
        print(result)
    
    # Run training
    print("\nStarting training...")
    
    # Dynamically set GPU based on argument
    # Note: This requires modifying the function's gpu parameter at deploy time
    # For now, we use H100 as default and recommend editing the decorator
    
    result = train_treerl.remote(
        base_model=base_model,
        sft_adapter=sft_adapter,
        num_epochs=epochs,
        chapter=chapter,
        num_rulings_per_epoch=num_rulings,
    )
    
    print("\n" + "=" * 70)
    print("Training Result:")
    print(json.dumps(result, indent=2))
    print("=" * 70)
    
    # List checkpoints
    checkpoints = list_checkpoints.remote()
    print(f"\nAvailable checkpoints: {checkpoints}")

