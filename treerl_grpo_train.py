#!/usr/bin/env python3
"""
TreeRL GRPO Training Script (Fixed for Unsloth)

Custom GRPO training loop for TreeRL with:
- 4-bit or BF16 precision via Unsloth (NO vLLM/fast_inference)
- Online rollouts using standard HuggingFace generation
- Proper LoRA adapter handling
- Per-step R(s) weighting per TreeRL paper

Key fixes from original:
1. Removed FP8 and fast_inference (vLLM) - use 4-bit instead
2. Removed FastLanguageModel.for_inference() calls during training
3. Fixed model mode management for train/eval switching
4. Simplified generation to work with Unsloth in training context

Usage:
    python treerl_grpo_train_fixed.py --chapter 84 --num-rulings 50 --epochs 3
"""

import os
import sys
import json
import gc
import math
import logging
import argparse
import random
import time
import shutil
import signal
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Add the api directory to path
script_dir = Path(__file__).parent
api_dir = script_dir / "api"
sys.path.insert(0, str(api_dir))


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TreeRLConfig:
    """Training configuration for TreeRL GRPO."""
    
    # Model settings
    base_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    sft_adapter: str = "orlandowhite/nemotron3_nano_sft"  # Continue from your SFT adapter
    merged_model_path: str = "./nemotron-merged"  # Pre-merged model for fast loading
    max_seq_length: int = 128000  
    load_in_4bit: bool = True  # Use 4-bit quantization (recommended for memory)
    offload_embedding: bool = False  # Disabled: causes device mismatch errors during generation
    rollout_max_new_tokens: int = 2048
    rollout_temperature: float = 0.7  # Small temp for diversity in rollouts
    rollout_top_p: float = 0.95
    
    # LoRA settings - MUST match your existing SFT adapter!
    lora_rank: int = 16
    lora_alpha: int = 32  # typically 2x rank
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    
    # Training settings
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 10
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # TreeRL settings
    beam_size: int = 4  # Reduced for faster rollouts
    max_questions: int = 3
    
    # Data settings
    chapter: str = "84"
    num_rulings_per_epoch: int = 20
    num_epochs: int = 3
    
    # Paths
    cross_rulings_file: str = "cross_rulings_dataset.json"
    output_dir: str = "treerl_checkpoints"
    log_file: str = "treerl_training.log"
    
    # Logging
    log_every_n_steps: int = 1
    save_every_n_epochs: int = 1
    
    # vLLM settings
    vllm_port: int = 8000
    vllm_max_model_len: int = 128000  # Full context length
    
    # Device
    device: str = "cuda"


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: TreeRLConfig) -> logging.Logger:
    """Configure logging for training."""
    logger = logging.getLogger("treerl_train")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(config.log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# MODEL LOADING (Merged Model + Fresh LoRA for Training)
# ============================================================================

def merge_adapter_to_base(
    base_model_name: str,
    adapter_name: str,
    output_path: str,
    logger: logging.Logger
) -> None:
    """
    Merge a LoRA adapter into the base model and save.
    This is done once to create a fast-loading merged model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    logger.info("=" * 60)
    logger.info("MERGING ADAPTER INTO BASE MODEL")
    logger.info("=" * 60)
    
    total_start = time.time()
    
    # Load tokenizer
    logger.info("[1/5] Loading tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    logger.info(f"       Done in {time.time() - t0:.2f}s")
    
    # Load base model
    logger.info("[2/5] Loading base model...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info(f"       Done in {time.time() - t0:.2f}s")
    
    # Load LoRA adapter
    logger.info("[3/5] Loading LoRA adapter...")
    t0 = time.time()
    model = PeftModel.from_pretrained(base_model, adapter_name)
    logger.info(f"       Done in {time.time() - t0:.2f}s")
    
    # Merge weights
    logger.info("[4/5] Merging LoRA weights into base model...")
    t0 = time.time()
    merged_model = model.merge_and_unload()
    logger.info(f"       Done in {time.time() - t0:.2f}s")
    
    # Save merged model
    logger.info(f"[5/5] Saving merged model to {output_path}...")
    t0 = time.time()
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info(f"       Done in {time.time() - t0:.2f}s")
    
    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"✓ Merge complete! Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    logger.info(f"✓ Merged model saved to: {output_path}")
    logger.info("=" * 60)
    
    # Free memory
    del merged_model, model, base_model
    gc.collect()
    torch.cuda.empty_cache()


def load_model_and_tokenizer(config: TreeRLConfig, logger: logging.Logger):
    """
    Load merged model with fresh LoRA for training.
    
    Assumes merged model already exists at merged_model_path.
    Adds FRESH LoRA adapters for continued training.
    
    Returns:
        model: The merged model with fresh LoRA adapters for training
        tokenizer: The tokenizer
    """
    logger.info(f"Loading merged model: {config.merged_model_path}")
    logger.info(f"  4-bit quantization: {config.load_in_4bit}")
    logger.info(f"  Max sequence length: {config.max_seq_length}")
    
    # Load the merged model with Unsloth
    try:
        from unsloth import FastLanguageModel
        
        t0 = time.time()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.merged_model_path,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            offload_embedding=config.offload_embedding,
            trust_remote_code=True,
        )
        logger.info(f"Merged model loaded in {time.time() - t0:.2f}s")
        
        # Add FRESH LoRA adapters for training
        logger.info(f"Adding fresh LoRA adapters (rank={config.lora_rank}, alpha={config.lora_alpha})...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            target_modules=list(config.lora_target_modules),
            lora_alpha=config.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        logger.info("Fresh LoRA adapters added successfully")
        
        USING_UNSLOTH = True
        
    except ImportError as e:
        logger.warning(f"Unsloth not available ({e}), falling back to standard loading")
        USING_UNSLOTH = False
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig
        
        # Standard loading without Unsloth
        model = AutoModelForCausalLM.from_pretrained(
            config.merged_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.merged_model_path,
            trust_remote_code=True,
        )
        
        # Add fresh LoRA
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=list(config.lora_target_modules),
            lora_dropout=0.0,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        
        logger.info("Model loaded with standard transformers + fresh LoRA")
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Log model stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")
    
    return model, tokenizer, USING_UNSLOTH


# ============================================================================
# VLLM SERVER MANAGEMENT FOR INFERENCE
# ============================================================================

class VLLMServerManager:
    """
    Manages a vLLM server for fast inference during rollouts.
    
    The vLLM server runs the merged model (no LoRA) for fast beam search.
    Training uses Unsloth with LoRA separately.
    """
    
    def __init__(
        self,
        model_path: str,
        port: int = 8000,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        logger: logging.Logger = None,
    ):
        self.model_path = model_path
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.logger = logger or logging.getLogger(__name__)
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://localhost:{port}"
    
    def start_async(self) -> bool:
        """Start the vLLM server process without waiting for it to be ready."""
        if self.process is not None:
            self.logger.info(f"vLLM server already started on port {self.port}")
            return True
        
        self.logger.info(f"Starting vLLM server with model: {self.model_path}")
        self.logger.info(f"  Port: {self.port}")
        self.logger.info(f"  Max model length: {self.max_model_len}")
        
        # Use 'vllm serve' command (vLLM 0.12.0+) instead of the old entry point
        # The old 'python -m vllm.entrypoints.openai.api_server' can cause
        # 'AttributeError: module vllm has no attribute sampling_params'
        cmd = [
            "vllm", "serve", self.model_path,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--async-scheduling",  # Recommended by NVIDIA for better performance
            "--disable-log-requests",
        ]
        
        self.logger.info(f"  Command: {' '.join(cmd)}")
        
        # Start server in background
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # Create new process group for clean shutdown
        )
        self._start_time = time.time()
        return True
    
    def wait_until_ready(self, timeout: int = 300) -> bool:
        """Wait for the vLLM server to be ready."""
        if self.process is None:
            self.logger.error("vLLM server not started")
            return False
        
        start_time = getattr(self, '_start_time', time.time())
        self.logger.info(f"Waiting for vLLM server to be ready (timeout={timeout}s)...")
        
        while time.time() - start_time < timeout:
            if self.is_running():
                elapsed = time.time() - start_time
                self.logger.info(f"✓ vLLM server ready in {elapsed:.1f}s")
                return True
            
            # Check if process died
            if self.process.poll() is not None:
                self.logger.error("vLLM server process died during startup")
                stdout, _ = self.process.communicate()
                if stdout:
                    self.logger.error(f"Server output: {stdout.decode()[-2000:]}")
                return False
            
            time.sleep(5)
        
        self.logger.error(f"vLLM server failed to start within {timeout}s")
        self.stop()
        return False
    
    def start(self, wait_timeout: int = 300) -> bool:
        """Start the vLLM server and wait for it to be ready."""
        if self.is_running():
            self.logger.info(f"vLLM server already running on port {self.port}")
            return True
        
        self.start_async()
        return self.wait_until_ready(timeout=wait_timeout)
    
    def is_running(self) -> bool:
        """Check if the vLLM server is responding."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def stop(self):
        """Stop the vLLM server."""
        if self.process:
            self.logger.info("Stopping vLLM server...")
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except Exception as e:
                self.logger.warning(f"Error stopping vLLM server: {e}")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.process = None
            self.logger.info("vLLM server stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def setup_vllm_environment(port: int = 8000):
    """Set environment variables for the LLMClient to use our vLLM server."""
    os.environ["CUSTOM_OPENAI_BASE_URL"] = f"http://localhost:{port}/v1"
    os.environ["CUSTOM_OPENAI_API_KEY"] = "sk-local"
    os.environ["CUSTOM_OPENAI_MODEL"] = "default"
    os.environ["CUSTOM_OPENAI_TIMEOUT"] = "120"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_chapter_rulings(config: TreeRLConfig, logger: logging.Logger) -> List[Dict]:
    """Load cross rulings filtered by chapter."""
    logger.info(f"Loading rulings from {config.cross_rulings_file}")
    
    if not os.path.exists(config.cross_rulings_file):
        logger.error(f"Rulings file not found: {config.cross_rulings_file}")
        return []
    
    with open(config.cross_rulings_file, 'r', encoding='utf-8') as f:
        all_rulings = json.load(f)
    
    # Filter by chapter
    chapter_rulings = [
        r for r in all_rulings 
        if r.get("hts_code", "").startswith(config.chapter)
    ]
    
    logger.info(f"Total rulings: {len(all_rulings)}")
    logger.info(f"Chapter {config.chapter} rulings: {len(chapter_rulings)}")
    
    return chapter_rulings


# ============================================================================
# STEP BOUNDARY DETECTION
# ============================================================================

def find_assistant_turn_boundaries(
    input_ids: torch.Tensor,
    tokenizer,
    messages: List[Dict[str, str]]
) -> List[Tuple[int, int]]:
    """
    Find token boundaries for each assistant turn in the conversation.
    
    Returns list of (start_idx, end_idx) tuples for each assistant response.
    """
    boundaries = []
    input_list = input_ids.tolist()
    
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        
        content = msg.get("content", "")
        if not content:
            continue
        
        # Tokenize this content to find it in the sequence
        content_tokens = tokenizer.encode(content, add_special_tokens=False)
        
        if len(content_tokens) < 3:
            continue
        
        # Search for the start of this content (use first few tokens as anchor)
        search_len = min(5, len(content_tokens))
        search_pattern = content_tokens[:search_len]
        
        for start_idx in range(len(input_list) - len(content_tokens) + 1):
            if input_list[start_idx:start_idx + search_len] == search_pattern:
                end_idx = start_idx + len(content_tokens)
                boundaries.append((start_idx, end_idx))
                break
    
    return boundaries


def build_token_weights(
    step_rewards: List[Dict],
    boundaries: List[Tuple[int, int]],
    seq_len: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Build per-token weight tensor from step rewards.
    
    Each token in an assistant turn gets weighted by R(s) for that step.
    """
    weights = torch.zeros(seq_len, device=device)
    
    # Map step index to R value
    step_to_R = {sr["step"]: sr["R"] for sr in step_rewards}
    
    # Assign weights to each boundary
    for step_idx, (start, end) in enumerate(boundaries):
        if step_idx in step_to_R:
            R = step_to_R[step_idx]
            weights[start:end] = R
    
    return weights


# ============================================================================
# GRPO LOSS FUNCTION
# ============================================================================

def compute_grpo_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    step_rewards: List[Dict],
    boundaries: List[Tuple[int, int]],
    device: str = "cuda"
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute GRPO loss with per-step R(s) weighting.
    
    Loss = -sum(R(s) * log_prob(token)) for tokens in step s
    """
    # CRITICAL: Ensure we're in training mode with gradients enabled
    assert model.training, "Model must be in training mode for loss computation"
    assert torch.is_grad_enabled(), "Gradients must be enabled for loss computation"
    
    # Forward pass - model should be in training mode
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    
    logits = outputs.logits  # [1, seq_len, vocab_size]
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(
        dim=-1, 
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # [1, seq_len-1]
    
    # Build per-token weights from step rewards
    adjusted_boundaries = [(max(0, s-1), max(0, e-1)) for s, e in boundaries]
    weights = build_token_weights(
        step_rewards, 
        adjusted_boundaries, 
        shift_labels.shape[1],
        device
    ).unsqueeze(0)  # [1, seq_len-1]
    
    # Apply mask and weights
    masked_log_probs = token_log_probs * shift_mask.float()
    weighted_log_probs = masked_log_probs * weights
    
    # Normalize by number of weighted tokens
    num_weighted = (weights.abs() > 0).sum().float()
    if num_weighted > 0:
        loss = -weighted_log_probs.sum() / num_weighted
    else:
        # Fallback: standard CE loss if no weights
        loss = -masked_log_probs.sum() / shift_mask.sum().float()
    
    # Verify loss requires grad (sanity check)
    assert loss.requires_grad, "Loss tensor must require gradients for training"
    
    # Compute metrics
    metrics = {
        "loss": loss.item(),
        "avg_log_prob": masked_log_probs.sum().item() / max(shift_mask.sum().item(), 1),
        "num_weighted_tokens": num_weighted.item(),
        "avg_weight": weights.abs().sum().item() / max(num_weighted.item(), 1),
        "max_weight": weights.max().item(),
        "min_weight": weights.min().item(),
    }
    
    return loss, metrics


# ============================================================================
# ONLINE ROLLOUT
# ============================================================================

def run_online_rollout(
    ruling: Dict,
    config: TreeRLConfig,
    logger: logging.Logger,
) -> List[Dict]:
    """
    Run beam search rollout for a single ruling and compute TreeRL rewards.
    
    Uses vLLM server (via LLMClient) for fast inference.
    
    Returns:
        List of training samples with messages and step_rewards
    """
    # Import TreeRL components
    try:
        from api.treerl_gold_trace import build_gold_trace, build_pred_trace_from_path
        from api.treerl_rewards import compute_leaf_reward
        from api.treerl_process_supervision import compute_treerl_rewards, emit_leaf_samples
        from llm_auto_responder import LLMAutoResponder
        from api.groq_tree_engine import HTSTree
    except ImportError as e:
        logger.error(f"Could not import TreeRL components: {e}")
        return []
    
    # Set beam size via environment
    os.environ["TREERL_BEAM_SIZE"] = str(config.beam_size)
    os.environ["TREERL_CHAPTER_BEAM_SIZE"] = str(config.beam_size)
    os.environ["DISABLE_CROSS_RULING_INJECTION"] = "true"
    
    product_description = ruling.get("short_product_description", "")
    gold_code = ruling.get("hts_code", "")
    
    try:
        logger.debug(f"  [rollout] Creating HTSTree...")
        hts_tree = HTSTree()
        
        # HTSTree uses LLMClient which connects to vLLM via CUSTOM_OPENAI_BASE_URL
        logger.debug(f"  [rollout] Using vLLM via LLMClient...")
        
        # Load HTS data
        logger.debug(f"  [rollout] Loading HTS data...")
        hts_data_file = script_dir / "api" / "hts_data.json"
        if hts_data_file.exists():
            with open(hts_data_file, "r", encoding="utf-8") as f:
                hts_data = json.load(f)
            hts_tree.build_from_json(hts_data)
        
        # Build gold trace
        logger.debug(f"  [rollout] Building gold trace for {gold_code}...")
        gold_trace = build_gold_trace(gold_code, hts_tree.navigator)
        
        # Initialize auto-responder
        logger.debug(f"  [rollout] Initializing auto-responder...")
        auto_responder = LLMAutoResponder(engine_name="groq", debug=False)
        if hasattr(auto_responder, "llm_client"):
            auto_responder.llm_client = local_llm
        
        # Run classification
        logger.info(f"  [rollout] Starting classification (max_q={config.max_questions}, beam={config.beam_size})...")
        if config.max_questions > 0:
            result = auto_responder.interactive_classify_with_auto_response(
                hts_tree=hts_tree,
                product_description=product_description,
                cross_ruling=ruling,
                max_questions=config.max_questions
            )
        else:
            result = hts_tree.start_classification(
                product=product_description,
                interactive=False,
                max_questions=0,
                use_multi_hypothesis=True,
                hypothesis_count=config.beam_size
            )
        logger.info(f"  [rollout] Classification complete")
        
        state = result.get("state", {})
        
        # Collect leaves
        leaves = []
        beam = state.get("beam", [])
        
        for path_data in beam:
            if isinstance(path_data, dict):
                classification_path = path_data.get("classification_path", [])
                trajectory = path_data.get("trajectory", [])
                path_id = path_data.get("path_id", "unknown")
            elif hasattr(path_data, "classification_path"):
                classification_path = path_data.classification_path
                trajectory = getattr(path_data, "trajectory", [])
                path_id = path_data.path_id
            else:
                continue
            
            pred_trace = build_pred_trace_from_path(classification_path)
            reward = compute_leaf_reward(pred_trace, gold_trace)
            
            leaves.append({
                "path_id": path_id,
                "pred_trace": pred_trace,
                "reward": reward,
                "trajectory": trajectory,
                "classification_path": classification_path,
                "source": "final_beam",
            })
        
        # Add pruned paths
        pruned_leaves = state.get("_treerl_pruned_leaves", [])
        for pruned in pruned_leaves:
            classification_path = pruned.get("classification_path", [])
            trajectory = pruned.get("trajectory", [])
            path_id = pruned.get("path_id", "unknown")
            
            pred_trace = build_pred_trace_from_path(classification_path)
            reward = compute_leaf_reward(pred_trace, gold_trace)
            
            leaves.append({
                "path_id": path_id,
                "pred_trace": pred_trace,
                "reward": reward,
                "trajectory": trajectory,
                "classification_path": classification_path,
                "source": "pruned",
            })
        
        if not leaves:
            logger.warning(f"No leaves collected for ruling: {product_description[:50]}")
            return []
        
        # Compute TreeRL rewards
        step_rewards, v_root = compute_treerl_rewards(leaves)
        
        # Emit training samples
        samples = emit_leaf_samples(
            leaves,
            step_rewards,
            gold_trace=gold_trace,
            gold_code=gold_code,
        )
        
        logger.debug(f"Rollout: {len(leaves)} leaves, {len(samples)} samples, V(root)={v_root:.3f}")
        
        return samples
        
    except Exception as e:
        logger.error(f"Rollout error for '{product_description[:50]}': {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model,
    tokenizer,
    optimizer,
    rulings: List[Dict],
    config: TreeRLConfig,
    logger: logging.Logger,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch with online rollouts.
    
    Rollouts use vLLM (via LLMClient).
    Training uses the Unsloth model with LoRA.
    """
    model.train()
    
    epoch_metrics = {
        "total_loss": 0.0,
        "num_samples": 0,
        "num_rulings": 0,
        "avg_reward": 0.0,
    }
    
    # Sample rulings for this epoch
    epoch_rulings = random.sample(
        rulings, 
        min(config.num_rulings_per_epoch, len(rulings))
    )
    
    logger.info(f"Epoch {epoch+1}: Processing {len(epoch_rulings)} rulings")
    
    accumulated_loss = 0.0
    accumulated_steps = 0
    
    optimizer.zero_grad()
    
    for ruling_idx, ruling in enumerate(epoch_rulings):
        product_desc = ruling.get("short_product_description", "")[:50]
        logger.info(f"  Ruling {ruling_idx+1}/{len(epoch_rulings)}: {product_desc}...")
        
        # Run online rollout using vLLM for inference
        samples = run_online_rollout(ruling, config, logger)
        
        if not samples:
            logger.warning(f"  No samples from rollout, skipping")
            continue
        
        epoch_metrics["num_rulings"] += 1
        
        # Train on each sample
        for sample in samples:
            messages = sample.get("messages", [])
            step_rewards = sample.get("step_rewards", [])
            leaf_reward = sample.get("leaf_reward", 0.0)
            
            if not messages or not step_rewards:
                continue
            
            # Tokenize the full trajectory
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
                    max_length=config.max_seq_length,
                    padding=False,
                )
                
                input_ids = inputs["input_ids"].to(config.device)
                attention_mask = inputs["attention_mask"].to(config.device)
                
            except Exception as e:
                logger.warning(f"Tokenization error: {e}")
                continue
            
            # Find assistant turn boundaries
            boundaries = find_assistant_turn_boundaries(
                input_ids[0], tokenizer, messages
            )
            
            # Compute GRPO loss
            try:
                model.train()  # Ensure training mode
                
                loss, metrics = compute_grpo_loss(
                    model,
                    input_ids,
                    attention_mask,
                    step_rewards,
                    boundaries,
                    config.device
                )
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / config.gradient_accumulation_steps
                scaled_loss.backward()
                
                accumulated_loss += loss.item()
                accumulated_steps += 1
                
                epoch_metrics["total_loss"] += loss.item()
                epoch_metrics["num_samples"] += 1
                epoch_metrics["avg_reward"] += leaf_reward
                
            except Exception as e:
                logger.error(f"Loss computation error: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
            
            # Gradient accumulation step
            if accumulated_steps >= config.gradient_accumulation_steps:
                # Verify gradients are flowing (first time only)
                if epoch_metrics["num_samples"] <= config.gradient_accumulation_steps:
                    lora_grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                    if lora_grads:
                        grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in lora_grads)).item()
                        logger.info(f"    Gradient check: {len(lora_grads)} LoRA params have grads, norm={grad_norm:.4f}")
                    else:
                        logger.warning("    WARNING: No gradients computed for LoRA parameters!")
                
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.max_grad_norm
                )
                
                optimizer.step()
                optimizer.zero_grad()
                
                avg_acc_loss = accumulated_loss / accumulated_steps
                logger.info(f"    Step loss: {avg_acc_loss:.4f}")
                
                accumulated_loss = 0.0
                accumulated_steps = 0
        
        # Log ruling progress
        if (ruling_idx + 1) % config.log_every_n_steps == 0:
            avg_loss = epoch_metrics["total_loss"] / max(epoch_metrics["num_samples"], 1)
            logger.info(
                f"  Progress: {ruling_idx+1}/{len(epoch_rulings)} rulings, "
                f"{epoch_metrics['num_samples']} samples, avg_loss={avg_loss:.4f}"
            )
        
        # Free memory periodically
        if (ruling_idx + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Handle remaining gradients
    if accumulated_steps > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    # Compute epoch averages
    if epoch_metrics["num_samples"] > 0:
        epoch_metrics["avg_loss"] = epoch_metrics["total_loss"] / epoch_metrics["num_samples"]
        epoch_metrics["avg_reward"] = epoch_metrics["avg_reward"] / epoch_metrics["num_samples"]
    else:
        epoch_metrics["avg_loss"] = 0.0
    
    return epoch_metrics


# ============================================================================
# EPOCH-END MERGE LOGIC
# ============================================================================

def merge_lora_into_merged_model(
    model,
    tokenizer,
    config: TreeRLConfig,
    logger: logging.Logger,
    epoch: int,
) -> None:
    """
    Merge the current LoRA weights into the merged model.
    
    After training LoRA for an epoch, we merge those weights back into
    the merged model so subsequent epochs and inference use the updated weights.
    """
    from peft import PeftModel
    
    logger.info("=" * 60)
    logger.info(f"MERGING EPOCH {epoch+1} LORA INTO MERGED MODEL")
    logger.info("=" * 60)
    
    t0 = time.time()
    
    # Save current LoRA adapter temporarily
    temp_adapter_dir = os.path.join(config.output_dir, f"temp_adapter_epoch_{epoch+1}")
    os.makedirs(temp_adapter_dir, exist_ok=True)
    
    logger.info(f"[1/4] Saving current LoRA adapter...")
    model.save_pretrained(temp_adapter_dir)
    logger.info(f"       Saved to {temp_adapter_dir}")
    
    # Free the current model from GPU memory
    logger.info("[2/4] Freeing GPU memory...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load merged model in FP16/BF16 (not quantized) for merging
    logger.info("[3/4] Loading merged model and new LoRA for merge...")
    from transformers import AutoModelForCausalLM
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.merged_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load the LoRA adapter
    peft_model = PeftModel.from_pretrained(base_model, temp_adapter_dir)
    
    # Merge and unload
    logger.info("[4/4] Merging LoRA weights and saving...")
    merged = peft_model.merge_and_unload()
    
    # Save back to merged_model_path (overwrite)
    merged.save_pretrained(config.merged_model_path)
    tokenizer.save_pretrained(config.merged_model_path)
    
    merge_time = time.time() - t0
    logger.info(f"✓ Merge complete in {merge_time:.2f}s")
    logger.info(f"✓ Updated merged model at: {config.merged_model_path}")
    
    # Cleanup
    del merged, peft_model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Clean up temp adapter
    shutil.rmtree(temp_adapter_dir, ignore_errors=True)


def reload_model_for_next_epoch(
    config: TreeRLConfig,
    logger: logging.Logger
):
    """
    Reload the merged model with fresh LoRA for the next epoch.
    Returns the new model and tokenizer.
    
    Note: vLLM server continues running - no need to restart it since
    the merged model was already updated by merge_lora_into_merged_model.
    We need to restart vLLM to pick up the new weights though.
    """
    logger.info("Reloading Unsloth model for next epoch...")
    
    model, tokenizer, using_unsloth = load_model_and_tokenizer(config, logger)
    
    return model, tokenizer, using_unsloth


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def ensure_merged_model_exists(config: TreeRLConfig, logger: logging.Logger) -> None:
    """Ensure the merged model exists before starting vLLM."""
    merged_config_path = os.path.join(config.merged_model_path, "config.json")
    if not os.path.exists(merged_config_path):
        if config.sft_adapter:
            logger.info(f"Merged model not found at {config.merged_model_path}")
            logger.info("Creating merged model from base + SFT adapter...")
            merge_adapter_to_base(
                config.base_model,
                config.sft_adapter,
                config.merged_model_path,
                logger
            )
        else:
            raise ValueError(f"No merged model at {config.merged_model_path} and no SFT adapter specified")
    else:
        logger.info(f"✓ Found existing merged model at {config.merged_model_path}")


def train(config: TreeRLConfig):
    """Main training function."""
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("=" * 70)
    logger.info("TREERL GRPO TRAINING (vLLM + UNSLOTH)")
    logger.info("=" * 70)
    logger.info(f"Config: {config}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Step 1: Ensure merged model exists (needed for vLLM)
    ensure_merged_model_exists(config, logger)
    
    # Step 2: Start vLLM server FIRST (let it load in background)
    logger.info("=" * 70)
    logger.info("STARTING VLLM SERVER FOR INFERENCE")
    logger.info("=" * 70)
    
    vllm_server = VLLMServerManager(
        model_path=config.merged_model_path,
        port=config.vllm_port,
        max_model_len=config.vllm_max_model_len,
        logger=logger,
    )
    
    # Start vLLM but don't wait yet - let it load while we load Unsloth
    vllm_server.start_async()
    logger.info("vLLM server starting in background...")
    
    # Step 3: Load Unsloth model while vLLM is starting
    logger.info("=" * 70)
    logger.info("LOADING UNSLOTH MODEL FOR TRAINING (parallel with vLLM)")
    logger.info("=" * 70)
    model, tokenizer, using_unsloth = load_model_and_tokenizer(config, logger)
    
    # Note: With Unsloth, model is already on the correct device
    if not using_unsloth:
        logger.info(f"Moving model to {config.device}...")
        model.to(config.device)
    logger.info("Unsloth model ready for training")
    
    # Step 4: Now wait for vLLM to be ready
    logger.info("Waiting for vLLM server to be ready...")
    if not vllm_server.wait_until_ready(timeout=300):
        logger.error("Failed to start vLLM server. Exiting.")
        vllm_server.stop()
        return
    
    # Configure LLMClient to use our vLLM server
    setup_vllm_environment(port=config.vllm_port)
    logger.info(f"✓ vLLM server running at http://localhost:{config.vllm_port}")
    logger.info(f"  LLMClient will use: {os.environ.get('CUSTOM_OPENAI_BASE_URL')}")
    
    # Load data
    rulings = load_chapter_rulings(config, logger)
    
    if not rulings:
        logger.error(f"No rulings found for chapter {config.chapter}")
        return
    
    # Setup optimizer - only train LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Verify LoRA setup
    lora_param_count = sum(p.numel() for p in trainable_params)
    frozen_param_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Training setup verification:")
    logger.info(f"  Trainable (LoRA) parameters: {lora_param_count:,}")
    logger.info(f"  Frozen (base) parameters: {frozen_param_count:,}")
    logger.info(f"  Optimizer param groups: {len(optimizer.param_groups)}")
    
    if lora_param_count == 0:
        logger.error("NO TRAINABLE PARAMETERS! LoRA setup may have failed.")
        return
    
    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    
    training_start = time.time()
    all_metrics = []
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'=' * 70}")
        
        # Train epoch (rollouts use vLLM, training uses Unsloth)
        epoch_metrics = train_epoch(
            model, tokenizer, optimizer,
            rulings, config, logger, epoch
        )
        
        epoch_time = time.time() - epoch_start
        epoch_metrics["epoch_time"] = epoch_time
        all_metrics.append(epoch_metrics)
        
        # Log epoch summary
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Rulings processed: {epoch_metrics['num_rulings']}")
        logger.info(f"  Samples trained: {epoch_metrics['num_samples']}")
        logger.info(f"  Average loss: {epoch_metrics.get('avg_loss', 0):.4f}")
        logger.info(f"  Average reward: {epoch_metrics.get('avg_reward', 0):.4f}")
        logger.info(f"  Time: {epoch_time:.1f}s")
        
        # Save checkpoint (LoRA adapter)
        if (epoch + 1) % config.save_every_n_epochs == 0:
            checkpoint_dir = os.path.join(
                config.output_dir, 
                f"checkpoint-epoch-{epoch + 1}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            logger.info(f"  Checkpoint saved: {checkpoint_dir}")
        
        # Merge LoRA into merged model and reload for next epoch
        # (Skip for the last epoch - we'll do final merge at the end)
        if epoch < config.num_epochs - 1:
            logger.info(f"\n  Merging epoch {epoch + 1} LoRA into merged model...")
            
            # Stop vLLM before merging (we'll restart with updated weights)
            logger.info("  Stopping vLLM server for merge...")
            vllm_server.stop()
            
            # Merge current LoRA into the merged model
            merge_lora_into_merged_model(
                model, tokenizer, config, logger, epoch
            )
            
            # Reload Unsloth model with fresh LoRA for next epoch
            model, tokenizer, using_unsloth = reload_model_for_next_epoch(
                config, logger
            )
            
            # Restart vLLM with the updated merged model
            logger.info("  Restarting vLLM server with updated weights...")
            if not vllm_server.start(wait_timeout=300):
                logger.error("Failed to restart vLLM server. Exiting.")
                return
            
            # Recreate optimizer for the new model's LoRA parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            logger.info(f"  ✓ Ready for epoch {epoch + 2}")
    
    # Training complete - cleanup
    total_time = time.time() - training_start
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Final average loss: {all_metrics[-1].get('avg_loss', 0):.4f}")
    
    # Stop vLLM server
    logger.info("\nStopping vLLM server...")
    vllm_server.stop()
    
    # Save final LoRA adapter
    final_lora_dir = os.path.join(config.output_dir, "final_lora")
    os.makedirs(final_lora_dir, exist_ok=True)
    model.save_pretrained(final_lora_dir)
    tokenizer.save_pretrained(final_lora_dir)
    logger.info(f"Final LoRA adapter saved: {final_lora_dir}")
    
    # Merge final LoRA into the merged model
    logger.info("\nMerging final LoRA into merged model...")
    merge_lora_into_merged_model(
        model, tokenizer, config, logger, config.num_epochs - 1
    )
    logger.info(f"✓ Final merged model at: {config.merged_model_path}")
    
    # Save training metrics
    metrics_file = os.path.join(config.output_dir, "training_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_file}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TreeRL GRPO Training (Fixed for Unsloth)")
    
    # Model args
    parser.add_argument("--base-model", type=str, 
                       default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                       help="Base model name or path")
    parser.add_argument("--sft-adapter", type=str, 
                       default="orlandowhite/nemotron3_nano_sft",
                       help="SFT LoRA adapter to merge into base (HF Hub or local path)")
    parser.add_argument("--no-sft-adapter", action="store_true",
                       help="Train fresh LoRA from base model (don't merge SFT adapter)")
    parser.add_argument("--merged-model-path", type=str,
                       default="./nemotron-merged",
                       help="Path to pre-merged model (will create if doesn't exist)")
    parser.add_argument("--max-seq-length", type=int, default=8192,
                       help="Maximum sequence length")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization (use BF16)")
    parser.add_argument("--rollout-max-new-tokens", type=int, default=2048,
                       help="Max new tokens per generation call during rollout")
    
    # LoRA args
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha")
    
    # Training args
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs")
    
    # Data args
    parser.add_argument("--chapter", type=str, default="84",
                       help="HTS chapter to train on")
    parser.add_argument("--num-rulings", type=int, default=20,
                       help="Number of rulings per epoch")
    parser.add_argument("--cross-rulings-file", type=str,
                       default="cross_rulings_dataset.json",
                       help="Path to cross rulings JSON")
    
    # TreeRL args
    parser.add_argument("--beam-size", type=int, default=4,
                       help="Beam size for rollouts")
    parser.add_argument("--max-questions", type=int, default=3,
                       help="Max Q&A turns per rollout")
    
    # Output args
    parser.add_argument("--output-dir", type=str, default="treerl_checkpoints",
                       help="Output directory for checkpoints")
    
    # vLLM args
    parser.add_argument("--vllm-port", type=int, default=8000,
                       help="Port for vLLM inference server")
    
    args = parser.parse_args()
    
    # Handle SFT adapter logic
    sft_adapter = args.sft_adapter if not args.no_sft_adapter else ""
    
    # Build config
    config = TreeRLConfig(
        base_model=args.base_model,
        sft_adapter=sft_adapter,
        merged_model_path=args.merged_model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        rollout_max_new_tokens=args.rollout_max_new_tokens,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        num_epochs=args.epochs,
        chapter=args.chapter,
        num_rulings_per_epoch=args.num_rulings,
        cross_rulings_file=args.cross_rulings_file,
        beam_size=args.beam_size,
        max_questions=args.max_questions,
        output_dir=args.output_dir,
        vllm_port=args.vllm_port,
    )
    
    train(config)


if __name__ == "__main__":
    main()
