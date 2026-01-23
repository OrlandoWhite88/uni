#!/usr/bin/env python3
# =============================================================================
# ENVIRONMENT VARIABLES - MUST BE SET BEFORE ANY IMPORTS
# =============================================================================
import os

# We manage vLLM externally

# Faster HF downloads if available
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Explicitly set vLLM device type (helps in some containers)
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")

# =============================================================================
# NOW SAFE TO IMPORT
# =============================================================================
"""
TreeRL GRPO Training Script with NVIDIA Nemotron-3-Nano-30B-A3B (vLLM + Transformers/PEFT)

ARCHITECTURE (vLLM LoRA - PR #30802):
- vLLM runs as external OpenAI-compatible server for rollouts (high throughput inference)
- Transformers + PEFT loads separately for training (with Flash Attention 2)
- vLLM serves base BF16 model with LoRA adapter via --enable-lora --lora-modules:
    * Start vLLM with base model + current LoRA adapter
    * Run rollouts via vLLM (uses LoRA adapter)
    * Stop vLLM before training
    * Train next LoRA adapter with Transformers + PEFT
    * Restart vLLM with updated LoRA adapter
    * Repeat

Nemotron-3-Nano reasoning (per NVIDIA Deployment Guide):
- Uses <think>...</think> in assistant outputs
- nano_v3 reasoning parser (with plugin) separates reasoning_content from content
- Reasoning budget: 2048 tokens for thinking, 2048 tokens for completion (configurable)
- Uses logits processor to enforce reasoning budget in a single request

vLLM server command (auto-generated, matches NVIDIA guide):
    vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
        --dtype auto \\
        --trust-remote-code \\
        --async-scheduling \\
        --served-model-name nemotron3nano \\
        --logits-processors custom_logit_processors.v1.nano_v3_logit_processors:ThinkingBudgetLogitsProcessor \\
        --reasoning-parser-plugin nano_v3_reasoning_parser.py \\
        --reasoning-parser nano_v3

API client uses OpenAI Python client with logit-processor reasoning budget control.
"""

import sys
import json
import math
import importlib.util
import logging
import argparse
import random
import time
import subprocess
import requests
import gc
import re
import threading
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from openai import OpenAI

# Add the api directory to path
script_dir = Path(__file__).parent
api_dir = script_dir / "api"
sys.path.insert(0, str(api_dir))


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TreeRLConfig:
    """Training configuration for TreeRL GRPO with Nemotron-3-Nano."""

    # ----------------------------
    # Base + training model IDs
    # ----------------------------
    # vLLM should serve NVIDIA BF16/FP8 repos or a local merged checkpoint dir.
    base_model_bf16: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    base_model_fp8: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

    # Training model (Transformers + PEFT with Flash Attention 2)
    # NOTE: We use the official BF16 model for training
    train_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    # Optional starting adapter to warm-start from (LoRA dir or HuggingFace repo)
    # Set to "" to start from base model (no adapter)
    sft_adapter: str = ""

    # Choose inference dtype
    # - "bf16" is simplest for merge->serve
    # - "fp8" is possible for serving, but merging adapters into FP8 weights is not recommended.
    #   This script will STILL merge into BF16 for correctness, then serve BF16 unless you override.
    inference_dtype: str = "bf16"   # "bf16" | "fp8"

    # ----------------------------
    # Context lengths
    # ----------------------------
    # NVIDIA examples often use 262144 default (256k).
    vllm_max_model_len: int = 262144

    # Training max per sample (be realistic; long sequences + MoE can OOM)
    train_max_seq_length: int = 90000


    # ----------------------------
    # Thinking / reasoning controls
    # ----------------------------
    # Thinking is enabled by default when --reasoning-parser nano_v3 is set

    # ----------------------------
    # Generation sampling (NVIDIA recs)
    # ----------------------------
    # General: temp=1.0 top_p=1.0
    # Balanced temperature for exploration + JSON consistency
    rollout_temperature: float = 0.6
    rollout_top_p: float = 0.95

    # Cap for output tokens (actual max_tokens computed dynamically)
    rollout_max_new_tokens_cap: int = 8192

    # ----------------------------
    # vLLM server settings
    # ----------------------------
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_tensor_parallel_size: int = 2  # Use both GPUs for inference (TP=2)
    vllm_max_num_seqs: int = 512  # High for throughput: 16 rulings × 8 path_workers × ~4 depth

    # Use a stable served name so client always uses same "model" string
    vllm_served_model_name: str = "nemotron3nano"

    # Tokenizer control for vLLM (prevents LoRA adapter from changing templates)
    # If empty, defaults to the base model when serving with LoRA.
    vllm_tokenizer: str = ""

    # Reasoning parser: use nano_v3 with custom plugin (required for Nemotron-3-Nano)
    # The nano_v3 parser correctly handles Nemotron's <think>...</think> format
    # and properly separates reasoning_content from content
    vllm_use_reasoning_parser: bool = True
    vllm_reasoning_parser_name: str = "nano_v3"  # Nemotron-specific parser
    vllm_reasoning_parser_plugin_filename: str = "nano_v3_reasoning_parser.py"
    
    # Reasoning budget: limit reasoning tokens to control generation length
    # Uses vLLM logits processor for strict budget control in a single request
    vllm_reasoning_budget: int = 2048  # 2k max tokens for <think>...</think>
    vllm_completion_budget: int = 2048  # 2k max tokens for actual response content
    vllm_use_logit_processor: bool = True  # Requires custom_logit_processors package
    vllm_thinking_budget_grace_period: int = 30
    # Default token IDs from NVIDIA guide for Nemotron-3-Nano
    vllm_thinking_end_token_ids: Tuple[int, ...] = (1871, 5565, 11483, 6139, 2016, 1536, 6934, 1338, 13)
    vllm_thinking_end_think_ids: Tuple[int, ...] = (13,)
    vllm_thinking_prompt_think_ids: Tuple[int, ...] = (12, 1010)
    
    # Tool calling flags (disabled by default - enable if needed)
    vllm_enable_auto_tool_choice: bool = False
    vllm_tool_call_parser: str = ""

    # ----------------------------
    # LoRA settings for training
    # ----------------------------
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Training settings
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 10
    gradient_accumulation_steps: int = 8  # Effective batch = micro_batch * grad_accum = 32
    max_grad_norm: float = 1.0
    train_micro_batch_size: int = 1  # Samples per forward pass (1 = no batching, safest for long sequences)

    # Advantage shaping / normalization
    advantage_method: str = "grpo"  # none | grpo | grpo_no_std | gdpo
    gdpo_reward_weights: Tuple[float, ...] = (1.0, 1.0)

    # ----------------------------
    # Reward / advantage scaling (stability knobs)
    # ----------------------------
    # TreeRL step rewards R(s) are intentionally reweighted by 1/sqrt(|L(s)|),
    # which can make per-token weights very small. Scaling helps avoid tiny gradients.
    step_reward_scale: float = 1.0

    # When advantage_method != "none", we multiply token weights by a per-trajectory
    # leaf-level advantage (GRPO/GDPO). This multiplier optionally scales that term.
    leaf_advantage_scale: float = 1.0

    # Optional clip on final per-token weights after scaling (0 disables).
    token_weight_clip: float = 0.0

    # TreeRL settings
    beam_size: int = 3
    max_questions: int = 8

    # Parallelization settings
    # Number of rulings to process concurrently during rollouts
    # vLLM handles batching internally - more concurrent rulings = better GPU utilization
    # With high max_num_seqs (256+), push this higher to saturate the GPU
    parallel_rollouts: int = 64  # Process all 16 rulings with 4× headroom for path workers

    # Benchmark evaluation settings
    # Run a larger evaluation every N batches to get cleaner signal
    benchmark_every_n_batches: int = 8  # Every 8 batches (0 to disable)
    benchmark_num_rulings: int = 50  # Fixed random seed (42) ensures same held-out set each time
    skip_initial_benchmark: bool = False  # Skip baseline benchmark before training starts

    # Data settings
    chapter: str = "84"
    rulings_per_batch: int = 16  # Higher = more concurrent classifications = better vLLM utilization
    accuracy_window_size: int = 10
    num_batches: int = 20
    num_epochs: int = 3
    train_all: bool = False
    start_batch: int = 0  # Skip to this batch number (for resuming mid-training)

    # Paths
    cross_rulings_file: str = "cross_rulings_dataset.json"
    output_dir: str = "treerl_checkpoints"
    log_file: str = "treerl_training.log"
    adapter_sync_dir: str = "treerl_checkpoints/adapter_sync"
    merged_models_dir: str = "treerl_checkpoints/merged_models"
    samples_dir: str = "treerl_checkpoints/samples"
    completions_log: str = "treerl_checkpoints/completions.jsonl"

    # Rollout caching (for dev/testing)
    save_rollouts: str = ""
    load_rollouts: str = ""

    # Logging
    log_every_n_steps: int = 1
    save_every_n_epochs: int = 1

    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = "treerl-grpo"
    wandb_run_name: str = ""
    wandb_entity: str = ""

    # Device
    device: str = "cuda"

    # Leaf reward shaping
    leaf_reward_weights: Tuple[float, ...] = (0.85, 0.15)
    leaf_reward_clip_0_1: bool = True

    # External vLLM architecture only
    use_fast_inference: bool = False

    # Safety margin for max_tokens calculation
    token_safety_margin: int = 512

    # vLLM LoRA serving (PR #30802)
    # When True, vLLM serves base model with LoRA adapter via --enable-lora
    # When False (legacy), merges LoRA into base model before serving
    use_vllm_lora: bool = True
    vllm_lora_name: str = "current_adapter"  # Name used in --lora-modules and API requests

    # DDP / Multi-GPU training settings
    use_ddp: bool = False  # Enable via torchrun or --use-ddp flag
    ddp_backend: str = "nccl"


# ============================================================================
# DISTRIBUTED SETUP
# ============================================================================

def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize distributed training if running under torchrun.
    Returns (world_size, rank, local_rank).
    If not distributed, returns (1, 0, 0).
    """
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return world_size, rank, local_rank
    return 1, 0, 0


def init_distributed(config: TreeRLConfig, logger: logging.Logger) -> Tuple[int, int, int]:
    """
    Initialize DDP if world_size > 1 or use_ddp is True.
    Returns (world_size, rank, local_rank).
    """
    world_size, rank, local_rank = setup_distributed()
    
    if world_size > 1 or config.use_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend=config.ddp_backend)
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Set device for this process
        torch.cuda.set_device(local_rank)
        config.device = f"cuda:{local_rank}"
        
        logger.info(f"DDP initialized: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    else:
        logger.info("Running in single-GPU mode (no DDP)")
    
    return world_size, rank, local_rank


def cleanup_distributed():
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    """Check if this is the main process (rank 0)."""
    return rank == 0


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: TreeRLConfig) -> logging.Logger:
    logger = logging.getLogger("treerl_train")
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate logging from propagating to root logger
    logger.propagate = False

    os.makedirs(config.output_dir, exist_ok=True)
    log_path = os.path.join(config.output_dir, os.path.basename(config.log_file))

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def free_gpu_memory(logger: Optional[logging.Logger] = None):
    if logger:
        logger.info("  Freeing GPU memory...")

    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        if logger:
            allocated_before = torch.cuda.memory_allocated() / 1e9
            reserved_before = torch.cuda.memory_reserved() / 1e9
            logger.info(f"    Before: {allocated_before:.2f}GB alloc, {reserved_before:.2f}GB reserved")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        if logger:
            allocated_after = torch.cuda.memory_allocated() / 1e9
            reserved_after = torch.cuda.memory_reserved() / 1e9
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            logger.info(f"    After:  {allocated_after:.2f}GB alloc, {reserved_after:.2f}GB reserved, {free_mem:.1f}GB free")


def wait_for_gpu_memory(logger: logging.Logger, target_free_gb: float = 20.0, timeout: int = 120):
    if not torch.cuda.is_available():
        return True

    start = time.time()
    while time.time() - start < timeout:
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        if free_mem >= target_free_gb:
            logger.info(f"  ✓ GPU memory available: {free_mem:.1f}GB free")
            return True
        logger.debug(f"  Waiting for GPU memory... {free_mem:.1f}GB free (target: {target_free_gb:.1f}GB)")
        time.sleep(2)
        free_gpu_memory()
    free_mem = torch.cuda.mem_get_info()[0] / 1e9
    logger.warning(f"  ⚠️ Timeout waiting for GPU memory. Current: {free_mem:.1f}GB free")
    return False


# ============================================================================
# MERGE LoRA -> BASE (for vLLM serving)
# ============================================================================

def _safe_rmtree(path: str):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def merge_lora_into_bf16_base(
    base_model_id: str,
    adapter_path: str,
    merged_out_dir: str,
    logger: logging.Logger,
) -> str:
    """
    Merge a PEFT LoRA adapter into a BF16 base model and save a merged HF checkpoint dir.
    vLLM will serve THIS merged_out_dir.

    Notes:
    - This requires loading the base model in BF16 (GPU memory heavy).
    - For stability, we always merge into BF16, even if you later choose FP8 serving.
    """
    if not adapter_path or not os.path.isdir(adapter_path):
        raise ValueError(f"adapter_path not found: {adapter_path}")

    os.makedirs(merged_out_dir, exist_ok=True)

    # If directory already looks like a HF model, reuse it
    if os.path.exists(os.path.join(merged_out_dir, "config.json")):
        logger.info(f"  ✓ Reusing existing merged model at: {merged_out_dir}")
        return merged_out_dir

    logger.info("=" * 70)
    logger.info("MERGING LoRA -> BF16 BASE FOR vLLM SERVING")
    logger.info("=" * 70)
    logger.info(f"  Base:   {base_model_id}")
    logger.info(f"  Adapter:{adapter_path}")
    logger.info(f"  Out:    {merged_out_dir}")

    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as e:
        raise RuntimeError(
            "Missing deps for merge. Install: pip install transformers peft safetensors"
        ) from e

    free_gpu_memory(logger)
    wait_for_gpu_memory(logger, target_free_gb=20.0, timeout=180)

    # Load base BF16
    logger.info("  Loading base model (BF16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    # Apply adapter
    logger.info("  Applying adapter ...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Merge
    logger.info("  Merging adapter into base ...")
    model = model.merge_and_unload()

    # Save merged checkpoint
    logger.info("  Saving merged checkpoint ...")
    model.save_pretrained(
        merged_out_dir,
        safe_serialization=True,
        max_shard_size="10GB",
    )
    tok.save_pretrained(merged_out_dir)

    # Cleanup
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logger.info(f"  ✓ Merge complete in {time.time() - t0:.1f}s: {merged_out_dir}")
    return merged_out_dir


# ============================================================================
# VLLM SERVER MANAGEMENT
# ============================================================================

class VLLMServerManager:
    """Manages vLLM server lifecycle with LoRA support (PR #30802)."""

    def __init__(self, config: TreeRLConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{config.vllm_host}:{config.vllm_port}"
        self._log_thread: Optional[threading.Thread] = None
        self._stop_logging = threading.Event()
        self._log_file_path = os.path.join(config.output_dir, "vllm_server.log")
        self._current_lora_path: Optional[str] = None  # Track current LoRA adapter path

    def _ensure_reasoning_parser_plugin(self) -> Optional[str]:
        """
        Ensure reasoning parser plugin exists locally.
        For nano_v3 (Nemotron), the plugin is REQUIRED for correct parsing.
        For deepseek_r1, no plugin is needed (built-in).
        """
        if not self.config.vllm_use_reasoning_parser:
            return None

        # deepseek_r1 is a built-in parser, no plugin needed
        if self.config.vllm_reasoning_parser_name == "deepseek_r1":
            return None

        # For custom parsers like nano_v3, plugin file is REQUIRED
        if not self.config.vllm_reasoning_parser_plugin_filename:
            self.logger.warning(f"  ⚠️ Parser {self.config.vllm_reasoning_parser_name} may need a plugin file")
            return None

        plugin_path = os.path.join(self.config.output_dir, self.config.vllm_reasoning_parser_plugin_filename)
        
        # Always re-download to ensure we have the latest version
        url = (
            "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/"
            "resolve/main/nano_v3_reasoning_parser.py"
        )
        
        if not os.path.exists(plugin_path):
            self.logger.info(f"  Downloading nano_v3 reasoning parser plugin...")
            self.logger.info(f"    URL: {url}")
        
        try:
            import urllib.request
            os.makedirs(os.path.dirname(plugin_path) if os.path.dirname(plugin_path) else ".", exist_ok=True)
            urllib.request.urlretrieve(url, plugin_path)
            self.logger.info(f"  ✓ Plugin ready: {plugin_path}")
            return plugin_path
        except Exception as e:
            if os.path.exists(plugin_path):
                self.logger.info(f"  Using existing plugin: {plugin_path}")
                return plugin_path
            self.logger.error(f"  ❌ Could not download reasoning parser plugin: {e}")
            self.logger.error(f"     Download manually: wget {url}")
            raise RuntimeError(f"nano_v3 reasoning parser plugin required but download failed: {e}")

    def start_server(self, model_to_serve: str, lora_adapter_path: Optional[str] = None) -> bool:
        """
        Start vLLM server serving model_to_serve (HF id or local dir).
        
        Args:
            model_to_serve: Base model HF id or local path
            lora_adapter_path: Optional LoRA adapter path (used when use_vllm_lora=True)
        """
        if self.is_running():
            self.logger.info("vLLM server already running")
            return True

        self.logger.info("Starting vLLM server (Nemotron-3-Nano)...")
        free_gpu_memory(self.logger)
        wait_for_gpu_memory(self.logger, target_free_gb=20.0, timeout=180)

        dtype = self.config.inference_dtype.lower().strip()
        if dtype not in ("bf16", "fp8"):
            dtype = "bf16"

        # FP8 env vars per NVIDIA guide
        if dtype == "fp8":
            os.environ["VLLM_USE_FLASHINFER_MOE_FP8"] = "1"
            os.environ["VLLM_FLASHINFER_MOE_BACKEND"] = "throughput"

        kv_cache_dtype = "fp8" if dtype == "fp8" else "auto"

        # Reasoning budget via logits processor (single-request control)
        if self.config.vllm_use_logit_processor:
            if importlib.util.find_spec("custom_logit_processors") is None:
                self.logger.warning(
                    "  ⚠️ custom_logit_processors not found; disabling logits processor. "
                    "Install via: cd ./tools/budget && pip install -e ."
                )
                self.config.vllm_use_logit_processor = False
            else:
                processor_args = {
                    "thinking_budget": self.config.vllm_reasoning_budget,
                    "thinking_budget_grace_period": self.config.vllm_thinking_budget_grace_period,
                    "end_token_ids": list(self.config.vllm_thinking_end_token_ids),
                    "end_think_ids": [list(self.config.vllm_thinking_end_think_ids)],
                    "prompt_think_ids": list(self.config.vllm_thinking_prompt_think_ids),
                }
                os.environ["THINKING_BUDGET_LOGITS_PROCESSOR_ARGS"] = json.dumps(processor_args)

        cmd = [
            "vllm", "serve", model_to_serve,
            "--host", self.config.vllm_host,
            "--port", str(self.config.vllm_port),
            "--dtype", "auto",  # Per NVIDIA guide - auto-detect dtype
            "--trust-remote-code",
            "--async-scheduling",
            "--kv-cache-dtype", kv_cache_dtype,
            "--tensor-parallel-size", str(self.config.vllm_tensor_parallel_size),
            "--max-num-seqs", str(self.config.vllm_max_num_seqs),
            "--max-model-len", str(self.config.vllm_max_model_len),
            "--served-model-name", self.config.vllm_served_model_name,
            "--disable-log-requests",  # Reduce log noise
        ]

        # Force tokenizer to stay on the base model when serving LoRA
        tokenizer_to_use = (self.config.vllm_tokenizer or "").strip()
        if not tokenizer_to_use and self.config.use_vllm_lora and lora_adapter_path:
            tokenizer_to_use = model_to_serve
        if tokenizer_to_use:
            cmd.extend(["--tokenizer", tokenizer_to_use])

        # vLLM LoRA support (PR #30802)
        if self.config.use_vllm_lora and lora_adapter_path:
            self._current_lora_path = lora_adapter_path
            # --enable-lora enables LoRA adapter support
            # --lora-modules format: name=path (can specify multiple)
            lora_module_spec = f"{self.config.vllm_lora_name}={lora_adapter_path}"
            cmd.extend([
                "--enable-lora",
                "--lora-modules", lora_module_spec,
                "--max-lora-rank", "64",  # Support higher rank LoRA adapters
            ])
            self.logger.info(f"  LoRA enabled: {lora_module_spec}")
        else:
            self._current_lora_path = None

        # Optional logits processor (requires custom_logit_processors package)
        if self.config.vllm_use_logit_processor:
            cmd.extend([
                "--logits-processors",
                "custom_logit_processors.v1.nano_v3_logit_processors:ThinkingBudgetLogitsProcessor",
            ])

        # Optional tool-calling + reasoning parser plugin flags (per NVIDIA guide)
        if self.config.vllm_enable_auto_tool_choice:
            cmd.extend(["--enable-auto-tool-choice"])
        if self.config.vllm_tool_call_parser:
            cmd.extend(["--tool-call-parser", self.config.vllm_tool_call_parser])

        # Reasoning parser configuration (per NVIDIA Nemotron guide)
        # nano_v3 is the recommended parser for Nemotron-3-Nano models
        # It correctly separates <think>...</think> reasoning from final content
        if self.config.vllm_use_reasoning_parser and self.config.vllm_reasoning_parser_name:
            parser_name = self.config.vllm_reasoning_parser_name
            
            if parser_name == "nano_v3":
                # nano_v3 requires plugin file - download if needed
                plugin_path = self._ensure_reasoning_parser_plugin()
                if plugin_path:
                    cmd.extend([
                        "--reasoning-parser-plugin", plugin_path,
                        "--reasoning-parser", "nano_v3",
                    ])
                    self.logger.info(f"  ✓ Reasoning parser: nano_v3 (Nemotron-specific, with plugin)")
                else:
                    raise RuntimeError("nano_v3 parser requires plugin file but it's unavailable")
            elif parser_name == "deepseek_r1":
                # deepseek_r1 is a built-in parser (fallback option)
                cmd.extend(["--reasoning-parser", "deepseek_r1"])
                self.logger.info(f"  ✓ Reasoning parser: deepseek_r1 (built-in fallback)")
            else:
                # Other built-in or custom parsers
                plugin_path = self._ensure_reasoning_parser_plugin()
                if plugin_path:
                    cmd.extend([
                        "--reasoning-parser-plugin", plugin_path,
                        "--reasoning-parser", parser_name,
                    ])
                else:
                    cmd.extend(["--reasoning-parser", parser_name])
                self.logger.info(f"  ✓ Reasoning parser: {parser_name}")

        self.logger.info(f"vLLM command: {' '.join(cmd)}")

        # Start process with real-time log streaming
        self._stop_logging.clear()
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        self._log_thread = threading.Thread(target=self._stream_logs, daemon=True)
        self._log_thread.start()

        return self._wait_for_ready(timeout=900)

    def _stream_logs(self):
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            with open(self._log_file_path, "a") as log_file:
                for line in iter(self.process.stdout.readline, ''):
                    if self._stop_logging.is_set():
                        break
                    line = line.rstrip()
                    if line:
                        log_file.write(line + "\n")
                        log_file.flush()
                        print(f"[vLLM] {line}")
        except Exception as e:
            print(f"[vLLM] Log streaming error: {e}")

    def _wait_for_ready(self, timeout: int = 900) -> bool:
        start = time.time()
        health_url = f"{self.base_url}/health"
        print(f"\n[vLLM] Waiting for server at {health_url} (timeout={timeout}s)...")

        while time.time() - start < timeout:
            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    elapsed = time.time() - start
                    self.logger.info(f"vLLM server ready in {elapsed:.1f}s")
                    print(f"\n[vLLM] ✓ Server ready in {elapsed:.1f}s\n")
                    return True
            except requests.exceptions.RequestException:
                pass

            if self.process and self.process.poll() is not None:
                self.logger.error("vLLM server process died!")
                print("\n[vLLM] ✗ Server process died!")
                return False

            time.sleep(2)

        self.logger.error(f"vLLM server failed to start within {timeout}s")
        print(f"\n[vLLM] ✗ Server failed to start within {timeout}s")
        return False

    def is_running(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def stop_server(self):
        if not self.process:
            return

        self.logger.info("Stopping vLLM server...")
        self._stop_logging.set()

        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.logger.warning("  Force killing vLLM process...")
            self.process.kill()
            self.process.wait()

        if self._log_thread and self._log_thread.is_alive():
            self._log_thread.join(timeout=2)

        self.process = None

        # Give GPU time to release memory
        time.sleep(5)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        free_gpu_memory(self.logger)
        time.sleep(3)
        self.logger.info("vLLM server stopped, GPU memory freed")


# ============================================================================
# VLLM INFERENCE CLIENT (Nemotron thinking with OpenAI API)
# ============================================================================

class VLLMInferenceClient:
    """
    vLLM-based LLM client for Nemotron-3-Nano with reasoning budget support.
    
    Uses the OpenAI Python client for cleaner API interactions and proper
    handling of Nemotron's reasoning_content field.
    """

    def __init__(self, config: TreeRLConfig, logger: logging.Logger, server_manager: VLLMServerManager):
        self.config = config
        self.logger = logger
        self.server_manager = server_manager
        self.base_url = f"http://{config.vllm_host}:{config.vllm_port}"

        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="null",  # vLLM doesn't require auth by default
        )

        try:
            from api.system_prompts_updated import UNIFIED_SYSTEM_PROMPT
            self.system_prompt = UNIFIED_SYSTEM_PROMPT.strip()
        except ImportError:
            self.system_prompt = "You are a helpful assistant specialized in HTS classification."

        self._system_prompt_injection: Optional[str] = None

        # Compatibility attributes expected by your engines
        self.log_prompts = False
        self.prompt_logger = logger
        
        # When using vLLM LoRA, use the LoRA adapter name for API requests
        # This tells vLLM to use the LoRA adapter instead of base model
        if config.use_vllm_lora and server_manager._current_lora_path:
            self.model_name = config.vllm_lora_name
        else:
            self.model_name = config.vllm_served_model_name

        # JSON retry settings - increase retries for model format inconsistency
        self._max_json_retries = 6

        # Token estimation (very rough, conservative)
        self._avg_chars_per_token = 3.5
        
        # Reasoning and completion budgets (2k each by default)
        self.reasoning_budget = config.vllm_reasoning_budget  # Max tokens for <think>
        self.completion_budget = config.vllm_completion_budget  # Max tokens for actual response

    def set_system_prompt_injection(self, prompt: Optional[str]) -> None:
        self._system_prompt_injection = prompt

    def clear_system_prompt_injection(self) -> None:
        self._system_prompt_injection = None

    def _current_system_prompt(self) -> str:
        return (self._system_prompt_injection or self.system_prompt).strip()

    def _strip_think_tags(self, text: str) -> str:
        """
        Strip all <think>...</think> blocks and orphaned </think> tags from text.
        Handles nested/multiple think blocks that the nano_v3 parser may miss.
        """
        if not text:
            return text
        
        # Remove complete <think>...</think> blocks (non-greedy, handles nested)
        # Use a loop to handle nested blocks
        prev_text = None
        while prev_text != text:
            prev_text = text
            text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
        
        # Remove any orphaned </think> tags
        text = re.sub(r'</think>', '', text, flags=re.IGNORECASE)
        
        # Remove any orphaned <think> tags (shouldn't happen but be safe)
        text = re.sub(r'<think>', '', text, flags=re.IGNORECASE)
        
        return text.strip()

    def _call_with_reasoning_budget(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> Tuple[str, str]:
        """
        Call vLLM API with strict reasoning budget control (two-call pattern).
        
        Per NVIDIA docs, this uses separate budgets:
        1. First call: Get reasoning content up to reasoning_budget (2k default)
        2. Second call: Get completion up to completion_budget (2k default)
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            
        Returns:
            Tuple of (reasoning_content, final_content)
        """
        gen_start = time.time()
        
        # Step 1: Get reasoning content up to reasoning_budget
        self.logger.debug(f"    [vLLM] Step 1: Reasoning (max {self.reasoning_budget} tokens)...")
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.reasoning_budget,
            temperature=temperature,
            top_p=self.config.rollout_top_p,
        )
        
        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None) or ""
        initial_content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason or "unknown"
        
        # Check if reasoning completed naturally (has </think> or got content)
        reasoning_complete = (
            "</think>" in reasoning_content or 
            "</think>" in initial_content or
            initial_content.strip()  # Got actual content = reasoning done
        )
        
        if reasoning_complete and initial_content.strip():
            # Reasoning completed within budget AND we got content
            content = self._strip_think_tags(initial_content)
            reasoning_clean = reasoning_content.replace("</think>", "").replace("<think>", "").strip()
            gen_elapsed = time.time() - gen_start
            self.logger.debug(f"    [vLLM] ✓ Complete in single call ({gen_elapsed:.1f}s)")
            return reasoning_clean, content
        
        # Reasoning hit budget or didn't get content - need second call
        # Close reasoning gracefully
        if reasoning_content and not reasoning_content.rstrip().endswith("</think>"):
            reasoning_content = f"{reasoning_content.rstrip()}.\n</think>\n\n"
        elif not reasoning_content and initial_content:
            # Parser might have put everything in content
            if "<think>" in initial_content and "</think>" not in initial_content:
                reasoning_content = f"{initial_content.rstrip()}.\n</think>\n\n"
                initial_content = ""
        
        # Step 2: Continue with completion using completion_budget
        self.logger.debug(f"    [vLLM] Step 2: Completion (max {self.completion_budget} tokens)...")
        
        # Build continuation messages - append the closed reasoning
        continued_messages = [m.copy() for m in messages]
        continued_messages.append({
            "role": "assistant", 
            "content": reasoning_content
        })
        
        try:
            continuation = self.client.chat.completions.create(
                model=self.model_name,
                messages=continued_messages,
                max_tokens=self.completion_budget,
                temperature=temperature,
                top_p=self.config.rollout_top_p,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            
            final_content = continuation.choices[0].message.content or ""
            final_content = self._strip_think_tags(final_content)
            
            gen_elapsed = time.time() - gen_start
            self.logger.debug(f"    [vLLM] ✓ Two-call complete ({gen_elapsed:.1f}s)")
            
            reasoning_clean = reasoning_content.replace("</think>", "").replace("<think>", "").strip()
            return reasoning_clean, final_content
            
        except Exception as e:
            self.logger.warning(f"    [vLLM] ⚠️ Completion call failed: {e}")
            # Fallback: return whatever we got from first call
            reasoning_clean = reasoning_content.replace("</think>", "").replace("<think>", "").strip()
            return reasoning_clean, self._strip_think_tags(initial_content)

    def _estimate_token_count(self, text: str) -> int:
        return int(len(text) / self._avg_chars_per_token) + 10

    def _estimate_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += self._estimate_token_count(msg.get("content", ""))
            total += 20
        total += 50
        return total

    def _calculate_max_tokens(self, messages: List[Dict[str, str]], requested_max: Optional[int] = None) -> int:
        input_tokens = self._estimate_messages_tokens(messages)
        available = self.config.vllm_max_model_len - input_tokens - self.config.token_safety_margin
        if requested_max is not None:
            safe_max = min(requested_max, available, self.config.rollout_max_new_tokens_cap)
        else:
            safe_max = min(available, self.config.rollout_max_new_tokens_cap)
        safe_max = max(safe_max, 256)
        if available < 1000:
            self.logger.warning(
                f"  ⚠️ Low available tokens: input≈{input_tokens}, available≈{available}, max_tokens={safe_max}"
            )
        return safe_max

    def _log_completion(self, request: Dict, response: Dict) -> None:
        try:
            log_path = self.config.completions_log
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            messages = request.get("messages", [])
            last_user_msg = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user_msg = m.get("content", "")[:2000]
                    break
            entry = {
                "timestamp": datetime.now().isoformat(),
                "request": {
                    "model": request.get("model"),
                    "max_tokens": request.get("max_tokens"),
                    "temperature": request.get("temperature"),
                    "messages_count": len(messages),
                    "last_user_msg_preview": last_user_msg,
                },
                "response": response,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.debug(f"Failed to log completion: {e}")

    def _extract_json_from_response(self, response_text: str) -> str:
        """
        Extract valid JSON from response text.
        Expects clean content from reasoning parser (no <think> tags).
        """
        if not response_text:
            raise ValueError("Empty response - retry needed")

        text = response_text.strip()
        
        # Handle markdown code blocks
        if "```json" in text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()
        elif "```" in text:
            match = re.search(r'```\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()

        # Try direct parse
        try:
            json.loads(text)
            return text
        except Exception:
            pass

        # Find first JSON object/array
        first_bracket = -1
        for i, c in enumerate(text):
            if c in '[{':
                first_bracket = i
                break
        
        if first_bracket == -1:
            raise ValueError(f"No JSON found in response:\n{text}")
        
        # Extract balanced JSON
        start_pos = first_bracket
        depth = 0
        end_pos = -1
        in_string = False
        escape_next = False
        
        for i in range(start_pos, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\':
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break
        
        if end_pos > start_pos:
            candidate = text[start_pos:end_pos + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass
        
        raise ValueError(f"Invalid JSON in response:\n{text}")

    def _call_vllm_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = True,
    ) -> str:
        """
        Single-call to vLLM server (no budget control).
        
        For budgeted calls, use _call_with_reasoning_budget() instead.
        
        Args:
            messages: Chat messages list
            temperature: Sampling temperature (0 = greedy)
            max_tokens: Maximum tokens to generate
            enable_thinking: Whether to enable <think> reasoning (default: True)
        
        Returns:
            Extracted content (with reasoning stripped if nano_v3 parser is working)
        """
        gen_start = time.time()
        safe_max_tokens = self._calculate_max_tokens(messages, max_tokens)

        effective_temp = temperature
        effective_top_p = self.config.rollout_top_p

        # Build extra_body for thinking control and logits processor args
        extra_body = {}
        if not enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.config.vllm_use_logit_processor:
            extra_body["vllm_xargs"] = {
                "thinking_budget": self.config.vllm_reasoning_budget,
                "thinking_budget_grace_period": self.config.vllm_thinking_budget_grace_period,
                "end_token_ids": json.dumps(list(self.config.vllm_thinking_end_token_ids)),
            }

        try:
            # Use OpenAI client for cleaner API interaction
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=safe_max_tokens,
                temperature=effective_temp,
                top_p=effective_top_p,
                extra_body=extra_body if extra_body else None,
            )
            
            # Log completion for debugging
            self._log_completion(
                {"model": self.model_name, "max_tokens": safe_max_tokens, 
                 "temperature": effective_temp, "messages": messages},
                {"choices": [{"message": {"content": response.choices[0].message.content}}]}
            )

            choice = response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason or "unknown"

            # Get usage stats
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            gen_elapsed = time.time() - gen_start
            tok_per_sec = completion_tokens / gen_elapsed if gen_elapsed > 0 else 0.0

            self.logger.debug(
                f"    [vLLM] {completion_tokens} tokens in {gen_elapsed:.1f}s "
                f"({tok_per_sec:.1f} tok/s, prompt={prompt_tokens}, max={safe_max_tokens}, finish={finish_reason})"
            )
            if finish_reason == "length":
                self.logger.warning(f"  ⚠️ Hit max_tokens ({safe_max_tokens})!")

            # nano_v3 reasoning parser should populate:
            # - message.content = final response (JSON, no <think> tags)
            # - message.reasoning_content = thinking content (Nemotron-specific)
            content = message.content or ""
            reasoning = getattr(message, 'reasoning_content', None) or ""
            
            # Robust fallback: if content is empty but reasoning exists, the parser might have
            # dumped everything into reasoning_content (can happen with some vLLM versions)
            if not content.strip() and reasoning.strip():
                # Check if reasoning contains the JSON block after </think>
                if "</think>" in reasoning:
                    parts = reasoning.split("</think>")
                    content = parts[-1].strip()
                    reasoning = "</think>".join(parts[:-1]) + "</think>"
                else:
                    # If no tags but content is empty, treat reasoning as the full raw output
                    content = reasoning
                    reasoning = ""

            # Post-process: strip any remaining <think> tags that parser missed
            content = self._strip_think_tags(content)
            
            # Check if we have valid content after stripping
            content_stripped = content.strip()
            if content_stripped:
                content_looks_like_json = content_stripped.startswith("{") or content_stripped.startswith("[")
                if content_looks_like_json:
                    self.logger.debug(f"    [vLLM] ✓ OK - clean JSON content")
                else:
                    self.logger.debug(f"    [vLLM] ✓ OK - content (non-JSON)")
                return content_stripped
            else:
                # Content is empty after stripping
                self.logger.warning(f"    [vLLM] ⚠️ Empty content after post-processing (reasoning exists: {bool(reasoning)})")
                return ""

        except Exception as e:
            self.logger.error(f"vLLM API error: {e}")
            raise

    def send_openai_request(
        self,
        prompt: str,
        requires_json: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        use_reasoning_budget: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Send a request to vLLM with reasoning budget control.
        
        Args:
            prompt: User prompt
            requires_json: Whether response must be valid JSON
            temperature: Sampling temperature
            max_tokens: Optional override for total output tokens
            use_reasoning_budget: Enforce reasoning budget via logits processor
        """
        user_content = (prompt or "").strip()

        messages = [
            {"role": "system", "content": self._current_system_prompt()},
            {"role": "user", "content": user_content},
        ]

        last_error = None
        for attempt in range(self._max_json_retries if requires_json else 1):
            try:
                if use_reasoning_budget and self.reasoning_budget > 0:
                    total_budget = self.reasoning_budget + self.completion_budget
                    safe_max = self._calculate_max_tokens(messages, total_budget)
                    if safe_max < total_budget:
                        self.logger.warning(
                            f"  ⚠️ Output budget reduced by context: "
                            f"requested={total_budget}, safe_max={safe_max}"
                        )
                    text = self._call_vllm_api(messages, temperature=temperature, max_tokens=safe_max)
                else:
                    text = self._call_vllm_api(messages, temperature=temperature, max_tokens=max_tokens)
                
                if not text.strip():
                    raise ValueError("Empty response from vLLM API")
                if requires_json:
                    cleaned = self._extract_json_from_response(text)
                    json.loads(cleaned)
                    return cleaned
                return text
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
                self.logger.warning(
                    f"JSON extraction failed (attempt {attempt + 1}/{self._max_json_retries}): {str(e)}"
                )
                if attempt < self._max_json_retries - 1:
                    continue
                raise last_error

    def send_trajectory_request(
        self,
        messages: List[Dict[str, str]],
        requires_json: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        use_reasoning_budget: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Send a trajectory request to vLLM with reasoning budget control.
        
        Args:
            messages: Full conversation messages
            requires_json: Whether response must be valid JSON
            temperature: Sampling temperature
            max_tokens: Optional override for total output tokens
            use_reasoning_budget: Enforce reasoning budget via logits processor
        """
        req_messages = [m.copy() for m in messages]

        last_error = None
        for attempt in range(self._max_json_retries if requires_json else 1):
            try:
                if use_reasoning_budget and self.reasoning_budget > 0:
                    total_budget = self.reasoning_budget + self.completion_budget
                    safe_max = self._calculate_max_tokens(req_messages, total_budget)
                    if safe_max < total_budget:
                        self.logger.warning(
                            f"  ⚠️ Output budget reduced by context: "
                            f"requested={total_budget}, safe_max={safe_max}"
                        )
                    text = self._call_vllm_api(req_messages, temperature=temperature, max_tokens=safe_max)
                else:
                    text = self._call_vllm_api(req_messages, temperature=temperature, max_tokens=max_tokens)
                
                if not text.strip():
                    raise ValueError("Empty response from vLLM API")
                if requires_json:
                    cleaned = self._extract_json_from_response(text)
                    json.loads(cleaned)
                    return cleaned
                return text
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
                self.logger.warning(
                    f"JSON extraction failed (attempt {attempt + 1}/{self._max_json_retries}): {str(e)}"
                )
                if attempt < self._max_json_retries - 1:
                    continue
                raise last_error

    # Compatibility aliases
    def send_vertex_ai_request(self, *args, **kwargs) -> str:
        return self.send_openai_request(*args, **kwargs)

    def send_groq_request(self, *args, **kwargs) -> str:
        return self.send_openai_request(*args, **kwargs)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_chapter_rulings(config: TreeRLConfig, logger: logging.Logger) -> List[Dict]:
    logger.info(f"Loading rulings from {config.cross_rulings_file}")
    if not os.path.exists(config.cross_rulings_file):
        logger.error(f"Rulings file not found: {config.cross_rulings_file}")
        return []
    with open(config.cross_rulings_file, 'r', encoding='utf-8') as f:
        all_rulings = json.load(f)
    chapter_rulings = [r for r in all_rulings if r.get("hts_code", "").startswith(config.chapter)]
    logger.info(f"Total rulings: {len(all_rulings)}")
    logger.info(f"Chapter {config.chapter} rulings: {len(chapter_rulings)}")
    return chapter_rulings


def is_gold_trace_valid(gold_code: str, gold_trace: List[Dict], normalize_code_fn) -> Tuple[bool, str]:
    """
    Check if a gold trace is valid for training.
    
    A valid trace means the tree reaches a terminal point for this gold code:
    1. The final trace code must be a PREFIX of the normalized gold code
    2. AND either exact match OR at least 6 digits (subheading level)
    
    Args:
        gold_code: The expected HTS code (e.g., "8414.30.0000")
        gold_trace: The trace built from the HTS tree
        normalize_code_fn: Function to normalize codes (strip dots/non-digits)
        
    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    if not gold_trace:
        return False, "empty trace"
    
    gold_normalized = normalize_code_fn(gold_code)
    if not gold_normalized:
        return False, "empty gold code"
    
    # Find the last code step in the trace (skip groups at the end)
    final_code = None
    for step in reversed(gold_trace):
        if step.get("kind") == "code":
            final_code = normalize_code_fn(step.get("code", ""))
            break
    
    if not final_code:
        return False, "no code step in trace"
    
    # Criterion 1: Final code must be a prefix of gold code
    if not gold_normalized.startswith(final_code):
        return False, f"trace code {final_code} is not prefix of gold {gold_normalized}"
    
    # Criterion 2: Either exact match OR at least 6 digits (subheading level)
    if final_code == gold_normalized:
        return True, "exact match"
    elif len(final_code) >= 6:
        return True, f"valid terminal at {len(final_code)} digits"
    else:
        return False, f"trace only {len(final_code)} digits (need ≥6 or exact match)"


def filter_rulings_with_valid_gold_trace(
    rulings: List[Dict],
    logger: logging.Logger,
) -> List[Dict]:
    """
    Pre-filter rulings to only those with VALID gold traces in the HTS tree.
    
    A valid gold trace means the tree reaches a proper terminal point for the gold code:
    - Final trace code is a prefix of the gold code
    - AND either exact match or at least 6 digits (subheading level)
    
    Args:
        rulings: List of ruling dicts with 'hts_code' field
        logger: Logger instance
    
    Returns:
        Filtered list of rulings with valid gold traces
    """
    try:
        from api.treerl_gold_trace import normalize_code, build_gold_trace
        from api.groq_tree_engine import HTSTree
    except ImportError as e:
        logger.warning(f"Could not import gold trace components for filtering: {e}")
        return rulings  # Return unfiltered if imports fail
    
    # Build HTS tree once for validation
    hts_tree = HTSTree()
    hts_data_file = script_dir / "api" / "hts_data.json"
    if hts_data_file.exists():
        with open(hts_data_file, "r", encoding="utf-8") as f:
            hts_data = json.load(f)
        hts_tree.build_from_json(hts_data)
    else:
        logger.warning(f"HTS data file not found: {hts_data_file}")
        return rulings
    
    valid_rulings = []
    invalid_rulings = []  # (gold_code, reason)
    
    for ruling in rulings:
        gold_code = ruling.get("hts_code", "")
        if not gold_code:
            continue
        
        gold_trace = build_gold_trace(gold_code, hts_tree.navigator)
        is_valid, reason = is_gold_trace_valid(gold_code, gold_trace, normalize_code)
        
        if is_valid:
            valid_rulings.append(ruling)
        else:
            invalid_rulings.append((gold_code, reason))
    
    # Show breakdown by reason
    reason_counts = {}
    for code, reason in invalid_rulings:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    reason_summary = ", ".join(f"{r}: {c}" for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]))
    
    logger.info(f"  ✓ Valid: {len(valid_rulings)}/{len(rulings)} | Skipped: {len(invalid_rulings)} ({reason_summary or 'none'})")
    
    # Show example skipped codes at debug level
    if invalid_rulings and len(invalid_rulings) <= 10:
        for code, reason in invalid_rulings:
            logger.debug(f"    Skipped {code}: {reason}")
    
    return valid_rulings


# ============================================================================
# SAMPLE SAVING / ROLLOUT CACHE
# ============================================================================

def save_samples_for_debug(
    samples: List[Dict],
    config: TreeRLConfig,
    logger: logging.Logger,
    epoch: int,
    ruling_desc: str = "",
) -> str:
    os.makedirs(config.samples_dir, exist_ok=True)
    timestamp = int(time.time())
    safe_desc = "".join(c if c.isalnum() else "_" for c in ruling_desc[:30])
    filename = f"samples_epoch{epoch}_{safe_desc}_{timestamp}.json"
    filepath = os.path.join(config.samples_dir, filename)

    serializable_samples = []
    for s in samples:
        serializable_samples.append({
            "messages": s.get("messages", []),
            "step_rewards": s.get("step_rewards", []),
            "gold_code": s.get("gold_code", ""),
            "pred_trace": s.get("pred_trace", []),
            "gold_trace": s.get("gold_trace", []),
            "path_id": s.get("path_id", ""),
            "leaf_reward": s.get("leaf_reward", 0),
            "reward_components": s.get("reward_components", None),
            "source": s.get("source", ""),
        })

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_samples, f, indent=2, ensure_ascii=False)

    logger.info(f"  Saved {len(samples)} samples to: {filepath}")
    return filepath


def save_mapping_debug(
    samples: List[Dict],
    config: TreeRLConfig,
    logger: logging.Logger,
    batch_num: int,
) -> str:
    """
    Save detailed mapping debug information for manual inspection.
    Shows which assistant turns mapped to which pred_trace steps and why others failed.
    """
    os.makedirs(config.samples_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"mapping_debug_batch{batch_num}_{timestamp}.json"
    filepath = os.path.join(config.samples_dir, filename)
    
    debug_data = []
    for sample_idx, s in enumerate(samples[:20]):  # Limit to first 20 samples for readability
        messages = s.get("messages", [])
        pred_trace = s.get("pred_trace", [])
        
        # Analyze each assistant turn
        turn_analysis = []
        last_user = ""
        assistant_idx = 0
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                last_user = content
            elif role == "assistant":
                # Detect task type
                user_lower = last_user[:500].lower()
                if "select_chapters" in user_lower:
                    task_type = "select_chapters"
                elif "rank_candidates" in user_lower:
                    task_type = "rank_candidates"
                elif "generate_question" in user_lower:
                    task_type = "generate_question"
                elif "process_answer" in user_lower:
                    task_type = "process_answer"
                else:
                    task_type = "other"
                
                # Try to extract
                extraction_result = None
                match_result = None
                
                try:
                    assistant_json = json.loads(content)
                    
                    if "top_selection" in assistant_json:
                        chapter = str(assistant_json["top_selection"])
                        normalized = _normalize_code(chapter)
                        if len(normalized) == 1:
                            normalized = normalized.zfill(2)
                        extraction_result = {"kind": "chapter", "code": normalized}
                        
                    elif "primary_selection" in assistant_json:
                        ps = assistant_json.get("primary_selection", {})
                        option_idx = ps.get("option_index")
                        
                        json_start = last_user.find("JSON INPUT:")
                        if json_start != -1:
                            json_str = last_user[json_start + len("JSON INPUT:"):].strip()
                            try:
                                user_json = json.loads(json_str)
                                children = user_json.get("data", {}).get("classification_tree", {}).get("children", [])
                                if children and option_idx and 1 <= option_idx <= len(children):
                                    child = children[option_idx - 1]
                                    if child.get("is_group"):
                                        extraction_result = {"kind": "group", "node_id": child.get("node_id")}
                                    else:
                                        extraction_result = {"kind": "code", "code": _normalize_code(child.get("code", ""))}
                                else:
                                    extraction_result = f"FAIL: option_idx={option_idx} out of range (len={len(children)})"
                            except:
                                extraction_result = "FAIL: couldn't parse user JSON"
                        else:
                            extraction_result = "FAIL: no 'JSON INPUT:' in user message"
                    else:
                        extraction_result = f"SKIP: non-selection keys={list(assistant_json.keys())[:3]}"
                        
                except json.JSONDecodeError:
                    extraction_result = "FAIL: assistant content not valid JSON"
                
                # Try to match if we got an identifier
                if isinstance(extraction_result, dict):
                    for step_idx, step in enumerate(pred_trace):
                        if extraction_result.get("kind") == "group":
                            if step.get("kind") == "group" and step.get("node_id") == extraction_result.get("node_id"):
                                match_result = f"MATCHED step {step_idx}"
                                break
                        elif extraction_result.get("kind") in ("chapter", "code"):
                            if step.get("kind") in ("chapter", "code"):
                                if step.get("code") == extraction_result.get("code"):
                                    match_result = f"MATCHED step {step_idx}"
                                    break
                    if match_result is None:
                        match_result = f"NO MATCH: identifier={extraction_result}, pred_trace_codes={[s.get('code') for s in pred_trace if s.get('code')]}"
                
                turn_analysis.append({
                    "assistant_idx": assistant_idx,
                    "task_type": task_type,
                    "extraction": extraction_result,
                    "match": match_result,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                })
                assistant_idx += 1
        
        debug_data.append({
            "sample_idx": sample_idx,
            "path_id": s.get("path_id", ""),
            "gold_code": s.get("gold_code", ""),
            "leaf_reward": s.get("leaf_reward", 0),
            "pred_trace": pred_trace,
            "num_assistant_turns": assistant_idx,
            "turn_analysis": turn_analysis,
        })
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  Saved mapping debug to: {filepath}")
    return filepath


def save_rollouts_to_file(
    all_samples: List[Dict],
    filepath: str,
    logger: logging.Logger,
    metadata: Optional[Dict] = None,
) -> str:
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    serializable_samples = []
    for s in all_samples:
        serializable_samples.append({
            "messages": s.get("messages", []),
            "step_rewards": s.get("step_rewards", []),
            "gold_code": s.get("gold_code", ""),
            "pred_trace": s.get("pred_trace", []),
            "gold_trace": s.get("gold_trace", []),
            "path_id": s.get("path_id", ""),
            "leaf_reward": s.get("leaf_reward", s.get("reward", 0)),
            "reward_components": s.get("reward_components", None),
            "source": s.get("source", ""),
        })
    output = {
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(serializable_samples),
        "samples": serializable_samples,
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved {len(serializable_samples)} rollout samples to: {filepath}")
    return filepath


def load_rollouts_from_file(filepath: str, logger: logging.Logger) -> List[Dict]:
    if not os.path.exists(filepath):
        logger.error(f"Rollouts file not found: {filepath}")
        return []
    logger.info(f"Loading rollout samples from: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        samples = data
    else:
        samples = data.get("samples", [])
        metadata = data.get("metadata", {})
        timestamp = data.get("timestamp", "unknown")
        logger.info(f"  Loaded from: {timestamp}")
        if metadata:
            logger.info(f"  Metadata: {metadata}")
    logger.info(f"✓ Loaded {len(samples)} rollout samples")
    return samples


def _extract_predicted_code_from_sample(sample: Dict, normalize: bool = True) -> str:
    """
    Extract the final predicted HTS code from a sample.
    
    Args:
        sample: Sample dict with classification_path or pred_trace
        normalize: If True, normalize to 10-digit format with dots (e.g., 8473.30.00.00)
    
    Returns:
        HTS code string, optionally formatted with proper dots
    """
    raw_code = ""
    
    # Try classification_path first (has full path)
    classification_path = sample.get("classification_path", [])
    if classification_path:
        for step in reversed(classification_path):
            code = step.get("code", "")
            if code and code != "[GROUP]":
                raw_code = code
                break
    
    # Try pred_trace
    if not raw_code:
        pred_trace = sample.get("pred_trace", [])
        for step in reversed(pred_trace):
            if step.get("kind") == "code":
                raw_code = step.get("code", "")
                break
    
    if not raw_code:
        return "unknown"
    
    if not normalize:
        return raw_code
    
    # Normalize to 10 digits and format with dots
    return format_hts_code(raw_code)


def display_rollout_stats(samples: List[Dict], logger: logging.Logger) -> None:
    """Display comprehensive rollout statistics with per-ruling breakdown."""
    if not samples:
        logger.warning("No samples to display stats for")
        return

    logger.info("\n" + "=" * 70)
    logger.info("ROLLOUT STATISTICS (Before Training)")
    logger.info("=" * 70)

    # Group by gold code
    by_ruling: Dict[str, List[Dict]] = {}
    for s in samples:
        gold = s.get("gold_code", "unknown")
        by_ruling.setdefault(gold, []).append(s)

    logger.info(f"\n📊 OVERVIEW")
    logger.info(f"  Total samples (beam paths): {len(samples)}")
    logger.info(f"  Unique rulings: {len(by_ruling)}")
    logger.info(f"  Avg paths per ruling: {len(samples) / max(len(by_ruling), 1):.1f}")

    # Collect aggregate stats
    all_leaf_rewards = []
    all_step_R = []
    perfect_count = partial_count = zero_count = 0

    for s in samples:
        leaf_r = s.get("leaf_reward", s.get("reward", 0)) or 0.0
        all_leaf_rewards.append(leaf_r)
        if leaf_r == 1.0:
            perfect_count += 1
        elif leaf_r > 0:
            partial_count += 1
        else:
            zero_count += 1
        for sr in s.get("step_rewards", []):
            all_step_R.append(sr.get("R", 0))

    logger.info(f"\n🎯 LEAF REWARDS (Prefix Match with Gold)")
    logger.info(f"  Perfect (=1.0): {perfect_count} ({100*perfect_count/max(len(samples), 1):.1f}%)")
    logger.info(f"  Partial (0<r<1): {partial_count} ({100*partial_count/max(len(samples), 1):.1f}%)")
    logger.info(f"  Zero (=0): {zero_count} ({100*zero_count/max(len(samples), 1):.1f}%)")
    if all_leaf_rewards:
        logger.info(f"  Min: {min(all_leaf_rewards):.3f}")
        logger.info(f"  Max: {max(all_leaf_rewards):.3f}")
        logger.info(f"  Mean: {sum(all_leaf_rewards)/len(all_leaf_rewards):.3f}")

    logger.info(f"\n📈 STEP REWARDS R(s) (TreeRL Process Supervision)")
    if all_step_R:
        logger.info(f"  Count: {len(all_step_R)}")
        logger.info(f"  Min: {min(all_step_R):.4f}")
        logger.info(f"  Max: {max(all_step_R):.4f}")
        logger.info(f"  Mean: {sum(all_step_R)/len(all_step_R):.4f}")

    # Per-ruling breakdown
    logger.info(f"\n" + "-" * 70)
    logger.info("📋 PER-RULING BREAKDOWN")
    logger.info("-" * 70)
    
    for ruling_idx, (gold_code, ruling_samples) in enumerate(sorted(by_ruling.items()), 1):
        # Sort samples by reward (best first)
        sorted_samples = sorted(
            ruling_samples, 
            key=lambda x: x.get("leaf_reward", x.get("reward", 0)) or 0.0, 
            reverse=True
        )
        
        rewards = [s.get("leaf_reward", s.get("reward", 0)) or 0.0 for s in sorted_samples]
        best_reward = max(rewards) if rewards else 0.0
        
        # Get best predicted code
        best_sample = sorted_samples[0] if sorted_samples else {}
        best_pred = _extract_predicted_code_from_sample(best_sample)
        
        # Get gold trace from first sample
        gold_trace = best_sample.get("gold_trace", [])
        gold_trace_parts = []
        for step in gold_trace:
            if step.get("kind") == "chapter":
                gold_trace_parts.append(f"CH{step.get('code', '??')}")
            elif step.get("kind") == "group":
                gold_trace_parts.append("[GRP]")
            else:
                gold_trace_parts.append(step.get("code", "??"))
        gold_trace_str = " > ".join(gold_trace_parts) if gold_trace_parts else "N/A"
        
        # Status emoji
        if best_reward >= 0.9999:
            status = "✅"  # Perfect match
        elif best_reward > 0:
            status = "🟡"  # Partial match
        else:
            status = "❌"  # No match
        
        # Count reward distribution for this ruling
        r_perfect = sum(1 for r in rewards if r >= 0.9999)
        r_partial = sum(1 for r in rewards if 0 < r < 0.9999)
        r_zero = sum(1 for r in rewards if r == 0)
        
        logger.info(f"\n  [{ruling_idx:2d}] {status} Gold: {format_hts_code(gold_code)} | {len(gold_trace)} steps | {gold_trace_str}")
        logger.info(f"       Best Pred: {best_pred} (reward={best_reward:.3f})")
        logger.info(f"       Paths: {len(ruling_samples)} | ✅{r_perfect} 🟡{r_partial} ❌{r_zero}")
        logger.info(f"       Rewards: [{', '.join(f'{r:.2f}' for r in rewards[:5])}{'...' if len(rewards) > 5 else ''}]")
        
        # Show top 3 distinct predictions if there's variation
        unique_preds = []
        seen_codes = set()
        for s in sorted_samples:
            pred = _extract_predicted_code_from_sample(s)
            if pred not in seen_codes:
                seen_codes.add(pred)
                r = s.get("leaf_reward", s.get("reward", 0)) or 0.0
                unique_preds.append((pred, r))
            if len(unique_preds) >= 3:
                break
        
        if len(unique_preds) > 1:
            logger.info(f"       Top preds: " + " | ".join(f"{p}({r:.2f})" for p, r in unique_preds))
    
    logger.info("\n" + "=" * 70)


# ============================================================================
# ONLINE ROLLOUT (vLLM phase) - unchanged logic, just uses Nemotron client
# ============================================================================

def run_online_rollout(
    ruling: Dict,
    config: TreeRLConfig,
    logger: logging.Logger,
    vllm_client: VLLMInferenceClient,
) -> List[Dict]:
    try:
        from api.treerl_gold_trace import build_gold_trace, build_pred_trace_from_path
        from api.treerl_rewards import compute_leaf_reward
        from api.treerl_process_supervision import compute_treerl_rewards, emit_leaf_samples
        from llm_auto_responder import LLMAutoResponder
        from api.groq_tree_engine import HTSTree
    except ImportError as e:
        logger.error(f"Could not import TreeRL components: {e}")
        return []

    def _code_digits_prefix_reward(pred_trace: List[Dict[str, Any]], gold_code: str) -> float:
        """
        Compare predicted code to gold code, both normalized to 10 digits.
        Uses normalize_hts_to_10_digits for consistent comparison.
        """
        gold_digits = normalize_hts_to_10_digits(gold_code)
        if not gold_digits:
            return 0.0
        
        # Find the final code from pred_trace
        pred_digits = ""
        for step in reversed(pred_trace or []):
            if step.get("kind") == "code":
                pred_digits = normalize_hts_to_10_digits(step.get("code", ""))
                if pred_digits:
                    break
        if not pred_digits:
            return 0.0
        
        # Compare digit by digit
        m = 0
        for a, b in zip(pred_digits, gold_digits):
            if a == b:
                m += 1
            else:
                break
        return m / 10  # Always compare against 10 digits

    def _aggregate_leaf_reward(components: Dict[str, float]) -> float:
        w = list(config.leaf_reward_weights or ())
        if len(w) < 2:
            w = [1.0, 0.0]
        w = w[:2]
        r = (w[0] * float(components.get("trace_prefix", 0.0))) + (w[1] * float(components.get("code_digits_prefix", 0.0)))
        if config.leaf_reward_clip_0_1:
            r = max(0.0, min(1.0, r))
        return r

    os.environ["TREERL_BEAM_SIZE"] = str(config.beam_size)
    os.environ["TREERL_CHAPTER_BEAM_SIZE"] = str(config.beam_size)
    os.environ["DISABLE_CROSS_RULING_INJECTION"] = "true"

    # CRITICAL: Disable SFT training collector to enable parallel beam processing
    # The classification engine uses ThreadPoolExecutor when this is false
    os.environ["COLLECT_TRAINING_DATA"] = "false"

    # Set parallel workers for within-classification beam parallelization
    os.environ["PATH_WORKERS"] = str(config.beam_size * 2)
    os.environ["CALIBRATE_WORKERS"] = str(config.beam_size * 2)

    product_description = ruling.get("short_product_description", "")
    gold_code = ruling.get("hts_code", "")

    try:
        logger.debug(f"  [rollout] Creating HTSTree...")
        hts_tree = HTSTree()

        logger.debug(f"  [rollout] Injecting vLLM client...")
        hts_tree.llm_client = vllm_client
        hts_tree.client = None
        if hasattr(hts_tree, "classification_engine") and hasattr(hts_tree.classification_engine, "llm"):
            hts_tree.classification_engine.llm = vllm_client
        if hasattr(hts_tree, "streaming_engine"):
            if hasattr(hts_tree.streaming_engine, "llm"):
                hts_tree.streaming_engine.llm = vllm_client
            if hasattr(hts_tree.streaming_engine, "llm_client"):
                hts_tree.streaming_engine.llm_client = vllm_client

        logger.debug(f"  [rollout] Loading HTS data...")
        hts_data_file = script_dir / "api" / "hts_data.json"
        if hts_data_file.exists():
            with open(hts_data_file, "r", encoding="utf-8") as f:
                hts_data = json.load(f)
            hts_tree.build_from_json(hts_data)

        logger.debug(f"  [rollout] Building gold trace for {gold_code}...")
        gold_trace = build_gold_trace(gold_code, hts_tree.navigator)
        
        # Build readable trace path for logging
        trace_path_parts = []
        for step in gold_trace:
            if step.get("kind") == "chapter":
                trace_path_parts.append(f"CH{step.get('code', '??')}")
            elif step.get("kind") == "group":
                trace_path_parts.append("[GRP]")
            else:
                trace_path_parts.append(step.get("code", "??"))
        trace_path_str = " > ".join(trace_path_parts)
        
        # Validate gold trace using same criterion as pre-filter
        from api.treerl_gold_trace import normalize_code
        is_valid, reason = is_gold_trace_valid(gold_code, gold_trace, normalize_code)
        
        if is_valid:
            logger.info(f"  [rollout] Gold: {gold_code} | {len(gold_trace)} steps | {trace_path_str}")
        else:
            logger.warning(f"  [rollout] ⚠️ SKIPPING - Gold code '{gold_code}' invalid: {reason}")
            logger.warning(f"  [rollout]    Trace: {trace_path_str}")
            return []

        logger.debug(f"  [rollout] Initializing auto-responder...")
        auto_responder = LLMAutoResponder(engine_name="groq", debug=False)
        if hasattr(auto_responder, "llm_client"):
            auto_responder.llm_client = vllm_client

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
        leaves = []
        beam = state.get("beam", [])
        
        # Log all beam paths (shows final codes for each path)
        if beam:
            logger.info(f"  [rollout] Beam paths ({len(beam)} total):")
            for i, path_data in enumerate(beam):
                if isinstance(path_data, dict):
                    cp = path_data.get("classification_path", [])
                    conf = path_data.get("cumulative_confidence", 0)
                elif hasattr(path_data, "classification_path"):
                    cp = path_data.classification_path
                    conf = getattr(path_data, "cumulative_confidence", 0)
                else:
                    continue
                # Build path string showing the codes at each level
                codes = [step.get("code", "?") for step in cp if step.get("code")]
                final_code = codes[-1] if codes else "?"
                path_str = " > ".join(codes)
                logger.info(f"    [{i+1}] {final_code} (conf={conf:.3f}) | {path_str}")

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
            reward_components = {
                "trace_prefix": compute_leaf_reward(pred_trace, gold_trace),
                "code_digits_prefix": _code_digits_prefix_reward(pred_trace, gold_code),
            }
            reward = _aggregate_leaf_reward(reward_components)

            leaves.append({
                "path_id": path_id,
                "pred_trace": pred_trace,
                "reward": reward,
                "reward_components": reward_components,
                "trajectory": trajectory,
                "classification_path": classification_path,
                "source": "final_beam",
            })

        pruned_leaves = state.get("_treerl_pruned_leaves", [])
        if pruned_leaves:
            logger.info(f"  [rollout] Pruned paths ({len(pruned_leaves)} total):")
            for i, pruned in enumerate(pruned_leaves[:5]):  # Show first 5
                cp = pruned.get("classification_path", [])
                codes = [step.get("code", "?") for step in cp if step.get("code")]
                final_code = codes[-1] if codes else "?"
                logger.info(f"    [P{i+1}] {final_code} | {' > '.join(codes)}")
            if len(pruned_leaves) > 5:
                logger.info(f"    ... and {len(pruned_leaves) - 5} more pruned paths")
        
        for pruned in pruned_leaves:
            classification_path = pruned.get("classification_path", [])
            trajectory = pruned.get("trajectory", [])
            path_id = pruned.get("path_id", "unknown")

            pred_trace = build_pred_trace_from_path(classification_path)
            reward_components = {
                "trace_prefix": compute_leaf_reward(pred_trace, gold_trace),
                "code_digits_prefix": _code_digits_prefix_reward(pred_trace, gold_code),
            }
            reward = _aggregate_leaf_reward(reward_components)

            leaves.append({
                "path_id": path_id,
                "pred_trace": pred_trace,
                "reward": reward,
                "reward_components": reward_components,
                "trajectory": trajectory,
                "classification_path": classification_path,
                "source": "pruned",
            })

        if not leaves:
            logger.warning(f"No leaves collected for ruling: {product_description[:50]}")
            return []

        step_rewards, v_root = compute_treerl_rewards(leaves)

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
# TRAINING FUNCTIONS (Transformers + PEFT phase - NO vLLM)
# ============================================================================

def _patch_nemotron_flash_attn_support(model_name: str, logger: logging.Logger) -> None:
    """
    Patch Nemotron model classes to enable Flash Attention 2 support.
    The remote modeling_nemotron_h.py doesn't declare _supports_flash_attn_2 = True,
    so we patch it at runtime before loading.
    """
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        
        for cls_name in ["NemotronHPreTrainedModel", "NemotronHForCausalLM"]:
            cls = get_class_from_dynamic_module(
                f"modeling_nemotron_h.{cls_name}",
                model_name,
                trust_remote_code=True,
            )
            if not getattr(cls, "_supports_flash_attn_2", False):
                cls._supports_flash_attn_2 = True
                cls._supports_sdpa = True
    except Exception:
        pass  # May already be supported


def load_training_model(
    config: TreeRLConfig,
    logger: logging.Logger,
    adapter_path: Optional[str] = None,
    local_rank: int = 0,
    use_ddp: bool = False,
):
    """
    Load Nemotron-3-Nano with pure Transformers + PEFT for training.
    Uses BF16 (Nemotron's native format) - 4-bit not supported due to custom model code.
    Requires Flash Attention 2 - will fail if not available.
    
    GPU modes:
    - DDP (torchrun): Each process loads full model to its GPU, gradients synchronized
    - Model Parallel (device_map="auto"): Single process, model sharded across all GPUs
    
    Args:
        local_rank: GPU index for DDP (0 if single-GPU)
        use_ddp: Whether to wrap model in DistributedDataParallel
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    train_model_name = config.train_model
    logger.info(f"Loading training model: {train_model_name}")

    # Detect GPU configuration
    num_gpus = torch.cuda.device_count()
    logger.info(f"  Available GPUs: {num_gpus}")

    # Patch Nemotron model classes to enable FA2 support (if not already enabled)
    _patch_nemotron_flash_attn_support(train_model_name, logger)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(train_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model in BF16 with Flash Attention 2
    # Note: 4-bit quantization not supported - Nemotron's custom _init_weights
    # has in-place ops incompatible with BitsAndBytes quantized tensors
    t0 = time.time()
    
    # For DDP (launched via torchrun), load to specific device
    if use_ddp:
        device = f"cuda:{local_rank}"
        logger.info(f"  🔀 DDP mode: loading full model to {device}")
        logger.info(f"     (Each process has full model copy, gradients synchronized)")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                train_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map={"": device},  # Load entire model to this device
                attn_implementation="flash_attention_2",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model with Flash Attention 2: {e}") from e
    else:
        # Model parallelism: shard across all available GPUs for larger batch sizes
        # With micro_batch_size > 1, we need extra VRAM for activations
        if num_gpus > 1:
            logger.info(f"  🔀 Model Parallel mode: sharding across {num_gpus} GPUs")
            logger.info(f"     Enables larger batch sizes by distributing memory across GPUs")
        else:
            logger.info(f"  Single GPU mode")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                train_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",  # Shard across available GPUs
                attn_implementation="flash_attention_2",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model with Flash Attention 2: {e}") from e
    
    logger.info(f"  Base model loaded in BF16 in {time.time() - t0:.1f}s")

    # Configure and apply LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    logger.info(f"  LoRA attached (rank={config.lora_rank})")

    # Ensure logits stay in model dtype (avoid fp32 [B,L,V] buffers during loss)
    # Some HF models return fp32 logits even when weights are bf16/fp16.
    target_logits_dtype = next(model.parameters()).dtype
    orig_forward = model.forward

    def _forward_cast_logits(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        try:
            if hasattr(out, "logits") and out.logits is not None and out.logits.dtype == torch.float32:
                out.logits = out.logits.to(target_logits_dtype)
        except Exception:
            pass
        return out

    model.forward = _forward_cast_logits

    # Resume from existing adapter if provided
    if adapter_path and os.path.isdir(adapter_path):
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            logger.info(f"  Loading adapter: {adapter_path}")
            try:
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file
                
                adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
                if os.path.exists(adapter_weights_path):
                    adapter_weights = load_file(adapter_weights_path)
                else:
                    adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
                    adapter_weights = torch.load(adapter_weights_path, map_location="cpu", weights_only=True)
                
                set_peft_model_state_dict(model, adapter_weights)
            except Exception as e:
                logger.warning(f"  Could not load adapter: {e}")

    # Enable training mode and gradient checkpointing
    model.train()
    
    # Enable gradient checkpointing for memory efficiency
    # use_reentrant=False is recommended for newer PyTorch and works better with QLoRA
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("  ✓ Gradient checkpointing enabled (use_reentrant=False)")
    except TypeError:
        # Older transformers version may not support kwargs
        try:
            model.gradient_checkpointing_enable()
            logger.info("  ✓ Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not enable gradient checkpointing: {e}")
    except Exception as e:
        logger.warning(f"  ⚠️ Could not enable gradient checkpointing: {e}")

    # Log trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Wrap in DDP if enabled
    if use_ddp and dist.is_initialized():
        device = f"cuda:{local_rank}"
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # Required for PEFT/LoRA
        )
        logger.info(f"  ✓ Model wrapped in DDP (device={device})")

    return model, tokenizer


def unload_training_model(model, tokenizer, logger: logging.Logger):
    logger.info("Unloading training model...")
    try:
        model.cpu()
    except Exception:
        pass
    del model
    del tokenizer
    for _ in range(5):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    free_gpu_memory(logger)
    time.sleep(3)
    logger.info("Training model unloaded, GPU memory freed")


# ----------------------------------------------------------------------------
# Token-weighted GRPO loss (unchanged)
# ----------------------------------------------------------------------------

def find_assistant_turn_boundaries(input_ids: torch.Tensor, tokenizer, messages: List[Dict[str, str]]) -> List[Tuple[int, int]]:
    """
    DEPRECATED: Use tokenize_with_assistant_boundaries instead.
    This fallback uses fragile substring matching.
    """
    boundaries = []
    input_list = input_ids.tolist()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not content:
            continue
        content_tokens = tokenizer.encode(content, add_special_tokens=False)
        if len(content_tokens) < 3:
            continue
        search_len = min(5, len(content_tokens))
        search_pattern = content_tokens[:search_len]
        for start_idx in range(len(input_list) - len(content_tokens) + 1):
            if input_list[start_idx:start_idx + search_len] == search_pattern:
                end_idx = start_idx + len(content_tokens)
                boundaries.append((start_idx, end_idx))
                break
    return boundaries


def tokenize_with_assistant_boundaries(
    messages: List[Dict[str, str]],
    tokenizer,
    max_length: int,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int]], bool]:
    """
    Tokenize messages and compute exact assistant turn boundaries.
    
    Uses incremental tokenization to guarantee correct boundaries:
    - Tokenize messages[:1], messages[:2], ..., messages[:n]
    - Boundary for message i is (len(tokens[:i]), len(tokens[:i+1]))
    
    This avoids the BPE context-dependency issue where tokenizing
    content in isolation produces different tokens than in context.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (for truncation)
        
    Returns:
        Tuple of:
        - inputs: Dict with 'input_ids' and 'attention_mask' tensors
        - boundaries: List of (start, end) token indices for assistant turns
        - truncated: Whether the sequence was truncated
    """
    boundaries = []
    prev_len = 0
    
    # Build boundaries incrementally
    for i in range(len(messages)):
        # Tokenize messages[0] through messages[i]
        partial_text = tokenizer.apply_chat_template(
            messages[:i+1], 
            tokenize=False, 
            add_generation_prompt=False
        )
        # Use encode to get token count (no tensors needed here)
        partial_ids = tokenizer.encode(partial_text, add_special_tokens=False)
        curr_len = len(partial_ids)
        
        if messages[i].get("role") == "assistant":
            content = messages[i].get("content", "")
            if content and curr_len > prev_len:
                boundaries.append((prev_len, curr_len))
        
        prev_len = curr_len
    
    # Final full tokenization with tensors
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        # IMPORTANT: apply_chat_template() already inserts special tokens.
        # Adding them again shifts indices and breaks assistant boundary alignment.
        add_special_tokens=False,
    )
    
    final_len = inputs["input_ids"].shape[1]
    truncated = (final_len == max_length)
    
    # Adjust boundaries if truncation occurred
    if truncated:
        # Filter out boundaries that are completely beyond the truncated length
        # and clip boundaries that partially extend beyond
        adjusted_boundaries = []
        for start, end in boundaries:
            if start >= final_len:
                continue  # Entire turn was truncated
            adjusted_end = min(end, final_len)
            if adjusted_end > start:
                adjusted_boundaries.append((start, adjusted_end))
        boundaries = adjusted_boundaries
    
    return inputs, boundaries, truncated


# ----------------------------------------------------------------------------
# Robust Step-to-Turn Mapping (replaces fragile regex-based approach)
# ----------------------------------------------------------------------------

def _normalize_code(code: str) -> str:
    """Strip dots and non-digit characters from an HTS code."""
    if not code:
        return ""
    return re.sub(r"\D", "", code)


def normalize_hts_to_10_digits(code: str) -> str:
    """
    Normalize HTS code to exactly 10 digits, padding with zeros.
    
    HTS codes are always 10 digits in their full form:
    - 847330 → 8473300000
    - 84733091 → 8473309100
    - 8473.30.00.00 → 8473300000
    
    This ensures consistent comparison between codes at different
    tree depths (e.g., subheading vs statistical suffix level).
    """
    digits = _normalize_code(code)
    if not digits:
        return ""
    # Pad to 10 digits (HTS standard), truncate if longer
    return digits.ljust(10, '0')[:10]


def format_hts_code(code: str) -> str:
    """
    Format HTS code as XXXX.XX.XX.XX (standard display format).
    Normalizes to 10 digits first.
    """
    digits = normalize_hts_to_10_digits(code)
    if not digits or len(digits) != 10:
        return code  # Return as-is if normalization fails
    return f"{digits[0:4]}.{digits[4:6]}.{digits[6:8]}.{digits[8:10]}"


def _extract_selected_identifier_from_turn(
    user_msg: str,
    assistant_msg: str
) -> Optional[Dict[str, Any]]:
    """
    Extract the selected code/node_id from an assistant turn by parsing JSON.
    
    Returns:
        Dict with:
        - 'kind': 'chapter' | 'code' | 'group'
        - 'code': normalized code (for chapter/code)
        - 'node_id': node ID (for groups)
        Or None if parsing fails.
    """
    try:
        assistant_json = json.loads(assistant_msg)
    except (json.JSONDecodeError, TypeError):
        return None

    # Some providers/models return a list instead of the named-dict format:
    #   [{"option_index": 1, ...}, {"option_index": 2, ...}, ...]
    # ClassificationEngine supports this via _unwrap_json_response; mirror that here
    # so step->turn mapping doesn't silently fail.
    if isinstance(assistant_json, list):
        if not assistant_json:
            return None
        first = assistant_json[0]
        if isinstance(first, dict):
            # List-of-selections format (no primary_selection key)
            if "option_index" in first and "primary_selection" not in first:
                assistant_json = {"primary_selection": first}
            else:
                # Wrapper list (e.g., Gemini-style) - take the first dict
                assistant_json = first
        else:
            return None
    
    # === SELECT_CHAPTERS: uses top_selection with chapter code directly ===
    if "top_selection" in assistant_json:
        chapter_code = assistant_json.get("top_selection", "")
        if chapter_code:
            # Chapters are always 2 digits - zero-pad if single digit
            normalized = _normalize_code(str(chapter_code))
            if len(normalized) == 1:
                normalized = normalized.zfill(2)  # "3" -> "03"
            return {
                "kind": "chapter",
                "code": normalized
            }
        return None
    
    # === RANK_CANDIDATES: uses option_index to lookup from user message ===
    if "primary_selection" in assistant_json:
        primary = assistant_json["primary_selection"]
        if not isinstance(primary, dict):
            return None
        option_index = primary.get("option_index")
        if option_index is None:
            return None
        
        try:
            option_index = int(option_index)
        except (ValueError, TypeError):
            return None
        
        # Parse user message to get classification_tree.children
        try:
            json_start = user_msg.find("JSON INPUT:")
            if json_start == -1:
                return None
            json_str = user_msg[json_start + len("JSON INPUT:"):].strip()
            user_json = json.loads(json_str)
            
            children = user_json.get("data", {}).get("classification_tree", {}).get("children", [])
            if not children or not (1 <= option_index <= len(children)):
                return None
            
            selected_child = children[option_index - 1]  # Convert 1-based to 0-based
            
            if selected_child.get("is_group"):
                # Groups use node_id as the unique identifier
                return {
                    "kind": "group",
                    "node_id": selected_child.get("node_id")
                }
            else:
                # Codes/headings use the code
                return {
                    "kind": "code",
                    "code": _normalize_code(selected_child.get("code", ""))
                }
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
    
    return None


def _match_identifier_to_trace(
    identifier: Dict[str, Any],
    pred_trace: List[Dict[str, Any]]
) -> Optional[int]:
    """
    Find the step index in pred_trace that matches the given identifier.
    
    Args:
        identifier: Dict with 'kind' and either 'code' or 'node_id'
        pred_trace: List of trace steps
        
    Returns:
        Step index (0-based) or None if not found.
    """
    if not identifier or not pred_trace:
        return None
    
    kind = identifier.get("kind")
    
    for step_idx, step in enumerate(pred_trace):
        step_kind = step.get("kind", "")
        
        if kind == "group":
            # Groups match by node_id (they have no meaningful code)
            if step_kind == "group" and step.get("node_id") == identifier.get("node_id"):
                return step_idx
        elif kind in ("chapter", "code"):
            # Chapters and codes match by normalized code
            if step_kind in ("chapter", "code"):
                if step.get("code", "") == identifier.get("code", ""):
                    return step_idx
    
    return None


def _build_assistant_step_mapping(
    messages: List[Dict[str, str]],
    pred_trace: List[Dict[str, Any]],
    debug: bool = False,
    stats: Optional[Dict[str, int]] = None,
) -> List[int]:
    """
    Build a mapping from assistant turn index to step index in pred_trace.
    
    This robustly parses each assistant's JSON response to determine what
    was selected, then matches against pred_trace using code/node_id.
    
    Args:
        messages: Full message trajectory
        pred_trace: Predicted trace for this trajectory
        debug: If True, print debug info for unmapped turns
        stats: Optional dict to accumulate mapping statistics
        
    Returns:
        List where assistant_step_map[i] = step index for i-th assistant turn
        (-1 if mapping failed for that turn)
    """
    assistant_step_map = []
    last_user_msg = ""
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            last_user_msg = content
        elif role == "assistant":
            # Detect task type from user message
            user_prefix = last_user_msg[:400].lower()
            if "select_chapters" in user_prefix:
                task_type = "select_chapters"
            elif "rank_candidates" in user_prefix:
                task_type = "rank_candidates"
            elif "generate_question" in user_prefix:
                task_type = "generate_question"
            elif "process_answer" in user_prefix:
                task_type = "process_answer"
            else:
                task_type = "other"
            
            # Try to extract what was selected in this turn
            identifier = _extract_selected_identifier_from_turn(last_user_msg, content)
            
            if identifier:
                step_idx = _match_identifier_to_trace(identifier, pred_trace)
                assistant_step_map.append(step_idx if step_idx is not None else -1)
                if stats is not None:
                    if step_idx is not None:
                        stats[f"mapped_{task_type}"] = stats.get(f"mapped_{task_type}", 0) + 1
                    else:
                        stats[f"extracted_no_match_{task_type}"] = stats.get(f"extracted_no_match_{task_type}", 0) + 1
                if debug and step_idx is None:
                    print(f"  [DEBUG] Mapped identifier but no trace match: {identifier}")
            else:
                if stats is not None:
                    stats[f"no_extract_{task_type}"] = stats.get(f"no_extract_{task_type}", 0) + 1
                if debug:
                    print(f"  [DEBUG] Unmapped turn: task={task_type}")
                assistant_step_map.append(-1)
    
    return assistant_step_map


def build_token_weights(
    step_rewards: List[Dict],
    boundaries: List[Tuple[int, int]],
    seq_len: int,
    device: str = "cuda",
    leaf_reward: Optional[float] = None,
    messages: Optional[List[Dict]] = None,
    pred_trace: Optional[List[Dict]] = None,
    assistant_step_map: Optional[List[int]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Build per-token weight tensor from step rewards.
    
    Uses robust mapping via pred_trace when available:
    1. Parse each assistant's JSON response to extract the selected code/node_id
    2. Match against pred_trace to find the corresponding step index
    3. Apply the step's R(s) reward to all tokens in that assistant turn
    
    Args:
        step_rewards: List of {step, R, ...} from TreeRL process supervision
        boundaries: List of (start, end) token indices for each assistant turn
        seq_len: Total sequence length
        device: Target device
        leaf_reward: Optional fallback reward
        messages: Optional message trajectory for mapping
        pred_trace: Optional predicted trace for robust step mapping
        dtype: Optional dtype for weights tensor
        
    Returns:
        Tensor of per-token weights [seq_len]
    """
    weights = torch.zeros(seq_len, device=device, dtype=dtype or torch.float32)
    if not boundaries:
        return weights

    step_to_R = {sr["step"]: sr["R"] for sr in step_rewards}

    if leaf_reward is not None:
        fallback_R = leaf_reward
    elif step_rewards:
        fallback_R = sum(sr.get("R", 0.0) for sr in step_rewards) / len(step_rewards)
    else:
        fallback_R = 0.0

    # No messages - use sequential fallback
    if not messages:
        for bound_idx, (start, end) in enumerate(boundaries):
            R = step_to_R.get(bound_idx, fallback_R)
            weights[start:end] = R
        return weights

    # Use robust mapping if pred_trace is available, unless caller already provided one.
    if assistant_step_map is None:
        if pred_trace:
            assistant_step_map = _build_assistant_step_mapping(messages, pred_trace)
        else:
            # Fallback to sequential mapping (less reliable but works when pred_trace unavailable)
            assistant_step_map = list(range(len([m for m in messages if m.get("role") == "assistant"])))

    for bound_idx, (start, end) in enumerate(boundaries):
        if bound_idx < len(assistant_step_map):
            step_idx = assistant_step_map[bound_idx]
            if step_idx is not None and step_idx >= 0:
                R = step_to_R.get(step_idx, fallback_R)
            else:
                R = 0.0
        else:
            R = fallback_R
        weights[start:end] = R

    return weights


def _mem_stats(tag: str, reset: bool = False) -> str:
    """Return memory stats string. If TREERL_LOG_MEMORY=1, prints; else no-op."""
    if not torch.cuda.is_available():
        return ""
    if os.environ.get("TREERL_LOG_MEMORY", "").lower() not in ("1", "true", "yes"):
        return ""
    if reset:
        torch.cuda.reset_peak_memory_stats()
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    return f"[mem:{tag}] alloc={alloc:.2f}GB peak={peak:.2f}GB"


def compute_grpo_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    step_rewards: List[Dict],
    boundaries: List[Tuple[int, int]],
    device: str = "cuda",
    leaf_reward: Optional[float] = None,
    messages: Optional[List[Dict]] = None,
    pred_trace: Optional[List[Dict]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    assert model.training
    assert torch.is_grad_enabled()

    log_mem = os.environ.get("TREERL_LOG_MEMORY", "").lower() in ("1", "true", "yes")
    _log = lambda msg: logger.info(msg) if logger else print(msg)

    if log_mem:
        _log(_mem_stats("start", reset=True))

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    if log_mem:
        _log(_mem_stats("after_forward"))

    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]
    if log_mem:
        _log(_mem_stats("after_shift"))
        _log(f"  shift_logits shape={list(shift_logits.shape)} dtype={shift_logits.dtype}")

    # Use cross_entropy with ignore_index to avoid explicit [B,L,V] log_softmax materialization.
    # CE expects [B,V,L] and target [B,L].
    ignore_index = -100
    labels = shift_labels.masked_fill(shift_mask == 0, ignore_index)
    nll = F.cross_entropy(
        shift_logits.transpose(1, 2),
        labels,
        reduction="none",
        ignore_index=ignore_index,
    )
    if log_mem:
        _log(_mem_stats("after_ce"))

    token_log_probs = -nll

    adjusted_boundaries = [(max(0, s-1), max(0, e-1)) for s, e in boundaries]
    weights = build_token_weights(
        step_rewards,
        adjusted_boundaries,
        shift_labels.shape[1],
        device,
        leaf_reward=leaf_reward,
        messages=messages,
        pred_trace=pred_trace,
        dtype=token_log_probs.dtype,
    ).unsqueeze(0)

    masked_log_probs = token_log_probs
    weighted_log_probs = masked_log_probs * weights

    num_weighted = (weights.abs() > 0).sum().float()
    if num_weighted > 0:
        loss = -weighted_log_probs.sum() / num_weighted
    else:
        # No weighted tokens -> no policy-gradient signal. Return a zero loss
        # (with a valid autograd graph) rather than falling back to full LM loss.
        loss = weighted_log_probs.sum() * 0.0

    metrics = {
        "loss": float(loss.item()),
        "avg_log_prob": float(masked_log_probs.sum().item() / max(shift_mask.sum().item(), 1)),
        "num_weighted_tokens": float(num_weighted.item()),
    }
    return loss, metrics


def _compute_leaf_advantages(samples: List[Dict[str, Any]], config: TreeRLConfig, logger: logging.Logger) -> Dict[str, float]:
    method = (config.advantage_method or "none").lower().strip()
    if method == "none":
        return {s.get("path_id", f"idx_{i}"): 1.0 for i, s in enumerate(samples)}

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        g = s.get("gold_code", "unknown")
        groups.setdefault(g, []).append(s)

    def _std(vals: List[float]) -> float:
        if not vals:
            return 0.0
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        return math.sqrt(max(var, 0.0))

    adv_by_path: Dict[str, float] = {}

    if method in ("grpo", "grpo_no_std"):
        for _, gs in groups.items():
            rs = [float(x.get("leaf_reward", x.get("reward", 0.0)) or 0.0) for x in gs]
            mu = sum(rs) / len(rs) if rs else 0.0
            sd = _std(rs)
            for x in gs:
                pid = x.get("path_id", "unknown")
                a = float(x.get("leaf_reward", x.get("reward", 0.0)) or 0.0) - mu
                if method == "grpo" and sd > 1e-6:
                    a = a / (sd + 1e-4)
                elif method == "grpo":
                    a = 0.0
                adv_by_path[pid] = a
        return adv_by_path

    if method == "gdpo":
        w = list(config.gdpo_reward_weights or ())
        if len(w) < 2:
            w = [1.0, 1.0]
        w = w[:2]

        def _ensure_components(x: Dict[str, Any]) -> Dict[str, float]:
            comps = x.get("reward_components")
            if isinstance(comps, dict) and ("trace_prefix" in comps or "code_digits_prefix" in comps):
                return {
                    "trace_prefix": float(comps.get("trace_prefix", x.get("leaf_reward", x.get("reward", 0.0)) or 0.0)),
                    "code_digits_prefix": float(comps.get("code_digits_prefix", 0.0) or 0.0),
                }
            return {
                "trace_prefix": float(x.get("leaf_reward", x.get("reward", 0.0)) or 0.0),
                "code_digits_prefix": 0.0,
            }

        pre_bn_adv: List[Tuple[str, float]] = []
        for _, gs in groups.items():
            comp0 = [float(_ensure_components(x).get("trace_prefix", 0.0)) for x in gs]
            comp1 = [float(_ensure_components(x).get("code_digits_prefix", 0.0)) for x in gs]

            mu0 = sum(comp0) / len(comp0) if comp0 else 0.0
            mu1 = sum(comp1) / len(comp1) if comp1 else 0.0
            sd0 = _std(comp0)
            sd1 = _std(comp1)

            for x in gs:
                pid = x.get("path_id", "unknown")
                comps = _ensure_components(x)
                r0 = float(comps.get("trace_prefix", 0.0))
                r1 = float(comps.get("code_digits_prefix", 0.0))

                a0 = (r0 - mu0) / (sd0 + 1e-4) if sd0 > 1e-6 else 0.0
                a1 = (r1 - mu1) / (sd1 + 1e-4) if sd1 > 1e-6 else 0.0
                pre = (w[0] * a0) + (w[1] * a1)
                pre_bn_adv.append((pid, pre))

        vals = [v for _, v in pre_bn_adv]
        mu = sum(vals) / len(vals) if vals else 0.0
        sd = _std(vals)
        for pid, v in pre_bn_adv:
            adv_by_path[pid] = (v - mu) / (sd + 1e-4) if sd > 1e-6 else 0.0

        return adv_by_path

    logger.warning(f"Unknown advantage_method='{config.advantage_method}', defaulting to none")
    return {s.get("path_id", f"idx_{i}"): 1.0 for i, s in enumerate(samples)}


def train_on_samples(
    samples: List[Dict],
    config: TreeRLConfig,
    logger: logging.Logger,
    adapter_path: Optional[str] = None,
    local_rank: int = 0,
    world_size: int = 1,
) -> Tuple[str, Dict[str, float]]:
    use_ddp = world_size > 1
    logger.info(f"Training on {len(samples)} samples...")
    logger.info(f"  Train max seq length: {config.train_max_seq_length}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    if use_ddp:
        logger.info(f"  DDP: world_size={world_size}, local_rank={local_rank}")

    model, tokenizer = load_training_model(config, logger, adapter_path, local_rank, use_ddp)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    model.train()

    adv_by_path_id = _compute_leaf_advantages(samples, config, logger)

    metrics = {
        "total_loss": 0.0,
        "num_samples": 0,  # trained samples (may be < len(samples) if we skip zero-weight samples)
        "skipped_truncated": 0,
        "skipped_error": 0,
        "skipped_zero_weight": 0,
    }

    accumulated_loss = 0.0
    accumulated_steps = 0
    optimizer.zero_grad()
    
    # Throughput tracking
    train_start_time = time.time()
    total_tokens_processed = 0
    micro_batch_size = config.train_micro_batch_size
    
    logger.info(f"  Micro batch size: {micro_batch_size} (effective batch = {micro_batch_size * config.gradient_accumulation_steps})")
    
    # Pre-tokenize all samples and filter
    # Uses incremental tokenization to compute exact assistant boundaries
    tokenized_samples = []
    mapping_stats: Dict[str, int] = {}  # Collect mapping diagnostics
    for sample_idx, sample in enumerate(samples):
        messages = sample.get("messages", [])
        if not messages:
            continue
        
        try:
            # Use robust incremental tokenization to get exact boundaries
            inputs, boundaries, truncated = tokenize_with_assistant_boundaries(
                messages, tokenizer, config.train_max_seq_length
            )
            seq_len = inputs["input_ids"].shape[1]
            
            if truncated:
                logger.debug(f"  Sample {sample_idx}: TRUNCATED at {seq_len} tokens - SKIPPING")
                metrics["skipped_truncated"] += 1
                continue
            
            path_id = sample.get("path_id", f"idx_{sample_idx}")
            pred_trace = sample.get("pred_trace", []) or []
            # Precompute assistant-turn -> pred_trace step mapping once (saves work + enables diagnostics)
            # Enable debug for first 3 samples when TREERL_DEBUG_MAPPING=1
            debug_mapping = os.environ.get("TREERL_DEBUG_MAPPING", "").lower() in ("1", "true") and sample_idx < 3
            assistant_step_map = _build_assistant_step_mapping(messages, pred_trace, debug=debug_mapping, stats=mapping_stats) if pred_trace else None
            tokenized_samples.append({
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "seq_len": seq_len,
                "messages": messages,
                "step_rewards": sample.get("step_rewards", []),
                "leaf_reward": sample.get("leaf_reward", sample.get("reward", None)),
                "pred_trace": pred_trace,
                "leaf_advantage": float(adv_by_path_id.get(path_id, 1.0)),
                "assistant_boundaries": boundaries,  # Pre-computed exact boundaries
                "assistant_step_map": assistant_step_map,
            })
        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            metrics["skipped_error"] += 1
    
    logger.info(f"  Tokenized {len(tokenized_samples)} samples (skipped {metrics['skipped_truncated']} truncated)")
    
    # Log mapping breakdown by task type (helps debug low mapping rates)
    if mapping_stats:
        # Group by outcome
        mapped_keys = [k for k in mapping_stats if k.startswith("mapped_")]
        extracted_no_match_keys = [k for k in mapping_stats if k.startswith("extracted_no_match_")]
        no_extract_keys = [k for k in mapping_stats if k.startswith("no_extract_")]
        
        total_mapped = sum(mapping_stats.get(k, 0) for k in mapped_keys)
        total_no_match = sum(mapping_stats.get(k, 0) for k in extracted_no_match_keys)
        total_no_extract = sum(mapping_stats.get(k, 0) for k in no_extract_keys)
        grand_total = total_mapped + total_no_match + total_no_extract
        
        if grand_total > 0:
            logger.info(f"  Mapping breakdown: {total_mapped} mapped, {total_no_match} extracted-but-no-trace-match, {total_no_extract} not-extractable")
            # Show per-task breakdown for selection tasks that should be extractable
            for task in ["select_chapters", "rank_candidates"]:
                m = mapping_stats.get(f"mapped_{task}", 0)
                nm = mapping_stats.get(f"extracted_no_match_{task}", 0)
                ne = mapping_stats.get(f"no_extract_{task}", 0)
                task_total = m + nm + ne
                if task_total > 0:
                    logger.info(f"    {task}: {m}/{task_total} mapped ({100*m/task_total:.0f}%), {nm} no-trace-match, {ne} not-extractable")

    # High-signal diagnostics: how much of the sequence is "assistant tokens", and how often
    # assistant turns map to a pred_trace step (needed for nonzero weights).
    try:
        total_tokens = sum(int(s.get("seq_len", 0)) for s in tokenized_samples)
        assistant_tokens = 0
        for s in tokenized_samples:
            bnds = s.get("assistant_boundaries") or []
            assistant_tokens += sum(max(0, int(en) - int(st)) for st, en in bnds)
        if total_tokens > 0:
            logger.info(f"  Assistant token fraction: {100.0 * assistant_tokens / total_tokens:.1f}%")

        mapped = 0
        total = 0
        # Also count MAPPED tokens (not just turns) to see true signal coverage
        mapped_tokens = 0
        for s in tokenized_samples:
            m = s.get("assistant_step_map")
            bnds = s.get("assistant_boundaries") or []
            if not m:
                continue
            total += len(m)
            for i, x in enumerate(m):
                if isinstance(x, int) and x >= 0:
                    mapped += 1
                    if i < len(bnds):
                        mapped_tokens += max(0, bnds[i][1] - bnds[i][0])
        if total > 0:
            logger.info(f"  Assistant turn mapping: {mapped}/{total} turns ({100.0 * mapped / total:.1f}%)")
            if assistant_tokens > 0:
                logger.info(f"  Mapped token coverage: {mapped_tokens}/{assistant_tokens} tokens ({100.0 * mapped_tokens / assistant_tokens:.1f}%)")
    except Exception:
        pass

    # Quick weight magnitude diagnostics (helps tune scaling/clipping)
    weight_diag = {
        "num_samples": 0,
        "nonzero_tokens": 0,
        "total_tokens": 0,
        "sum_abs": 0.0,
        "max_abs": 0.0,
        "clipped_tokens": 0,
        "zero_weight_samples": 0,
    }
    
    # Process in mini-batches
    num_batches = (len(tokenized_samples) + micro_batch_size - 1) // micro_batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * micro_batch_size
        batch_end = min(batch_start + micro_batch_size, len(tokenized_samples))
        batch_samples = tokenized_samples[batch_start:batch_end]
        
        if not batch_samples:
            continue
        
        try:
            # Pad batch to same length
            max_len = max(s["seq_len"] for s in batch_samples)
            batch_input_ids = []
            batch_attention_mask = []
            batch_weights = []
            kept_samples = []
            
            for s in batch_samples:
                # Use pre-computed exact boundaries (from incremental tokenization)
                boundaries = s.get("assistant_boundaries", [])
                # Adjust for shifted loss computation (predicting next token)
                adjusted_boundaries = [(max(0, st - 1), max(0, en - 1)) for st, en in boundaries]
                
                weights = build_token_weights(
                    s["step_rewards"],
                    adjusted_boundaries,
                    s["seq_len"] - 1,  # Shifted for loss
                    "cpu",
                    leaf_reward=s["leaf_reward"],
                    messages=s["messages"],
                    pred_trace=s["pred_trace"],
                    assistant_step_map=s.get("assistant_step_map"),
                    dtype=torch.float32,
                )
                # Apply scaling + advantage
                # Step 1: Scale step rewards (process supervision signal)
                step_scale = float(getattr(config, "step_reward_scale", 1.0))
                if step_scale != 1.0:
                    weights = weights * step_scale
                
                # Step 2: Apply advantage as a multiplicative modifier (1 + advantage * scale)
                # This ensures step rewards are PRESERVED even when advantage=0 (all trajectories same quality)
                # - advantage > 0: amplify weights (reinforce better-than-average trajectories)
                # - advantage < 0: reduce weights (penalize worse-than-average trajectories)
                # - advantage = 0: keep original step rewards (don't zero out!)
                if config.advantage_method and config.advantage_method.lower() != "none":
                    adv = float(s["leaf_advantage"])
                    adv_scale = float(getattr(config, "leaf_advantage_scale", 1.0))
                    # Multiplicative factor: (1 + advantage * scale)
                    # With typical normalized advantages in [-2, 2] and scale=1.0, this gives [−1, 3]
                    # Clip the factor to stay positive (minimum 0.1 to preserve some signal)
                    adv_factor = max(0.1, 1.0 + adv * adv_scale)
                    weights = weights * adv_factor
                # Optional clip on final weights
                clip = float(getattr(config, "token_weight_clip", 0.0) or 0.0)
                if clip > 0:
                    weights = torch.clamp(weights, -clip, clip)

                # Diagnostics
                nz = int((weights.abs() > 0).sum().item())
                weight_diag["num_samples"] += 1
                weight_diag["nonzero_tokens"] += nz
                weight_diag["total_tokens"] += int(weights.numel())
                abs_sum = float(weights.abs().sum().item())
                weight_diag["sum_abs"] += abs_sum
                weight_diag["max_abs"] = max(weight_diag["max_abs"], float(weights.abs().max().item()) if weights.numel() else 0.0)
                if clip > 0:
                    clipped = int((weights.abs() >= (clip - 1e-6)).sum().item())
                    weight_diag["clipped_tokens"] += clipped

                # If ALL weights are zero for this sample, skip it.
                # Training on full-sequence LM loss here would be incorrect (it would train on user tokens).
                if nz == 0:
                    weight_diag["zero_weight_samples"] += 1
                    metrics["skipped_zero_weight"] += 1
                    continue

                # Pad input_ids and attention_mask (only for kept samples)
                pad_len = max_len - s["seq_len"]
                padded_ids = F.pad(s["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
                padded_mask = F.pad(s["attention_mask"], (0, pad_len), value=0)
                batch_input_ids.append(padded_ids)
                batch_attention_mask.append(padded_mask)
                
                # Pad weights
                weights = F.pad(weights, (0, max_len - 1 - len(weights)), value=0.0)
                batch_weights.append(weights)
                kept_samples.append(s)

            if not kept_samples:
                continue
            
            # Stack into batch tensors
            batch_input_ids = torch.cat(batch_input_ids, dim=0).to(config.device)
            batch_attention_mask = torch.cat(batch_attention_mask, dim=0).to(config.device)
            batch_weights = torch.stack(batch_weights, dim=0).to(config.device)
            
            # Forward pass
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, return_dict=True)
            logits = outputs.logits
            
            # Get the device where logits are (may differ from input device with device_map="auto")
            compute_device = logits.device
            
            # Compute loss - move everything to logits device
            shift_logits = logits[:, :-1, :]
            shift_labels = batch_input_ids[:, 1:].to(compute_device)
            shift_mask = batch_attention_mask[:, 1:].to(compute_device)
            
            # Cross-entropy per token
            ignore_index = -100
            labels = shift_labels.masked_fill(shift_mask == 0, ignore_index)
            nll = F.cross_entropy(
                shift_logits.transpose(1, 2),
                labels,
                reduction="none",
                ignore_index=ignore_index,
            )
            token_log_probs = -nll
            
            # Apply weights and compute mean loss (move weights to compute device)
            batch_weights_device = batch_weights.to(device=compute_device, dtype=token_log_probs.dtype)
            weighted_log_probs = token_log_probs * batch_weights_device
            
            # Normalize by WEIGHT MASS (sum of |weights|) not count of nonzero tokens
            # This ensures loss scale is invariant to:
            # - sequence length (longer sequences don't dilute gradients)
            # - number of weighted tokens (more supervision = proportionally more gradient)
            weight_mass = batch_weights_device.abs().sum()
            
            if weight_mass <= 1e-8:
                # No meaningful weights - skip (shouldn't happen with new advantage logic)
                metrics["skipped_zero_weight"] += len(kept_samples)
                del batch_input_ids, batch_attention_mask, batch_weights, outputs, logits
                continue
            loss = -weighted_log_probs.sum() / weight_mass
            
            scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()
            
            batch_tokens = sum(s["seq_len"] for s in kept_samples)
            accumulated_loss += float(loss.item())
            accumulated_steps += 1
            total_tokens_processed += batch_tokens
            
            metrics["total_loss"] += float(loss.item()) * len(kept_samples)
            metrics["num_samples"] += len(kept_samples)
            
            del batch_input_ids, batch_attention_mask, batch_weights, outputs, logits, loss, scaled_loss
            
            # Throughput logging
            if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
                elapsed = time.time() - train_start_time
                samples_done = metrics["num_samples"]
                samples_per_sec = samples_done / elapsed if elapsed > 0 else 0.0
                tokens_per_sec = total_tokens_processed / elapsed
                logger.info(f"    [batch {batch_idx+1}/{num_batches}] {samples_per_sec:.2f} samples/s, {tokens_per_sec:.0f} tok/s")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM on batch {batch_idx} (max_len={max_len}, batch_size={len(batch_samples)}): {e}")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            accumulated_loss = 0.0
            accumulated_steps = 0
            metrics["skipped_error"] += len(batch_samples)
            continue
        
        except Exception as e:
            import traceback
            logger.error(f"Batch loss computation error: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            metrics["skipped_error"] += len(batch_samples)
            continue
        
        # Optimizer step every grad_accum batches
        if accumulated_steps >= config.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
            avg_acc_loss = accumulated_loss / accumulated_steps
            logger.info(f"  Optimizer step: loss={avg_acc_loss:.4f}")
            
            accumulated_loss = 0.0
            accumulated_steps = 0
            torch.cuda.empty_cache()

    # Log weight magnitude diagnostics after processing all batches
    if weight_diag["num_samples"] > 0 and weight_diag["total_tokens"] > 0:
        nz_frac = weight_diag["nonzero_tokens"] / max(weight_diag["total_tokens"], 1)
        mean_abs = weight_diag["sum_abs"] / max(weight_diag["total_tokens"], 1)
        mean_abs_nz = weight_diag["sum_abs"] / max(weight_diag["nonzero_tokens"], 1)
        clip = float(getattr(config, "token_weight_clip", 0.0) or 0.0)
        clipped_frac_nz = (weight_diag["clipped_tokens"] / max(weight_diag["nonzero_tokens"], 1)) if clip > 0 else 0.0
        # Total weight mass (used for loss normalization)
        total_weight_mass = weight_diag["sum_abs"]
        logger.info(
            f"  Token-weight stats: nonzero={100*nz_frac:.1f}% | weight_mass={total_weight_mass:.1f} | "
            f"mean|w|={mean_abs:.4f} | mean|w|_nz={mean_abs_nz:.3f} | max|w|={weight_diag['max_abs']:.3f} | "
            f"clipped_nz={100*clipped_frac_nz:.1f}% | zero_weight_samples={int(weight_diag.get('zero_weight_samples', 0))} "
            f"(step_scale={getattr(config, 'step_reward_scale', 1.0)}, adv_scale={getattr(config, 'leaf_advantage_scale', 1.0)}, clip={clip})"
        )

    if accumulated_steps > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    # Synchronize before saving in DDP
    if use_ddp:
        dist.barrier()

    timestamp = int(time.time())
    new_adapter_path = os.path.join(config.adapter_sync_dir, f"adapter_{timestamp}")

    # Only save on main process (rank 0)
    if is_main_process(local_rank):
        os.makedirs(new_adapter_path, exist_ok=True)
        # Unwrap DDP model for saving
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(new_adapter_path)
        tokenizer.save_pretrained(new_adapter_path)
        logger.info(f"LoRA adapter saved to: {new_adapter_path}")

    # Wait for save to complete before proceeding
    if use_ddp:
        dist.barrier()

    metrics["avg_loss"] = metrics["total_loss"] / metrics["num_samples"] if metrics["num_samples"] else 0.0
    logger.info(
        f"  Training complete: trained {metrics['num_samples']}/{len(samples)} samples "
        f"(skipped_zero_weight={metrics['skipped_zero_weight']}, "
        f"skipped_truncated={metrics['skipped_truncated']}, skipped_error={metrics['skipped_error']})"
    )

    unload_training_model(model, tokenizer, logger)
    return new_adapter_path, metrics


# ============================================================================
# ACCURACY METRICS
# ============================================================================

def compute_batch_accuracy(samples: List[Dict], logger: logging.Logger) -> Dict[str, float]:
    if not samples:
        return {
            "exact_match_rate": 0.0,
            "avg_best_reward": 0.0,
            "avg_reward": 0.0,
            "num_rulings": 0,
            "num_exact_matches": 0,
        }

    by_ruling: Dict[str, List[Dict]] = {}
    for s in samples:
        gold = s.get("gold_code", "unknown")
        by_ruling.setdefault(gold, []).append(s)

    num_rulings = len(by_ruling)
    num_exact_matches = 0
    best_rewards = []
    all_rewards = []

    for _, ruling_samples in by_ruling.items():
        rewards = [float(s.get("leaf_reward", s.get("reward", 0.0)) or 0.0) for s in ruling_samples]
        all_rewards.extend(rewards)
        best_reward = max(rewards) if rewards else 0.0
        best_rewards.append(best_reward)
        if best_reward >= 0.9999:
            num_exact_matches += 1

    exact_match_rate = num_exact_matches / num_rulings if num_rulings else 0.0
    avg_best_reward = sum(best_rewards) / len(best_rewards) if best_rewards else 0.0
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    return {
        "exact_match_rate": exact_match_rate,
        "avg_best_reward": avg_best_reward,
        "avg_reward": avg_reward,
        "num_rulings": num_rulings,
        "num_exact_matches": num_exact_matches,
    }


# ============================================================================
# BENCHMARK EVALUATION
# ============================================================================

def run_benchmark_evaluation(
    benchmark_rulings: List[Dict],
    config: TreeRLConfig,
    logger: logging.Logger,
    vllm_client: "VLLMInferenceClient",
    global_batch_num: int,
) -> Dict[str, float]:
    """
    Run evaluation on a held-out benchmark set for cleaner accuracy signal.
    Uses parallel rollouts for efficiency.
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"📋 BENCHMARK EVALUATION (batch {global_batch_num})")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Evaluating {len(benchmark_rulings)} held-out rulings...")
    logger.info(f"  Parallel workers: {config.parallel_rollouts}")

    all_benchmark_samples = []
    completed_rulings = []  # Track (gold_code, best_reward) for each ruling
    errors_count = 0
    start_time = time.time()
    last_progress_time = start_time

    def run_single_rollout(ruling_idx_ruling):
        """Worker function for parallel rollout execution."""
        ruling_idx, ruling = ruling_idx_ruling
        gold_code = ruling.get("hts_code", "unknown")
        product_desc = ruling.get("short_product_description", "")[:40]
        try:
            samples = run_online_rollout(ruling, config, logger, vllm_client)
            return ruling_idx, gold_code, product_desc, samples, None
        except Exception as e:
            return ruling_idx, gold_code, product_desc, [], str(e)

    # Process benchmark rulings in parallel
    with ThreadPoolExecutor(max_workers=config.parallel_rollouts) as executor:
        futures = {
            executor.submit(run_single_rollout, (idx, ruling)): idx
            for idx, ruling in enumerate(benchmark_rulings)
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            ruling_idx, gold_code, product_desc, samples, error = future.result()

            if error:
                errors_count += 1
                logger.debug(f"  Benchmark [{completed}/{len(benchmark_rulings)}]: {gold_code} ❌ {error}")
            elif samples:
                all_benchmark_samples.extend(samples)
                # Track best reward for this ruling
                best_reward = max((s.get("leaf_reward", 0) or 0 for s in samples), default=0)
                completed_rulings.append((gold_code, best_reward))
            
            # Progress update every 15 seconds
            current_time = time.time()
            if current_time - last_progress_time >= 15:
                elapsed = current_time - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(benchmark_rulings) - completed) / rate if rate > 0 else 0
                
                # Calculate running stats
                if completed_rulings:
                    running_exact = sum(1 for _, r in completed_rulings if r >= 0.9999)
                    running_avg_best = sum(r for _, r in completed_rulings) / len(completed_rulings)
                    logger.info(f"  ⏳ Progress: {completed}/{len(benchmark_rulings)} ({100*completed/len(benchmark_rulings):.0f}%) | "
                               f"Exact: {running_exact}/{len(completed_rulings)} ({100*running_exact/len(completed_rulings):.0f}%) | "
                               f"Avg best: {running_avg_best:.3f} | "
                               f"ETA: {eta:.0f}s")
                else:
                    logger.info(f"  ⏳ Progress: {completed}/{len(benchmark_rulings)} ({100*completed/len(benchmark_rulings):.0f}%) | ETA: {eta:.0f}s")
                last_progress_time = current_time

    total_time = time.time() - start_time
    
    # Compute benchmark accuracy
    benchmark_metrics = compute_batch_accuracy(all_benchmark_samples, logger)

    logger.info(f"\n📊 BENCHMARK RESULTS (batch {global_batch_num}):")
    logger.info(f"  {'─' * 50}")
    logger.info(f"  Rulings evaluated: {benchmark_metrics['num_rulings']} ({errors_count} errors)")
    logger.info(f"  Total samples: {len(all_benchmark_samples)}")
    logger.info(f"  Time: {total_time:.1f}s ({total_time/max(len(benchmark_rulings),1):.1f}s/ruling)")
    logger.info(f"  {'─' * 50}")
    logger.info(f"  🎯 Exact match rate: {benchmark_metrics['exact_match_rate']:.1%} ({benchmark_metrics['num_exact_matches']}/{benchmark_metrics['num_rulings']})")
    logger.info(f"  📈 Avg BEST reward:  {benchmark_metrics['avg_best_reward']:.4f}")
    logger.info(f"  📊 Avg reward:       {benchmark_metrics['avg_reward']:.4f}")
    
    # Show reward distribution
    if completed_rulings:
        perfect = sum(1 for _, r in completed_rulings if r >= 0.9999)
        partial = sum(1 for _, r in completed_rulings if 0 < r < 0.9999)
        zero = sum(1 for _, r in completed_rulings if r == 0)
        logger.info(f"  Distribution: ✅ {perfect} perfect | 🟡 {partial} partial | ❌ {zero} zero")
    
    logger.info(f"{'=' * 70}\n")

    return benchmark_metrics


# ============================================================================
# WANDB
# ============================================================================

def init_wandb(config: TreeRLConfig, logger: logging.Logger) -> Optional[Any]:
    if not config.use_wandb:
        return None
    try:
        import wandb
        run_name = config.wandb_run_name or f"treerl-ch{config.chapter}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity if config.wandb_entity else None,
            name=run_name,
            config=vars(config),
        )
        logger.info(f"✓ Wandb initialized: {run_name}")
        return wandb
    except ImportError:
        logger.warning("wandb not installed. Run: pip install wandb")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return None


def log_to_wandb(wandb_run, metrics: Dict[str, float], step: int, prefix: str = ""):
    if wandb_run is None:
        return
    try:
        log_dict = {(f"{prefix}/{k}" if prefix else k): v for k, v in metrics.items()}
        wandb_run.log(log_dict, step=step)
    except Exception:
        pass


# ============================================================================
# MAIN TRAINING LOOP (merge -> vLLM -> rollout -> stop -> train)
# ============================================================================

def train(config: TreeRLConfig):
    logger = setup_logging(config)

    # Initialize distributed training if enabled
    world_size, rank, local_rank = init_distributed(config, logger)
    use_ddp = world_size > 1

    logger.info("=" * 70)
    logger.info("TREERL GRPO TRAINING with Nemotron-3-Nano-30B-A3B")
    logger.info("=" * 70)
    logger.info(f"Train model: {config.train_model}")
    logger.info(f"Inference dtype: {config.inference_dtype}")
    logger.info(f"Thinking: enabled by default (reasoning parser configured)")
    
    # GPU configuration summary
    num_gpus = torch.cuda.device_count()
    logger.info(f"\n📊 GPU CONFIGURATION:")
    logger.info(f"  Available GPUs: {num_gpus}")
    if use_ddp:
        logger.info(f"  Training mode: DDP (world_size={world_size}, rank={rank}, local_rank={local_rank})")
        logger.info(f"    → Each process has full model copy, gradients synchronized")
    elif num_gpus > 1:
        logger.info(f"  Training mode: Model Parallel (device_map='auto')")
        logger.info(f"    → Single process, model sharded across {num_gpus} GPUs")
        logger.info(f"    → Both GPUs will show activity during training")
        logger.info(f"    → For true DDP: torchrun --nproc_per_node={num_gpus} nemotron_train.py --use-ddp ...")
    else:
        logger.info(f"  Training mode: Single GPU")
    logger.info(f"  vLLM inference: TP={config.vllm_tensor_parallel_size} (uses {config.vllm_tensor_parallel_size} GPUs)")
    
    logger.info(f"\nvLLM max len: {config.vllm_max_model_len}")
    logger.info(f"vLLM max_num_seqs: {config.vllm_max_num_seqs}")
    logger.info(f"vLLM served name: {config.vllm_served_model_name}")
    logger.info(f"vLLM reasoning parser: {config.vllm_use_reasoning_parser}")
    logger.info(f"Parallel rollouts: {config.parallel_rollouts}")
    logger.info(f"Benchmark every N batches: {config.benchmark_every_n_batches} (0=disabled)")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.adapter_sync_dir, exist_ok=True)
    os.makedirs(config.merged_models_dir, exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)

    wandb_run = init_wandb(config, logger)

    rulings = load_chapter_rulings(config, logger)
    if not rulings:
        logger.error(f"No rulings found for chapter {config.chapter}")
        return

    # Pre-filter rulings to only those with COMPLETE gold traces in HTS tree
    # This is done early so we know the actual pool size for batch calculations
    logger.info(f"\n🔍 Pre-filtering rulings for valid gold traces...")
    valid_rulings = filter_rulings_with_valid_gold_trace(rulings, logger)
    if not valid_rulings:
        logger.error("No rulings with valid gold traces found!")
        return

    if config.train_all:
        num_batches_per_epoch = math.ceil(len(valid_rulings) / config.rulings_per_batch)
    else:
        num_batches_per_epoch = config.num_batches
    
    logger.info(f"📊 Training configuration:")
    logger.info(f"   Valid rulings: {len(valid_rulings)}")
    logger.info(f"   Batches per epoch: {num_batches_per_epoch}")
    logger.info(f"   Rulings per batch: {config.rulings_per_batch}")

    # Initialize adapter path - handle HuggingFace adapter download
    current_adapter_path = None
    if config.sft_adapter:
        if os.path.isdir(config.sft_adapter):
            # Local path
            current_adapter_path = config.sft_adapter
            logger.info(f"📦 Using local SFT adapter: {current_adapter_path}")
        elif "/" in config.sft_adapter and not config.sft_adapter.startswith("/"):
            # HuggingFace repo ID (e.g., "orlandowhite/nemotron_sft")
            logger.info(f"📦 Downloading SFT adapter from HuggingFace: {config.sft_adapter}")
            try:
                from huggingface_hub import snapshot_download

                # Download to a local cache dir
                hf_cache_dir = os.path.join(config.adapter_sync_dir, "hf_sft_adapter")
                os.makedirs(hf_cache_dir, exist_ok=True)

                current_adapter_path = snapshot_download(
                    repo_id=config.sft_adapter,
                    local_dir=hf_cache_dir,
                    local_dir_use_symlinks=False,
                )
                logger.info(f"  ✓ Downloaded to: {current_adapter_path}")

                # Verify it's a valid adapter
                adapter_config = os.path.join(current_adapter_path, "adapter_config.json")
                if os.path.exists(adapter_config):
                    logger.info(f"  ✓ Valid LoRA adapter found")
                else:
                    logger.warning(f"  ⚠️ No adapter_config.json found, may not be a LoRA adapter")

            except Exception as e:
                logger.error(f"❌ Failed to download HuggingFace adapter: {e}")
                logger.warning(f"  Continuing without SFT adapter...")
                current_adapter_path = None
        else:
            logger.warning(f"⚠️ SFT adapter path not found: {config.sft_adapter}")
            current_adapter_path = None

    vllm_manager = VLLMServerManager(config, logger)

    # Prepare benchmark held-out set (if enabled)
    benchmark_rulings = []
    if config.benchmark_every_n_batches > 0 and config.benchmark_num_rulings > 0:
        # Ensure we have enough valid rulings for benchmark (use valid_rulings, not raw rulings!)
        if len(valid_rulings) < config.benchmark_num_rulings:
            logger.error(f"❌ Not enough valid rulings for benchmark!")
            logger.error(f"   Required: {config.benchmark_num_rulings}, Available: {len(valid_rulings)}")
            logger.error(f"   Either increase valid rulings or reduce benchmark_num_rulings")
            raise ValueError(f"Insufficient rulings for benchmark: need {config.benchmark_num_rulings}, have {len(valid_rulings)}")
        
        # Use a fixed random seed for reproducible benchmark set
        # IMPORTANT: Sample from valid_rulings (pre-filtered) not rulings (raw)
        benchmark_rng = random.Random(42)
        benchmark_rulings = benchmark_rng.sample(valid_rulings, config.benchmark_num_rulings)
        
        logger.info(f"\n📋 Benchmark set: {len(benchmark_rulings)} held-out rulings (fixed seed=42)")
        logger.info(f"   Will evaluate every {config.benchmark_every_n_batches} batches")

    training_start = time.time()
    all_metrics = []
    global_batch_num = config.start_batch  # Start from specified batch (for resuming)

    if config.start_batch > 0:
        logger.info(f"\n⏭️  RESUMING from batch {config.start_batch}")
        logger.info(f"   Skipping first {config.start_batch} batches")

    # =============================================
    # INITIAL BASELINE BENCHMARK (before any training)
    # =============================================
    if benchmark_rulings and config.benchmark_every_n_batches > 0 and not config.skip_initial_benchmark:
        logger.info(f"\n{'=' * 70}")
        logger.info("📊 BASELINE BENCHMARK (before RL training)")
        logger.info(f"{'=' * 70}")

        # For baseline, we need to start vLLM with the initial model/adapter
        base_for_serve = config.base_model_fp8 if config.inference_dtype.lower() == "fp8" else config.base_model_bf16

        if config.use_vllm_lora:
            # vLLM LoRA mode: serve base model with LoRA adapter
            baseline_model = base_for_serve
            baseline_lora = current_adapter_path  # May be None if no adapter yet
            logger.info(f"  Using vLLM LoRA mode: base={baseline_model}, lora={baseline_lora}")
        else:
            # Legacy merge mode
            baseline_lora = None
            if current_adapter_path:
                # Merge adapter for baseline
                merged_dir = os.path.join(config.merged_models_dir, "merged_baseline")
                _safe_rmtree(merged_dir)
                baseline_model = merge_lora_into_bf16_base(
                    base_model_id=config.base_model_bf16,
                    adapter_path=current_adapter_path,
                    merged_out_dir=merged_dir,
                    logger=logger,
                )
            else:
                baseline_model = base_for_serve

        if vllm_manager.start_server(model_to_serve=baseline_model, lora_adapter_path=baseline_lora):
            baseline_client = VLLMInferenceClient(config, logger, vllm_manager)

            baseline_metrics = run_benchmark_evaluation(
                benchmark_rulings,
                config,
                logger,
                baseline_client,
                global_batch_num=0,
            )

            # Log baseline to wandb
            log_to_wandb(wandb_run, {
                "exact_match_rate": baseline_metrics['exact_match_rate'],
                "num_exact_matches": baseline_metrics['num_exact_matches'],
                "avg_best_reward": baseline_metrics['avg_best_reward'],
                "avg_reward": baseline_metrics['avg_reward'],
                "num_rulings": baseline_metrics['num_rulings'],
            }, step=0, prefix="benchmark")

            # Stop vLLM to free memory
            vllm_manager.stop_server()
        else:
            logger.error("Failed to start vLLM for baseline benchmark!")
    elif benchmark_rulings and config.skip_initial_benchmark:
        logger.info(f"\n⏭️  Skipping initial baseline benchmark (--skip-initial-benchmark)")

    accuracy_rolling_samples = []

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        epoch_samples_total = 0
        epoch_exact_matches = 0
        epoch_rulings_total = 0

        if config.train_all:
            epoch_rulings_order = valid_rulings.copy()
            random.shuffle(epoch_rulings_order)

        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'=' * 70}")

        for batch_num in range(num_batches_per_epoch):
            global_batch_num += 1

            # Skip batches if resuming from a later position
            if global_batch_num <= config.start_batch:
                if global_batch_num == config.start_batch:
                    logger.info(f"⏭️  Skipped to batch {config.start_batch}, starting training...")
                continue

            batch_start = time.time()

            logger.info(f"\n{'─' * 50}")
            logger.info(f"BATCH {batch_num + 1}/{num_batches_per_epoch} (Global: {global_batch_num})")
            logger.info(f"{'─' * 50}")

            all_batch_samples = []
            batch_rulings = []

            if config.load_rollouts:
                logger.info(f"\n--- Loading cached rollouts (skipping vLLM) ---")
                all_batch_samples = load_rollouts_from_file(config.load_rollouts, logger)
                if not all_batch_samples:
                    logger.error("Failed to load rollouts from file!")
                    return
            else:
                if config.train_all:
                    start_idx = batch_num * config.rulings_per_batch
                    end_idx = min(start_idx + config.rulings_per_batch, len(epoch_rulings_order))
                    batch_rulings = epoch_rulings_order[start_idx:end_idx]
                    if not batch_rulings:
                        continue
                else:
                    batch_rulings = random.sample(valid_rulings, min(config.rulings_per_batch, len(valid_rulings)))

                # =========================================================
                # PHASE 1: PREPARE MODEL TO SERVE
                # =========================================================
                logger.info(f"\n--- Phase 1: Preparing model for vLLM ---")

                # Choose base model for serve
                base_for_serve = config.base_model_fp8 if config.inference_dtype.lower() == "fp8" else config.base_model_bf16

                if config.use_vllm_lora:
                    # vLLM LoRA mode (PR #30802): serve base model with LoRA adapter
                    model_to_serve = base_for_serve
                    lora_to_serve = current_adapter_path  # May be None if no adapter yet
                    if current_adapter_path:
                        logger.info(f"  vLLM LoRA mode: base={model_to_serve}")
                        logger.info(f"  LoRA adapter: {current_adapter_path}")
                    else:
                        logger.info(f"  No adapter yet; serving base model: {model_to_serve}")
                else:
                    # Legacy merge mode
                    lora_to_serve = None
                    base_for_merge = config.base_model_bf16  # always merge into BF16

                    if current_adapter_path:
                        logger.info(f"  Current adapter: {current_adapter_path}")

                        # Create unique merged dir per batch
                        merged_dir = os.path.join(config.merged_models_dir, f"merged_e{epoch+1}_b{batch_num+1}")
                        _safe_rmtree(merged_dir)

                        # Merge adapter into BF16 base (recommended)
                        merged_model_dir = merge_lora_into_bf16_base(
                            base_model_id=base_for_merge,
                            adapter_path=current_adapter_path,
                            merged_out_dir=merged_dir,
                            logger=logger,
                        )

                        # Serve merged BF16 dir (most robust)
                        model_to_serve = merged_model_dir
                        logger.info(f"  Serving merged checkpoint dir: {model_to_serve}")
                    else:
                        model_to_serve = base_for_serve
                        logger.info(f"  No adapter yet; serving base model: {model_to_serve}")

                # =========================================================
                # PHASE 2: START vLLM
                # =========================================================
                logger.info(f"\n--- Phase 2: Starting vLLM server ---")
                lora_path_for_server = lora_to_serve if config.use_vllm_lora else None
                if not vllm_manager.start_server(model_to_serve=model_to_serve, lora_adapter_path=lora_path_for_server):
                    logger.error("Failed to start vLLM server!")
                    return

                vllm_client = VLLMInferenceClient(config, logger, vllm_manager)

                # =========================================================
                # PHASE 3: RUN ROLLOUTS (PARALLEL)
                # =========================================================
                logger.info(f"\n--- Phase 3: Running rollouts for {len(batch_rulings)} rulings ---")
                logger.info(f"  Parallel workers: {config.parallel_rollouts}")

                def run_single_rollout(ruling_idx_ruling):
                    """Worker function for parallel rollout execution."""
                    ruling_idx, ruling = ruling_idx_ruling
                    product_desc = ruling.get("short_product_description", "")[:50]
                    try:
                        samples = run_online_rollout(ruling, config, logger, vllm_client)
                        return ruling_idx, product_desc, samples, None
                    except Exception as e:
                        return ruling_idx, product_desc, [], str(e)

                # Process rulings in parallel using ThreadPoolExecutor
                # vLLM handles request batching internally for optimal GPU utilization
                with ThreadPoolExecutor(max_workers=config.parallel_rollouts) as executor:
                    futures = {
                        executor.submit(run_single_rollout, (idx, ruling)): idx
                        for idx, ruling in enumerate(batch_rulings)
                    }

                    completed = 0
                    for future in as_completed(futures):
                        completed += 1
                        ruling_idx, product_desc, samples, error = future.result()

                        if error:
                            logger.warning(f"  [{completed}/{len(batch_rulings)}] Ruling {ruling_idx+1}: {product_desc}... ❌ Error: {error}")
                        elif samples:
                            all_batch_samples.extend(samples)
                            logger.info(f"  [{completed}/{len(batch_rulings)}] Ruling {ruling_idx+1}: {product_desc}... ✓ {len(samples)} samples (total: {len(all_batch_samples)})")
                        else:
                            logger.warning(f"  [{completed}/{len(batch_rulings)}] Ruling {ruling_idx+1}: {product_desc}... ⚠️ No samples")

                # =========================================================
                # PHASE 4: STOP vLLM
                # =========================================================
                logger.info(f"\n--- Phase 4: Stopping vLLM server ---")
                vllm_manager.stop_server()

                if config.save_rollouts and all_batch_samples:
                    base, ext = os.path.splitext(config.save_rollouts)
                    rollout_file = f"{base}_e{epoch + 1}_b{batch_num + 1}{ext}"
                    save_rollouts_to_file(
                        all_batch_samples,
                        rollout_file,
                        logger,
                        metadata={
                            "epoch": epoch + 1,
                            "batch": batch_num + 1,
                            "global_batch": global_batch_num,
                            "num_rulings": len(batch_rulings),
                            "chapter": config.chapter,
                            "beam_size": config.beam_size,
                            "served_model_name": config.vllm_served_model_name,
                        },
                    )

            # =========================================================
            # ACCURACY METRICS (Rolling)
            # =========================================================
            for s in all_batch_samples:
                accuracy_rolling_samples.append(s)

            # Keep last N unique rulings
            unique_gold_codes = []
            for s in accuracy_rolling_samples:
                g = s.get("gold_code")
                if g and g not in unique_gold_codes:
                    unique_gold_codes.append(g)
            if len(unique_gold_codes) > config.accuracy_window_size:
                keep = set(unique_gold_codes[-config.accuracy_window_size:])
                accuracy_rolling_samples = [s for s in accuracy_rolling_samples if s.get("gold_code") in keep]

            accuracy_metrics = compute_batch_accuracy(accuracy_rolling_samples, logger)
            logger.info(f"\n📊 ROLLING ACCURACY (window={config.accuracy_window_size} rulings):")
            logger.info(f"  Exact match rate: {accuracy_metrics['exact_match_rate']:.1%} ({accuracy_metrics['num_exact_matches']}/{accuracy_metrics['num_rulings']})")
            logger.info(f"  Avg best reward: {accuracy_metrics['avg_best_reward']:.4f}")
            logger.info(f"  Avg reward: {accuracy_metrics['avg_reward']:.4f}")

            current_batch_accuracy = compute_batch_accuracy(all_batch_samples, logger)
            epoch_samples_total += len(all_batch_samples)
            epoch_exact_matches += current_batch_accuracy["num_exact_matches"]
            epoch_rulings_total += current_batch_accuracy["num_rulings"]

            if all_batch_samples:
                save_samples_for_debug(all_batch_samples, config, logger, epoch=epoch + 1, ruling_desc=f"e{epoch+1}_b{batch_num+1}")
                # Save detailed mapping debug for manual inspection
                save_mapping_debug(all_batch_samples, config, logger, batch_num=batch_num + 1)
                display_rollout_stats(all_batch_samples, logger)

            # =========================================================
            # PHASE 5: TRAIN WITH TRANSFORMERS + PEFT
            # =========================================================
            if not all_batch_samples:
                logger.warning(f"No samples collected for batch {batch_num + 1}, skipping training")
                continue

            logger.info(f"\n--- Phase 5: Training on {len(all_batch_samples)} samples ---")
            new_adapter_path, train_metrics = train_on_samples(
                all_batch_samples,
                config,
                logger,
                adapter_path=current_adapter_path,
                local_rank=local_rank,
                world_size=world_size,
            )

            # Ensure GPU memory is released
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            time.sleep(3)
            free_gpu_memory(logger)

            # Update adapter
            previous_adapter_path = current_adapter_path
            current_adapter_path = new_adapter_path

            avg_loss = float(train_metrics.get("avg_loss", 0.0))
            logger.info(f"\n--- Phase 6: Model updated ---")
            logger.info(f"  📤 Previous adapter: {previous_adapter_path or 'None'}")
            logger.info(f"  📥 New LoRA adapter: {new_adapter_path}")

            batch_metrics = {
                "epoch": epoch + 1,
                "batch": batch_num + 1,
                "global_batch": global_batch_num,
                "num_rulings": len(batch_rulings),
                "num_samples": int(train_metrics.get("num_samples", 0)),
                "avg_loss": avg_loss,
                "exact_match_rate": float(accuracy_metrics["exact_match_rate"]),
                "num_exact_matches": int(accuracy_metrics["num_exact_matches"]),
                "avg_best_reward": float(accuracy_metrics["avg_best_reward"]),
                "avg_reward": float(accuracy_metrics["avg_reward"]),
                "adapter_path": new_adapter_path,
                "skipped_truncated": int(train_metrics.get("skipped_truncated", 0)),
                "skipped_error": int(train_metrics.get("skipped_error", 0)),
            }
            all_metrics.append(batch_metrics)

            log_to_wandb(wandb_run, {
                "loss": avg_loss,
                "exact_match_rate": float(accuracy_metrics["exact_match_rate"]),
                "avg_best_reward": float(accuracy_metrics["avg_best_reward"]),
                "avg_reward": float(accuracy_metrics["avg_reward"]),
                "num_samples": int(train_metrics.get("num_samples", 0)),
            }, step=global_batch_num, prefix="batch")

            # =========================================================
            # BENCHMARK EVALUATION (every N batches)
            # =========================================================
            if (config.benchmark_every_n_batches > 0 and
                benchmark_rulings and
                global_batch_num % config.benchmark_every_n_batches == 0):

                logger.info(f"\n--- Running Benchmark Evaluation (batch {global_batch_num}) ---")

                base_for_serve = config.base_model_fp8 if config.inference_dtype.lower() == "fp8" else config.base_model_bf16

                if config.use_vllm_lora:
                    # vLLM LoRA mode: serve base model with LoRA adapter
                    benchmark_model = base_for_serve
                    benchmark_lora = current_adapter_path
                    logger.info(f"  vLLM LoRA benchmark: base={benchmark_model}, lora={benchmark_lora}")
                else:
                    # Legacy merge mode
                    benchmark_lora = None
                    benchmark_merged_dir = os.path.join(config.merged_models_dir, f"merged_benchmark_b{global_batch_num}")
                    _safe_rmtree(benchmark_merged_dir)

                    if current_adapter_path:
                        benchmark_model = merge_lora_into_bf16_base(
                            base_model_id=config.base_model_bf16,
                            adapter_path=current_adapter_path,
                            merged_out_dir=benchmark_merged_dir,
                            logger=logger,
                        )
                    else:
                        benchmark_model = base_for_serve

                # Start vLLM for benchmark
                if vllm_manager.start_server(model_to_serve=benchmark_model, lora_adapter_path=benchmark_lora if config.use_vllm_lora else None):
                    benchmark_client = VLLMInferenceClient(config, logger, vllm_manager)

                    benchmark_metrics = run_benchmark_evaluation(
                        benchmark_rulings,
                        config,
                        logger,
                        benchmark_client,
                        global_batch_num,
                    )

                    # Log benchmark to wandb
                    log_to_wandb(wandb_run, {
                        "exact_match_rate": benchmark_metrics['exact_match_rate'],
                        "num_exact_matches": benchmark_metrics['num_exact_matches'],
                        "avg_best_reward": benchmark_metrics['avg_best_reward'],
                        "avg_reward": benchmark_metrics['avg_reward'],
                        "num_rulings": benchmark_metrics['num_rulings'],
                    }, step=global_batch_num, prefix="benchmark")

                    # Stop vLLM to free memory for next training batch
                    vllm_manager.stop_server()
                else:
                    logger.error("Failed to start vLLM for benchmark!")

            batch_time = time.time() - batch_start
            logger.info(f"\n{'─' * 50}")
            logger.info(f"Batch {batch_num + 1} Summary:")
            logger.info(f"  Rulings: {len(batch_rulings)}")
            logger.info(f"  Samples: {len(all_batch_samples)}")
            logger.info(f"  Trained: {train_metrics.get('num_samples', 0)}")
            logger.info(f"  Loss: {avg_loss:.4f}")
            logger.info(f"  Exact match: {accuracy_metrics['exact_match_rate']:.1%}")
            logger.info(f"  Time: {batch_time:.1f}s ({batch_time/60:.1f}m)")
            logger.info(f"{'─' * 50}")

        epoch_time = time.time() - epoch_start
        epoch_exact_match_rate = epoch_exact_matches / epoch_rulings_total if epoch_rulings_total > 0 else 0.0

        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch + 1} COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"  Total rulings: {epoch_rulings_total}")
        logger.info(f"  Total samples: {epoch_samples_total}")
        logger.info(f"  Epoch exact match rate: {epoch_exact_match_rate:.1%} ({epoch_exact_matches}/{epoch_rulings_total})")
        logger.info(f"  Time: {epoch_time/60:.1f}m")

        log_to_wandb(wandb_run, {
            "exact_match_rate": epoch_exact_match_rate,
            "num_exact_matches": epoch_exact_matches,
            "total_rulings": epoch_rulings_total,
            "total_samples": epoch_samples_total,
        }, step=global_batch_num, prefix="epoch")

        # Save checkpoint at end of epoch
        if (epoch + 1) % config.save_every_n_epochs == 0 and current_adapter_path:
            checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-epoch-{epoch + 1}")
            _safe_rmtree(checkpoint_dir)
            shutil.copytree(current_adapter_path, checkpoint_dir)
            logger.info(f"  Checkpoint saved: {checkpoint_dir}")

    total_time = time.time() - training_start
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total batches: {global_batch_num}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")

    if current_adapter_path:
        final_adapter_dir = os.path.join(config.output_dir, "final_adapter")
        _safe_rmtree(final_adapter_dir)
        shutil.copytree(current_adapter_path, final_adapter_dir)
        logger.info(f"Final adapter saved: {final_adapter_dir}")

    metrics_file = os.path.join(config.output_dir, "training_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_file}")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    # Cleanup distributed training
    cleanup_distributed()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TreeRL GRPO Training with Nemotron-3-Nano (vLLM + Transformers/PEFT)"
    )

    # Nemotron model args
    parser.add_argument("--train-model", type=str, default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                        help="Training model (HuggingFace model ID)")
    parser.add_argument("--base-model-bf16", type=str, default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                        help="BF16 base model for merging/serving")
    parser.add_argument("--base-model-fp8", type=str, default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                        help="FP8 base model for serving (not used for merges)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp8"],
                        help="vLLM inference dtype (serving). Merges are done into BF16.")

    parser.add_argument("--sft-adapter", type=str, default="orlandowhite/nemotron3_nano_sft", 
                        help="Starting LoRA adapter (HuggingFace repo or local path)")
    parser.add_argument("--no-sft-adapter", "--fresh-start", action="store_true",
                        help="Start from fresh LoRA (no SFT adapter), train from base model")

    # Context lengths
    parser.add_argument("--vllm-max-len", type=int, default=262144, help="vLLM max model len")
    parser.add_argument("--train-max-seq-length", type=int, default=90000, help="Training max seq length")


    # vLLM args
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=512, help="vLLM max-num-seqs (higher = more throughput)")
    parser.add_argument("--vllm-tp", type=int, default=2, help="vLLM tensor parallel size (2 for 2-GPU)")
    parser.add_argument("--served-model-name", type=str, default="nemotron3nano",
                        help="Stable model name used in OpenAI requests")
    parser.add_argument("--vllm-tokenizer", type=str, default="",
                        help="Tokenizer model/path for vLLM (defaults to base model for LoRA)")

    parser.add_argument("--no-reasoning-parser", action="store_true",
                        help="Disable reasoning parser")
    parser.add_argument("--reasoning-parser", type=str, default="deepseek_r1",
                        choices=["deepseek_r1", "nano_v3"],
                        help="Reasoning parser to use (deepseek_r1 is usually more robust)")

    # LoRA args
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--no-vllm-lora", action="store_true",
                        help="Disable vLLM LoRA (use legacy merge architecture instead)")
    parser.add_argument("--vllm-lora-name", type=str, default="current_adapter",
                        help="Name for LoRA adapter in vLLM --lora-modules")

    # DDP / Multi-GPU training
    parser.add_argument("--use-ddp", action="store_true",
                        help="Enable DDP even if world_size=1 (for debugging)")
    parser.add_argument("--ddp-backend", type=str, default="nccl",
                        help="DDP backend (nccl for GPU, gloo for CPU)")

    # Training args
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Samples per forward pass (batched training)")

    # Batch/Epoch structure
    parser.add_argument("--rulings-per-batch", type=int, default=16, help="Rulings per batch (higher = better vLLM utilization)")
    parser.add_argument("--accuracy-window", type=int, default=10, help="Accuracy rolling window (unique rulings)")
    parser.add_argument("--num-batches", type=int, default=20, help="Batches per epoch")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--start-batch", type=int, default=0, help="Skip to this batch number (for resuming)")

    # Parallelization
    parser.add_argument("--parallel-rollouts", type=int, default=64, help="Number of concurrent rollouts (higher with high max-num-seqs)")

    # Benchmark evaluation
    parser.add_argument("--benchmark-every-n-batches", type=int, default=8, help="Run benchmark every N batches (0 to disable)")
    parser.add_argument("--benchmark-num-rulings", type=int, default=50, help="Number of held-out rulings for benchmark")
    parser.add_argument("--skip-initial-benchmark", action="store_true", help="Skip baseline benchmark before training starts")

    # Advantage normalization
    parser.add_argument("--advantage-method", type=str, default="grpo",
                        choices=["none", "grpo", "grpo_no_std", "gdpo"])
    parser.add_argument("--gdpo-reward-weights", type=str, default="1.0,1.0")

    # Reward/advantage scaling (stability knobs)
    parser.add_argument("--step-reward-scale", type=float, default=1.0,
                        help="Multiply TreeRL step rewards R(s) by this factor (helps avoid tiny gradients)")
    parser.add_argument("--leaf-advantage-scale", type=float, default=1.0,
                        help="When advantage_method!=none, multiply leaf advantages by this factor")
    parser.add_argument("--token-weight-clip", type=float, default=0.0,
                        help="Clip final per-token weights to [-clip, clip] (0 disables)")

    # Data args
    parser.add_argument("--chapter", type=str, default="84", help="HTS chapter to train on")
    parser.add_argument("--cross-rulings-file", type=str, default="cross_rulings_dataset.json")
    parser.add_argument("--train-all", action="store_true")

    # Wandb args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="treerl-grpo")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")

    # TreeRL args
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--max-questions", type=int, default=8)

    # Output args
    parser.add_argument("--output-dir", type=str, default="treerl_checkpoints")

    # Rollout caching
    parser.add_argument("--save-rollouts", type=str, default="")
    parser.add_argument("--load-rollouts", type=str, default="")

    args = parser.parse_args()

    config = TreeRLConfig(
        base_model_bf16=args.base_model_bf16,
        base_model_fp8=args.base_model_fp8,
        train_model=args.train_model,
        sft_adapter="" if args.no_sft_adapter else args.sft_adapter,
        inference_dtype=args.dtype,
        vllm_max_model_len=args.vllm_max_len,
        train_max_seq_length=args.train_max_seq_length,
        vllm_port=args.vllm_port,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_tensor_parallel_size=args.vllm_tp,
        vllm_served_model_name=args.served_model_name,
        vllm_tokenizer=args.vllm_tokenizer,
        vllm_use_reasoning_parser=not args.no_reasoning_parser,
        vllm_reasoning_parser_name=args.reasoning_parser if not args.no_reasoning_parser else "",
        vllm_reasoning_parser_plugin_filename="nano_v3_reasoning_parser.py" if args.reasoning_parser == "nano_v3" else "",
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_vllm_lora=not args.no_vllm_lora,
        vllm_lora_name=args.vllm_lora_name,
        use_ddp=args.use_ddp,
        ddp_backend=args.ddp_backend,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        train_micro_batch_size=args.micro_batch_size,
        rulings_per_batch=args.rulings_per_batch,
        accuracy_window_size=args.accuracy_window,
        num_batches=args.num_batches,
        num_epochs=args.epochs,
        start_batch=args.start_batch,
        parallel_rollouts=args.parallel_rollouts,
        benchmark_every_n_batches=args.benchmark_every_n_batches,
        benchmark_num_rulings=args.benchmark_num_rulings,
        skip_initial_benchmark=args.skip_initial_benchmark,
        advantage_method=args.advantage_method,
        gdpo_reward_weights=tuple(float(x.strip()) for x in args.gdpo_reward_weights.split(",") if x.strip()),
        step_reward_scale=args.step_reward_scale,
        leaf_advantage_scale=args.leaf_advantage_scale,
        token_weight_clip=args.token_weight_clip,
        chapter=args.chapter,
        cross_rulings_file=args.cross_rulings_file,
        train_all=args.train_all,
        beam_size=args.beam_size,
        max_questions=args.max_questions,
        output_dir=args.output_dir,
        save_rollouts=args.save_rollouts,
        load_rollouts=args.load_rollouts,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        use_fast_inference=False,
    )

    # Ensure derived dirs follow output_dir
    config.adapter_sync_dir = os.path.join(config.output_dir, "adapter_sync")
    config.merged_models_dir = os.path.join(config.output_dir, "merged_models")
    config.samples_dir = os.path.join(config.output_dir, "samples")
    config.completions_log = os.path.join(config.output_dir, "completions.jsonl")
    config.log_file = os.path.join(config.output_dir, "treerl_training.log")

    train(config)


if __name__ == "__main__":
    main()
