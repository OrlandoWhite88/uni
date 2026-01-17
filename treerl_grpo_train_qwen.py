#!/usr/bin/env python3
# =============================================================================
# ENVIRONMENT VARIABLES - MUST BE SET BEFORE ANY IMPORTS
# =============================================================================
import os

os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"  # See any errors

# DISABLE Unsloth vLLM standby - we manage vLLM externally
# This prevents conflicts between external vLLM server and Unsloth's internal vLLM
os.environ["UNSLOTH_VLLM_STANDBY"] = "0"

# CUDA memory management - standard settings for training
# Don't use expandable_segments which can cause issues
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    alloc_conf = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    if "expandable_segments" in alloc_conf:
        parts = [p for p in alloc_conf.split(",") if "expandable_segments" not in p]
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts) if parts else ""

# Fast HF downloads
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Explicitly set vLLM device type (fixes device detection in some containers)
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")

# =============================================================================
# NOW SAFE TO IMPORT
# =============================================================================
"""
TreeRL GRPO Training Script with Qwen3-14B (vLLM + Unsloth)

ARCHITECTURE: External vLLM for inference, Pure Unsloth for training
- vLLM runs as external server for rollouts (high throughput inference)
- Unsloth loads model separately for training (fast_inference=False)
- LoRA adapters saved and loaded via vLLM's --lora-modules

Model: Qwen3-14B with Thinking Mode
- Uses <think>...</think> tags for chain-of-thought reasoning
- Supports vLLM LoRA for fast adapter switching

Per-Epoch Flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │  PHASE 1: Start vLLM server with base model + LoRA adapter      │
    │           (--lora-modules for hot LoRA loading)                 │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  PHASE 2: Run beam search rollouts for ALL rulings in batch     │
    │           Collect training samples from each ruling             │
    │           (Qwen3 thinking mode enabled for better reasoning)    │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  PHASE 3: Stop vLLM server, free GPU memory completely          │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  PHASE 4: Load Unsloth (fast_inference=False), train LoRA       │
    │           Pure training mode - no internal vLLM                 │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  PHASE 5: Save LoRA adapter for next vLLM cycle                 │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
                            [Next Epoch]

Qwen3 Recommended Settings:
- Thinking mode: temp=0.6, top_p=0.95, top_k=20
- Non-thinking: temp=0.7, top_p=0.8, top_k=20
- LoRA rank: 32, alpha: 64 (2x rank)
- Max seq length: 32768 (native), extendable with YaRN

Usage:
    python treerl_grpo_train_fixed.py --chapter 84 --num-rulings 20 --epochs 3
    
    # With custom settings
    python treerl_grpo_train_fixed.py --lora-rank 32 --no-thinking --train-max-seq-length 4096
"""

import sys
import json
import math
import logging
import argparse
import random
import time
import subprocess
import signal
import requests
import gc
import re
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """Training configuration for TreeRL GRPO with Qwen3-14B."""
    
    # Model settings
    # base_model: The canonical model for training (Unsloth)
    # vllm_model: Model for vLLM inference (can be FP8 variant for faster inference)
    # LoRA adapters are compatible as long as base architecture matches
    base_model: str = "Qwen/Qwen3-14B"  # Model for training (Unsloth will quantize)
    train_model: str = "Qwen/Qwen3-14B"  # Alias for base_model (for compatibility)
    vllm_model: str = ""  # Model for vLLM inference (empty = use base_model)
    sft_adapter: str = "treerl_checkpoints/adapter_sync/adapter_1768520829"  # Starting LoRA adapter (HuggingFace or local path)

    
    # Context length settings
    # Qwen3-14B native: 32768, extendable to 128k+ with YaRN
    # YaRN factor 4.0 = 131072 context (32768 * 4)
    max_seq_length: int = 131072  # vLLM max context (with YaRN)
    train_max_seq_length: int = 131072  # Max tokens per training sample
    load_in_4bit: bool = True  # 4-bit quantization for training
    
    # YaRN rope scaling for extended context
    # Set to 0 to disable, or 4.0 for 128k context
    yarn_factor: float = 4.0  # 4.0 = 131072 context
    yarn_original_max_position_embeddings: int = 32768  # Qwen3 native
    
    # Qwen3 reasoning/thinking mode settings
    enable_thinking: bool = True  # Enable thinking mode for better accuracy
    json_thinking: bool = False  # Enable thinking for JSON responses (adds tokens but may improve quality)
    
    # Generation settings (Qwen3 recommended)
    # Thinking mode: temp=0.6, top_p=0.95, top_k=20
    # Non-thinking: temp=0.7, top_p=0.8, top_k=20
    rollout_temperature: float = 0.4  # 0.6 for thinking, 0.7 for non-thinking
    rollout_top_p: float = 0.95  # 0.95 for thinking, 0.8 for non-thinking
    rollout_top_k: int = 20
    rollout_min_p: float = 0.0  # Qwen3 default
    
    # IMPORTANT: max_new_tokens will be calculated dynamically based on input length
    # This is a fallback/cap value, not the actual value used
    rollout_max_new_tokens_cap: int = 6000  # Cap for output tokens (reasoning)
    json_max_tokens: int = 5000  # Cap for JSON responses (chapter/score selection) - needs room for thinking
    
    # vLLM settings
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    # Lower GPU utilization to leave headroom for memory fragmentation
    # 0.80 = 112GB on 140GB GPU, leaving ~28GB buffer
    vllm_gpu_memory_utilization: float = 0.80
    vllm_max_model_len: int = 131072  # Extended with YaRN (32768 * 4)
    
    # vLLM LoRA support - Qwen3 fully supports this
    vllm_enable_lora: bool = True
    vllm_max_lora_rank: int = 64
    vllm_max_loras: int = 4  # Max concurrent LoRA adapters
    
    # NOTE: vLLM reasoning parser flags vary by version and often don't exist
    # Qwen3 handles <think> tags natively in its output - no special vLLM config needed
    # Set to empty string to disable (recommended for compatibility)
    vllm_reasoning_parser: str = ""  # Options: "", "deepseek_r1", "qwen3" (if supported)
    
    # LoRA settings for training
    lora_rank: int = 64  # Must match SFT adapter (orlandowhite/qwen3_sft uses rank 64)
    lora_alpha: int = 128  # 2x rank
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    
    # Training settings
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Advantage shaping / normalization
    advantage_method: str = "gdpo"  # none | grpo | gdpo
    gdpo_reward_weights: Tuple[float, ...] = (1.0, 1.0)
    gdpo_eps: float = 1e-6
    # Whether to scale the loss by GDPO per-leaf advantages
    # Per GDPO paper: use normalized advantages for loss scaling, NOT for TreeRL values
    apply_leaf_advantage_scaling: bool = True  # Enable by default for GDPO
    
    # TreeRL settings
    beam_size: int = 4
    max_questions: int = 5
    
    # Parallelization settings
    # Number of rulings to process concurrently during rollouts
    # vLLM handles batching internally - more concurrent rulings = better GPU utilization
    # Set based on GPU memory: 2-4 for 24GB, 4-8 for 48GB, 8-16 for 80GB+
    parallel_rollouts: int = 8
    
    # Benchmark evaluation settings
    # Run a larger evaluation every N batches to get cleaner signal
    benchmark_every_n_batches: int = 0  # 0 to disable (disabled by default)
    benchmark_num_rulings: int = 50  # Number of rulings to evaluate (held-out from training)
    
    # Data settings
    chapter: str = "84"
    # Batch = process N rulings, run rollouts, train on all samples
    # Epoch = run multiple batches (or all rulings if train_all=True)
    rulings_per_batch: int = 5  # Number of rulings per batch
    accuracy_window_size: int = 20  # Primary rolling window size (rulings)
    accuracy_window_sizes: Tuple[int, ...] = (10, 20, 50)  # Multiple windows for diagnostics
    num_batches: int = 20  # Number of batches per epoch (ignored if train_all=True)
    num_epochs: int = 3  # Number of full epochs
    train_all: bool = False  # If True, train on ALL rulings in chapter (not random sampling)
    start_batch: int = 0  # Skip to this batch number (for resuming mid-training)
    
    # Paths
    cross_rulings_file: str = "cross_rulings_dataset.json"
    output_dir: str = "treerl_checkpoints"
    log_file: str = "treerl_training.log"
    adapter_sync_dir: str = "treerl_checkpoints/adapter_sync"
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
    wandb_run_name: str = ""  # Auto-generated if empty
    wandb_entity: str = ""  # Your wandb username/team
    wandb_resume: bool = False  # Resume a previous run
    wandb_run_id: str = ""  # Run ID to resume (required if wandb_resume=True)
    
    # Device
    device: str = "cuda"

    # Leaf reward shaping
    leaf_reward_weights: Tuple[float, ...] = (0.85, 0.15)
    leaf_reward_clip_0_1: bool = True
    
    # Length penalty to prevent hitting max tokens
    # Penalty ramps up as output approaches max_tokens
    length_penalty_enabled: bool = False
    length_penalty_max_tokens: int = 6000  # Target max (should match rollout_max_new_tokens_cap)
    length_penalty_start_ratio: float = 0.5  # Start penalizing at 3000 tokens (3000/6000)
    length_penalty_max: float = 0.9  # Maximum penalty at max_tokens (0.9 = reward * 0.1)
    
    # IMPORTANT: Disable fast_inference - we use external vLLM
    # Setting this False prevents Unsloth from loading its own vLLM
    use_fast_inference: bool = False  # MUST BE FALSE for this architecture
    
    # Safety margin for max_tokens calculation
    # Reserves tokens for special tokens, chat template overhead, etc.
    token_safety_margin: int = 512


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: TreeRLConfig) -> logging.Logger:
    """Configure logging for training."""
    logger = logging.getLogger("treerl_train")
    logger.setLevel(logging.DEBUG)
    
    # Create output directory if needed
    os.makedirs(config.output_dir, exist_ok=True)
    
    log_path = os.path.join(config.output_dir, os.path.basename(config.log_file))
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def free_gpu_memory(logger: Optional[logging.Logger] = None):
    """Aggressively free GPU memory with full CUDA context cleanup."""
    if logger:
        logger.info("  Freeing GPU memory...")
    
    # Force Python garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        # Log before cleanup
        if logger:
            allocated_before = torch.cuda.memory_allocated() / 1e9
            reserved_before = torch.cuda.memory_reserved() / 1e9
            logger.info(f"    Before: {allocated_before:.2f}GB alloc, {reserved_before:.2f}GB reserved")
        
        # Synchronize all CUDA streams
        torch.cuda.synchronize()
        
        # Clear all cached memory
        torch.cuda.empty_cache()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Additional sync
        torch.cuda.synchronize()
        
        # Log after cleanup
        if logger:
            allocated_after = torch.cuda.memory_allocated() / 1e9
            reserved_after = torch.cuda.memory_reserved() / 1e9
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            logger.info(f"    After:  {allocated_after:.2f}GB alloc, {reserved_after:.2f}GB reserved, {free_mem:.1f}GB free")


def wait_for_gpu_memory(logger: logging.Logger, target_free_gb: float = 100.0, timeout: int = 60):
    """Wait for GPU memory to be released up to a timeout."""
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
# VLLM SERVER MANAGEMENT
# ============================================================================

class VLLMServerManager:
    """Manages vLLM server lifecycle with proper LoRA support."""
    
    def __init__(self, config: TreeRLConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{config.vllm_host}:{config.vllm_port}"
        self._current_model_path: Optional[str] = None
        self._current_lora_name: Optional[str] = None
        self._log_thread: Optional[threading.Thread] = None
        self._stop_logging = threading.Event()
    
    def start_server(
        self, 
        model_path: Optional[str] = None, 
        lora_path: Optional[str] = None,
        lora_name: str = "adapter"
    ) -> bool:
        """
        Start vLLM server with specified model and optional LoRA adapter.
        
        Args:
            model_path: Path to model to serve. If None, uses config.base_model.
            lora_path: Path to LoRA adapter directory (optional).
            lora_name: Name to register the LoRA adapter under.
        """
        if self.is_running():
            self.logger.info("vLLM server already running")
            return True
        
        self.logger.info("Starting vLLM server...")
        free_gpu_memory(self.logger)
        
        # Ensure GPU is ready - wait for enough free memory
        # vLLM with 0.85 util on 140GB GPU needs ~119GB, so require at least 125GB free
        if not wait_for_gpu_memory(self.logger, target_free_gb=125.0, timeout=120):
            self.logger.warning("  ⚠️ GPU memory may not be fully released, attempting anyway...")
        
        # Extra cleanup attempt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        
        # Use provided model, or vllm_model if set, or fall back to base_model
        # vllm_model allows using FP8 variant for faster inference
        serve_model = model_path or self.config.vllm_model or self.config.base_model
        
        # Build command
        cmd = [
            "vllm", "serve", serve_model,
            "--host", self.config.vllm_host,
            "--port", str(self.config.vllm_port),
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(self.config.vllm_gpu_memory_utilization),
            "--max-model-len", str(self.config.vllm_max_model_len),
            # Note: --disable-log-requests is deprecated, but still works
            # Use --enable-log-requests=false in newer versions if needed
        ]
        
        # YaRN rope scaling for extended context (Qwen3: 32k native → 128k+ with YaRN)
        # Required for contexts > 32768 tokens
        # NOTE: Newer vLLM (0.8+) uses --hf-overrides instead of --rope-scaling
        if self.config.yarn_factor and self.config.yarn_factor > 1.0:
            # Build nested JSON for hf-overrides
            rope_scaling_config = {
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": self.config.yarn_factor,
                    "original_max_position_embeddings": self.config.yarn_original_max_position_embeddings,
                }
            }
            hf_overrides_json = json.dumps(rope_scaling_config, separators=(',', ':'))
            cmd.extend([
                "--hf-overrides", hf_overrides_json,
            ])
            self.logger.info(f"  YaRN enabled: factor={self.config.yarn_factor}, max_len={self.config.vllm_max_model_len}")
        
        # NOTE: --enable-reasoning flag does NOT exist in older vLLM versions
        # For vLLM 0.9.0+, you can add: --reasoning-parser qwen3
        # For vLLM 0.8.x, you can try: --enable-reasoning --reasoning-parser deepseek_r1
        # Qwen3 handles <think> tags natively - no special vLLM config strictly needed
        
        # LoRA support
        if self.config.vllm_enable_lora:
            cmd.extend([
                "--enable-lora",
                "--max-lora-rank", str(self.config.vllm_max_lora_rank),
                "--max-loras", str(self.config.vllm_max_loras),
            ])
            
            # CRITICAL FIX: Actually pass the LoRA adapter to vLLM!
            if lora_path and os.path.isdir(lora_path):
                # Verify adapter files exist
                adapter_config = os.path.join(lora_path, "adapter_config.json")
                if os.path.exists(adapter_config):
                    cmd.extend([
                        "--lora-modules", f"{lora_name}={lora_path}",
                    ])
                    self._current_lora_name = lora_name
                    self.logger.info(f"  Loading LoRA adapter: {lora_name}={lora_path}")
                else:
                    self.logger.warning(f"  LoRA path exists but no adapter_config.json: {lora_path}")
        
        self._current_model_path = serve_model
        
        self.logger.info(f"vLLM command: {' '.join(cmd)}")
        
        # Start server process with real-time log streaming
        self._stop_logging.clear()
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._log_file_path = os.path.join(self.config.output_dir, "vllm_server.log")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        
        # Start log streaming thread
        self._log_thread = threading.Thread(target=self._stream_logs, daemon=True)
        self._log_thread.start()
        
        return self._wait_for_ready(timeout=900)
    
    def _stream_logs(self):
        """Stream vLLM logs to console and file."""
        try:
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
        """Wait for vLLM server to be ready."""
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
        """Check if vLLM server is running."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def stop_server(self):
        """Stop the vLLM server and free GPU memory."""
        if self.process:
            self.logger.info("Stopping vLLM server...")
            
            # Stop log streaming
            self._stop_logging.set()
            
            # Graceful termination
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.logger.warning("  Force killing vLLM process...")
                self.process.kill()
                self.process.wait()
            
            # Wait for log thread
            if self._log_thread and self._log_thread.is_alive():
                self._log_thread.join(timeout=2)
            
            self.process = None
            self._current_model_path = None
            self._current_lora_name = None
            
            # Give GPU time to release memory - CRITICAL for switching to training
            self.logger.info("Waiting for GPU memory release...")
            time.sleep(5)
            
            # Aggressive cleanup
            gc.collect()
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
            
            # Wait longer for GPU to fully release
            time.sleep(5)
            
            # Wait for GPU to be mostly free (with lower threshold)
            wait_for_gpu_memory(self.logger, target_free_gb=120.0, timeout=60)
            
            self.logger.info("vLLM server stopped, GPU memory freed")
    
    def load_lora_adapter(self, lora_path: str, lora_name: str = "adapter") -> bool:
        """
        Dynamically load a LoRA adapter via vLLM API.
        
        Requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=true environment variable.
        """
        if not self.is_running():
            self.logger.error("Cannot load LoRA - vLLM server not running")
            return False
        
        url = f"{self.base_url}/v1/load_lora_adapter"
        payload = {
            "lora_name": lora_name,
            "lora_path": lora_path,
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                self._current_lora_name = lora_name
                self.logger.info(f"✓ LoRA adapter loaded: {lora_name}")
                return True
            else:
                self.logger.error(f"Failed to load LoRA: {resp.status_code} - {resp.text}")
                return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error loading LoRA: {e}")
            return False
    
    @property
    def current_model(self) -> Optional[str]:
        return self._current_model_path
    
    @property
    def current_lora(self) -> Optional[str]:
        return self._current_lora_name


# ============================================================================
# VLLM INFERENCE CLIENT
# ============================================================================

class VLLMInferenceClient:
    """vLLM-based LLM client for fast inference with Qwen3 thinking mode."""

    def __init__(
        self,
        config: TreeRLConfig,
        logger: logging.Logger,
        server_manager: VLLMServerManager,
    ):
        self.config = config
        self.logger = logger
        self.server_manager = server_manager
        self.base_url = f"http://{config.vllm_host}:{config.vllm_port}"
        
        # Initialize OpenAI client pointing to vLLM server
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="not-needed",  # vLLM doesn't require API key
            )
            logger.info(f"  ✓ OpenAI client initialized for vLLM at {self.base_url}")
        except ImportError:
            logger.warning("  ⚠️ openai package not installed, falling back to requests")
            self.openai_client = None
        
        # Try to load system prompt
        try:
            from api.system_prompts_updated import UNIFIED_SYSTEM_PROMPT
            self.system_prompt = UNIFIED_SYSTEM_PROMPT.strip()
        except ImportError:
            self.system_prompt = "You are a helpful assistant specialized in HTS classification."
        
        self._system_prompt_injection: Optional[str] = None
        
        # CRITICAL: These attributes are required by classification_engine.py
        self.log_prompts = False
        self.prompt_logger = logger
        
        # Additional compatibility attributes that external code may expect
        self.client = None  # Some code checks for this
        self.model_name = config.base_model
        
        # JSON extraction settings
        self._json_requirements = (
            "\n\n=== OUTPUT FORMAT ===\n"
            "You MUST respond with ONLY a valid JSON object or array.\n"
            "Do NOT include any reasoning, explanation, or text before or after the JSON.\n"
            "Do NOT wrap the JSON in markdown code blocks.\n"
            "Start your response with [ or { and end with ] or }.\n"
            "===================\n"
        )
        self._max_json_retries = 4
        
        # Qwen3 thinking tags
        self._think_start = "<think>"
        self._think_end = "</think>"
        
        # Token estimation cache
        self._avg_chars_per_token = 3.5  # Conservative estimate for Qwen3

    def set_system_prompt_injection(self, prompt: Optional[str]) -> None:
        self._system_prompt_injection = prompt

    def clear_system_prompt_injection(self) -> None:
        self._system_prompt_injection = None

    def _current_system_prompt(self) -> str:
        return (self._system_prompt_injection or self.system_prompt).strip()
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for a string (conservative)."""
        return int(len(text) / self._avg_chars_per_token) + 10
    
    def _estimate_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate total tokens for a message list including chat template overhead."""
        total = 0
        for msg in messages:
            # Content tokens
            total += self._estimate_token_count(msg.get("content", ""))
            # Chat template overhead per message (role tags, special tokens)
            total += 20
        # Add buffer for system prompt and formatting
        total += 50
        return total
    
    def _calculate_max_tokens(
        self, 
        messages: List[Dict[str, str]], 
        requested_max: Optional[int] = None,
        is_json: bool = False,
    ) -> int:
        """
        Calculate safe max_tokens based on input length and model context.
        
        CRITICAL FIX: Prevents "max_tokens too large" errors.
        For JSON responses, uses a much lower cap to avoid runaway thinking.
        """
        # Estimate input tokens
        input_tokens = self._estimate_messages_tokens(messages)
        
        # Calculate available space
        available = self.config.vllm_max_model_len - input_tokens - self.config.token_safety_margin
        
        # Use lower cap for JSON responses (chapter selection, scores)
        # These should be short responses, not 16K token essays
        cap = self.config.json_max_tokens if is_json else self.config.rollout_max_new_tokens_cap
        
        # Apply cap
        if requested_max is not None:
            safe_max = min(requested_max, available, cap)
        else:
            safe_max = min(available, cap)
        
        # Debug: log when cap is limiting factor
        if safe_max == cap and available > cap:
            self.logger.debug(f"    [max_tokens] Using cap={cap} (json={is_json}), available={available}")
        
        # Ensure minimum viable output
        safe_max = max(safe_max, 256)  # At least 256 tokens
        
        if available < 1000:
            self.logger.warning(
                f"  ⚠️ Low available tokens: input≈{input_tokens}, "
                f"available≈{available}, using max_tokens={safe_max}"
            )
        
        return safe_max
    
    def _log_completion(self, request: Dict, response: Dict) -> None:
        """Log full raw completion to JSONL file for debugging."""
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

    def _extract_thinking_and_content(self, response_text: str) -> Tuple[str, str]:
        """
        Extract thinking content and final response from Qwen3 output.
        
        Qwen3 format: <think>reasoning here</think>final answer here
        
        Returns:
            (thinking_content, final_content)
        """
        if not response_text:
            return "", ""
        
        text = response_text.strip()
        
        # Find </think> tag to split thinking from content
        think_end_pos = text.find(self._think_end)
        
        if think_end_pos != -1:
            think_start_pos = text.find(self._think_start)
            if think_start_pos != -1:
                thinking = text[think_start_pos + len(self._think_start):think_end_pos].strip()
            else:
                thinking = text[:think_end_pos].strip()
            
            content = text[think_end_pos + len(self._think_end):].strip()
            return thinking, content
        else:
            # No thinking tags - check if truncated
            if text.startswith(self._think_start):
                return text[len(self._think_start):].strip(), ""
            return "", text
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from model response, handling chain-of-thought/reasoning."""
        if not response_text:
            raise ValueError("No response text to parse.")

        # Extract content after </think> if present (for thinking mode)
        thinking, content = self._extract_thinking_and_content(response_text)
        text = content.strip() if content else response_text.strip()

        # Remove markdown code blocks
        if "```json" in text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()
        elif "```" in text:
            match = re.search(r'```\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()

        # Try parsing the full text first
        try:
            json.loads(text)
            return text
        except Exception:
            pass

        # Find JSON with bracket matching
        bracket_positions = []
        for i, c in enumerate(text):
            if c == '[':
                bracket_positions.append(('array', i))
            elif c == '{':
                bracket_positions.append(('object', i))
        
        for json_type, start_pos in bracket_positions:
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
                    
                if c == '[' or c == '{':
                    depth += 1
                elif c == ']' or c == '}':
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
                    continue

        # Fallback: simple find
        start_a = text.find("[")
        end_a = text.rfind("]")
        if start_a != -1 and end_a != -1 and end_a > start_a:
            candidate = text[start_a:end_a + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        start_o = text.find("{")
        end_o = text.rfind("}")
        if start_o != -1 and end_o != -1 and end_o > start_o:
            candidate = text[start_o:end_o + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        raise ValueError(f"Failed to extract valid JSON from response: {response_text[:500]}...")

    def _prepare_messages_for_call(
        self,
        messages: List[Dict[str, str]],
        requires_json: bool,
    ) -> List[Dict[str, str]]:
        """
        Ensure message history fits within context window.
        
        We cannot exceed vLLM's max context length. If it does:
        - For JSON calls: keep only system + last user (minimal prompt)
        - For non-JSON: keep system + last 2 user/assistant turns
        - If still too long, hard-truncate the last user content
        """
        if not messages:
            return messages

        max_len = self.config.vllm_max_model_len
        if max_len <= 0:
            return messages

        input_tokens = self._estimate_messages_tokens(messages)
        if input_tokens + self.config.token_safety_margin <= max_len:
            return messages

        system_msg = messages[0] if messages[0].get("role") == "system" else None

        # Find last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            trimmed = [m for m in [system_msg, messages[-1]] if m]
        else:
            last_user_msg = messages[last_user_idx]
            if requires_json:
                trimmed = [m for m in [system_msg, last_user_msg] if m]
            else:
                tail = messages[max(0, last_user_idx - 2):]
                trimmed = [m for m in [system_msg] if m] + tail

        # If still too long, hard-truncate the last user message content
        if self._estimate_messages_tokens(trimmed) + self.config.token_safety_margin > max_len:
            for i in range(len(trimmed) - 1, -1, -1):
                if trimmed[i].get("role") == "user":
                    content = trimmed[i].get("content", "")
                    if isinstance(content, str) and len(content) > 200:
                        while len(content) > 200 and (
                            self._estimate_messages_tokens(trimmed) + self.config.token_safety_margin > max_len
                        ):
                            content = content[: int(len(content) * 0.8)]
                            trimmed[i]["content"] = content
                    break

        self.logger.warning(
            f"  ⚠️ Context too long; trimming messages. "
            f"tokens≈{input_tokens}, max_len={max_len}, json={requires_json}"
        )
        return trimmed

    def _call_vllm_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int] = None,
        requires_json: bool = False,
    ) -> str:
        """
        Call vLLM OpenAI-compatible API with Qwen3 support.
        
        CRITICAL: For JSON requests, disable thinking mode to get clean JSON output.
        Uses OpenAI client for cleaner handling.
        """
        gen_start = time.time()
        
        # Use LoRA adapter model name if available, else base model
        if self.server_manager.current_lora:
            model_name = self.server_manager.current_lora
        else:
            model_name = self.server_manager.current_model or self.config.base_model
        
        # Ensure input fits context window
        messages = self._prepare_messages_for_call(messages, requires_json)
        
        # Calculate safe max_tokens
        safe_max_tokens = self._calculate_max_tokens(messages, max_tokens, is_json=requires_json)
        
        # Qwen3 sampling parameters
        if requires_json:
            # JSON mode: lower temp, optionally enable thinking
            effective_temp = 0.7
            effective_top_p = 0.8
            enable_thinking = self.config.json_thinking  # Configurable: thinking for JSON
        else:
            # Reasoning mode: higher temp, enable thinking
            effective_temp = temperature if temperature > 0 else self.config.rollout_temperature
            effective_top_p = self.config.rollout_top_p
            enable_thinking = self.config.enable_thinking
        
        try:
            if self.openai_client:
                # Use OpenAI client (cleaner, handles edge cases better)
                extra_body = {
                    "top_k": self.config.rollout_top_k,
                    # CRITICAL: Control thinking mode via chat_template_kwargs
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                }
                if hasattr(self.config, 'rollout_min_p') and self.config.rollout_min_p > 0:
                    extra_body["min_p"] = self.config.rollout_min_p
                
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=safe_max_tokens,
                    temperature=effective_temp,
                    top_p=effective_top_p,
                    extra_body=extra_body,
                )
                
                choice = response.choices[0]
                output_text = choice.message.content or ""
                reasoning_text = getattr(choice.message, 'reasoning_content', None) or ""
                finish_reason = choice.finish_reason or "unknown"
                
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                
            else:
                # Fallback to requests (legacy)
                url = f"{self.base_url}/v1/chat/completions"
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": safe_max_tokens,
                    "temperature": effective_temp,
                    "top_p": effective_top_p,
                    "top_k": self.config.rollout_top_k,
                    # CRITICAL: Control thinking mode
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                }
                if hasattr(self.config, 'rollout_min_p') and self.config.rollout_min_p > 0:
                    payload["min_p"] = self.config.rollout_min_p
                
                resp = requests.post(url, json=payload, timeout=300)
                resp.raise_for_status()
                result = resp.json()
                
                choice = result.get("choices", [{}])[0]
                message = choice.get("message", {})
                output_text = message.get("content") or ""
                reasoning_text = message.get("reasoning_content") or ""
                finish_reason = choice.get("finish_reason", "unknown")
                
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
            
            # Log completion
            self._log_completion(
                {"model": model_name, "max_tokens": safe_max_tokens, "messages_count": len(messages)},
                {"finish_reason": finish_reason, "completion_tokens": completion_tokens}
            )
            
            gen_elapsed = time.time() - gen_start
            tok_per_sec = completion_tokens / gen_elapsed if gen_elapsed > 0 else 0
            
            self.logger.debug(
                f"    [vLLM] {completion_tokens} tokens in {gen_elapsed:.1f}s "
                f"({tok_per_sec:.1f} tok/s, prompt={prompt_tokens}, max={safe_max_tokens}, finish={finish_reason})"
            )
            
            if finish_reason == "length":
                self.logger.warning(f"  ⚠️ Hit max_tokens ({safe_max_tokens})! json_request={requires_json}")
                if requires_json:
                    self.logger.warning(f"  ⚠️ JSON request truncated!")
            
            # Handle Qwen3 reasoning response format (when using --reasoning-parser)
            if not output_text and reasoning_text:
                output_text = f"{self._think_start}{reasoning_text}{self._think_end}"
                self.logger.debug("  Reconstructed response from reasoning_content")
            
            # Debug logging (only when thinking is enabled and present)
            if output_text and self._think_start in output_text:
                thinking, final = self._extract_thinking_and_content(output_text)
                self.logger.debug(f"    [vLLM] Thinking: {len(thinking)} chars, Final: {len(final)} chars")
            
            if output_text:
                self.logger.debug(f"    [vLLM Response] len={len(output_text)}")
            else:
                self.logger.warning(f"vLLM returned empty content (tokens={completion_tokens})")
            
            return output_text.strip()
            
        except Exception as e:
            self.logger.error(f"vLLM API error: {e}")
            raise

    def send_openai_request(
        self,
        prompt: str,
        requires_json: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Send a single-turn request with retry for JSON extraction."""
        user_content = prompt.strip()
        
        # For JSON requests, append format requirements
        if requires_json:
            user_content = f"{user_content.rstrip()}{self._json_requirements}"
        
        messages = [
            {"role": "system", "content": self._current_system_prompt()},
            {"role": "user", "content": user_content},
        ]

        last_error = None
        for attempt in range(self._max_json_retries if requires_json else 1):
            try:
                retry_temp = temperature + (attempt * 0.1) if requires_json else temperature
                
                text = self._call_vllm_api(
                    messages,
                    temperature=min(retry_temp, 1.0),
                    max_tokens=max_tokens,
                    requires_json=requires_json,
                )

                if not text or not text.strip():
                    raise ValueError("Empty response from vLLM API")

                if requires_json:
                    cleaned = self._extract_json_from_response(text)
                    json.loads(cleaned)  # Validate
                    return cleaned
                return text
                
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
                self.logger.warning(f"JSON extraction failed (attempt {attempt + 1}/{self._max_json_retries}): {str(e)[:100]}")
                if attempt < self._max_json_retries - 1:
                    continue
                raise last_error

    def send_trajectory_request(
        self,
        messages: List[Dict[str, str]],
        requires_json: bool = False,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Send a multi-turn request with retry for JSON extraction."""
        req_messages = [m.copy() for m in messages]
        if requires_json:
            for i in range(len(req_messages) - 1, -1, -1):
                if req_messages[i].get("role") == "user":
                    req_messages[i]["content"] = f"{req_messages[i]['content'].rstrip()}{self._json_requirements}"
                    break

        last_error = None
        for attempt in range(self._max_json_retries if requires_json else 1):
            try:
                retry_temp = temperature + (attempt * 0.1) if requires_json else temperature
                
                text = self._call_vllm_api(
                    req_messages,
                    temperature=min(retry_temp, 1.0),
                    max_tokens=max_tokens,
                    requires_json=requires_json,
                )

                if not text or not text.strip():
                    raise ValueError("Empty response from vLLM API")

                if requires_json:
                    cleaned = self._extract_json_from_response(text)
                    json.loads(cleaned)  # Validate
                    return cleaned
                return text
                
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
                self.logger.warning(f"JSON extraction failed (attempt {attempt + 1}/{self._max_json_retries}): {str(e)[:100]}")
                if attempt < self._max_json_retries - 1:
                    continue
                raise last_error

    # Compatibility aliases
    def send_vertex_ai_request(self, *args, **kwargs) -> str:
        return self.send_openai_request(*args, **kwargs)

    def send_groq_request(
        self, 
        prompt: str, 
        requires_json: bool = False, 
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        return self.send_openai_request(
            prompt=prompt, 
            requires_json=requires_json, 
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_chapter_rulings(config: TreeRLConfig, logger: logging.Logger) -> List[Dict]:
    """Load cross rulings filtered by chapter, excluding rulings with codes not in the HTS tree."""
    logger.info(f"Loading rulings from {config.cross_rulings_file}")
    
    if not os.path.exists(config.cross_rulings_file):
        logger.error(f"Rulings file not found: {config.cross_rulings_file}")
        return []
    
    with open(config.cross_rulings_file, 'r', encoding='utf-8') as f:
        all_rulings = json.load(f)
    
    chapter_rulings = [
        r for r in all_rulings 
        if r.get("hts_code", "").startswith(config.chapter)
    ]
    
    logger.info(f"Total rulings: {len(all_rulings)}")
    logger.info(f"Chapter {config.chapter} rulings (before validation): {len(chapter_rulings)}")
    
    # Validate rulings against HTS tree - skip codes that don't exist
    try:
        from api.groq_tree_engine import HTSTree
        from api.treerl_gold_trace import build_gold_trace
        
        hts_tree = HTSTree()
        hts_data_file = Path(config.cross_rulings_file).parent / "api" / "hts_data.json"
        if not hts_data_file.exists():
            hts_data_file = Path(__file__).parent / "api" / "hts_data.json"
        
        if hts_data_file.exists():
            with open(hts_data_file, "r", encoding="utf-8") as f:
                hts_data = json.load(f)
            hts_tree.build_from_json(hts_data)
            
            valid_rulings = []
            skipped_codes = []
            
            for ruling in chapter_rulings:
                gold_code = ruling.get("hts_code", "")
                gold_trace = build_gold_trace(gold_code, hts_tree.navigator)
                
                # Only keep rulings where the gold code was properly found (>2 steps)
                if len(gold_trace) > 2:
                    valid_rulings.append(ruling)
                else:
                    skipped_codes.append(gold_code)
            
            if skipped_codes:
                logger.warning(f"⚠️ Skipped {len(skipped_codes)} rulings with codes not in HTS tree:")
                for code in skipped_codes[:10]:
                    logger.warning(f"    - {code}")
                if len(skipped_codes) > 10:
                    logger.warning(f"    ... and {len(skipped_codes) - 10} more")
            
            logger.info(f"Chapter {config.chapter} rulings (after validation): {len(valid_rulings)}")
            return valid_rulings
        else:
            logger.warning(f"HTS data file not found, skipping validation: {hts_data_file}")
            
    except Exception as e:
        logger.warning(f"Could not validate rulings against HTS tree: {e}")
    
    return chapter_rulings


# ============================================================================
# SAMPLE SAVING (for debugging)
# ============================================================================

def save_samples_for_debug(
    samples: List[Dict],
    config: TreeRLConfig,
    logger: logging.Logger,
    epoch: int,
    ruling_desc: str = "",
) -> str:
    """Save collected samples to disk for debugging."""
    os.makedirs(config.samples_dir, exist_ok=True)
    
    timestamp = int(time.time())
    safe_desc = "".join(c if c.isalnum() else "_" for c in ruling_desc[:30])
    filename = f"samples_epoch{epoch}_{safe_desc}_{timestamp}.json"
    filepath = os.path.join(config.samples_dir, filename)
    
    serializable_samples = []
    for s in samples:
        sample_copy = {
            "messages": s.get("messages", []),
            "step_rewards": s.get("step_rewards", []),
            "gold_code": s.get("gold_code", ""),
            "pred_trace": s.get("pred_trace", []),
            "gold_trace": s.get("gold_trace", []),
            "path_id": s.get("path_id", ""),
            "leaf_reward": s.get("leaf_reward", 0),
            "reward_components": s.get("reward_components", None),
            "source": s.get("source", ""),
        }
        serializable_samples.append(sample_copy)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  Saved {len(samples)} samples to: {filepath}")
    return filepath


def save_rollouts_to_file(
    all_samples: List[Dict],
    filepath: str,
    logger: logging.Logger,
    metadata: Optional[Dict] = None,
) -> str:
    """Save all rollout samples to a single file for later reuse."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    serializable_samples = []
    for s in all_samples:
        sample_copy = {
            "messages": s.get("messages", []),
            "step_rewards": s.get("step_rewards", []),
            "gold_code": s.get("gold_code", ""),
            "pred_trace": s.get("pred_trace", []),
            "gold_trace": s.get("gold_trace", []),
            "path_id": s.get("path_id", ""),
            "leaf_reward": s.get("leaf_reward", s.get("reward", 0)),
            "reward_components": s.get("reward_components", None),
            "source": s.get("source", ""),
        }
        serializable_samples.append(sample_copy)
    
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
    """Load rollout samples from a previously saved file."""
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


def display_rollout_stats(samples: List[Dict], logger: logging.Logger) -> None:
    """Display comprehensive stats after rollout phase."""
    if not samples:
        logger.warning("No samples to display stats for")
        return
    
    logger.info("\n" + "=" * 70)
    logger.info("ROLLOUT STATISTICS (Before Training)")
    logger.info("=" * 70)
    
    by_ruling = {}
    for s in samples:
        gold = s.get("gold_code", "unknown")
        if gold not in by_ruling:
            by_ruling[gold] = []
        by_ruling[gold].append(s)
    
    logger.info(f"\n📊 OVERVIEW")
    logger.info(f"  Total samples (beam paths): {len(samples)}")
    logger.info(f"  Unique rulings: {len(by_ruling)}")
    logger.info(f"  Avg paths per ruling: {len(samples) / max(len(by_ruling), 1):.1f}")
    
    all_leaf_rewards = []
    all_step_R = []
    perfect_count = 0
    partial_count = 0
    zero_count = 0
    
    for s in samples:
        leaf_r = s.get("leaf_reward", s.get("reward", 0))
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
        neg_count = sum(1 for r in all_step_R if r < 0)
        pos_count = sum(1 for r in all_step_R if r >= 0)
        logger.info(f"  Negative: {neg_count} ({100*neg_count/len(all_step_R):.1f}%)")
        logger.info(f"  Positive: {pos_count} ({100*pos_count/len(all_step_R):.1f}%)")
    
    logger.info(f"\n🔍 BEAM PATHS vs GOLD (per ruling)")
    logger.info("-" * 70)
    
    for i, (gold_code, ruling_samples) in enumerate(list(by_ruling.items())[:10]):
        gold_trace = ruling_samples[0].get("gold_trace", [])
        gold_path = " > ".join([
            t.get("code", f"grp:{t.get('node_id', '?')}")[:8] 
            for t in gold_trace
        ])
        
        best_sample = max(ruling_samples, key=lambda s: s.get("leaf_reward", s.get("reward", 0)))
        best_reward = best_sample.get("leaf_reward", best_sample.get("reward", 0))
        pred_trace = best_sample.get("pred_trace", [])
        pred_path = " > ".join([
            t.get("code", f"grp:{t.get('node_id', '?')}")[:8]
            for t in pred_trace
        ])
        
        match_depth = 0
        for j, (g, p) in enumerate(zip(gold_trace, pred_trace)):
            g_key = g.get("code") or g.get("node_id")
            p_key = p.get("code") or p.get("node_id")
            if str(g_key) == str(p_key):
                match_depth = j + 1
            else:
                break
        
        status = "✓ PERFECT" if best_reward == 1.0 else (f"◐ {match_depth}/{len(gold_trace)}" if best_reward > 0 else "✗ MISS")
        
        logger.info(f"\n  [{i+1}] Gold: {gold_code}")
        logger.info(f"      Gold path:  {gold_path}")
        logger.info(f"      Best pred:  {pred_path}")
        logger.info(f"      Best leaf_r: {best_reward:.3f} | Paths: {len(ruling_samples)} | {status}")
        
        step_Rs = [f"{sr.get('R', 0):.2f}" for sr in best_sample.get("step_rewards", [])]
        if step_Rs:
            logger.info(f"      Step R(s): [{', '.join(step_Rs)}]")
    
    if len(by_ruling) > 10:
        logger.info(f"\n  ... and {len(by_ruling) - 10} more rulings")
    
    logger.info("\n" + "=" * 70)


# ============================================================================
# ONLINE ROLLOUT (vLLM phase)
# ============================================================================

def run_online_rollout(
    ruling: Dict,
    config: TreeRLConfig,
    logger: logging.Logger,
    vllm_client: VLLMInferenceClient,
) -> List[Dict]:
    """Run rollout for a single ruling using vLLM."""
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
        """Secondary gold-anchored shaping signal."""
        from api.treerl_gold_trace import normalize_code

        gold_digits = normalize_code(gold_code)
        if not gold_digits:
            return 0.0

        pred_digits = ""
        for step in reversed(pred_trace or []):
            if step.get("kind") == "code":
                pred_digits = normalize_code(step.get("code", ""))
                if pred_digits:
                    break

        if not pred_digits:
            return 0.0

        m = 0
        for a, b in zip(pred_digits, gold_digits):
            if a == b:
                m += 1
            else:
                break
        return m / max(len(gold_digits), 1)

    def _aggregate_leaf_reward(components: Dict[str, float]) -> float:
        """Aggregate multiple leaf reward components."""
        w = list(config.leaf_reward_weights or ())
        if len(w) < 2:
            w = [1.0, 0.0]
        w = w[:2]
        r = (w[0] * float(components.get("trace_prefix", 0.0))) + (w[1] * float(components.get("code_digits_prefix", 0.0)))
        if config.leaf_reward_clip_0_1:
            r = max(0.0, min(1.0, r))
        return r

    def _compute_length_penalty(trajectory: List[Dict[str, Any]]) -> float:
        """
        Compute length penalty based on trajectory token count.
        
        Returns a multiplier in [1 - max_penalty, 1.0]:
        - 1.0 = no penalty (short output)
        - 1 - max_penalty = maximum penalty (at or above max_tokens)
        
        Penalty ramps linearly from start_ratio to 1.0 of max_tokens.
        """
        if not config.length_penalty_enabled or not trajectory:
            return 1.0
        
        # Estimate token count from trajectory content
        # Use ~4 chars per token as rough estimate, or count actual content
        total_chars = 0
        for msg in trajectory:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
        
        # Rough token estimate (4 chars per token)
        estimated_tokens = total_chars / 4.0
        
        max_tokens = config.length_penalty_max_tokens
        start_tokens = max_tokens * config.length_penalty_start_ratio
        
        if estimated_tokens <= start_tokens:
            return 1.0  # No penalty
        
        if estimated_tokens >= max_tokens:
            # At or beyond max: apply maximum penalty
            return 1.0 - config.length_penalty_max
        
        # Linear ramp between start and max
        progress = (estimated_tokens - start_tokens) / (max_tokens - start_tokens)
        penalty = progress * config.length_penalty_max
        
        return 1.0 - penalty

    def _compute_gdpo_advantages(
        leaves: List[Dict[str, Any]],
        weights: Tuple[float, ...],
        eps: float,
    ) -> None:
        """
        Compute GDPO-style decoupled normalized ADVANTAGES per leaf.
        
        Per GDPO paper: Normalize each reward component independently (decoupled),
        then sum weighted normalized values to get advantages.
        
        CRITICAL: These are ADVANTAGES for loss scaling, NOT replacement rewards.
        TreeRL should still use RAW rewards for V(s) computation.
        
        Stores "gdpo_advantage" on each leaf for optional loss scaling.
        """
        if not leaves:
            return
        
        components = ["trace_prefix", "code_digits_prefix"]
        values_by_comp = {}
        for comp in components:
            values_by_comp[comp] = [
                float(leaf.get("reward_components", {}).get(comp, 0.0) or 0.0)
                for leaf in leaves
            ]
        
        w = list(weights or ())
        if len(w) < len(components):
            w += [1.0] * (len(components) - len(w))
        w = w[:len(components)]

        # GDPO: Normalize each component independently (decoupled normalization)
        normalized_by_comp = {}
        for comp, vals in values_by_comp.items():
            if not vals:
                normalized_by_comp[comp] = []
                continue
            mu = sum(vals) / len(vals)
            var = sum((v - mu) ** 2 for v in vals) / len(vals)
            sd = math.sqrt(max(var, 0.0))
            if sd < eps:
                # No variance - all same value, advantage = 0
                normalized_by_comp[comp] = [0.0 for _ in vals]
            else:
                normalized_by_comp[comp] = [(v - mu) / (sd + eps) for v in vals]

        # Compute per-leaf GDPO advantage (weighted sum of normalized components)
        gdpo_advantages = []
        for i in range(len(leaves)):
            total = 0.0
            for comp_idx, comp in enumerate(components):
                comp_vals = normalized_by_comp.get(comp, [])
                if i < len(comp_vals):
                    total += float(w[comp_idx]) * float(comp_vals[i])
            gdpo_advantages.append(total)
        
        # Optional: Batch-wise normalization of advantages (per GDPO paper Eq. 6)
        if len(gdpo_advantages) > 1:
            adv_mu = sum(gdpo_advantages) / len(gdpo_advantages)
            adv_var = sum((a - adv_mu) ** 2 for a in gdpo_advantages) / len(gdpo_advantages)
            adv_sd = math.sqrt(max(adv_var, 0.0))
            if adv_sd > eps:
                gdpo_advantages = [(a - adv_mu) / (adv_sd + eps) for a in gdpo_advantages]

        # Store GDPO advantages on leaves (for optional loss scaling)
        # Do NOT replace "reward" - TreeRL needs raw rewards for V(s)
        for leaf, adv in zip(leaves, gdpo_advantages):
            leaf["gdpo_advantage"] = float(adv)
    
    os.environ["TREERL_BEAM_SIZE"] = str(config.beam_size)
    os.environ["TREERL_CHAPTER_BEAM_SIZE"] = str(config.beam_size)
    os.environ["DISABLE_CROSS_RULING_INJECTION"] = "true"
    
    # CRITICAL: Disable SFT training collector to enable parallel beam processing
    # The classification engine uses ThreadPoolExecutor when this is false
    os.environ["COLLECT_TRAINING_DATA"] = "false"
    
    # Set parallel workers for within-classification beam parallelization
    # This batches multiple beam paths' LLM calls together
    os.environ["PATH_WORKERS"] = str(config.beam_size * 2)  # 2x beam size for good batching
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
        logger.info(f"  [rollout] Gold trace has {len(gold_trace)} steps for {gold_code}")
        
        # SKIP rulings where gold code is not properly found in the HTS tree
        # A trace of <=2 steps means only chapter (+ maybe heading prefix) was found
        if len(gold_trace) <= 2:
            logger.warning(f"  [rollout] ⚠️ SKIPPING - Gold code '{gold_code}' not found in HTS tree!")
            logger.warning(f"  [rollout]    Gold trace: {gold_trace}")
            return []
        
        logger.debug(f"  [rollout] Initializing auto-responder...")
        auto_responder = LLMAutoResponder(engine_name="groq", debug=False)
        if hasattr(auto_responder, "llm_client"):
            auto_responder.llm_client = vllm_client
        
        logger.info(f"  [rollout] Starting classification (max_q={config.max_questions}, beam={config.beam_size})...")
        logger.info(f"  [rollout] Product: {product_description[:100]}...")
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
        
        # DIAGNOSTIC: Check if classification actually completed properly
        if result.get("final"):
            final_code = result.get("final_code") or result.get("classification", {}).get("final_code", "N/A")
            logger.info(f"  [rollout] Final result: {final_code}")
        else:
            logger.warning(f"  [rollout] ⚠️ Classification not final! Has pending question?")
        
        state = result.get("state", {})
        leaves = []
        beam = state.get("beam", [])
        
        # DIAGNOSTIC: Log beam state
        logger.info(f"  [rollout] Beam has {len(beam)} paths")
        if not beam:
            logger.warning(f"  [rollout] ⚠️ EMPTY BEAM! Classification may have failed.")
            logger.warning(f"  [rollout]    Result keys: {list(result.keys())}")
            logger.warning(f"  [rollout]    State keys: {list(state.keys()) if state else 'None'}")
        
        for path_idx, path_data in enumerate(beam):
            if isinstance(path_data, dict):
                classification_path = path_data.get("classification_path", [])
                trajectory = path_data.get("trajectory", [])
                path_id = path_data.get("path_id", "unknown")
                is_complete = path_data.get("is_complete", False)
            elif hasattr(path_data, "classification_path"):
                classification_path = path_data.classification_path
                trajectory = getattr(path_data, "trajectory", [])
                path_id = path_data.path_id
                is_complete = getattr(path_data, "is_complete", False)
            else:
                logger.warning(f"  [rollout] Path {path_idx}: Unknown type {type(path_data)}")
                continue
            
            # DIAGNOSTIC: Log each path's depth
            if path_idx < 3:  # Only log first 3 to avoid spam
                path_codes = [s.get("code", "[GRP]") for s in classification_path]
                logger.info(f"  [rollout] Path {path_idx}: {len(classification_path)} steps, complete={is_complete}, codes={path_codes[:5]}")
            
            pred_trace = build_pred_trace_from_path(classification_path)
            reward_components = {
                "trace_prefix": compute_leaf_reward(pred_trace, gold_trace),
                "code_digits_prefix": _code_digits_prefix_reward(pred_trace, gold_code),
            }
            base_reward = _aggregate_leaf_reward(reward_components)
            
            # Apply length penalty to discourage long outputs
            length_multiplier = _compute_length_penalty(trajectory)
            reward = base_reward * length_multiplier
            
            # Log if penalty applied (INFO level so user can see it working)
            if length_multiplier < 0.95:
                logger.info(f"  [rollout] ⚠️ Path {path_id}: LENGTH PENALTY {1-length_multiplier:.0%} applied (reward {base_reward:.3f} → {reward:.3f})")
            
            leaves.append({
                "path_id": path_id,
                "pred_trace": pred_trace,
                "reward": reward,
                "reward_raw": base_reward,
                "reward_components": reward_components,
                "length_penalty": length_multiplier,
                "trajectory": trajectory,
                "classification_path": classification_path,
                "source": "final_beam",
            })
        
        pruned_leaves = state.get("_treerl_pruned_leaves", [])
        for pruned in pruned_leaves:
            classification_path = pruned.get("classification_path", [])
            trajectory = pruned.get("trajectory", [])
            path_id = pruned.get("path_id", "unknown")
            
            pred_trace = build_pred_trace_from_path(classification_path)
            reward_components = {
                "trace_prefix": compute_leaf_reward(pred_trace, gold_trace),
                "code_digits_prefix": _code_digits_prefix_reward(pred_trace, gold_code),
            }
            base_reward = _aggregate_leaf_reward(reward_components)
            
            # Apply length penalty to discourage long outputs
            length_multiplier = _compute_length_penalty(trajectory)
            reward = base_reward * length_multiplier
            
            leaves.append({
                "path_id": path_id,
                "pred_trace": pred_trace,
                "reward": reward,
                "reward_raw": base_reward,
                "reward_components": reward_components,
                "length_penalty": length_multiplier,
                "trajectory": trajectory,
                "classification_path": classification_path,
                "source": "pruned",
            })
        
        if not leaves:
            logger.warning(f"No leaves collected for ruling: {product_description[:50]}")
            return []
        
        # CRITICAL FIX: Always use RAW rewards for TreeRL V(s) computation
        # GDPO normalization is for advantage-based loss scaling, NOT for value baselines
        # Per GDPO paper: normalization is applied to advantages, not to replace rewards
        
        # Compute GDPO advantages (stored on leaves for optional loss scaling)
        if config.advantage_method and config.advantage_method.lower() == "gdpo":
            _compute_gdpo_advantages(
                leaves,
                config.gdpo_reward_weights,
                config.gdpo_eps,
            )

        # TreeRL uses RAW rewards for proper V(root) and step reward computation
        step_rewards, v_root = compute_treerl_rewards(leaves, reward_key="reward")
        
        # DIAGNOSTIC: Check trajectory presence before emitting
        leaves_with_traj = sum(1 for leaf in leaves if leaf.get("trajectory"))
        if leaves_with_traj < len(leaves):
            logger.warning(f"  [rollout] ⚠️ Only {leaves_with_traj}/{len(leaves)} leaves have trajectories!")
        
        samples = emit_leaf_samples(
            leaves,
            step_rewards,
            gold_trace=gold_trace,
            gold_code=gold_code,
        )
        
        logger.info(f"  [rollout] Rollout: {len(leaves)} leaves, {len(samples)} samples, V(root)={v_root:.3f}")
        if len(samples) < len(leaves):
            logger.warning(f"  [rollout] ⚠️ Fewer samples than leaves! Some may lack trajectories.")
        
        return samples
        
    except Exception as e:
        logger.error(f"Rollout error for '{product_description[:50]}': {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return []


# ============================================================================
# TRAINING FUNCTIONS (Unsloth phase - NO vLLM)
# ============================================================================

def load_training_model(config: TreeRLConfig, logger: logging.Logger, adapter_path: Optional[str] = None):
    """
    Load model with Unsloth for training.
    
    CRITICAL: Uses fast_inference=False to avoid loading internal vLLM.
    This keeps training completely separate from the external vLLM server.
    
    IMPORTANT: Always use FastLanguageModel.get_peft_model() first to set up
    Unsloth's gradient checkpointing properly, then load adapter weights if resuming.
    Using PeftModel.from_pretrained() directly causes gradient checkpointing errors.
    """
    logger.info("=" * 70)
    logger.info("LOADING QWEN3-14B WITH UNSLOTH (Training Mode)")
    logger.info("=" * 70)
    
    from unsloth import FastLanguageModel
    
    train_load_seq = config.train_max_seq_length
    train_model_name = config.train_model or config.base_model
    
    logger.info(f"Loading training model: {train_model_name}")
    logger.info(f"  seq_length={train_load_seq}, 4bit={config.load_in_4bit}")
    logger.info(f"  fast_inference=False (external vLLM architecture)")
    
    t0 = time.time()
    logger.info("  Calling FastLanguageModel.from_pretrained()...")
    sys.stdout.flush()
    
    # Load model WITHOUT fast_inference to avoid internal vLLM
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=train_model_name,
        max_seq_length=train_load_seq,
        dtype=None,  # Auto-detect
        load_in_4bit=config.load_in_4bit,
        # CRITICAL: fast_inference=False prevents Unsloth from loading vLLM
        # This is required for the external vLLM architecture
        # token="hf_...",  # Uncomment if using gated models
    )
    
    logger.info(f"  ✓ Base model loaded in {time.time() - t0:.1f}s")
    
    # ALWAYS attach LoRA with Unsloth's method first (sets up gradient checkpointing properly)
    # Then load adapter weights if resuming - this avoids gradient checkpointing errors
    logger.info("  Attaching LoRA modules with Unsloth...")
    t0 = time.time()
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
    logger.info(f"  ✓ LoRA attached in {time.time() - t0:.1f}s (rank={config.lora_rank}, alpha={config.lora_alpha})")
    
    # If resuming from a previous adapter, load the weights
    if adapter_path and os.path.isdir(adapter_path):
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            logger.info(f"  Resuming from adapter: {adapter_path}")
            t0 = time.time()
            
            loaded = False
            
            # Method 1: Try PEFT's load_adapter (most robust)
            try:
                from peft import set_peft_model_state_dict
                
                # Load adapter weights - try safetensors first, then bin
                adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
                if os.path.exists(adapter_weights_path):
                    from safetensors.torch import load_file
                    adapter_weights = load_file(adapter_weights_path)
                    logger.info(f"    Loading from safetensors: {len(adapter_weights)} tensors")
                else:
                    adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
                    if os.path.exists(adapter_weights_path):
                        adapter_weights = torch.load(adapter_weights_path, map_location="cpu", weights_only=True)
                        logger.info(f"    Loading from bin: {len(adapter_weights)} tensors")
                    else:
                        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")
                
                # Use PEFT's method to set adapter state dict
                set_peft_model_state_dict(model, adapter_weights)
                loaded = True
                logger.info(f"  ✓ Adapter weights loaded via PEFT in {time.time() - t0:.1f}s")
                
            except Exception as e:
                logger.warning(f"    PEFT load failed: {e}, trying direct state_dict load...")
            
            # Method 2: Fallback to direct state_dict load
            if not loaded:
                try:
                    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
                    if os.path.exists(adapter_weights_path):
                        from safetensors.torch import load_file
                        adapter_weights = load_file(adapter_weights_path)
                    else:
                        adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
                        adapter_weights = torch.load(adapter_weights_path, map_location="cpu", weights_only=True)
                    
                    # Load weights into model (strict=False to handle any key mismatches)
                    missing, unexpected = model.load_state_dict(adapter_weights, strict=False)
                    if missing:
                        logger.debug(f"    Missing keys (expected for base model): {len(missing)}")
                    if unexpected:
                        logger.warning(f"    Unexpected keys: {unexpected[:5]}...")
                    loaded = True
                    logger.info(f"  ✓ Adapter weights loaded via state_dict in {time.time() - t0:.1f}s")
                except Exception as e:
                    logger.error(f"    Failed to load adapter weights: {e}")
            
            if not loaded:
                logger.warning(f"  ⚠️ Could not load adapter weights, starting fresh")
        else:
            logger.warning(f"  Adapter path exists but no config found: {adapter_path}")
    
    # Enable training mode
    FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Print trainable parameters
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
    
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        free = torch.cuda.mem_get_info()[0] / 1e9
        logger.info(f"GPU: {alloc:.1f}GB used, {free:.1f}GB free")
    
    logger.info("✓ Training model ready (Qwen3 + Unsloth)")
    
    return model, tokenizer


def unload_training_model(model, tokenizer, logger: logging.Logger):
    """Unload training model and free GPU memory (gentle approach to avoid crashes)."""
    logger.info("Unloading training model...")
    
    # Give GPU time to finish any pending operations
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Delete model directly (don't move to CPU - can cause issues)
    del model
    gc.collect()
    
    # Delete tokenizer
    del tokenizer
    gc.collect()
    
    # Wait before CUDA cleanup
    time.sleep(2)
    
    # Gentle CUDA cleanup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Light garbage collection
    gc.collect()
    gc.collect()
    
    # Wait for memory to stabilize
    time.sleep(3)
    
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1e9
        logger.info(f"  GPU memory after unload: {free_mem:.1f}GB free")
    
    logger.info("Training model unloaded")


def find_assistant_turn_boundaries(
    input_ids: torch.Tensor,
    tokenizer,
    messages: List[Dict[str, str]]
) -> List[Tuple[int, int]]:
    """Find token boundaries for each assistant turn."""
    boundaries = []
    input_list = input_ids.tolist()
    
    for i, msg in enumerate(messages):
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


def _get_path_depth(user_content: str) -> int:
    """Extract tree depth from path_so_far in a rank_candidates user message."""
    match = re.search(r'"path_so_far":\s*"([^"]+)"', user_content)
    if not match:
        return -1
    
    path = match.group(1)
    depth = path.count(' > ') + 1
    return depth


def build_token_weights(
    step_rewards: List[Dict],
    boundaries: List[Tuple[int, int]],
    seq_len: int,
    device: str = "cuda",
    leaf_reward: Optional[float] = None,
    messages: Optional[List[Dict]] = None,
) -> torch.Tensor:
    """Build per-token weight tensor from step rewards."""
    weights = torch.zeros(seq_len, device=device)
    
    if not boundaries:
        return weights
    
    step_to_R = {sr["step"]: sr["R"] for sr in step_rewards}
    max_step = max(step_to_R.keys()) if step_to_R else 0
    
    if leaf_reward is not None:
        fallback_R = leaf_reward
    elif step_rewards:
        fallback_R = sum(sr.get("R", 0.0) for sr in step_rewards) / len(step_rewards)
    else:
        fallback_R = 0.0
    
    if not messages:
        for bound_idx, (start, end) in enumerate(boundaries):
            R = step_to_R.get(bound_idx, fallback_R)
            weights[start:end] = R
        return weights
    
    assistant_step_map = []
    current_depth = 0
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            if "rank_candidates" in content:
                depth = _get_path_depth(content)
                if depth > 0:
                    current_depth = depth
        
        elif role == "assistant":
            is_chapter_selection = (
                '"chapters"' in content or 
                '"top_selection"' in content or
                'chapter-level' in content.lower()
            )
            is_rank_candidates = '"primary_selection"' in content and not is_chapter_selection
            
            if is_chapter_selection:
                assistant_step_map.append(0)
            elif is_rank_candidates:
                step_idx = min(current_depth, max_step)
                assistant_step_map.append(step_idx)
            else:
                assistant_step_map.append(-1)
    
    for bound_idx, (start, end) in enumerate(boundaries):
        if bound_idx < len(assistant_step_map):
            step_idx = assistant_step_map[bound_idx]
            if step_idx >= 0:
                R = step_to_R.get(step_idx, fallback_R)
            else:
                R = 0.0  # Q&A turn - exclude from training
        else:
            R = fallback_R
        
        weights[start:end] = R
    
    return weights


def compute_grpo_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    step_rewards: List[Dict],
    boundaries: List[Tuple[int, int]],
    device: str = "cuda",
    leaf_reward: Optional[float] = None,
    messages: Optional[List[Dict]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute GRPO loss with per-step R(s) weighting."""
    assert model.training, "Model must be in training mode"
    assert torch.is_grad_enabled(), "Gradients must be enabled"
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, 
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    adjusted_boundaries = [(max(0, s-1), max(0, e-1)) for s, e in boundaries]
    weights = build_token_weights(
        step_rewards, 
        adjusted_boundaries, 
        shift_labels.shape[1],
        device,
        leaf_reward=leaf_reward,
        messages=messages,
    ).unsqueeze(0)
    
    masked_log_probs = token_log_probs * shift_mask.float()
    weighted_log_probs = masked_log_probs * weights
    
    num_weighted = (weights.abs() > 0).sum().float()
    if num_weighted > 0:
        loss = -weighted_log_probs.sum() / num_weighted
    else:
        loss = -masked_log_probs.sum() / shift_mask.sum().float()
    
    assert loss.requires_grad, "Loss must require gradients"
    
    metrics = {
        "loss": loss.item(),
        "avg_log_prob": masked_log_probs.sum().item() / max(shift_mask.sum().item(), 1),
        "num_weighted_tokens": num_weighted.item(),
    }
    
    return loss, metrics


def train_on_samples(
    samples: List[Dict],
    config: TreeRLConfig,
    logger: logging.Logger,
    adapter_path: Optional[str] = None,
) -> Tuple[str, Dict[str, float]]:
    """
    Train on collected samples using Unsloth.
    
    Returns:
        (new_adapter_path, metrics)
    """
    logger.info(f"Training on {len(samples)} samples...")
    logger.info(f"  Train max seq length: {config.train_max_seq_length}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    
    # Load training model (no internal vLLM)
    model, tokenizer = load_training_model(config, logger, adapter_path)
    
    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    model.train()

    # GDPO advantages are pre-computed during rollout and stored on samples
    # Only used if apply_leaf_advantage_scaling is enabled
    use_gdpo_scaling = config.apply_leaf_advantage_scaling and config.advantage_method == "gdpo"
    if use_gdpo_scaling:
        # Log GDPO advantage stats
        gdpo_advs = [s.get("gdpo_advantage", 0.0) for s in samples if "gdpo_advantage" in s]
        if gdpo_advs:
            logger.info(f"  GDPO advantages: mean={sum(gdpo_advs)/len(gdpo_advs):.4f}, "
                       f"min={min(gdpo_advs):.4f}, max={max(gdpo_advs):.4f}")
    
    metrics = {
        "total_loss": 0.0,
        "num_samples": 0,
        "skipped_truncated": 0,
        "skipped_error": 0,
    }
    
    accumulated_loss = 0.0
    accumulated_steps = 0
    optimizer.zero_grad()
    
    for sample_idx, sample in enumerate(samples):
        messages = sample.get("messages", [])
        step_rewards = sample.get("step_rewards", [])
        leaf_reward = sample.get("leaf_reward", sample.get("reward", None))
        path_id = sample.get("path_id", f"idx_{sample_idx}")
        
        # GDPO advantage is pre-computed during rollout
        # Per TreeRL/GDPO papers: negative advantages are CORRECT
        # They signal "decrease probability of this path" (gradient descent on negative loss)
        gdpo_adv = sample.get("gdpo_advantage", 0.0)
        # Clamp to prevent extreme scaling (stabilization)
        leaf_advantage = max(-3.0, min(3.0, gdpo_adv)) if use_gdpo_scaling else 1.0
        
        if not messages:
            continue
        
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config.train_max_seq_length,
                padding=False,
            )
            input_ids = inputs["input_ids"].to(config.device)
            attention_mask = inputs["attention_mask"].to(config.device)
            seq_len = input_ids.shape[1]

            # CRITICAL FIX: Skip truncated samples - they have incomplete trajectories
            if seq_len == config.train_max_seq_length:
                logger.warning(f"  Sample {sample_idx}: TRUNCATED at {seq_len} tokens - SKIPPING")
                metrics["skipped_truncated"] += 1
                continue

        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            metrics["skipped_error"] += 1
            continue
        
        boundaries = find_assistant_turn_boundaries(
            input_ids[0], tokenizer, messages
        )
        
        try:
            loss, loss_metrics = compute_grpo_loss(
                model,
                input_ids,
                attention_mask,
                step_rewards,
                boundaries,
                config.device,
                leaf_reward=leaf_reward,
                messages=messages,
            )

            # Apply GDPO leaf-level advantage scaling (optional)
            if use_gdpo_scaling:
                loss = loss * leaf_advantage
            
            scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()
            
            accumulated_loss += loss.item()
            accumulated_steps += 1
            
            metrics["total_loss"] += loss.item()
            metrics["num_samples"] += 1
            
            del input_ids, attention_mask, loss, scaled_loss
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM on sample {sample_idx} (seq_len={seq_len}): {e}")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            accumulated_loss = 0.0
            accumulated_steps = 0
            metrics["skipped_error"] += 1
            continue
            
        except Exception as e:
            import traceback
            logger.error(f"Loss computation error: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            metrics["skipped_error"] += 1
            continue
        
        if sample_idx % 4 == 0:
            torch.cuda.empty_cache()
        
        if accumulated_steps >= config.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.max_grad_norm
            )
            
            optimizer.step()
            optimizer.zero_grad()
            
            avg_acc_loss = accumulated_loss / accumulated_steps
            logger.info(f"  Step {metrics['num_samples']}: loss={avg_acc_loss:.4f}")
            
            accumulated_loss = 0.0
            accumulated_steps = 0
            
            torch.cuda.empty_cache()
    
    # Handle remaining gradients
    if accumulated_steps > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    # Save LoRA adapter
    timestamp = int(time.time())
    new_adapter_path = os.path.join(config.adapter_sync_dir, f"adapter_{timestamp}")
    os.makedirs(new_adapter_path, exist_ok=True)
    
    model.save_pretrained(new_adapter_path)
    tokenizer.save_pretrained(new_adapter_path)
    logger.info(f"LoRA adapter saved to: {new_adapter_path}")
    
    # Compute final metrics
    if metrics["num_samples"] > 0:
        metrics["avg_loss"] = metrics["total_loss"] / metrics["num_samples"]
    else:
        metrics["avg_loss"] = 0.0
    
    logger.info(f"  Training complete: {metrics['num_samples']} samples, "
                f"{metrics['skipped_truncated']} truncated, {metrics['skipped_error']} errors")
    
    # Unload model
    unload_training_model(model, tokenizer, logger)
    
    return new_adapter_path, metrics


def _compute_leaf_advantages(
    samples: List[Dict[str, Any]],
    config: TreeRLConfig,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Compute scalar advantage per sample for GRPO/GDPO."""
    method = (config.advantage_method or "none").lower().strip()
    if method == "none":
        return {s.get("path_id", f"idx_{i}"): 1.0 for i, s in enumerate(samples)}

    # Group by ruling
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
        for gold, gs in groups.items():
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
        for gold, gs in groups.items():
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
            if sd > 1e-6:
                adv_by_path[pid] = (v - mu) / (sd + 1e-4)
            else:
                adv_by_path[pid] = 0.0

        if vals:
            logger.info(
                f"Advantage (GDPO) stats: mean={mu:.4f}, std={sd:.4f}, "
                f"min={min(vals):.4f}, max={max(vals):.4f}"
            )
        return adv_by_path

    logger.warning(f"Unknown advantage_method='{config.advantage_method}', defaulting to none")
    return {s.get("path_id", f"idx_{i}"): 1.0 for i, s in enumerate(samples)}


# ============================================================================
# ACCURACY METRICS
# ============================================================================

def compute_batch_accuracy(samples: List[Dict], logger: logging.Logger, quiet: bool = False) -> Dict[str, float]:
    """
    Compute accuracy metrics for a batch of samples.
    
    Args:
        samples: List of sample dictionaries
        logger: Logger instance
        quiet: If True, don't log anything (for internal calculations)
    
    Returns:
        Dict with:
        - exact_match_rate: % of rulings where best path exactly matches gold (leaf_reward == 1.0)
        - avg_best_reward: Average of the best leaf_reward per ruling
        - avg_reward: Average leaf_reward across all samples
        - num_rulings: Number of unique rulings in batch
        - num_exact_matches: Count of rulings with exact match
    """
    if not samples:
        return {
            "exact_match_rate": 0.0,
            "avg_best_reward": 0.0,
            "avg_reward": 0.0,
            "num_rulings": 0,
            "num_exact_matches": 0,
        }
    
    # Group samples by ruling (gold_code)
    by_ruling: Dict[str, List[Dict]] = {}
    for s in samples:
        gold = s.get("gold_code", "unknown")
        by_ruling.setdefault(gold, []).append(s)
    
    num_rulings = len(by_ruling)
    num_exact_matches = 0
    best_rewards = []
    all_rewards = []
    
    for gold_code, ruling_samples in by_ruling.items():
        # Get rewards for all paths
        rewards = [s.get("leaf_reward", s.get("reward", 0.0)) or 0.0 for s in ruling_samples]
        all_rewards.extend(rewards)
        
        # Best path's reward for this ruling
        best_reward = max(rewards)
        best_rewards.append(best_reward)
        
        # Exact match = best path has reward == 1.0
        if best_reward >= 0.9999:  # Use small epsilon for float comparison
            num_exact_matches += 1
    
    exact_match_rate = num_exact_matches / num_rulings if num_rulings > 0 else 0.0
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
    vllm_client: VLLMInferenceClient,
    global_batch_num: int,
) -> Dict[str, float]:
    """
    Run a larger benchmark evaluation to get cleaner accuracy signal.
    
    This processes multiple rulings and computes aggregate metrics.
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"🎯 BENCHMARK EVALUATION (batch {global_batch_num})")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Evaluating on {len(benchmark_rulings)} rulings...")
    
    all_samples = []
    
    def run_single_rollout(ruling_idx_ruling):
        """Worker function for parallel rollout execution."""
        ruling_idx, ruling = ruling_idx_ruling
        product_desc = ruling.get("short_product_description", "")[:50]
        try:
            samples = run_online_rollout(ruling, config, logger, vllm_client)
            return ruling_idx, product_desc, samples, None
        except Exception as e:
            return ruling_idx, product_desc, [], str(e)
    
    # Process rulings in parallel
    with ThreadPoolExecutor(max_workers=config.parallel_rollouts) as executor:
        futures = {
            executor.submit(run_single_rollout, (idx, ruling)): idx 
            for idx, ruling in enumerate(benchmark_rulings)
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            ruling_idx, product_desc, samples, error = future.result()
            
            if samples:
                all_samples.extend(samples)
            
            if completed % 10 == 0:
                logger.info(f"  Benchmark progress: {completed}/{len(benchmark_rulings)}")
    
    # Compute metrics
    metrics = compute_batch_accuracy(all_samples, logger)
    
    logger.info(f"\n📊 BENCHMARK RESULTS (n={metrics['num_rulings']} rulings):")
    logger.info(f"  ┌─────────────────────────────────────────┐")
    logger.info(f"  │ Exact Match Rate: {metrics['exact_match_rate']:>6.1%} ({metrics['num_exact_matches']}/{metrics['num_rulings']}) │")
    logger.info(f"  │ Avg Best Reward:  {metrics['avg_best_reward']:>6.4f}               │")
    logger.info(f"  │ Avg Reward:       {metrics['avg_reward']:>6.4f}               │")
    logger.info(f"  └─────────────────────────────────────────┘")
    logger.info(f"{'=' * 70}\n")
    
    return metrics


# ============================================================================
# WANDB INTEGRATION
# ============================================================================

def init_wandb(config: TreeRLConfig, logger: logging.Logger) -> Optional[Any]:
    """Initialize wandb if enabled."""
    if not config.use_wandb:
        return None
    
    try:
        import wandb
        
        run_name = config.wandb_run_name or f"treerl-ch{config.chapter}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb_config = {
            "base_model": config.base_model,
            "chapter": config.chapter,
            "rulings_per_batch": config.rulings_per_batch,
            "num_batches": config.num_batches,
            "num_epochs": config.num_epochs,
            "beam_size": config.beam_size,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "learning_rate": config.learning_rate,
            "advantage_method": config.advantage_method,
            "gdpo_reward_weights": config.gdpo_reward_weights,
            "leaf_reward_weights": config.leaf_reward_weights,
        }
        
        # Support resuming a previous run
        if config.wandb_resume and config.wandb_run_id:
            logger.info(f"📊 Resuming wandb run: {config.wandb_run_id}")
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity if config.wandb_entity else None,
                id=config.wandb_run_id,
                resume="must",  # Fail if run doesn't exist
                config=wandb_config,
            )
            logger.info(f"✓ Wandb resumed: {config.wandb_run_id}")
        else:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity if config.wandb_entity else None,
                name=run_name,
                config=wandb_config,
            )
            logger.info(f"✓ Wandb initialized: {run_name}")
        
        return wandb
    except ImportError:
        logger.warning("wandb not installed. Run: pip install wandb")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return None


def log_to_wandb(
    wandb_run,
    metrics: Dict[str, float],
    step: int,
    prefix: str = "",
):
    """Log metrics to wandb if available."""
    if wandb_run is None:
        return
    
    try:
        log_dict = {}
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            log_dict[key] = v
        wandb_run.log(log_dict, step=step)
    except Exception:
        pass  # Silently ignore wandb errors


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(config: TreeRLConfig):
    """
    Main training function with batched online RL for Qwen3-14B.
    
    Architecture: External vLLM for inference + Pure Unsloth for training
    
    Terminology:
    - Batch: Process N rulings, run rollouts, train on collected samples
    - Epoch: Run through num_batches batches
    """
    
    logger = setup_logging(config)
    
    logger.info("=" * 70)
    logger.info("TREERL GRPO TRAINING with Qwen3-14B")
    logger.info("=" * 70)
    logger.info(f"Training model: {config.base_model}")
    logger.info(f"vLLM model: {config.vllm_model or config.base_model}")
    if config.vllm_model and config.vllm_model != config.base_model:
        logger.info(f"  ⚡ Using different model for inference (FP8/AWQ)")
    logger.info(f"Thinking mode: {config.enable_thinking}")
    logger.info(f"vLLM LoRA: {config.vllm_enable_lora}")
    logger.info(f"fast_inference: {config.use_fast_inference} (should be False)")
    logger.info(f"Max model len: {config.vllm_max_model_len}")
    logger.info(f"Parallel rollouts: {config.parallel_rollouts} (concurrent rulings)")
    
    # Verify configuration
    if config.use_fast_inference:
        logger.warning("⚠️ fast_inference=True is set but should be False for external vLLM architecture!")
        logger.warning("   Continuing anyway, but this may cause issues.")
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.adapter_sync_dir, exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)
    
    # Initialize wandb
    wandb_run = init_wandb(config, logger)
    
    # Load data
    rulings = load_chapter_rulings(config, logger)
    if not rulings:
        logger.error(f"No rulings found for chapter {config.chapter}")
        return
    
    # Calculate number of batches
    if config.train_all:
        # Train on ALL rulings - calculate batches needed
        num_batches_per_epoch = math.ceil(len(rulings) / config.rulings_per_batch)
        logger.info(f"\n📊 Training Schedule (TRAIN ALL MODE):")
        logger.info(f"  Total rulings in chapter {config.chapter}: {len(rulings)}")
        logger.info(f"  Rulings per batch: {config.rulings_per_batch}")
        logger.info(f"  Batches per epoch: {num_batches_per_epoch}")
        logger.info(f"  Number of epochs: {config.num_epochs}")
        logger.info(f"  Total batches: {num_batches_per_epoch * config.num_epochs}")
    else:
        # Random sampling mode
        num_batches_per_epoch = config.num_batches
        logger.info(f"\n📊 Training Schedule (Random Sampling):")
        logger.info(f"  Rulings per batch: {config.rulings_per_batch}")
        logger.info(f"  Batches per epoch: {num_batches_per_epoch}")
        logger.info(f"  Number of epochs: {config.num_epochs}")
        logger.info(f"  Total batches: {num_batches_per_epoch * config.num_epochs}")
    
    # Initialize paths - handle HuggingFace adapter download
    current_adapter_path = None
    if config.sft_adapter:
        if os.path.isdir(config.sft_adapter):
            # Local path
            current_adapter_path = config.sft_adapter
            logger.info(f"📦 Using local SFT adapter: {current_adapter_path}")
        elif "/" in config.sft_adapter and not config.sft_adapter.startswith("/"):
            # HuggingFace repo ID (e.g., "orlandowhite/qwen3_sft")
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
    
    # Initialize vLLM server manager
    vllm_manager = VLLMServerManager(config, logger)
    
    # Prepare benchmark held-out set (if enabled)
    benchmark_rulings = []
    if config.benchmark_every_n_batches > 0 and config.benchmark_num_rulings > 0:
        # Use a fixed random seed for reproducible benchmark set
        benchmark_rng = random.Random(42)
        benchmark_rulings = benchmark_rng.sample(
            rulings, 
            min(config.benchmark_num_rulings, len(rulings))
        )
        logger.info(f"\n📋 Benchmark set: {len(benchmark_rulings)} held-out rulings")
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
    if benchmark_rulings and config.benchmark_every_n_batches > 0:
        logger.info(f"\n{'=' * 70}")
        logger.info("📊 BASELINE BENCHMARK (before RL training)")
        logger.info(f"{'=' * 70}")
        
        # Start vLLM with initial adapter for baseline
        if not vllm_manager.start_server(
            model_path=config.vllm_model or config.base_model,
            lora_path=current_adapter_path,
            lora_name="adapter"
        ):
            logger.error("Failed to start vLLM for baseline benchmark!")
        else:
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
    
    # Rolling buffer for stable accuracy monitoring
    # Stores samples from the last N rulings
    accuracy_rolling_samples = []
    accuracy_rolling_rulings_seen = set()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        epoch_samples_total = 0
        epoch_exact_matches = 0
        epoch_rulings_total = 0
        
        # Shuffle rulings at start of each epoch (for train_all mode)
        if config.train_all:
            epoch_rulings_order = rulings.copy()
            random.shuffle(epoch_rulings_order)
            logger.info(f"\n  Shuffled {len(epoch_rulings_order)} rulings for epoch {epoch + 1}")
        
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
            if config.train_all:
                start_idx = batch_num * config.rulings_per_batch
                end_idx = min(start_idx + config.rulings_per_batch, len(rulings))
                logger.info(f"BATCH {batch_num + 1}/{num_batches_per_epoch} (Global: {global_batch_num}) | Rulings {start_idx+1}-{end_idx}/{len(rulings)}")
            else:
                logger.info(f"BATCH {batch_num + 1}/{num_batches_per_epoch} (Global: {global_batch_num})")
            logger.info(f"{'─' * 50}")
            
            # VALIDATION: Log adapter chain for debugging
            if current_adapter_path:
                logger.info(f"🔗 ADAPTER CHAIN: Using adapter from previous batch")
                logger.info(f"   Path: {current_adapter_path}")
                if os.path.isdir(current_adapter_path):
                    adapter_config = os.path.join(current_adapter_path, "adapter_config.json")
                    logger.info(f"   ✓ Adapter exists: {os.path.exists(adapter_config)}")
                else:
                    logger.warning(f"   ⚠️ Adapter path does not exist!")
            else:
                logger.info(f"🔗 ADAPTER CHAIN: Starting fresh (no previous adapter)")
            
            all_batch_samples = []
            batch_rulings = []
            
            if config.load_rollouts:
                # Load from file - skip vLLM
                logger.info(f"\n--- Loading cached rollouts (skipping vLLM) ---")
                all_batch_samples = load_rollouts_from_file(config.load_rollouts, logger)
                
                if not all_batch_samples:
                    logger.error("Failed to load rollouts from file!")
                    return
            else:
                # Select rulings for this batch
                if config.train_all:
                    # Sequential iteration through all rulings
                    start_idx = batch_num * config.rulings_per_batch
                    end_idx = min(start_idx + config.rulings_per_batch, len(epoch_rulings_order))
                    batch_rulings = epoch_rulings_order[start_idx:end_idx]
                    
                    if not batch_rulings:
                        logger.info(f"  No more rulings for batch {batch_num + 1}, skipping")
                        continue
                else:
                    # Random sampling mode
                    batch_rulings = random.sample(
                        rulings, 
                        min(config.rulings_per_batch, len(rulings))
                    )
                
                # =============================================
                # PHASE 1: START VLLM WITH BASE MODEL + LORA
                # =============================================
                logger.info(f"\n--- Phase 1: Starting vLLM server ---")
                vllm_serve_model = config.vllm_model or config.base_model
                logger.info(f"  vLLM model: {vllm_serve_model}")
                if current_adapter_path:
                    logger.info(f"  LoRA adapter: {current_adapter_path}")
                
                if not vllm_manager.start_server(
                    model_path=config.base_model, 
                    lora_path=current_adapter_path,
                    lora_name="adapter"
                ):
                    logger.error("Failed to start vLLM server!")
                    return
                
                vllm_client = VLLMInferenceClient(config, logger, vllm_manager)
                
                # =============================================
                # PHASE 2: RUN ROLLOUTS (PARALLEL)
                # =============================================
                logger.info(f"\n--- Phase 2: Running rollouts for {len(batch_rulings)} rulings ---")
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
                
                # =============================================
                # PHASE 3: STOP VLLM
                # =============================================
                logger.info(f"\n--- Phase 3: Stopping vLLM server ---")
                vllm_manager.stop_server()
                
                # Save rollouts if requested
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
                            "model": config.base_model,
                        }
                    )
            
            # =============================================
            # COMPUTE ACCURACY METRICS (Rolling Window)
            # =============================================
            # Add current samples to rolling buffer
            for s in all_batch_samples:
                accuracy_rolling_samples.append(s)
            
            # Keep track of unique rulings in buffer
            current_batch_gold_codes = {s.get("gold_code") for s in all_batch_samples if s.get("gold_code")}
            accuracy_rolling_rulings_seen.update(current_batch_gold_codes)
            
            # Prune rolling buffer to maintain window size
            # We group by gold_code and keep only the most recent 'accuracy_window_size' rulings
            rolling_by_ruling = {}
            for s in accuracy_rolling_samples:
                gold = s.get("gold_code", "unknown")
                # We want to keep the order of when they were added, but group them
                rolling_by_ruling.setdefault(gold, []).append(s)
            
            # Get the list of gold codes in the order they were first seen in the buffer
            # Actually, to be a true sliding window, we take the last N unique gold codes
            unique_gold_codes = []
            for s in accuracy_rolling_samples:
                g = s.get("gold_code")
                if g and g not in unique_gold_codes:
                    unique_gold_codes.append(g)
            
            if len(unique_gold_codes) > config.accuracy_window_size:
                # Keep only the last N unique rulings
                gold_codes_to_keep = unique_gold_codes[-config.accuracy_window_size:]
                accuracy_rolling_samples = [s for s in accuracy_rolling_samples if s.get("gold_code") in gold_codes_to_keep]
                accuracy_rolling_rulings_seen = set(gold_codes_to_keep)
            
            # Compute accuracy on the rolling window
            accuracy_metrics = compute_batch_accuracy(accuracy_rolling_samples, logger)
            
            logger.info(f"\n📊 ROLLING ACCURACY (window={config.accuracy_window_size} rulings):")
            logger.info(f"  Exact match rate: {accuracy_metrics['exact_match_rate']:.1%} ({accuracy_metrics['num_exact_matches']}/{accuracy_metrics['num_rulings']})")
            logger.info(f"  Avg best reward: {accuracy_metrics['avg_best_reward']:.4f}")
            logger.info(f"  Avg reward: {accuracy_metrics['avg_reward']:.4f}")
            
            # Current batch-specific accuracy for logging (not used for wandb tracking)
            current_batch_accuracy = compute_batch_accuracy(all_batch_samples, logger)
            
            # Update epoch totals based on actual batch rulings (not rolling window)
            epoch_samples_total += len(all_batch_samples)
            epoch_exact_matches += current_batch_accuracy['num_exact_matches']
            epoch_rulings_total += current_batch_accuracy['num_rulings']
            
            # Debug: save samples and display stats
            if all_batch_samples:
                save_samples_for_debug(
                    all_batch_samples,
                    config,
                    logger,
                    epoch=epoch + 1,
                    ruling_desc=f"e{epoch+1}_b{batch_num+1}",
                )
                display_rollout_stats(all_batch_samples, logger)
            
            # =============================================
            # PHASE 4: TRAIN WITH UNSLOTH (NO INTERNAL VLLM)
            # =============================================
            if not all_batch_samples:
                logger.warning(f"No samples collected for batch {batch_num + 1}, skipping training")
                continue
            
            logger.info(f"\n--- Phase 4: Training on {len(all_batch_samples)} samples ---")
            
            # VALIDATION: Log which adapter training will resume from
            if current_adapter_path:
                logger.info(f"  📥 Training will RESUME from: {current_adapter_path}")
            else:
                logger.info(f"  📥 Training will START FRESH (no adapter to resume)")
            
            # Verify CUDA is ready
            logger.info("  Verifying CUDA context for Unsloth...")
            if torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1, device="cuda")
                    del test_tensor
                    torch.cuda.synchronize()
                    
                    free_mem = torch.cuda.mem_get_info()[0] / 1e9
                    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                    logger.info(f"  ✓ CUDA ready: {free_mem:.1f}GB free of {total_mem:.1f}GB total")
                    
                    if free_mem < 20:
                        logger.warning(f"  ⚠️ Low GPU memory - waiting for cleanup...")
                        wait_for_gpu_memory(logger, target_free_gb=50.0, timeout=120)
                        
                except Exception as e:
                    logger.error(f"  ❌ CUDA verification failed: {e}")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            new_adapter_path, train_metrics = train_on_samples(
                all_batch_samples,
                config,
                logger,
                adapter_path=current_adapter_path,
            )
            
            # CRITICAL: Ensure GPU memory is fully released before next batch's vLLM
            logger.info("  Ensuring GPU memory is fully released after training...")
            gc.collect()
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
            
            # Wait for memory to stabilize
            time.sleep(5)
            free_gpu_memory(logger)
            
            # =============================================
            # PHASE 5: UPDATE PATHS
            # =============================================
            previous_adapter_path = current_adapter_path  # For logging
            current_adapter_path = new_adapter_path
            
            avg_loss = train_metrics.get("avg_loss", 0)
            
            logger.info(f"\n--- Phase 5: Model updated ---")
            logger.info(f"  📤 Previous adapter: {previous_adapter_path or 'None (fresh start)'}")
            logger.info(f"  📥 New LoRA adapter: {new_adapter_path}")
            logger.info(f"  ✓ Next batch will use: {current_adapter_path}")
            
            # Record batch metrics
            batch_metrics = {
                "epoch": epoch + 1,
                "batch": batch_num + 1,
                "global_batch": global_batch_num,
                "num_rulings": len(batch_rulings),
                "num_samples": train_metrics.get("num_samples", 0),
                "avg_loss": avg_loss,
                "exact_match_rate": accuracy_metrics['exact_match_rate'],
                "num_exact_matches": accuracy_metrics['num_exact_matches'],
                "avg_best_reward": accuracy_metrics['avg_best_reward'],
                "avg_reward": accuracy_metrics['avg_reward'],
                "adapter_path": new_adapter_path,
                "skipped_truncated": train_metrics.get("skipped_truncated", 0),
                "skipped_error": train_metrics.get("skipped_error", 0),
            }
            all_metrics.append(batch_metrics)
            
            # Log to wandb with comprehensive metrics
            wandb_metrics = {
                # Training metrics
                "loss": avg_loss,
                "num_samples": train_metrics.get("num_samples", 0),
                "num_rulings": len(batch_rulings),
                "skipped_truncated": train_metrics.get("skipped_truncated", 0),
                "skipped_error": train_metrics.get("skipped_error", 0),
                # Current batch accuracy (noisy but immediate signal)
                "batch_exact_match_rate": current_batch_accuracy['exact_match_rate'],
                "batch_avg_best_reward": current_batch_accuracy['avg_best_reward'],
            }
            
            # Log multiple rolling window accuracies for diagnostic comparison
            for window_size in config.accuracy_window_sizes:
                # Compute accuracy over this window size
                gold_codes_for_window = unique_gold_codes[-window_size:] if len(unique_gold_codes) >= window_size else unique_gold_codes
                samples_for_window = [s for s in accuracy_rolling_samples if s.get("gold_code") in gold_codes_for_window]
                
                if samples_for_window:
                    window_metrics = compute_batch_accuracy(samples_for_window, logger, quiet=True)
                    wandb_metrics[f"rolling_{window_size}_exact_match_rate"] = window_metrics['exact_match_rate']
                    wandb_metrics[f"rolling_{window_size}_avg_best_reward"] = window_metrics['avg_best_reward']
                    wandb_metrics[f"rolling_{window_size}_avg_reward"] = window_metrics['avg_reward']
            
            # Reward distribution for current batch (diagnostic)
            if all_batch_samples:
                rewards = [s.get("reward", 0) for s in all_batch_samples]
                best_rewards_per_ruling = {}
                for s in all_batch_samples:
                    gold = s.get("gold_code", "unknown")
                    r = s.get("reward", 0)
                    best_rewards_per_ruling[gold] = max(best_rewards_per_ruling.get(gold, 0), r)
                
                wandb_metrics["reward_mean"] = sum(rewards) / len(rewards) if rewards else 0
                wandb_metrics["reward_max"] = max(rewards) if rewards else 0
                wandb_metrics["reward_min"] = min(rewards) if rewards else 0
                wandb_metrics["reward_std"] = (sum((r - wandb_metrics["reward_mean"])**2 for r in rewards) / len(rewards))**0.5 if len(rewards) > 1 else 0
                
                best_rs = list(best_rewards_per_ruling.values())
                wandb_metrics["best_reward_mean"] = sum(best_rs) / len(best_rs) if best_rs else 0
            
            log_to_wandb(wandb_run, wandb_metrics, step=global_batch_num, prefix="train")
            
            # =============================================
            # BENCHMARK EVALUATION (every N batches)
            # =============================================
            if (config.benchmark_every_n_batches > 0 and 
                benchmark_rulings and 
                global_batch_num % config.benchmark_every_n_batches == 0):
                
                logger.info(f"\n--- Running Benchmark Evaluation (batch {global_batch_num}) ---")
                
                # Start vLLM with current adapter for benchmark
                if not vllm_manager.is_running():
                    vllm_manager.start_server(
                        model_path=config.vllm_model or config.base_model,
                        lora_path=current_adapter_path,
                        lora_name="adapter"
                    )
                    benchmark_client = VLLMInferenceClient(config, logger, vllm_manager)
                else:
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
            
            # Batch complete
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
        
        # Epoch complete
        epoch_time = time.time() - epoch_start
        epoch_exact_match_rate = epoch_exact_matches / epoch_rulings_total if epoch_rulings_total > 0 else 0.0
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch + 1} COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"  Batches: {config.num_batches}")
        logger.info(f"  Total rulings: {epoch_rulings_total}")
        logger.info(f"  Total samples: {epoch_samples_total}")
        logger.info(f"  Epoch exact match rate: {epoch_exact_match_rate:.1%} ({epoch_exact_matches}/{epoch_rulings_total})")
        logger.info(f"  Time: {epoch_time/60:.1f}m")
        
        # Log epoch summary to wandb
        log_to_wandb(wandb_run, {
            "exact_match_rate": epoch_exact_match_rate,
            "num_exact_matches": epoch_exact_matches,
            "total_rulings": epoch_rulings_total,
            "total_samples": epoch_samples_total,
        }, step=global_batch_num, prefix="epoch")
        
        # Save checkpoint at end of epoch
        if (epoch + 1) % config.save_every_n_epochs == 0 and current_adapter_path:
            import shutil
            checkpoint_dir = os.path.join(
                config.output_dir, 
                f"checkpoint-epoch-{epoch + 1}"
            )
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            shutil.copytree(current_adapter_path, checkpoint_dir)
            logger.info(f"  Checkpoint saved: {checkpoint_dir}")
    
    # Training complete
    total_time = time.time() - training_start
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total batches: {global_batch_num}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    if all_metrics:
        final_exact_match = all_metrics[-1].get('exact_match_rate', 0)
        logger.info(f"Final batch exact match: {final_exact_match:.1%}")
        logger.info(f"Final batch loss: {all_metrics[-1].get('avg_loss', 0):.4f}")
    
    # Save final models
    import shutil
    if current_adapter_path:
        final_adapter_dir = os.path.join(config.output_dir, "final_adapter")
        if os.path.exists(final_adapter_dir):
            shutil.rmtree(final_adapter_dir)
        shutil.copytree(current_adapter_path, final_adapter_dir)
        logger.info(f"Final adapter saved: {final_adapter_dir}")
    
    # Save training metrics
    metrics_file = os.path.join(config.output_dir, "training_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_file}")
    
    # Close wandb
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TreeRL GRPO Training with Qwen3-14B (External vLLM + Unsloth)"
    )
    
    # Model args
    parser.add_argument("--base-model", type=str, 
                       default="Qwen/Qwen3-14B",
                       help="Base model for training (Unsloth)")
    parser.add_argument("--vllm-model", type=str, 
                       default="",
                       help="Model for vLLM inference (default: same as base-model). Use FP8 variant for faster inference, e.g. Qwen/Qwen3-14B-FP8")
    parser.add_argument("--sft-adapter", type=str, 
                       default="treerl_checkpoints/adapter_sync/adapter_1768520829",
                       help="Starting LoRA adapter path")
    parser.add_argument("--max-seq-length", type=int, default=131072,
                       help="Max context length for vLLM (131072 with YaRN, 32768 native)")
    parser.add_argument("--train-max-seq-length", type=int, default=131072,
                       help="Max sequence length per training sample")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization for training")
    
    # YaRN rope scaling for extended context
    parser.add_argument("--yarn-factor", type=float, default=4.0,
                       help="YaRN scaling factor (4.0 for 128k context, 0 to disable)")
    
    # Qwen3 thinking mode
    parser.add_argument("--no-thinking", action="store_true",
                       help="Disable Qwen3 thinking mode")
    parser.add_argument("--json-thinking", action="store_true",
                       help="Enable thinking for JSON responses (more tokens but may improve quality)")
    
    # vLLM args
    parser.add_argument("--vllm-port", type=int, default=8000,
                       help="vLLM server port")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.80,
                       help="vLLM GPU memory utilization")
    parser.add_argument("--no-vllm-lora", action="store_true",
                       help="Disable vLLM LoRA support")
    parser.add_argument("--vllm-reasoning-parser", type=str, default="",
                       help="vLLM reasoning parser (leave empty for compatibility, or try 'deepseek_r1')")
    
    # LoRA args
    parser.add_argument("--lora-rank", type=int, default=64,
                       help="LoRA rank (must match SFT adapter)")
    parser.add_argument("--lora-alpha", type=int, default=128,
                       help="LoRA alpha (typically 2x rank)")
    
    # Training args
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # Batch/Epoch structure
    parser.add_argument("--rulings-per-batch", type=int, default=5,
                       help="Number of rulings per batch")
    parser.add_argument("--accuracy-window", type=int, default=10,
                       help="Track accuracy over the last N rulings (sliding window)")
    parser.add_argument("--num-batches", type=int, default=20,
                       help="Number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--start-batch", type=int, default=0,
                       help="Skip to this batch number (for resuming mid-training)")

    # Advantage normalization
    parser.add_argument("--advantage-method", type=str, default="gdpo",
                       choices=["none", "grpo", "grpo_no_std", "gdpo"],
                       help="Advantage normalization method")
    parser.add_argument("--gdpo-reward-weights", type=str, default="1.0,1.0",
                       help="GDPO reward component weights")
    parser.add_argument("--leaf-reward-weights", type=str, default="0.85,0.15",
                       help="Leaf reward aggregation weights")
    parser.add_argument("--gdpo-eps", type=float, default=1e-6,
                       help="GDPO normalization epsilon")
    parser.add_argument("--no-gdpo-scaling", action="store_true",
                       help="Disable GDPO advantage scaling (uses raw TreeRL R(s) only)")
    
    # Data args
    parser.add_argument("--chapter", type=str, default="84",
                       help="HTS chapter to train on")
    parser.add_argument("--cross-rulings-file", type=str,
                       default="cross_rulings_dataset.json",
                       help="Path to cross rulings JSON")
    parser.add_argument("--train-all", action="store_true",
                       help="Train on ALL rulings in chapter (not random sampling)")
    
    # Wandb args
    parser.add_argument("--wandb", action="store_true",
                       help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="treerl-grpo",
                       help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default="",
                       help="Wandb run name (auto-generated if empty)")
    parser.add_argument("--wandb-entity", type=str, default="",
                       help="Wandb entity (username/team)")
    parser.add_argument("--wandb-resume", action="store_true",
                       help="Resume a previous wandb run")
    parser.add_argument("--wandb-run-id", type=str, default="",
                       help="Wandb run ID to resume (e.g., 'fr6uebpz')")
    
    # TreeRL args
    parser.add_argument("--beam-size", type=int, default=4,
                       help="Beam size for rollouts")
    parser.add_argument("--max-questions", type=int, default=5,
                       help="Max Q&A turns per rollout")
    parser.add_argument("--parallel-rollouts", type=int, default=8,
                       help="Number of rulings to process concurrently (vLLM batches internally)")
    
    # Benchmark evaluation
    parser.add_argument("--benchmark-every", type=int, default=0,
                       help="Run benchmark evaluation every N batches (0 to disable)")
    parser.add_argument("--benchmark-rulings", type=int, default=0,
                       help="Number of held-out rulings for benchmark evaluation")
    
    # Output args
    parser.add_argument("--output-dir", type=str, default="treerl_checkpoints",
                       help="Output directory")
    
    # Rollout caching
    parser.add_argument("--save-rollouts", type=str, default="",
                       help="Save rollout samples to file")
    parser.add_argument("--load-rollouts", type=str, default="",
                       help="Load rollout samples from file (skip vLLM)")
    
    args = parser.parse_args()
    
    # Build config
    # vllm_model can be FP8 variant for faster inference (e.g., Qwen/Qwen3-14B-FP8)
    # LoRA adapters remain compatible as long as base architecture matches
    config = TreeRLConfig(
        base_model=args.base_model,
        train_model=args.base_model,
        vllm_model=args.vllm_model if args.vllm_model else args.base_model,
        sft_adapter=args.sft_adapter,
        max_seq_length=args.max_seq_length,
        vllm_max_model_len=args.max_seq_length,
        train_max_seq_length=args.train_max_seq_length,
        load_in_4bit=not args.no_4bit,
        yarn_factor=args.yarn_factor,
        enable_thinking=not args.no_thinking,
        json_thinking=args.json_thinking,
        vllm_port=args.vllm_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_util,
        vllm_enable_lora=not args.no_vllm_lora,
        vllm_reasoning_parser=args.vllm_reasoning_parser,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        rulings_per_batch=args.rulings_per_batch,
        accuracy_window_size=args.accuracy_window,
        num_batches=args.num_batches,
        num_epochs=args.epochs,
        start_batch=args.start_batch,
        advantage_method=args.advantage_method,
        gdpo_reward_weights=tuple(float(x.strip()) for x in args.gdpo_reward_weights.split(",") if x.strip()),
        leaf_reward_weights=tuple(float(x.strip()) for x in args.leaf_reward_weights.split(",") if x.strip()),
        gdpo_eps=args.gdpo_eps,
        apply_leaf_advantage_scaling=not args.no_gdpo_scaling,
        chapter=args.chapter,
        cross_rulings_file=args.cross_rulings_file,
        train_all=args.train_all,
        beam_size=args.beam_size,
        max_questions=args.max_questions,
        parallel_rollouts=args.parallel_rollouts,
        benchmark_every_n_batches=args.benchmark_every,
        benchmark_num_rulings=args.benchmark_rulings,
        output_dir=args.output_dir,
        save_rollouts=args.save_rollouts,
        load_rollouts=args.load_rollouts,
        # Wandb
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        wandb_resume=args.wandb_resume,
        wandb_run_id=args.wandb_run_id,
        # CRITICAL: External vLLM architecture
        use_fast_inference=False,
    )
    
    train(config)


if __name__ == "__main__":
    main()