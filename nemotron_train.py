#!/usr/bin/env python3
# =============================================================================
# ENVIRONMENT VARIABLES - MUST BE SET BEFORE ANY IMPORTS
# =============================================================================
import os

os.environ.setdefault("UNSLOTH_ENABLE_LOGGING", "1")

# We manage vLLM externally
os.environ["UNSLOTH_VLLM_STANDBY"] = "0"

# Faster HF downloads if available
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Explicitly set vLLM device type (helps in some containers)
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")

# =============================================================================
# NOW SAFE TO IMPORT
# =============================================================================
"""
TreeRL GRPO Training Script with NVIDIA Nemotron-3-Nano-30B-A3B (vLLM + Unsloth)

ARCHITECTURE (NO vLLM LoRA):
- vLLM runs as external server for rollouts (high throughput inference)
- Unsloth loads separately for training (fast_inference=False)
- Since vLLM LoRA is not supported/used for this model, we:
    * Merge the latest LoRA adapter into a BF16 base checkpoint
    * Start vLLM on the merged checkpoint directory
    * Stop vLLM before training
    * Train next LoRA adapter
    * Repeat

Nemotron-3-Nano reasoning:
- Uses <think>...</think> in assistant outputs
- Thinking mode control via flags in user turns:
    /think or /no_think

vLLM Nemotron Nano 3 recommended flags (per NVIDIA guide):
- --async-scheduling
- FP8: --kv-cache-dtype fp8 + FlashInfer MoE FP8 env vars
- Optional reasoning parser plugin:
    --reasoning-parser-plugin nano_v3_reasoning_parser.py --reasoning-parser nano_v3
"""

import sys
import json
import math
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
    """Training configuration for TreeRL GRPO with Nemotron-3-Nano."""

    # ----------------------------
    # Base + training model IDs
    # ----------------------------
    # vLLM should serve NVIDIA BF16/FP8 repos or a local merged checkpoint dir.
    base_model_bf16: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    base_model_fp8: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

    # Unsloth training model (Unsloth supports fine-tuning Nemotron).
    # NOTE: We use the official BF16 model because 4-bit quantization 
    # is incompatible with Nemotron's Mamba2 layers.
    train_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    # Optional starting adapter to warm-start from (LoRA dir)
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
    train_max_seq_length: int = 65536

    # Quantization for training
    # NOTE: 4-bit disabled by default for Nemotron because Mamba2 layers
    # are incompatible with bitsandbytes 4-bit quantization
    load_in_4bit: bool = False

    # ----------------------------
    # Thinking / reasoning controls
    # ----------------------------
    enable_thinking: bool = True  # appends /think vs /no_think to user turns

    # ----------------------------
    # Generation sampling (NVIDIA recs)
    # ----------------------------
    # General: temp=1.0 top_p=1.0
    # Tool calling: temp=0.6 top_p=0.95
    rollout_temperature: float = 0.6
    rollout_top_p: float = 0.95

    # Cap for output tokens (actual max_tokens computed dynamically)
    rollout_max_new_tokens_cap: int = 16384

    # ----------------------------
    # vLLM server settings
    # ----------------------------
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_tensor_parallel_size: int = 1
    vllm_max_num_seqs: int = 8

    # Use a stable served name so client always uses same "model" string
    vllm_served_model_name: str = "nemotron3nano"

    # Optional reasoning parser plugin usage
    vllm_use_reasoning_parser: bool = True
    vllm_reasoning_parser_name: str = "nano_v3"
    vllm_reasoning_parser_plugin_filename: str = "nano_v3_reasoning_parser.py"

    # Tool calling flags (safe to keep on even if not used)
    vllm_enable_auto_tool_choice: bool = True
    vllm_tool_call_parser: str = "qwen3_coder"

    # ----------------------------
    # LoRA settings for training
    # ----------------------------
    lora_rank: int = 32
    lora_alpha: int = 64
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

    # Advantage shaping / normalization
    advantage_method: str = "gdpo"  # none | grpo | grpo_no_std | gdpo
    gdpo_reward_weights: Tuple[float, ...] = (1.0, 1.0)

    # TreeRL settings
    beam_size: int = 4
    max_questions: int = 3

    # Data settings
    chapter: str = "84"
    rulings_per_batch: int = 5
    accuracy_window_size: int = 10
    num_batches: int = 20
    num_epochs: int = 3
    train_all: bool = False

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


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: TreeRLConfig) -> logging.Logger:
    logger = logging.getLogger("treerl_train")
    logger.setLevel(logging.DEBUG)

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
    """Manages vLLM server lifecycle (NO LoRA)."""

    def __init__(self, config: TreeRLConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{config.vllm_host}:{config.vllm_port}"
        self._log_thread: Optional[threading.Thread] = None
        self._stop_logging = threading.Event()
        self._log_file_path = os.path.join(config.output_dir, "vllm_server.log")

    def _ensure_reasoning_parser_plugin(self) -> Optional[str]:
        """
        Ensure nano_v3_reasoning_parser.py exists locally.
        If missing, we DON'T hard-fail; we just skip plugin flags.
        """
        if not self.config.vllm_use_reasoning_parser:
            return None

        plugin_path = os.path.join(self.config.output_dir, self.config.vllm_reasoning_parser_plugin_filename)
        if os.path.exists(plugin_path):
            return plugin_path

        # Try to download from HF URL (as per NVIDIA docs)
        # This is best-effort; if it fails, we'll run without plugin flags.
        url = (
            "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/"
            "resolve/main/nano_v3_reasoning_parser.py"
        )
        self.logger.info(f"  Reasoning parser plugin not found, attempting download: {url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, plugin_path)
            self.logger.info(f"  ✓ Downloaded plugin to: {plugin_path}")
            return plugin_path
        except Exception as e:
            self.logger.warning(f"  ⚠️ Could not download reasoning parser plugin ({e}). Continuing without it.")
            return None

    def start_server(self, model_to_serve: str) -> bool:
        """
        Start vLLM server serving model_to_serve (HF id or local dir).
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

        cmd = [
            "vllm", "serve", model_to_serve,
            "--host", self.config.vllm_host,
            "--port", str(self.config.vllm_port),
            "--trust-remote-code",
            "--async-scheduling",
            "--kv-cache-dtype", kv_cache_dtype,
            "--tensor-parallel-size", str(self.config.vllm_tensor_parallel_size),
            "--max-num-seqs", str(self.config.vllm_max_num_seqs),
            "--max-model-len", str(self.config.vllm_max_model_len),
            "--served-model-name", self.config.vllm_served_model_name,
        ]

        # Optional tool-calling + reasoning parser plugin flags (per NVIDIA guide)
        if self.config.vllm_enable_auto_tool_choice:
            cmd.extend(["--enable-auto-tool-choice"])
        if self.config.vllm_tool_call_parser:
            cmd.extend(["--tool-call-parser", self.config.vllm_tool_call_parser])

        plugin_path = self._ensure_reasoning_parser_plugin()
        if plugin_path and self.config.vllm_reasoning_parser_name:
            cmd.extend([
                "--reasoning-parser-plugin", plugin_path,
                "--reasoning-parser", self.config.vllm_reasoning_parser_name,
            ])

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
# VLLM INFERENCE CLIENT (Nemotron thinking)
# ============================================================================

class VLLMInferenceClient:
    """vLLM-based LLM client (Nemotron-3-Nano)."""

    def __init__(self, config: TreeRLConfig, logger: logging.Logger, server_manager: VLLMServerManager):
        self.config = config
        self.logger = logger
        self.server_manager = server_manager
        self.base_url = f"http://{config.vllm_host}:{config.vllm_port}"

        try:
            from api.system_prompts_updated import UNIFIED_SYSTEM_PROMPT
            self.system_prompt = UNIFIED_SYSTEM_PROMPT.strip()
        except ImportError:
            self.system_prompt = "You are a helpful assistant specialized in HTS classification."

        self._system_prompt_injection: Optional[str] = None

        # Compatibility attributes expected by your engines
        self.log_prompts = False
        self.prompt_logger = logger
        self.client = None
        self.model_name = config.vllm_served_model_name

        self._json_requirements = (
            "\n\n=== OUTPUT FORMAT ===\n"
            "You MUST respond with ONLY a valid JSON object or array.\n"
            "Do NOT include any reasoning, explanation, or text before or after the JSON.\n"
            "Do NOT wrap the JSON in markdown code blocks.\n"
            "Start your response with [ or { and end with ] or }.\n"
            "===================\n"
        )
        self._max_json_retries = 4

        # Nemotron thinking tags
        self._think_start = "<think>"
        self._think_end = "</think>"

        # Token estimation (very rough, conservative)
        self._avg_chars_per_token = 3.5

    def set_system_prompt_injection(self, prompt: Optional[str]) -> None:
        self._system_prompt_injection = prompt

    def clear_system_prompt_injection(self) -> None:
        self._system_prompt_injection = None

    def _current_system_prompt(self) -> str:
        return (self._system_prompt_injection or self.system_prompt).strip()

    def _apply_thinking_flag(self, user_content: str) -> str:
        """
        Append /think or /no_think to the user prompt to control reasoning mode.
        """
        c = (user_content or "").rstrip()
        # Avoid duplicating flags
        if re.search(r"(^|\n)\s*/(no_?think|think)\s*$", c, flags=re.IGNORECASE):
            return c
        flag = "/think" if self.config.enable_thinking else "/no_think"
        return f"{c}\n{flag}"

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

    def _extract_thinking_and_content(self, response_text: str) -> Tuple[str, str]:
        if not response_text:
            return "", ""
        text = response_text.strip()
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
            if text.startswith(self._think_start):
                return text[len(self._think_start):].strip(), ""
            return "", text

    def _extract_json_from_response(self, response_text: str) -> str:
        if not response_text:
            raise ValueError("No response text to parse.")

        _, content = self._extract_thinking_and_content(response_text)
        text = content.strip() if content else response_text.strip()

        if "```json" in text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()
        elif "```" in text:
            match = re.search(r'```\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()

        try:
            json.loads(text)
            return text
        except Exception:
            pass

        bracket_positions = []
        for i, c in enumerate(text):
            if c == '[':
                bracket_positions.append(('array', i))
            elif c == '{':
                bracket_positions.append(('object', i))

        for _, start_pos in bracket_positions:
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
                    continue

        raise ValueError(f"Failed to extract valid JSON from response: {response_text[:500]}...")

    def _call_vllm_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int] = None,
        requires_json: bool = False,
    ) -> str:
        gen_start = time.time()
        url = f"{self.base_url}/v1/chat/completions"

        safe_max_tokens = self._calculate_max_tokens(messages, max_tokens)

        effective_temp = temperature if temperature > 0 else self.config.rollout_temperature
        effective_top_p = self.config.rollout_top_p

        payload = {
            "model": self.config.vllm_served_model_name,
            "messages": messages,
            "max_tokens": safe_max_tokens,
            "temperature": effective_temp,
            "top_p": effective_top_p,
        }

        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json()
            self._log_completion(payload, result)

            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {}) or {}
            finish_reason = choice.get("finish_reason", "unknown")

            usage = result.get("usage", {}) or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            gen_elapsed = time.time() - gen_start
            tok_per_sec = completion_tokens / gen_elapsed if gen_elapsed > 0 else 0.0

            self.logger.debug(
                f"    [vLLM] {completion_tokens} tokens in {gen_elapsed:.1f}s "
                f"({tok_per_sec:.1f} tok/s, prompt={prompt_tokens}, max={safe_max_tokens}, finish={finish_reason})"
            )
            if finish_reason == "length":
                self.logger.warning(f"  ⚠️ Hit max_tokens ({safe_max_tokens})!")

            # Nemotron reasoning parser plugin may return:
            # - message["content"] (final)
            # - message["reasoning_content"] (thinking)
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""

            if reasoning and content:
                output_text = f"{self._think_start}{reasoning}{self._think_end}{content}"
            elif content:
                output_text = content
            elif reasoning:
                output_text = f"{self._think_start}{reasoning}{self._think_end}"
            else:
                output_text = choice.get("text") or ""

            return (output_text or "").strip()

        except requests.exceptions.RequestException as e:
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
        user_content = (prompt or "").strip()

        # Thinking control flag
        user_content = self._apply_thinking_flag(user_content)

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
                    f"JSON extraction failed (attempt {attempt + 1}/{self._max_json_retries}): {str(e)[:120]}"
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
        **kwargs: Any,
    ) -> str:
        req_messages = [m.copy() for m in messages]

        # Append thinking control to the last user turn
        for i in range(len(req_messages) - 1, -1, -1):
            if req_messages[i].get("role") == "user":
                req_messages[i]["content"] = self._apply_thinking_flag(req_messages[i].get("content", ""))
                if requires_json:
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
                    f"JSON extraction failed (attempt {attempt + 1}/{self._max_json_retries}): {str(e)[:120]}"
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


def display_rollout_stats(samples: List[Dict], logger: logging.Logger) -> None:
    if not samples:
        logger.warning("No samples to display stats for")
        return

    logger.info("\n" + "=" * 70)
    logger.info("ROLLOUT STATISTICS (Before Training)")
    logger.info("=" * 70)

    by_ruling = {}
    for s in samples:
        gold = s.get("gold_code", "unknown")
        by_ruling.setdefault(gold, []).append(s)

    logger.info(f"\n📊 OVERVIEW")
    logger.info(f"  Total samples (beam paths): {len(samples)}")
    logger.info(f"  Unique rulings: {len(by_ruling)}")
    logger.info(f"  Avg paths per ruling: {len(samples) / max(len(by_ruling), 1):.1f}")

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
# TRAINING FUNCTIONS (Unsloth phase - NO vLLM)
# ============================================================================

def load_training_model(config: TreeRLConfig, logger: logging.Logger, adapter_path: Optional[str] = None):
    """
    Load Nemotron-3-Nano with Unsloth for training.
    """
    logger.info("=" * 70)
    logger.info("LOADING NEMOTRON-3-NANO WITH UNSLOTH (Training Mode)")
    logger.info("=" * 70)

    from unsloth import FastLanguageModel

    train_load_seq = config.train_max_seq_length
    train_model_name = config.train_model

    logger.info(f"Loading training model: {train_model_name}")
    logger.info(f"  seq_length={train_load_seq}, 4bit={config.load_in_4bit}")
    logger.info(f"  fast_inference=False (external vLLM architecture)")

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=train_model_name,
        max_seq_length=train_load_seq,
        dtype=torch.bfloat16,  # Force BF16 for Nemotron
        load_in_4bit=config.load_in_4bit,
        trust_remote_code=True,  # Required for Nemotron custom architecture
    )
    logger.info(f"  ✓ Base model loaded in {time.time() - t0:.1f}s")

    logger.info("  Attaching LoRA modules with Unsloth...")
    t0 = time.time()
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=list(config.lora_target_modules),
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        # NOTE: Gradient checkpointing disabled for Nemotron because Mamba2 CUDA
        # kernels are incompatible with Unsloth's gradient checkpointing
        use_gradient_checkpointing=False,
        random_state=3407,
    )
    logger.info(f"  ✓ LoRA attached in {time.time() - t0:.1f}s (rank={config.lora_rank}, alpha={config.lora_alpha})")

    # Resume adapter
    if adapter_path and os.path.isdir(adapter_path):
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            logger.info(f"  Resuming from adapter: {adapter_path}")
            t0 = time.time()
            loaded = False
            try:
                from peft import set_peft_model_state_dict
                adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
                if os.path.exists(adapter_weights_path):
                    from safetensors.torch import load_file
                    adapter_weights = load_file(adapter_weights_path)
                else:
                    adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
                    adapter_weights = torch.load(adapter_weights_path, map_location="cpu", weights_only=True)
                set_peft_model_state_dict(model, adapter_weights)
                loaded = True
                logger.info(f"  ✓ Adapter weights loaded via PEFT in {time.time() - t0:.1f}s")
            except Exception as e:
                logger.warning(f"    PEFT load failed: {e}, trying direct state_dict load...")

            if not loaded:
                try:
                    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
                    if os.path.exists(adapter_weights_path):
                        from safetensors.torch import load_file
                        adapter_weights = load_file(adapter_weights_path)
                    else:
                        adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
                        adapter_weights = torch.load(adapter_weights_path, map_location="cpu", weights_only=True)
                    model.load_state_dict(adapter_weights, strict=False)
                    loaded = True
                    logger.info(f"  ✓ Adapter weights loaded via state_dict in {time.time() - t0:.1f}s")
                except Exception as e:
                    logger.error(f"    Failed to load adapter weights: {e}")

            if not loaded:
                logger.warning(f"  ⚠️ Could not load adapter weights, starting fresh")
        else:
            logger.warning(f"  Adapter path exists but no config found: {adapter_path}")

    # NOTE: Gradient checkpointing disabled for Nemotron (Mamba2 incompatible)
    FastLanguageModel.for_training(model, use_gradient_checkpointing=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    logger.info("✓ Training model ready (Nemotron-3-Nano + Unsloth)")
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


def _get_path_depth(user_content: str) -> int:
    match = re.search(r'"path_so_far":\s*"([^"]+)"', user_content)
    if not match:
        return -1
    path = match.group(1)
    return path.count(' > ') + 1


def build_token_weights(
    step_rewards: List[Dict],
    boundaries: List[Tuple[int, int]],
    seq_len: int,
    device: str = "cuda",
    leaf_reward: Optional[float] = None,
    messages: Optional[List[Dict]] = None,
) -> torch.Tensor:
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
    for msg in messages:
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
                R = 0.0
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
    assert model.training
    assert torch.is_grad_enabled()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

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
) -> Tuple[str, Dict[str, float]]:
    logger.info(f"Training on {len(samples)} samples...")
    logger.info(f"  Train max seq length: {config.train_max_seq_length}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")

    model, tokenizer = load_training_model(config, logger, adapter_path)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    model.train()

    adv_by_path_id = _compute_leaf_advantages(samples, config, logger)

    metrics = {"total_loss": 0.0, "num_samples": 0, "skipped_truncated": 0, "skipped_error": 0}

    accumulated_loss = 0.0
    accumulated_steps = 0
    optimizer.zero_grad()

    for sample_idx, sample in enumerate(samples):
        messages = sample.get("messages", [])
        step_rewards = sample.get("step_rewards", [])
        leaf_reward = sample.get("leaf_reward", sample.get("reward", None))
        path_id = sample.get("path_id", f"idx_{sample_idx}")
        leaf_advantage = float(adv_by_path_id.get(path_id, 1.0))

        if not messages:
            continue

        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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

            if seq_len == config.train_max_seq_length:
                logger.warning(f"  Sample {sample_idx}: TRUNCATED at {seq_len} tokens - SKIPPING")
                metrics["skipped_truncated"] += 1
                continue

        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            metrics["skipped_error"] += 1
            continue

        boundaries = find_assistant_turn_boundaries(input_ids[0], tokenizer, messages)

        try:
            loss, _ = compute_grpo_loss(
                model,
                input_ids,
                attention_mask,
                step_rewards,
                boundaries,
                config.device,
                leaf_reward=leaf_reward,
                messages=messages,
            )

            if config.advantage_method and config.advantage_method.lower() != "none":
                loss = loss * leaf_advantage

            scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()

            accumulated_loss += float(loss.item())
            accumulated_steps += 1

            metrics["total_loss"] += float(loss.item())
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            avg_acc_loss = accumulated_loss / accumulated_steps
            logger.info(f"  Step {metrics['num_samples']}: loss={avg_acc_loss:.4f}")

            accumulated_loss = 0.0
            accumulated_steps = 0
            torch.cuda.empty_cache()

    if accumulated_steps > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    timestamp = int(time.time())
    new_adapter_path = os.path.join(config.adapter_sync_dir, f"adapter_{timestamp}")
    os.makedirs(new_adapter_path, exist_ok=True)
    model.save_pretrained(new_adapter_path)
    tokenizer.save_pretrained(new_adapter_path)
    logger.info(f"LoRA adapter saved to: {new_adapter_path}")

    metrics["avg_loss"] = metrics["total_loss"] / metrics["num_samples"] if metrics["num_samples"] else 0.0
    logger.info(
        f"  Training complete: {metrics['num_samples']} samples, "
        f"{metrics['skipped_truncated']} truncated, {metrics['skipped_error']} errors"
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

    logger.info("=" * 70)
    logger.info("TREERL GRPO TRAINING with Nemotron-3-Nano-30B-A3B")
    logger.info("=" * 70)
    logger.info(f"Train model (Unsloth): {config.train_model}")
    logger.info(f"Inference dtype: {config.inference_dtype}")
    logger.info(f"Thinking enabled: {config.enable_thinking}")
    logger.info(f"vLLM max len: {config.vllm_max_model_len}")
    logger.info(f"vLLM served name: {config.vllm_served_model_name}")
    logger.info(f"vLLM reasoning parser: {config.vllm_use_reasoning_parser}")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.adapter_sync_dir, exist_ok=True)
    os.makedirs(config.merged_models_dir, exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)

    wandb_run = init_wandb(config, logger)

    rulings = load_chapter_rulings(config, logger)
    if not rulings:
        logger.error(f"No rulings found for chapter {config.chapter}")
        return

    if config.train_all:
        num_batches_per_epoch = math.ceil(len(rulings) / config.rulings_per_batch)
    else:
        num_batches_per_epoch = config.num_batches

    current_adapter_path = config.sft_adapter if config.sft_adapter else None

    vllm_manager = VLLMServerManager(config, logger)

    training_start = time.time()
    all_metrics = []
    global_batch_num = 0

    accuracy_rolling_samples = []

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        epoch_samples_total = 0
        epoch_exact_matches = 0
        epoch_rulings_total = 0

        if config.train_all:
            epoch_rulings_order = rulings.copy()
            random.shuffle(epoch_rulings_order)

        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'=' * 70}")

        for batch_num in range(num_batches_per_epoch):
            global_batch_num += 1
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
                    batch_rulings = random.sample(rulings, min(config.rulings_per_batch, len(rulings)))

                # =========================================================
                # PHASE 1: PREPARE MODEL TO SERVE (MERGE if adapter exists)
                # =========================================================
                logger.info(f"\n--- Phase 1: Preparing model for vLLM ---")

                # Choose base model for merge/serve
                base_for_merge = config.base_model_bf16  # always merge into BF16
                base_for_serve = config.base_model_fp8 if config.inference_dtype.lower() == "fp8" else config.base_model_bf16

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
                if not vllm_manager.start_server(model_to_serve=model_to_serve):
                    logger.error("Failed to start vLLM server!")
                    return

                vllm_client = VLLMInferenceClient(config, logger, vllm_manager)

                # =========================================================
                # PHASE 3: RUN ROLLOUTS
                # =========================================================
                logger.info(f"\n--- Phase 3: Running rollouts for {len(batch_rulings)} rulings ---")
                for ruling_idx, ruling in enumerate(batch_rulings):
                    product_desc = ruling.get("short_product_description", "")[:50]
                    logger.info(f"\n  Ruling {ruling_idx+1}/{len(batch_rulings)}: {product_desc}...")
                    samples = run_online_rollout(ruling, config, logger, vllm_client)
                    if samples:
                        all_batch_samples.extend(samples)
                        logger.info(f"    → Collected {len(samples)} samples (total: {len(all_batch_samples)})")
                    else:
                        logger.warning(f"    → No samples collected")

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
                display_rollout_stats(all_batch_samples, logger)

            # =========================================================
            # PHASE 5: TRAIN WITH UNSLOTH
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


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TreeRL GRPO Training with Nemotron-3-Nano (merge->serve vLLM + Unsloth)"
    )

    # Nemotron model args
    parser.add_argument("--train-model", type=str, default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                        help="Training model for Unsloth")
    parser.add_argument("--base-model-bf16", type=str, default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                        help="BF16 base model for merging/serving")
    parser.add_argument("--base-model-fp8", type=str, default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                        help="FP8 base model for serving (not used for merges)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp8"],
                        help="vLLM inference dtype (serving). Merges are done into BF16.")

    parser.add_argument("--sft-adapter", type=str, default="", help="Starting LoRA adapter path (optional)")

    # Context lengths
    parser.add_argument("--vllm-max-len", type=int, default=262144, help="vLLM max model len")
    parser.add_argument("--train-max-seq-length", type=int, default=65536, help="Training max seq length")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization for training")

    # Thinking control
    parser.add_argument("--no-thinking", action="store_true", help="Append /no_think instead of /think")

    # vLLM args
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=8, help="vLLM max-num-seqs")
    parser.add_argument("--vllm-tp", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--served-model-name", type=str, default="nemotron3nano",
                        help="Stable model name used in OpenAI requests")

    parser.add_argument("--no-reasoning-parser", action="store_true",
                        help="Disable nano_v3 reasoning parser plugin flags")

    # LoRA args
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")

    # Training args
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")

    # Batch/Epoch structure
    parser.add_argument("--rulings-per-batch", type=int, default=5, help="Rulings per batch")
    parser.add_argument("--accuracy-window", type=int, default=10, help="Accuracy rolling window (unique rulings)")
    parser.add_argument("--num-batches", type=int, default=20, help="Batches per epoch")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")

    # Advantage normalization
    parser.add_argument("--advantage-method", type=str, default="gdpo",
                        choices=["none", "grpo", "grpo_no_std", "gdpo"])
    parser.add_argument("--gdpo-reward-weights", type=str, default="1.0,1.0")

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
    parser.add_argument("--max-questions", type=int, default=3)

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
        sft_adapter=args.sft_adapter,
        inference_dtype=args.dtype,
        vllm_max_model_len=args.vllm_max_len,
        train_max_seq_length=args.train_max_seq_length,
        load_in_4bit=not args.no_4bit,
        enable_thinking=not args.no_thinking,
        vllm_port=args.vllm_port,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_tensor_parallel_size=args.vllm_tp,
        vllm_served_model_name=args.served_model_name,
        vllm_use_reasoning_parser=not args.no_reasoning_parser,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        rulings_per_batch=args.rulings_per_batch,
        accuracy_window_size=args.accuracy_window,
        num_batches=args.num_batches,
        num_epochs=args.epochs,
        advantage_method=args.advantage_method,
        gdpo_reward_weights=tuple(float(x.strip()) for x in args.gdpo_reward_weights.split(",") if x.strip()),
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
