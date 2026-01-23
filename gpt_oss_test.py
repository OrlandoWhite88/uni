#!/usr/bin/env python3
# =============================================================================
# ENVIRONMENT VARIABLES - MUST BE SET BEFORE ANY IMPORTS
# =============================================================================
import os

# Faster HF downloads if available
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Explicitly set vLLM device type
os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")

# ============================================================================
# FIX: Force Unsloth to enable Flex Attention for GPT-OSS
# Bug: Unsloth checks for "gpt_oss" (underscore) but model name has "gpt-oss" (hyphen)
# Location: /usr/local/lib/python3.12/dist-packages/unsloth_zoo/temporary_patches/gpt_oss.py:741
# Without this fix, attention falls back to O(n²) eager mode causing OOM on long sequences
# MUST use hard assignment, not setdefault - Unsloth modifies this var!
# ============================================================================
os.environ["UNSLOTH_MODEL_NAME"] = "gpt_oss"  # FORCE - do not use setdefault!
os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "1"  # Enable flex attention

# PyTorch CUDA memory allocator settings to reduce fragmentation
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

# =============================================================================
# NOW SAFE TO IMPORT
# =============================================================================
"""
TreeRL GRPO Training Script with GPT-OSS 20B QLoRA (vLLM + Unsloth)

ARCHITECTURE:
- vLLM runs as external server for rollouts (high throughput inference with LoRA)
- Unsloth loads separately for training (2x faster, 60% less VRAM)
- vLLM serves base model with LoRA adapter via --enable-lora --lora-modules

GPT-OSS Features:
- Uses OpenAI Harmony format with tags: <|start|>, <|message|>, <|return|>
- Reasoning effort levels: low, medium, high (controlled via chat_template_kwargs)
- MoE architecture optimized for reasoning tasks

vLLM GPT-OSS recommended flags:
- --async-scheduling
- --tool-call-parser openai --enable-auto-tool-choice (for function calling)
- FP8 KV cache: --kv-cache-dtype fp8 (on Hopper/Blackwell)
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
    """Training configuration for TreeRL GRPO with GPT-OSS 20B."""

    # ----------------------------
    # Model IDs
    # ----------------------------
    # vLLM serves the base model, training uses Unsloth's quantized version
    base_model_vllm: str = "openai/gpt-oss-20b"  # For vLLM serving
    
    # Unsloth training models (4-bit quantized for QLoRA)
    train_model_4bit: str = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"  # BitsAndBytes 4-bit
    train_model_mxfp4: str = "unsloth/gpt-oss-20b"  # MXFP4 format
    
    # Choose quantization format: "bnb" (bitsandbytes) or "mxfp4"
    quantization_format: str = "bnb"

    # Optional starting adapter to warm-start from (empty = start from scratch)
    sft_adapter: str = ""

    # ----------------------------
    # Context lengths
    # ----------------------------
    # GPT-OSS supports up to 128k context
    vllm_max_model_len: int = 131072

    # Training max per sample (just below 128k context)
    train_max_seq_length: int = 126976

    # ----------------------------
    # Reasoning effort (GPT-OSS specific)
    # ----------------------------
    # Options: "low", "medium", "high"
    # Higher = more reasoning tokens = better quality but slower
    reasoning_effort: str = "medium"

    # ----------------------------
    # Generation sampling
    # ----------------------------
    # GPT-OSS recommended: temp=0.6, top_p=0.95 for tool calling
    # General: temp=1.0, top_p=1.0
    rollout_temperature: float = 0.6
    rollout_top_p: float = 0.95

    # Cap for output tokens
    rollout_max_new_tokens_cap: int = 16384

    # ----------------------------
    # vLLM server settings
    # ----------------------------
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_tensor_parallel_size: int = 1
    vllm_max_num_seqs: int = 8

    # Use a stable served name
    vllm_served_model_name: str = "gpt-oss-20b"

    # Tokenizer control for vLLM
    vllm_tokenizer: str = ""

    # Tool calling flags (GPT-OSS has built-in tool support)
    vllm_enable_auto_tool_choice: bool = True
    vllm_tool_call_parser: str = "openai"

    # KV cache dtype (fp8 for Hopper/Blackwell, auto otherwise)
    vllm_kv_cache_dtype: str = "auto"

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
    learning_rate: float = 2e-4  # Unsloth recommended
    weight_decay: float = 0.01
    warmup_steps: int = 5
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Advantage shaping / normalization
    advantage_method: str = "none"  # none | grpo | grpo_no_std | gdpo
    gdpo_reward_weights: Tuple[float, ...] = (1.0, 1.0)

    # TreeRL settings
    beam_size: int = 4
    max_questions: int = 3

    # Parallelization settings
    parallel_rollouts: int = 10

    # Benchmark evaluation settings
    benchmark_every_n_batches: int = 0
    benchmark_num_rulings: int = 50

    # Data settings
    chapter: str = "84"
    rulings_per_batch: int = 5
    accuracy_window_size: int = 10
    num_batches: int = 20
    num_epochs: int = 3
    train_all: bool = False
    start_batch: int = 0

    # Paths
    cross_rulings_file: str = "cross_rulings_dataset.json"
    output_dir: str = "gpt_oss_checkpoints"
    log_file: str = "gpt_oss_training.log"
    adapter_sync_dir: str = "gpt_oss_checkpoints/adapter_sync"
    samples_dir: str = "gpt_oss_checkpoints/samples"
    completions_log: str = "gpt_oss_checkpoints/completions.jsonl"

    # Rollout caching
    save_rollouts: str = ""
    load_rollouts: str = ""

    # Logging
    log_every_n_steps: int = 1
    save_every_n_epochs: int = 1

    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = "treerl-gpt-oss"
    wandb_run_name: str = ""
    wandb_entity: str = ""

    # Device
    device: str = "cuda"

    # Leaf reward shaping
    leaf_reward_weights: Tuple[float, ...] = (0.85, 0.15)
    leaf_reward_clip_0_1: bool = True

    # Safety margin for max_tokens calculation
    token_safety_margin: int = 512

    # vLLM LoRA serving
    use_vllm_lora: bool = True
    vllm_lora_name: str = "current_adapter"

    # Unsloth config
    unsloth_max_lora_rank: int = 64


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: TreeRLConfig) -> logging.Logger:
    logger = logging.getLogger("gpt_oss_train")
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
# VLLM SERVER MANAGEMENT
# ============================================================================

class VLLMServerManager:
    """Manages vLLM server lifecycle with LoRA support for GPT-OSS."""

    def __init__(self, config: TreeRLConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{config.vllm_host}:{config.vllm_port}"
        self._log_thread: Optional[threading.Thread] = None
        self._stop_logging = threading.Event()
        self._log_file_path = os.path.join(config.output_dir, "vllm_server.log")
        self._current_lora_path: Optional[str] = None

    def start_server(self, model_to_serve: str, lora_adapter_path: Optional[str] = None) -> bool:
        """
        Start vLLM server serving GPT-OSS model.
        
        Args:
            model_to_serve: Base model HF id or local path
            lora_adapter_path: Optional LoRA adapter path
        """
        if self.is_running():
            self.logger.info("vLLM server already running")
            return True

        self.logger.info("Starting vLLM server (GPT-OSS 20B)...")
        free_gpu_memory(self.logger)
        wait_for_gpu_memory(self.logger, target_free_gb=20.0, timeout=180)

        cmd = [
            "vllm", "serve", model_to_serve,
            "--host", self.config.vllm_host,
            "--port", str(self.config.vllm_port),
            "--trust-remote-code",
            "--async-scheduling",
            "--kv-cache-dtype", self.config.vllm_kv_cache_dtype,
            "--tensor-parallel-size", str(self.config.vllm_tensor_parallel_size),
            "--max-num-seqs", str(self.config.vllm_max_num_seqs),
            "--max-model-len", str(self.config.vllm_max_model_len),
            "--served-model-name", self.config.vllm_served_model_name,
        ]

        # Tokenizer control
        tokenizer_to_use = (self.config.vllm_tokenizer or "").strip()
        if not tokenizer_to_use and self.config.use_vllm_lora and lora_adapter_path:
            tokenizer_to_use = model_to_serve
        if tokenizer_to_use:
            cmd.extend(["--tokenizer", tokenizer_to_use])

        # vLLM LoRA support
        if self.config.use_vllm_lora and lora_adapter_path:
            self._current_lora_path = lora_adapter_path
            lora_module_spec = f"{self.config.vllm_lora_name}={lora_adapter_path}"
            cmd.extend([
                "--enable-lora",
                "--lora-modules", lora_module_spec,
                "--max-lora-rank", str(self.config.unsloth_max_lora_rank),
            ])
            self.logger.info(f"  LoRA enabled: {lora_module_spec}")
        else:
            self._current_lora_path = None

        # GPT-OSS tool calling support
        if self.config.vllm_enable_auto_tool_choice:
            cmd.extend(["--enable-auto-tool-choice"])
        if self.config.vllm_tool_call_parser:
            cmd.extend(["--tool-call-parser", self.config.vllm_tool_call_parser])

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
# VLLM INFERENCE CLIENT (GPT-OSS with Harmony format)
# ============================================================================

class VLLMInferenceClient:
    """vLLM-based LLM client for GPT-OSS with Harmony format support."""

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

        # Compatibility attributes
        self.log_prompts = False
        self.prompt_logger = logger
        self.client = None
        
        # Model name for API requests
        if config.use_vllm_lora and server_manager._current_lora_path:
            self.model_name = config.vllm_lora_name
        else:
            self.model_name = config.vllm_served_model_name

        # JSON retry settings
        self._max_json_retries = 4

        # Token estimation
        self._avg_chars_per_token = 3.5

    def set_system_prompt_injection(self, prompt: Optional[str]) -> None:
        self._system_prompt_injection = prompt

    def clear_system_prompt_injection(self) -> None:
        self._system_prompt_injection = None

    def _current_system_prompt(self) -> str:
        return (self._system_prompt_injection or self.system_prompt).strip()

    def _strip_harmony_tags(self, text: str) -> str:
        """
        Strip Harmony format tags from GPT-OSS output if needed.
        GPT-OSS uses tags like <|start|>, <|message|>, <|return|>
        """
        if not text:
            return text
        
        # Remove Harmony-specific tags
        patterns = [
            r'<\|start\|>',
            r'<\|message\|>',
            r'<\|return\|>',
            r'<\|end\|>',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()

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
        """Extract valid JSON from response text."""
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
    ) -> str:
        gen_start = time.time()
        url = f"{self.base_url}/v1/chat/completions"

        safe_max_tokens = self._calculate_max_tokens(messages, max_tokens)

        effective_temp = temperature
        effective_top_p = self.config.rollout_top_p

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": safe_max_tokens,
            "temperature": effective_temp,
            "top_p": effective_top_p,
        }
        
        # GPT-OSS reasoning effort can be controlled via chat_template_kwargs
        # This is handled by the tokenizer's chat template

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

            # GPT-OSS returns content and possibly reasoning_content
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            
            # Post-process: strip any Harmony tags
            content = self._strip_harmony_tags(content)
            
            content_stripped = content.strip()
            if content_stripped:
                return content_stripped
            else:
                self.logger.warning(f"    [vLLM] ⚠️ Empty content (reasoning exists: {bool(reasoning)})")
                return ""

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

        messages = [
            {"role": "system", "content": self._current_system_prompt()},
            {"role": "user", "content": user_content},
        ]

        last_error = None
        for attempt in range(self._max_json_retries if requires_json else 1):
            try:
                text = self._call_vllm_api(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
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
        **kwargs: Any,
    ) -> str:
        req_messages = [m.copy() for m in messages]

        last_error = None
        for attempt in range(self._max_json_retries if requires_json else 1):
            try:
                text = self._call_vllm_api(
                    req_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
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
# ONLINE ROLLOUT (vLLM phase)
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
    os.environ["COLLECT_TRAINING_DATA"] = "false"
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
        logger.info(f"  [rollout] Gold trace has {len(gold_trace)} steps for {gold_code}")

        if len(gold_trace) <= 2:
            logger.warning(f"  [rollout] ⚠️ SKIPPING - Gold code '{gold_code}' not found in HTS tree!")
            logger.warning(f"  [rollout]    Gold trace: {gold_trace}")
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
# TRAINING FUNCTIONS (Unsloth for GPT-OSS QLoRA)
# ============================================================================

def load_training_model(config: TreeRLConfig, logger: logging.Logger, adapter_path: Optional[str] = None):
    """
    Load GPT-OSS 20B with Unsloth for QLoRA training.
    Uses 4-bit quantization for memory efficiency.
    """
    # CRITICAL FIX: Reset env var to clean state right before import
    # Unsloth appends to this variable during import, causing corruption on re-imports
    # We must force it back to "gpt_oss" to pass the string check in Unsloth
    if "UNSLOTH_MODEL_NAME" in os.environ:
        logger.warning(f"  Cleaning corrupted UNSLOTH_MODEL_NAME: {os.environ['UNSLOTH_MODEL_NAME']}")
    
    os.environ["UNSLOTH_MODEL_NAME"] = "gpt_oss"
    os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "1"
    
    logger.info(f"  Forced clean env: UNSLOTH_MODEL_NAME={os.environ['UNSLOTH_MODEL_NAME']}")
    
    # Reloading unsloth is necessary if it was already imported with bad env vars
    import sys
    if 'unsloth' in sys.modules:
        logger.info("  Reloading unsloth module to apply env var fix...")
        import importlib
        import unsloth
        importlib.reload(unsloth)
    
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        logger.error("Unsloth not installed! Install with: pip install unsloth")
        raise ImportError("Unsloth required but not installed") from e

    # Choose model based on quantization format
    if config.quantization_format == "bnb":
        train_model_name = config.train_model_4bit
        load_in_4bit = True
        logger.info(f"Using BitsAndBytes 4-bit quantization")
    else:
        train_model_name = config.train_model_mxfp4
        load_in_4bit = True  # MXFP4 also uses load_in_4bit
        logger.info(f"Using MXFP4 quantization")
    
    logger.info(f"Loading training model with Unsloth: {train_model_name}")

    t0 = time.time()
    
    # Load with Unsloth's FastLanguageModel for GPT-OSS
    # CRITICAL: Must specify attn_implementation to actually USE Flash Attention
    # Otherwise it falls back to eager O(n²) attention causing OOM on long sequences
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=train_model_name,
        max_seq_length=config.train_max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,  # We want QLoRA
        dtype=None,  # Auto detection
        attn_implementation="flash_attention_2",  # Force FA2 to avoid O(n²) memory
        # token = "hf_...",  # If using gated models
    )
    
    logger.info(f"  Base model loaded with Unsloth in {time.time() - t0:.1f}s")
    
    # Verify Flex/Flash Attention setup
    try:
        attn_impl = getattr(model.config, '_attn_implementation', None)
        logger.info(f"  Attention config reports: {attn_impl or 'not set'}")
        
        # NOTE: Unsloth often reports 'eager' in config even when Flex Attention is active
        # The real check is whether the environment variables are set correctly
        
        current_model_name = os.environ.get("UNSLOTH_MODEL_NAME", "")
        current_flex = os.environ.get("UNSLOTH_ENABLE_FLEX_ATTENTION", "")
        
        if "gpt_oss" in current_model_name and current_flex == "1":
            logger.info(f"  ✓ Flex Attention environment active")
            logger.info(f"    UNSLOTH_MODEL_NAME={current_model_name}")
        else:
            logger.warning(f"  ⚠️ Flex Attention environment vars missing or incorrect!")
            logger.warning(f"    UNSLOTH_MODEL_NAME={current_model_name}")
            logger.warning(f"    UNSLOTH_ENABLE_FLEX_ATTENTION={current_flex}")
            
    except Exception as e:
        logger.debug(f"  Could not verify attention implementation: {e}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA with Unsloth's optimized method
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=list(config.lora_target_modules),
        lora_alpha=config.lora_alpha,
        lora_dropout=0,  # Unsloth optimizes for dropout=0
        bias="none",  # Unsloth optimizes for bias="none"
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    logger.info(f"  LoRA attached with Unsloth (rank={config.lora_rank})")
    logger.info(f"  ✓ Unsloth gradient checkpointing enabled (30% less VRAM)")

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
                logger.info(f"  ✓ Adapter weights loaded")
            except Exception as e:
                logger.warning(f"  Could not load adapter: {e}")

    # Enable training mode
    model.train()

    # Log trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

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
# Token-weighted GRPO loss
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
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    weights = torch.zeros(seq_len, device=device, dtype=dtype or torch.float32)
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
    logger: Optional[logging.Logger] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    assert model.training
    assert torch.is_grad_enabled()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    ignore_index = -100
    labels = shift_labels.masked_fill(shift_mask == 0, ignore_index)
    nll = F.cross_entropy(
        shift_logits.transpose(1, 2),
        labels,
        reduction="none",
        ignore_index=ignore_index,
    )

    token_log_probs = -nll

    adjusted_boundaries = [(max(0, s-1), max(0, e-1)) for s, e in boundaries]
    weights = build_token_weights(
        step_rewards,
        adjusted_boundaries,
        shift_labels.shape[1],
        device,
        leaf_reward=leaf_reward,
        messages=messages,
        dtype=token_log_probs.dtype,
    ).unsqueeze(0)

    masked_log_probs = token_log_probs
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
            # GPT-OSS uses Harmony format - apply chat template with reasoning effort
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False,
                # reasoning_effort=config.reasoning_effort,  # If supported by tokenizer
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
                logger=logger,
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
# BENCHMARK EVALUATION
# ============================================================================

def run_benchmark_evaluation(
    benchmark_rulings: List[Dict],
    config: TreeRLConfig,
    logger: logging.Logger,
    vllm_client: "VLLMInferenceClient",
    global_batch_num: int,
) -> Dict[str, float]:
    """Run evaluation on a held-out benchmark set."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"📋 BENCHMARK EVALUATION (batch {global_batch_num})")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Evaluating {len(benchmark_rulings)} held-out rulings...")

    all_benchmark_samples = []

    def run_single_rollout(ruling_idx_ruling):
        ruling_idx, ruling = ruling_idx_ruling
        product_desc = ruling.get("short_product_description", "")[:50]
        try:
            samples = run_online_rollout(ruling, config, logger, vllm_client)
            return ruling_idx, product_desc, samples, None
        except Exception as e:
            return ruling_idx, product_desc, [], str(e)

    with ThreadPoolExecutor(max_workers=config.parallel_rollouts) as executor:
        futures = {
            executor.submit(run_single_rollout, (idx, ruling)): idx
            for idx, ruling in enumerate(benchmark_rulings)
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            ruling_idx, product_desc, samples, error = future.result()

            if error:
                logger.debug(f"  Benchmark [{completed}/{len(benchmark_rulings)}]: {product_desc}... ❌ {error}")
            elif samples:
                all_benchmark_samples.extend(samples)

    benchmark_metrics = compute_batch_accuracy(all_benchmark_samples, logger)

    logger.info(f"\n📊 BENCHMARK RESULTS:")
    logger.info(f"  Exact match rate: {benchmark_metrics['exact_match_rate']:.1%} ({benchmark_metrics['num_exact_matches']}/{benchmark_metrics['num_rulings']})")
    logger.info(f"  Avg best reward: {benchmark_metrics['avg_best_reward']:.4f}")
    logger.info(f"  Avg reward: {benchmark_metrics['avg_reward']:.4f}")
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
        run_name = config.wandb_run_name or f"gpt-oss-ch{config.chapter}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
# MAIN TRAINING LOOP
# ============================================================================

def train(config: TreeRLConfig):
    logger = setup_logging(config)

    logger.info("=" * 70)
    logger.info("TREERL GRPO TRAINING with GPT-OSS 20B QLoRA")
    logger.info("=" * 70)
    logger.info(f"vLLM base model: {config.base_model_vllm}")
    logger.info(f"Training model: {config.train_model_4bit if config.quantization_format == 'bnb' else config.train_model_mxfp4}")
    logger.info(f"Quantization: {config.quantization_format}")
    logger.info(f"Reasoning effort: {config.reasoning_effort}")
    logger.info(f"vLLM max len: {config.vllm_max_model_len}")
    logger.info(f"vLLM served name: {config.vllm_served_model_name}")
    logger.info(f"Parallel rollouts: {config.parallel_rollouts}")
    logger.info(f"No SFT base: Starting from scratch")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.adapter_sync_dir, exist_ok=True)
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

    # No SFT adapter - start from scratch
    current_adapter_path = None
    if config.sft_adapter:
        logger.warning(f"SFT adapter specified but this config is for training from scratch")
        logger.warning(f"Ignoring sft_adapter: {config.sft_adapter}")

    vllm_manager = VLLMServerManager(config, logger)

    # Prepare benchmark held-out set
    benchmark_rulings = []
    if config.benchmark_every_n_batches > 0 and config.benchmark_num_rulings > 0:
        benchmark_rng = random.Random(42)
        benchmark_rulings = benchmark_rng.sample(
            rulings,
            min(config.benchmark_num_rulings, len(rulings))
        )
        logger.info(f"\n📋 Benchmark set: {len(benchmark_rulings)} held-out rulings")
        logger.info(f"   Will evaluate every {config.benchmark_every_n_batches} batches")

    training_start = time.time()
    all_metrics = []
    global_batch_num = config.start_batch

    if config.start_batch > 0:
        logger.info(f"\n⏭️  RESUMING from batch {config.start_batch}")

    # Initial baseline benchmark
    if benchmark_rulings and config.benchmark_every_n_batches > 0:
        logger.info(f"\n{'=' * 70}")
        logger.info("📊 BASELINE BENCHMARK (before RL training)")
        logger.info(f"{'=' * 70}")

        if vllm_manager.start_server(model_to_serve=config.base_model_vllm, lora_adapter_path=None):
            baseline_client = VLLMInferenceClient(config, logger, vllm_manager)

            baseline_metrics = run_benchmark_evaluation(
                benchmark_rulings,
                config,
                logger,
                baseline_client,
                global_batch_num=0,
            )

            log_to_wandb(wandb_run, {
                "exact_match_rate": baseline_metrics['exact_match_rate'],
                "num_exact_matches": baseline_metrics['num_exact_matches'],
                "avg_best_reward": baseline_metrics['avg_best_reward'],
                "avg_reward": baseline_metrics['avg_reward'],
                "num_rulings": baseline_metrics['num_rulings'],
            }, step=0, prefix="benchmark")

            vllm_manager.stop_server()
        else:
            logger.error("Failed to start vLLM for baseline benchmark!")

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
                logger.info(f"\n--- Loading cached rollouts ---")
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

                # Phase 1: Start vLLM
                logger.info(f"\n--- Phase 1: Starting vLLM server ---")
                
                lora_to_serve = current_adapter_path if config.use_vllm_lora else None
                if not vllm_manager.start_server(model_to_serve=config.base_model_vllm, lora_adapter_path=lora_to_serve):
                    logger.error("Failed to start vLLM server!")
                    return

                vllm_client = VLLMInferenceClient(config, logger, vllm_manager)

                # Phase 2: Run rollouts
                logger.info(f"\n--- Phase 2: Running rollouts for {len(batch_rulings)} rulings ---")

                def run_single_rollout(ruling_idx_ruling):
                    ruling_idx, ruling = ruling_idx_ruling
                    product_desc = ruling.get("short_product_description", "")[:50]
                    try:
                        samples = run_online_rollout(ruling, config, logger, vllm_client)
                        return ruling_idx, product_desc, samples, None
                    except Exception as e:
                        return ruling_idx, product_desc, [], str(e)

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
                            logger.info(f"  [{completed}/{len(batch_rulings)}] Ruling {ruling_idx+1}: {product_desc}... ✓ {len(samples)} samples")
                        else:
                            logger.warning(f"  [{completed}/{len(batch_rulings)}] Ruling {ruling_idx+1}: {product_desc}... ⚠️ No samples")

                # Phase 3: Stop vLLM
                logger.info(f"\n--- Phase 3: Stopping vLLM server ---")
                vllm_manager.stop_server()

                if config.save_rollouts and all_batch_samples:
                    base, ext = os.path.splitext(config.save_rollouts)
                    rollout_file = f"{base}_e{epoch + 1}_b{batch_num + 1}{ext}"
                    save_rollouts_to_file(all_batch_samples, rollout_file, logger)

            # Accuracy metrics
            for s in all_batch_samples:
                accuracy_rolling_samples.append(s)

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
            logger.info(f"  Exact match rate: {accuracy_metrics['exact_match_rate']:.1%}")
            logger.info(f"  Avg best reward: {accuracy_metrics['avg_best_reward']:.4f}")

            current_batch_accuracy = compute_batch_accuracy(all_batch_samples, logger)
            epoch_samples_total += len(all_batch_samples)
            epoch_exact_matches += current_batch_accuracy["num_exact_matches"]
            epoch_rulings_total += current_batch_accuracy["num_rulings"]

            if all_batch_samples:
                save_samples_for_debug(all_batch_samples, config, logger, epoch=epoch + 1, ruling_desc=f"e{epoch+1}_b{batch_num+1}")
                display_rollout_stats(all_batch_samples, logger)

            # Phase 4: Train
            if not all_batch_samples:
                logger.warning(f"No samples collected for batch {batch_num + 1}, skipping training")
                continue

            logger.info(f"\n--- Phase 4: Training on {len(all_batch_samples)} samples ---")
            new_adapter_path, train_metrics = train_on_samples(
                all_batch_samples,
                config,
                logger,
                adapter_path=current_adapter_path,
            )

            # Free GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            time.sleep(3)
            free_gpu_memory(logger)

            # Update adapter
            previous_adapter_path = current_adapter_path
            current_adapter_path = new_adapter_path

            avg_loss = float(train_metrics.get("avg_loss", 0.0))
            logger.info(f"\n--- Phase 5: Model updated ---")
            logger.info(f"  📤 Previous adapter: {previous_adapter_path or 'None (base model)'}")
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
            }
            all_metrics.append(batch_metrics)

            log_to_wandb(wandb_run, {
                "loss": avg_loss,
                "exact_match_rate": float(accuracy_metrics["exact_match_rate"]),
                "avg_best_reward": float(accuracy_metrics["avg_best_reward"]),
                "num_samples": int(train_metrics.get("num_samples", 0)),
            }, step=global_batch_num, prefix="batch")

            # Benchmark evaluation
            if (config.benchmark_every_n_batches > 0 and
                benchmark_rulings and
                global_batch_num % config.benchmark_every_n_batches == 0):

                logger.info(f"\n--- Running Benchmark Evaluation ---")

                lora_for_benchmark = current_adapter_path if config.use_vllm_lora else None
                if vllm_manager.start_server(model_to_serve=config.base_model_vllm, lora_adapter_path=lora_for_benchmark):
                    benchmark_client = VLLMInferenceClient(config, logger, vllm_manager)

                    benchmark_metrics = run_benchmark_evaluation(
                        benchmark_rulings,
                        config,
                        logger,
                        benchmark_client,
                        global_batch_num,
                    )

                    log_to_wandb(wandb_run, {
                        "exact_match_rate": benchmark_metrics['exact_match_rate'],
                        "avg_best_reward": benchmark_metrics['avg_best_reward'],
                    }, step=global_batch_num, prefix="benchmark")

                    vllm_manager.stop_server()

            batch_time = time.time() - batch_start
            logger.info(f"\n{'─' * 50}")
            logger.info(f"Batch {batch_num + 1} Summary:")
            logger.info(f"  Rulings: {len(batch_rulings)}")
            logger.info(f"  Samples: {len(all_batch_samples)}")
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
        logger.info(f"  Epoch exact match rate: {epoch_exact_match_rate:.1%}")
        logger.info(f"  Time: {epoch_time/60:.1f}m")

        log_to_wandb(wandb_run, {
            "exact_match_rate": epoch_exact_match_rate,
            "total_rulings": epoch_rulings_total,
        }, step=global_batch_num, prefix="epoch")

        # Save checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0 and current_adapter_path:
            checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-epoch-{epoch + 1}")
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
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
        shutil.rmtree(final_adapter_dir, ignore_errors=True)
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
        description="TreeRL GRPO Training with GPT-OSS 20B QLoRA (vLLM + Unsloth)"
    )

    # Model args
    parser.add_argument("--base-model-vllm", type=str, default="openai/gpt-oss-20b",
                        help="Base model for vLLM serving")
    parser.add_argument("--quantization", type=str, default="bnb", choices=["bnb", "mxfp4"],
                        help="Quantization format: bnb (bitsandbytes 4-bit) or mxfp4")

    # Reasoning effort
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="GPT-OSS reasoning effort level")

    # Context lengths
    parser.add_argument("--vllm-max-len", type=int, default=131072, help="vLLM max model len")
    parser.add_argument("--train-max-seq-length", type=int, default=126976, help="Training max seq length (just below 128k)")

    # vLLM args
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--vllm-max-num-seqs", type=int, default=8, help="vLLM max-num-seqs")
    parser.add_argument("--vllm-tp", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--vllm-kv-cache-dtype", type=str, default="auto",
                        choices=["auto", "fp8"], help="KV cache dtype (fp8 for Hopper/Blackwell)")
    parser.add_argument("--served-model-name", type=str, default="gpt-oss-20b")

    # LoRA args
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--no-vllm-lora", action="store_true",
                        help="Disable vLLM LoRA serving")

    # Training args
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")

    # Batch/Epoch structure
    parser.add_argument("--rulings-per-batch", type=int, default=5, help="Rulings per batch")
    parser.add_argument("--accuracy-window", type=int, default=10, help="Accuracy rolling window")
    parser.add_argument("--num-batches", type=int, default=20, help="Batches per epoch")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--start-batch", type=int, default=0, help="Skip to this batch number")

    # Parallelization
    parser.add_argument("--parallel-rollouts", type=int, default=4, help="Number of concurrent rollouts")

    # Benchmark evaluation
    parser.add_argument("--benchmark-every-n-batches", type=int, default=0, help="Run benchmark every N batches")
    parser.add_argument("--benchmark-num-rulings", type=int, default=50, help="Number of held-out rulings")

    # Advantage normalization
    parser.add_argument("--advantage-method", type=str, default="none",
                        choices=["none", "grpo", "grpo_no_std", "gdpo"])

    # Data args
    parser.add_argument("--chapter", type=str, default="84", help="HTS chapter to train on")
    parser.add_argument("--cross-rulings-file", type=str, default="cross_rulings_dataset.json")
    parser.add_argument("--train-all", action="store_true")

    # Wandb args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="treerl-gpt-oss")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")

    # TreeRL args
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--max-questions", type=int, default=3)

    # Output args
    parser.add_argument("--output-dir", type=str, default="gpt_oss_checkpoints")

    # Rollout caching
    parser.add_argument("--save-rollouts", type=str, default="")
    parser.add_argument("--load-rollouts", type=str, default="")

    args = parser.parse_args()

    config = TreeRLConfig(
        base_model_vllm=args.base_model_vllm,
        quantization_format=args.quantization,
        reasoning_effort=args.reasoning_effort,
        vllm_max_model_len=args.vllm_max_len,
        train_max_seq_length=args.train_max_seq_length,
        vllm_port=args.vllm_port,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_tensor_parallel_size=args.vllm_tp,
        vllm_kv_cache_dtype=args.vllm_kv_cache_dtype,
        vllm_served_model_name=args.served_model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_vllm_lora=not args.no_vllm_lora,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        rulings_per_batch=args.rulings_per_batch,
        accuracy_window_size=args.accuracy_window,
        num_batches=args.num_batches,
        num_epochs=args.epochs,
        start_batch=args.start_batch,
        parallel_rollouts=args.parallel_rollouts,
        benchmark_every_n_batches=args.benchmark_every_n_batches,
        benchmark_num_rulings=args.benchmark_num_rulings,
        advantage_method=args.advantage_method,
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
        sft_adapter="",  # No SFT adapter - start from scratch
    )

    # Ensure derived dirs follow output_dir
    config.adapter_sync_dir = os.path.join(config.output_dir, "adapter_sync")
    config.samples_dir = os.path.join(config.output_dir, "samples")
    config.completions_log = os.path.join(config.output_dir, "completions.jsonl")
    config.log_file = os.path.join(config.output_dir, "gpt_oss_training.log")

    train(config)


if __name__ == "__main__":
    main()