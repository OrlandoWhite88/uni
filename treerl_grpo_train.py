#!/usr/bin/env python3
"""
TreeRL GRPO Training Script with vLLM + Unsloth

Architecture (Sequential GPU sharing - Batched RL):

Per-Epoch Flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │  PHASE 1: Load vLLM with merged SFT/LoRA model                  │
    │           (first epoch uses pre-merged SFT adapter)             │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  PHASE 2: Run beam search rollouts for ALL rulings in batch     │
    │           Collect training samples from each ruling             │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  PHASE 3: Stop vLLM server, free GPU memory                     │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  PHASE 4: Load Unsloth, train LoRA on ALL collected samples     │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │  PHASE 5: Export LoRA adapter + merge into base for next vLLM   │
    │           Unload Unsloth, free GPU memory                       │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
                            [Next Epoch]

This ensures vLLM and Unsloth never compete for GPU memory.

Usage:
    python treerl_grpo_train.py --chapter 84 --num-rulings 20 --epochs 3
"""

import os
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
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import threading

# Enable expandable segments for better CUDA memory management with long contexts
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    sft_adapter: str = "orlandowhite/nemotron3_nano_sft"
    max_seq_length: int = 55000  # For Unsloth model loading
    # Training cap: keep trajectories under 32k to avoid OOM
    train_max_seq_length: int = 32000  # Max tokens per training sample
    load_in_4bit: bool = True
    rollout_max_new_tokens: int = 2048
    rollout_temperature: float = 0.7
    rollout_top_p: float = 0.95
    
    # vLLM settings
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_gpu_memory_utilization: float = 0.90
    vllm_max_model_len: int = 55000
    # NOTE: vLLM LoRA disabled - Nemotron-H conv1d/Mamba layers unsupported until PR #30802 merges
    vllm_enable_lora: bool = False
    vllm_max_lora_rank: int = 64
    
    # LoRA settings
    lora_rank: int = 16
    lora_alpha: int = 32
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
    beam_size: int = 4
    max_questions: int = 3
    
    # Data settings
    chapter: str = "84"
    num_rulings_per_epoch: int = 20
    num_epochs: int = 3
    
    # Paths
    cross_rulings_file: str = "cross_rulings_dataset.json"
    output_dir: str = "treerl_checkpoints"
    log_file: str = "treerl_training.log"
    adapter_sync_dir: str = "treerl_checkpoints/adapter_sync"
    samples_dir: str = "treerl_checkpoints/samples"  # Debug: save collected samples
    
    # Logging
    log_every_n_steps: int = 1
    save_every_n_epochs: int = 1
    
    # Device
    device: str = "cuda"


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: TreeRLConfig) -> logging.Logger:
    """Configure logging for training."""
    logger = logging.getLogger("treerl_train")
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(config.log_file)
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


def free_gpu_memory():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# VLLM SERVER MANAGEMENT
# ============================================================================

class VLLMServerManager:
    """Manages vLLM server lifecycle."""
    
    def __init__(self, config: TreeRLConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{config.vllm_host}:{config.vllm_port}"
        self._current_model_path: Optional[str] = None
        self._log_thread: Optional[threading.Thread] = None
        self._stop_logging = threading.Event()
    
    def start_server(self, model_path: Optional[str] = None) -> bool:
        """Start vLLM server with specified model.
        
        Args:
            model_path: Path to model to serve. If None, uses config.base_model.
                       For Nemotron-H, pass a merged model path (LoRA unsupported).
        """
        if self.is_running():
            self.logger.info("vLLM server already running")
            return True
        
        self.logger.info("Starting vLLM server...")
        free_gpu_memory()
        
        # Use provided model or default to base
        serve_model = model_path or self.config.base_model
        
        # Build command
        cmd = [
            "vllm", "serve", serve_model,
            "--host", self.config.vllm_host,
            "--port", str(self.config.vllm_port),
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(self.config.vllm_gpu_memory_utilization),
            "--max-model-len", str(self.config.vllm_max_model_len),
            "--disable-log-requests",
        ]
        
        # LoRA support (disabled for Nemotron-H until vLLM PR #30802 merges)
        if self.config.vllm_enable_lora:
            cmd.extend([
                "--enable-lora",
                "--max-lora-rank", str(self.config.vllm_max_lora_rank),
            ])
        
        self._current_model_path = serve_model
        
        self.logger.info(f"vLLM command: {' '.join(cmd)}")
        
        # Start server process with real-time log streaming
        self._stop_logging.clear()
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
        
        return self._wait_for_ready(timeout=300)
    
    def _stream_logs(self):
        """Stream vLLM logs to console and file."""
        with open(self._log_file_path, "a") as log_file:
            for line in iter(self.process.stdout.readline, ''):
                if self._stop_logging.is_set():
                    break
                line = line.rstrip()
                if line:
                    # Write to file
                    log_file.write(line + "\n")
                    log_file.flush()
                    # Print to console with prefix
                    print(f"[vLLM] {line}")
    
    def _wait_for_ready(self, timeout: int = 300) -> bool:
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
            
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            
            # Wait for log thread to finish
            if self._log_thread and self._log_thread.is_alive():
                self._log_thread.join(timeout=2)
            
            self.process = None
            self._current_model_path = None
            
            # Give GPU time to release memory
            time.sleep(2)
            free_gpu_memory()
            self.logger.info("vLLM server stopped, GPU memory freed")
    
    @property
    def current_model(self) -> Optional[str]:
        return self._current_model_path


# ============================================================================
# VLLM INFERENCE CLIENT
# ============================================================================

class VLLMInferenceClient:
    """vLLM-based LLM client for fast inference during rollouts."""

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
        
        try:
            from api.system_prompts_updated import UNIFIED_SYSTEM_PROMPT
            self.system_prompt = UNIFIED_SYSTEM_PROMPT.strip()
        except ImportError:
            self.system_prompt = "You are a helpful assistant."
        
        self._system_prompt_injection: Optional[str] = None
        self.log_prompts = False
        self.prompt_logger = logger
        
        self._json_requirements = (
            "\n\n=== OUTPUT FORMAT ===\n"
            "You MUST respond with ONLY a valid JSON object or array.\n"
            "Do NOT include any reasoning, explanation, or text before or after the JSON.\n"
            "Do NOT wrap the JSON in markdown code blocks.\n"
            "Start your response with [ or { and end with ] or }.\n"
            "===================\n"
        )
        self._max_json_retries = 4

    def set_system_prompt_injection(self, prompt: Optional[str]) -> None:
        self._system_prompt_injection = prompt

    def clear_system_prompt_injection(self) -> None:
        self._system_prompt_injection = None

    def _current_system_prompt(self) -> str:
        return (self._system_prompt_injection or self.system_prompt).strip()

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from model response, handling chain-of-thought/reasoning."""
        import re
        
        if not response_text:
            raise ValueError("No response text to parse.")

        text = response_text.strip()

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

        # Try to find JSON array - use bracket matching for robustness
        # This handles cases where model outputs reasoning before JSON
        bracket_positions = []
        for i, c in enumerate(text):
            if c == '[':
                bracket_positions.append(('array', i))
            elif c == '{':
                bracket_positions.append(('object', i))
        
        # Try each potential JSON start position
        for json_type, start_pos in bracket_positions:
            end_char = ']' if json_type == 'array' else '}'
            
            # Find matching bracket with depth tracking
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

        raise ValueError(f"Failed to extract valid JSON from response: {response_text[:300]}...")

    def _call_vllm_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        requires_json: bool = False,
    ) -> str:
        """Call vLLM OpenAI-compatible API."""
        gen_start = time.time()
        
        url = f"{self.base_url}/v1/chat/completions"
        
        # Use whatever model vLLM is currently serving
        model_name = self.server_manager.current_model or self.config.base_model
        
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.config.rollout_top_p,
        }
        
        # Enable JSON mode for structured output
        if requires_json:
            payload["response_format"] = {"type": "json_object"}
        
        self.logger.debug(f"    [vLLM] Calling API...")
        
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json()
            
            output_text = result["choices"][0]["message"]["content"].strip()
            
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            gen_elapsed = time.time() - gen_start
            tok_per_sec = completion_tokens / gen_elapsed if gen_elapsed > 0 else 0
            
            self.logger.debug(
                f"    [vLLM] Generated {completion_tokens} tokens in {gen_elapsed:.1f}s "
                f"({tok_per_sec:.1f} tok/s, prompt={prompt_tokens})"
            )
            
            return output_text
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"vLLM API error: {e}")
            raise

    def send_openai_request(
        self,
        prompt: str,
        requires_json: bool = False,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Send a single-turn request with retry for JSON extraction."""
        user_content = prompt.strip()
        if requires_json:
            user_content = f"{user_content.rstrip()}{self._json_requirements}"

        messages = [
            {"role": "system", "content": self._current_system_prompt()},
            {"role": "user", "content": user_content},
        ]

        last_error = None
        for attempt in range(self._max_json_retries if requires_json else 1):
            try:
                # Increase temperature on retries to get different outputs
                retry_temp = temperature + (attempt * 0.1) if requires_json else temperature
                
                text = self._call_vllm_api(
                    messages,
                    temperature=min(retry_temp, 1.0),
                    max_tokens=self.config.rollout_max_new_tokens,
                    requires_json=requires_json,
                )

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
                # Increase temperature on retries to get different outputs
                retry_temp = temperature + (attempt * 0.1) if requires_json else temperature
                
                text = self._call_vllm_api(
                    req_messages,
                    temperature=min(retry_temp, 1.0),
                    max_tokens=self.config.rollout_max_new_tokens,
                    requires_json=requires_json,
                )

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

    def send_vertex_ai_request(self, *args, **kwargs) -> str:
        return self.send_openai_request(*args, **kwargs)

    def send_groq_request(self, prompt: str, requires_json: bool = False, temperature: float = 0.0) -> str:
        return self.send_openai_request(prompt=prompt, requires_json=requires_json, temperature=temperature)


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
    
    chapter_rulings = [
        r for r in all_rulings 
        if r.get("hts_code", "").startswith(config.chapter)
    ]
    
    logger.info(f"Total rulings: {len(all_rulings)}")
    logger.info(f"Chapter {config.chapter} rulings: {len(chapter_rulings)}")
    
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
    
    # Prepare samples for JSON serialization
    serializable_samples = []
    for s in samples:
        sample_copy = {
            "messages": s.get("messages", []),
            "step_rewards": s.get("step_rewards", []),
            "gold_code": s.get("gold_code", ""),
            "pred_trace": s.get("pred_trace", []),
            "gold_trace": s.get("gold_trace", []),
            "path_id": s.get("path_id", ""),
            "leaf_reward": s.get("leaf_reward", 0),  # Fractional prefix match
            "source": s.get("source", ""),
        }
        serializable_samples.append(sample_copy)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  Saved {len(samples)} samples to: {filepath}")
    return filepath


def display_rollout_stats(
    samples: List[Dict],
    logger: logging.Logger,
) -> None:
    """
    Display comprehensive stats after rollout phase, before training.
    
    Shows:
    - Beam paths vs gold target comparisons
    - Reward distributions
    - Step reward statistics
    - V(root) if available
    """
    if not samples:
        logger.warning("No samples to display stats for")
        return
    
    logger.info("\n" + "=" * 70)
    logger.info("ROLLOUT STATISTICS (Before Training)")
    logger.info("=" * 70)
    
    # Group samples by gold_code (ruling)
    by_ruling = {}
    for s in samples:
        gold = s.get("gold_code", "unknown")
        if gold not in by_ruling:
            by_ruling[gold] = []
        by_ruling[gold].append(s)
    
    logger.info(f"\n📊 OVERVIEW")
    logger.info(f"  Total samples (beam paths): {len(samples)}")
    logger.info(f"  Unique rulings: {len(by_ruling)}")
    logger.info(f"  Avg paths per ruling: {len(samples) / len(by_ruling):.1f}")
    
    # Collect all rewards and step rewards
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
    
    # Leaf reward stats
    logger.info(f"\n🎯 LEAF REWARDS (Prefix Match with Gold)")
    logger.info(f"  Perfect (=1.0): {perfect_count} ({100*perfect_count/len(samples):.1f}%)")
    logger.info(f"  Partial (0<r<1): {partial_count} ({100*partial_count/len(samples):.1f}%)")
    logger.info(f"  Zero (=0): {zero_count} ({100*zero_count/len(samples):.1f}%)")
    if all_leaf_rewards:
        logger.info(f"  Min: {min(all_leaf_rewards):.3f}")
        logger.info(f"  Max: {max(all_leaf_rewards):.3f}")
        logger.info(f"  Mean: {sum(all_leaf_rewards)/len(all_leaf_rewards):.3f}")
    
    # Step reward stats
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
    
    # Per-ruling breakdown (show first few)
    logger.info(f"\n🔍 BEAM PATHS vs GOLD (per ruling)")
    logger.info("-" * 70)
    
    for i, (gold_code, ruling_samples) in enumerate(list(by_ruling.items())[:10]):
        # Get gold trace
        gold_trace = ruling_samples[0].get("gold_trace", [])
        gold_path = " > ".join([
            t.get("code", f"grp:{t.get('node_id', '?')}")[:8] 
            for t in gold_trace
        ])
        
        # Find best prediction
        best_sample = max(ruling_samples, key=lambda s: s.get("leaf_reward", s.get("reward", 0)))
        best_reward = best_sample.get("leaf_reward", best_sample.get("reward", 0))
        pred_trace = best_sample.get("pred_trace", [])
        pred_path = " > ".join([
            t.get("code", f"grp:{t.get('node_id', '?')}")[:8]
            for t in pred_trace
        ])
        
        # Count matches at each level
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
        
        # Show step rewards for best path
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
            reward = compute_leaf_reward(pred_trace, gold_trace)
            
            leaves.append({
                "path_id": path_id,
                "pred_trace": pred_trace,
                "reward": reward,
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
# TRAINING FUNCTIONS (Unsloth phase)
# ============================================================================

def load_training_model(config: TreeRLConfig, logger: logging.Logger, adapter_path: Optional[str] = None):
    """Load model with Unsloth for training."""
    logger.info("=" * 70)
    logger.info("LOADING TRAINING MODEL WITH UNSLOTH")
    logger.info("=" * 70)
    
    from unsloth import FastLanguageModel
    
    # Determine which adapter to load
    load_adapter = adapter_path or config.sft_adapter
    
    if load_adapter:
        logger.info(f"Loading model with adapter: {load_adapter}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=load_adapter,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            trust_remote_code=True,
            # offload_embedding=True,  # Disabled - causes device mismatch errors
            # unsloth_tiled_mlp=True,  # Disabled - incompatible with Nemotron-H MLP
        )
        logger.info(f"Model + adapter loaded (4bit={config.load_in_4bit})")
        
        # Enable Unsloth gradient checkpointing for memory efficiency
        # This offloads activations to CPU RAM, enabling 10x longer contexts
        FastLanguageModel.for_training(model, use_gradient_checkpointing=True)
        logger.info("Enabled Unsloth gradient checkpointing (activation offloading)")
        
        # Ensure LoRA params are trainable (should already be, but explicit is safer)
        lora_param_count = 0
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
                lora_param_count += 1
        logger.debug(f"Set requires_grad=True for {lora_param_count} LoRA parameters")
        
    else:
        logger.info("Loading base model and creating fresh LoRA adapters")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=config.max_seq_length,
            load_in_4bit=config.load_in_4bit,
            trust_remote_code=True,
            # offload_embedding=True,  # Disabled - causes device mismatch errors
            # unsloth_tiled_mlp=True,  # Disabled - incompatible with Nemotron-H MLP
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            target_modules=list(config.lora_target_modules),
            lora_alpha=config.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Offloads activations to CPU
            random_state=3407,
        )
        logger.info(f"Fresh LoRA adapters added (rank={config.lora_rank}, 4bit={config.load_in_4bit})")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Log GPU memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU memory after model load: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
    
    return model, tokenizer


def unload_training_model(model, tokenizer, logger: logging.Logger):
    """Unload training model and free GPU memory."""
    logger.info("Unloading training model...")
    del model
    del tokenizer
    free_gpu_memory()
    logger.info("Training model unloaded, GPU memory freed")


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
    """
    Extract tree depth from path_so_far in a rank_candidates user message.
    
    path_so_far looks like: "84 - Chapter desc > 8481 - Heading desc > 8481.80 - Subheading"
    Depth = number of '>' separators + 1 (for the chapter)
    
    Returns:
        Depth in the tree (1 = at chapter, selecting heading, 2 = at heading, etc.)
        Returns -1 if path_so_far not found.
    """
    import re
    # Find path_so_far in JSON
    match = re.search(r'"path_so_far":\s*"([^"]+)"', user_content)
    if not match:
        return -1
    
    path = match.group(1)
    # Count '>' separators - each one represents a level traversed
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
    """
    Build per-token weight tensor from step rewards.
    
    TreeRL process supervision: Maps classification step rewards to assistant turns.
    
    Mapping logic:
    - select_chapters_stage1/stage2 responses → step 0 (chapter selection)
    - rank_candidates responses → step based on tree depth from path_so_far
    - Q&A responses → weight 0 (excluded from training gradient)
    
    This correctly handles:
    - Re-selections at the same level after Q&A (same step reward)
    - Group nodes in the trace (included in step count)
    - Q&A turns are NOT trained on (weight = 0)
    
    Args:
        step_rewards: List of {step, R, trace_prefix, ...} from process supervision
        boundaries: Token boundaries for each assistant turn
        seq_len: Total sequence length
        device: Device for tensor
        leaf_reward: Fallback reward for edge cases
        messages: Full message list to identify turn types and tree depth
    """
    weights = torch.zeros(seq_len, device=device)
    
    if not boundaries:
        return weights
    
    # Build step index to R mapping
    step_to_R = {sr["step"]: sr["R"] for sr in step_rewards}
    max_step = max(step_to_R.keys()) if step_to_R else 0
    
    # Compute fallback reward for non-classification turns
    if leaf_reward is not None:
        fallback_R = leaf_reward
    elif step_rewards:
        fallback_R = sum(sr.get("R", 0.0) for sr in step_rewards) / len(step_rewards)
    else:
        fallback_R = 0.0
    
    # If we don't have messages, use simple sequential mapping
    if not messages:
        for bound_idx, (start, end) in enumerate(boundaries):
            R = step_to_R.get(bound_idx, fallback_R)
            weights[start:end] = R
        return weights
    
    # Build mapping from message index to (assistant_turn_idx, step_idx)
    # We need to look at USER messages to get path_so_far for depth
    assistant_step_map = []  # List of step indices for each assistant turn
    current_depth = 0  # Track tree depth from user prompts
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            # Check if this is a rank_candidates prompt with path_so_far
            if "rank_candidates" in content:
                depth = _get_path_depth(content)
                if depth > 0:
                    current_depth = depth
        
        elif role == "assistant":
            # Identify turn type
            is_chapter_selection = (
                '"chapters"' in content or 
                '"top_selection"' in content or
                'chapter-level' in content.lower()
            )
            is_rank_candidates = '"primary_selection"' in content and not is_chapter_selection
            
            if is_chapter_selection:
                # Chapter selection stages all get step 0
                assistant_step_map.append(0)
            elif is_rank_candidates:
                # Use current_depth from preceding user message
                # Depth 1 = selecting at chapter level = step 1
                # Depth 2 = selecting at heading level = step 2
                # etc.
                step_idx = current_depth
                # Clamp to valid range
                step_idx = min(step_idx, max_step)
                assistant_step_map.append(step_idx)
            else:
                # Q&A or other - mark as -1 to use fallback
                assistant_step_map.append(-1)
    
    # Apply weights to token boundaries
    # Q&A turns (step_idx == -1) get weight 0 - excluded from training
    for bound_idx, (start, end) in enumerate(boundaries):
        if bound_idx < len(assistant_step_map):
            step_idx = assistant_step_map[bound_idx]
            if step_idx >= 0:
                # Classification decision - use step reward
                R = step_to_R.get(step_idx, fallback_R)
            else:
                # Q&A turn - exclude from training (weight = 0)
                R = 0.0
        else:
            # Extra boundaries without messages - use fallback
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
) -> Tuple[str, Optional[str], Dict[str, float]]:
    """
    Train on collected samples using Unsloth.
    
    Uses microbatch size of 1 with gradient accumulation to avoid OOM.
    
    Returns:
        (new_adapter_path, merged_model_path, metrics)
    """
    logger.info(f"Training on {len(samples)} samples...")
    logger.info(f"  Train max seq length: {config.train_max_seq_length}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    
    # Load training model
    model, tokenizer = load_training_model(config, logger, adapter_path)
    
    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    model.train()
    
    metrics = {
        "total_loss": 0.0,
        "num_samples": 0,
        "skipped_too_long": 0,
    }
    
    accumulated_loss = 0.0
    accumulated_steps = 0
    optimizer.zero_grad()
    
    for sample_idx, sample in enumerate(samples):
        messages = sample.get("messages", [])
        step_rewards = sample.get("step_rewards", [])
        leaf_reward = sample.get("leaf_reward", sample.get("reward", None))
        
        if not messages:
            continue
        
        def _try_tokenize(msgs: List[Dict[str, str]]):
            text_local = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs_local = tokenizer(
                text_local,
                return_tensors="pt",
                truncation=True,
                max_length=config.train_max_seq_length,
                padding=False,
            )
            return text_local, inputs_local

        def _slim_messages_for_training(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
            """
            Reduce token count while preserving:
            - path_so_far in rank_candidates user messages (needed for step mapping)
            - primary_selection option_index/code in assistant messages
            - enough structure to keep the model conditioned.

            Strategy:
            - For assistant JSON, drop large reasoning fields ('thinking', 'detailed_internal_reasoning')
            - For user JSON INPUT blocks, keep only a minimal subset and shrink large arrays / long strings
            - For chapter notes blocks, remove the notes payload
            """
            import re
            import json as _json

            def _truncate(s: str, n: int) -> str:
                s = s or ""
                return s if len(s) <= n else (s[:n] + "…")

            slimmed: List[Dict[str, str]] = []
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content", "") or ""

                if role == "assistant":
                    # Strip huge reasoning fields to reduce context
                    try:
                        data = _json.loads(content)
                        if isinstance(data, dict):
                            data.pop("thinking", None)
                            data.pop("detailed_internal_reasoning", None)
                            # Also truncate nested reasoning strings
                            for k in ("primary_selection", "alternative_1", "alternative_2"):
                                if isinstance(data.get(k), dict) and "reasoning" in data[k]:
                                    data[k]["reasoning"] = _truncate(str(data[k]["reasoning"]), 160)
                            content = _json.dumps(data, ensure_ascii=False)
                    except Exception:
                        pass
                    slimmed.append({"role": role, "content": content})
                    continue

                if role == "user":
                    # Remove giant chapter notes payloads (stage2)
                    if "RELEVANT NOTES" in content and "TASK: select_chapters_stage2" in content:
                        # Keep header + JSON INPUT only
                        json_idx = content.find("JSON INPUT:")
                        kept = content[: json_idx + len("JSON INPUT:")] if json_idx != -1 else "TASK: select_chapters_stage2"
                        # Keep the JSON input itself if present
                        json_blob = content[json_idx + len("JSON INPUT:") :] if json_idx != -1 else ""
                        content = kept + "\n{ \"notes_omitted\": true }\n" + _truncate(json_blob.strip(), 1200)
                        slimmed.append({"role": role, "content": content})
                        continue

                    # Try to shrink JSON INPUT payloads
                    if "JSON INPUT:" in content:
                        head, _, tail = content.partition("JSON INPUT:")
                        # Tail should start with JSON; try to parse the first JSON object in tail.
                        json_start = tail.find("{")
                        if json_start != -1:
                            raw = tail[json_start:]
                            try:
                                data = _json.loads(raw)
                                if isinstance(data, dict) and "data" in data:
                                    task = data.get("task", "")
                                    d = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
                                    out = {"task": task, "data": {}}
                                    # Keep conditioning essentials
                                    if "product_text" in d:
                                        out["data"]["product_text"] = _truncate(str(d.get("product_text")), 500)
                                    if "path_so_far" in d:
                                        # MUST keep this for step mapping
                                        out["data"]["path_so_far"] = str(d.get("path_so_far"))
                                    if "select_count" in d:
                                        out["data"]["select_count"] = d.get("select_count")
                                    # Shrink candidate lists aggressively
                                    tree = d.get("classification_tree")
                                    if isinstance(tree, dict) and isinstance(tree.get("children"), list):
                                        children = tree["children"]
                                        # Keep only small fields; cap list length
                                        slim_children = []
                                        for c in children[:12]:
                                            if not isinstance(c, dict):
                                                continue
                                            slim_children.append({
                                                "index": c.get("index"),
                                                "code": c.get("code"),
                                                "is_group": c.get("is_group"),
                                                "node_id": c.get("node_id"),
                                                "description": _truncate(str(c.get("description", "")), 60),
                                            })
                                        out["data"]["classification_tree"] = {"children": slim_children, "children_truncated": len(children) > len(slim_children)}
                                    content = head + "JSON INPUT:\n" + _json.dumps(out, ensure_ascii=False)
                            except Exception:
                                pass

                    # Generic trimming for very long user messages
                    if len(content) > 6000:
                        content = _truncate(content, 6000)
                    slimmed.append({"role": role, "content": content})
                    continue

                # System or other: keep but trim huge blocks
                if len(content) > 6000:
                    content = content[:6000] + "…"
                slimmed.append({"role": role, "content": content})

            return slimmed

        try:
            _, inputs = _try_tokenize(messages)
            input_ids = inputs["input_ids"].to(config.device)
            attention_mask = inputs["attention_mask"].to(config.device)
            seq_len = input_ids.shape[1]

            # If too long, slim messages and retry tokenization
            if seq_len > config.train_max_seq_length:
                logger.warning(f"  Sample {sample_idx}: {seq_len} tokens > {config.train_max_seq_length} — slimming trajectory...")
                slimmed_messages = _slim_messages_for_training(messages)
                _, inputs2 = _try_tokenize(slimmed_messages)
                input_ids = inputs2["input_ids"].to(config.device)
                attention_mask = inputs2["attention_mask"].to(config.device)
                seq_len2 = input_ids.shape[1]
                logger.warning(f"  Sample {sample_idx}: after slimming → {seq_len2} tokens")
                messages = slimmed_messages
                seq_len = seq_len2

            if seq_len >= config.train_max_seq_length:
                logger.debug(f"  Sample {sample_idx}: truncated to {seq_len} tokens")

        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
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
            
            scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()
            
            accumulated_loss += loss.item()
            accumulated_steps += 1
            
            metrics["total_loss"] += loss.item()
            metrics["num_samples"] += 1
            
            # Clear intermediate tensors to free memory
            del input_ids, attention_mask, loss, scaled_loss
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM on sample {sample_idx} (seq_len={seq_len}): {e}")
            # Clear cache and skip this sample
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            accumulated_loss = 0.0
            accumulated_steps = 0
            continue
            
        except Exception as e:
            import traceback
            logger.error(f"Loss computation error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue
        
        # Clear GPU cache periodically
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
            
            # Clear cache after optimizer step
            torch.cuda.empty_cache()
    
    # Handle remaining gradients
    if accumulated_steps > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    # Save adapter (for checkpointing/resuming)
    timestamp = int(time.time())
    new_adapter_path = os.path.join(config.adapter_sync_dir, f"adapter_{timestamp}")
    os.makedirs(new_adapter_path, exist_ok=True)
    model.save_pretrained(new_adapter_path)
    tokenizer.save_pretrained(new_adapter_path)
    logger.info(f"Adapter saved to: {new_adapter_path}")
    
    # Merge LoRA into base model for vLLM (Nemotron-H LoRA unsupported in vLLM)
    merged_path = os.path.join(config.adapter_sync_dir, f"merged_{timestamp}")
    logger.info(f"Merging LoRA weights for vLLM serving...")
    try:
        model.save_pretrained_merged(
            merged_path,
            tokenizer,
            save_method="merged_16bit",
        )
        logger.info(f"Merged model saved to: {merged_path}")
    except Exception as e:
        logger.error(f"Failed to merge model: {e}")
        merged_path = None
    
    # Compute final metrics
    if metrics["num_samples"] > 0:
        metrics["avg_loss"] = metrics["total_loss"] / metrics["num_samples"]
    else:
        metrics["avg_loss"] = 0.0
    
    # Unload model
    unload_training_model(model, tokenizer, logger)
    
    return new_adapter_path, merged_path, metrics


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def merge_sft_adapter(config: TreeRLConfig, logger: logging.Logger) -> Optional[str]:
    """Pre-merge SFT adapter for initial vLLM serving.
    
    Caches the merged model - if it already exists, skips re-merging.
    """
    if not config.sft_adapter:
        logger.info("No SFT adapter specified, using base model")
        return None
    
    merged_path = os.path.join(config.adapter_sync_dir, "sft_merged")
    
    # Check if already merged (look for config.json as indicator)
    merged_config = os.path.join(merged_path, "config.json")
    if os.path.exists(merged_config):
        logger.info(f"✓ SFT merged model already exists: {merged_path}")
        logger.info("  (delete this folder to force re-merge)")
        return merged_path
    
    logger.info(f"Pre-merging SFT adapter: {config.sft_adapter}")
    logger.info("  (this only happens once, cached for future runs)")
    
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.sft_adapter,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        trust_remote_code=True,
    )
    
    logger.info(f"Merging SFT adapter to: {merged_path}")
    
    try:
        model.save_pretrained_merged(
            merged_path,
            tokenizer,
            save_method="merged_16bit",
        )
        logger.info(f"SFT merged model saved: {merged_path}")
    except Exception as e:
        logger.error(f"Failed to merge SFT adapter: {e}")
        return None
    
    # Cleanup
    del model, tokenizer
    free_gpu_memory()
    
    return merged_path


def train(config: TreeRLConfig):
    """
    Main training function with batched online RL.
    
    Flow per epoch:
    1. Start vLLM with merged model
    2. Run rollouts for ALL rulings in batch (collect samples)
    3. Stop vLLM (free GPU)
    4. Load Unsloth, train on all samples
    5. Export LoRA adapter + merge for next vLLM cycle
    6. Repeat
    """
    
    logger = setup_logging(config)
    
    logger.info("=" * 70)
    logger.info("TREERL GRPO TRAINING (Batched: rollout batch → train → repeat)")
    logger.info("=" * 70)
    logger.info(f"Config: {config}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.adapter_sync_dir, exist_ok=True)
    os.makedirs(config.samples_dir, exist_ok=True)
    
    # Load data
    rulings = load_chapter_rulings(config, logger)
    if not rulings:
        logger.error(f"No rulings found for chapter {config.chapter}")
        return
    
    # =============================================
    # PHASE 0: PRE-MERGE SFT ADAPTER (for first rollout)
    # =============================================
    logger.info("\n--- Phase 0: Pre-merging SFT adapter for vLLM ---")
    current_vllm_model = merge_sft_adapter(config, logger)
    current_adapter_path = config.sft_adapter if config.sft_adapter else None
    
    # Initialize vLLM server manager
    vllm_manager = VLLMServerManager(config, logger)
    
    training_start = time.time()
    all_metrics = []
    global_step = 0
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"EPOCH {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'=' * 70}")
        
        # Sample rulings for this epoch
        epoch_rulings = random.sample(
            rulings, 
            min(config.num_rulings_per_epoch, len(rulings))
        )
        
        # =============================================
        # PHASE 1: START VLLM WITH MERGED MODEL
        # =============================================
        logger.info(f"\n--- Phase 1: Starting vLLM server ---")
        logger.info(f"  Model: {current_vllm_model or config.base_model}")
        
        if not vllm_manager.start_server(model_path=current_vllm_model):
            logger.error("Failed to start vLLM server!")
            return
        
        vllm_client = VLLMInferenceClient(config, logger, vllm_manager)
        
        # =============================================
        # PHASE 2: RUN ROLLOUTS FOR ALL RULINGS IN BATCH
        # =============================================
        logger.info(f"\n--- Phase 2: Running rollouts for {len(epoch_rulings)} rulings ---")
        
        all_epoch_samples = []
        
        for ruling_idx, ruling in enumerate(epoch_rulings):
            global_step += 1
            product_desc = ruling.get("short_product_description", "")[:50]
            
            logger.info(f"\n  Ruling {ruling_idx+1}/{len(epoch_rulings)}: {product_desc}...")
            
            samples = run_online_rollout(ruling, config, logger, vllm_client)
            
            if samples:
                all_epoch_samples.extend(samples)
                logger.info(f"    → Collected {len(samples)} samples (total: {len(all_epoch_samples)})")
            else:
                logger.warning(f"    → No samples collected")
        
        # =============================================
        # PHASE 3: STOP VLLM (FREE GPU)
        # =============================================
        logger.info(f"\n--- Phase 3: Stopping vLLM server ---")
        vllm_manager.stop_server()
        
        # =============================================
        # DEBUG: SAVE SAMPLES + DISPLAY STATS
        # =============================================
        if all_epoch_samples:
            save_samples_for_debug(
                all_epoch_samples,
                config,
                logger,
                epoch=epoch + 1,
                ruling_desc=f"epoch{epoch+1}_batch",
            )
            # Display comprehensive rollout stats before training
            display_rollout_stats(all_epoch_samples, logger)
        
        # =============================================
        # PHASE 4: TRAIN WITH UNSLOTH
        # =============================================
        if not all_epoch_samples:
            logger.warning(f"No samples collected for epoch {epoch + 1}, skipping training")
            continue
        
        logger.info(f"\n--- Phase 4: Training on {len(all_epoch_samples)} samples ---")
        
        new_adapter_path, merged_model_path, train_metrics = train_on_samples(
            all_epoch_samples,
            config,
            logger,
            adapter_path=current_adapter_path,
        )
        
        # =============================================
        # PHASE 5: UPDATE PATHS FOR NEXT CYCLE
        # =============================================
        current_adapter_path = new_adapter_path
        current_vllm_model = merged_model_path
        
        avg_loss = train_metrics.get("avg_loss", 0)
        
        logger.info(f"\n--- Phase 5: Model updated ---")
        logger.info(f"  Adapter: {new_adapter_path}")
        logger.info(f"  Merged model: {merged_model_path}")
        
        # Record epoch metrics
        all_metrics.append({
            "epoch": epoch + 1,
            "num_rulings": len(epoch_rulings),
            "num_samples": train_metrics.get("num_samples", 0),
            "avg_loss": avg_loss,
            "adapter_path": new_adapter_path,
            "merged_model_path": merged_model_path,
        })
        
        # =============================================
        # EPOCH COMPLETE
        # =============================================
        epoch_time = time.time() - epoch_start
        
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Rulings processed: {len(epoch_rulings)}")
        logger.info(f"  Total samples: {len(all_epoch_samples)}")
        logger.info(f"  Average loss: {avg_loss:.4f}")
        logger.info(f"  Time: {epoch_time:.1f}s ({epoch_time/60:.1f}m)")
        logger.info(f"{'=' * 50}")
        
        # Save checkpoint
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
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    if all_metrics:
        logger.info(f"Final average loss: {all_metrics[-1].get('avg_loss', 0):.4f}")
    
    # Save final models
    import shutil
    if current_adapter_path:
        final_adapter_dir = os.path.join(config.output_dir, "final_adapter")
        if os.path.exists(final_adapter_dir):
            shutil.rmtree(final_adapter_dir)
        shutil.copytree(current_adapter_path, final_adapter_dir)
        logger.info(f"Final adapter saved: {final_adapter_dir}")
    
    if current_vllm_model:
        final_merged_dir = os.path.join(config.output_dir, "final_merged")
        if os.path.exists(final_merged_dir):
            shutil.rmtree(final_merged_dir)
        shutil.copytree(current_vllm_model, final_merged_dir)
        logger.info(f"Final merged model saved: {final_merged_dir}")
    
    # Save training metrics
    metrics_file = os.path.join(config.output_dir, "training_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_file}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TreeRL GRPO Training (vLLM + Unsloth)")
    
    # Model args
    parser.add_argument("--base-model", type=str, 
                       default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                       help="Base model name or path")
    parser.add_argument("--sft-adapter", type=str, 
                       default="orlandowhite/nemotron3_nano_sft",
                       help="SFT LoRA adapter to continue training from")
    parser.add_argument("--no-sft-adapter", action="store_true",
                       help="Train fresh LoRA from base model")
    parser.add_argument("--max-seq-length", type=int, default=55000,
                       help="Maximum sequence length for model loading")
    parser.add_argument("--train-max-seq-length", type=int, default=32000,
                       help="Maximum sequence length per training sample")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--rollout-max-new-tokens", type=int, default=2048,
                       help="Max new tokens per generation during rollout")
    
    # vLLM args
    parser.add_argument("--vllm-port", type=int, default=8000,
                       help="vLLM server port")
    parser.add_argument("--vllm-max-model-len", type=int, default=55000,
                       help="vLLM max context length")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.90,
                       help="vLLM GPU memory utilization")
    
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
    
    args = parser.parse_args()
    
    sft_adapter = args.sft_adapter if not args.no_sft_adapter else ""
    
    config = TreeRLConfig(
        base_model=args.base_model,
        sft_adapter=sft_adapter,
        max_seq_length=args.max_seq_length,
        train_max_seq_length=args.train_max_seq_length,
        load_in_4bit=not args.no_4bit,
        rollout_max_new_tokens=args.rollout_max_new_tokens,
        vllm_port=args.vllm_port,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_gpu_memory_utilization=args.vllm_gpu_util,
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
    )
    
    train(config)


if __name__ == "__main__":
    main()
