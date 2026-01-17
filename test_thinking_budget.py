#!/usr/bin/env python3
"""
Test script for vLLM Thinking Budget Processor

This script tests whether we can cap Qwen3's thinking tokens using
a custom LogitsProcessor in vLLM.

Usage:
    python test_thinking_budget.py [--max-thinking-tokens N] [--model MODEL]
    
Example:
    python test_thinking_budget.py --max-thinking-tokens 100
    python test_thinking_budget.py --max-thinking-tokens 0  # No thinking at all
"""

import argparse
import subprocess
import time
import sys
import os
import signal
import requests
from pathlib import Path

# ============================================================================
# THINKING BUDGET PROCESSOR
# ============================================================================

PROCESSOR_CODE = '''
"""Thinking Budget Processor for vLLM"""
import torch
from typing import Dict, Any

try:
    # vLLM v1 API (newer versions)
    from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate
    VLLM_V1 = True
except ImportError:
    # Fallback for older vLLM versions
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    VLLM_V1 = False
    BatchUpdate = None


class ThinkingBudgetProcessor(LogitsProcessor):
    """
    Caps thinking tokens in vLLM generation for Qwen3 models.
    
    After max_thinking_tokens are generated, forces </think> to end thinking mode.
    Includes smooth transition by boosting </think> probability near the limit.
    
    Configure via:
    - Constructor kwarg: max_thinking_tokens=N
    - Environment variable: THINKING_BUDGET_MAX_TOKENS=N
    """
    
    def __init__(self, vllm_config=None, device=None, is_pin_memory=False, max_thinking_tokens: int = None):
        import os
        # Priority: constructor arg > env var > default
        if max_thinking_tokens is not None:
            self.max_thinking_tokens = max_thinking_tokens
        else:
            self.max_thinking_tokens = int(os.environ.get("THINKING_BUDGET_MAX_TOKENS", "500"))
        
        print(f"[ThinkingBudgetProcessor] Initialized with max_thinking_tokens={self.max_thinking_tokens}")
        self.device = device
        self.neg_inf = float("-inf")
        
        # Token IDs - will be set on first call if not available from config
        self.think_end_id = None
        self.nl_id = None
        self._tokenizer = None
        
        # Try to get tokenizer from vllm_config
        if vllm_config is not None:
            try:
                if hasattr(vllm_config, 'tokenizer'):
                    self._tokenizer = vllm_config.tokenizer
                elif hasattr(vllm_config, 'model_config') and hasattr(vllm_config.model_config, 'tokenizer'):
                    self._tokenizer = vllm_config.model_config.tokenizer
            except Exception:
                pass
        
        # Per-request state tracking
        self.req_state: Dict[int, Dict[str, Any]] = {}
        self._initialized = False
    
    def _initialize_tokens(self, vocab_size: int = None):
        """Lazy initialization of token IDs."""
        if self._initialized:
            return
            
        if self._tokenizer is not None:
            try:
                self.think_end_id = self._tokenizer.encode("</think>", add_special_tokens=False)[0]
                self.nl_id = self._tokenizer.encode("\\n", add_special_tokens=False)[0]
                self._initialized = True
                return
            except Exception:
                pass
        
        # Fallback: Common token IDs for Qwen3 models
        # These are typical values but may need adjustment for your specific model
        # </think> is often around 151668 in Qwen3 vocab
        # \\n is often 198 or similar
        self.think_end_id = 151668  # Qwen3 </think>
        self.nl_id = 198  # Common newline token
        self._initialized = True
        print(f"[ThinkingBudget] Using fallback token IDs: </think>={self.think_end_id}, \\\\n={self.nl_id}")
    
    def update_state(self, batch_update):
        """Track requests as they come and go in the batch (vLLM v1 API)."""
        if batch_update is None:
            return
        for req in batch_update.added:
            idx = req.request_index if hasattr(req, 'request_index') else req
            self.req_state[idx] = {"count": 0, "stopped": False}
        for idx in batch_update.removed:
            self.req_state.pop(idx, None)
    
    def __call__(self, token_ids, logits):
        """Called for each generation step (older vLLM API)."""
        return self.apply(logits)
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Modify logits to enforce thinking budget."""
        self._initialize_tokens(logits.shape[-1])
        
        # Handle single request case (no batch tracking)
        if not self.req_state:
            self.req_state[0] = {"count": 0, "stopped": False}
        
        for req_idx, state in list(self.req_state.items()):
            if state["stopped"]:
                continue
            
            # Bounds check
            if req_idx >= logits.shape[0]:
                continue
                
            count = state["count"]
            
            # Skip if no limit set
            if self.max_thinking_tokens is None or self.max_thinking_tokens < 0:
                state["count"] += 1
                continue
            
            # Handle zero budget case - immediately end thinking
            if self.max_thinking_tokens == 0 and not state["stopped"]:
                logits[req_idx] = self.neg_inf
                logits[req_idx, self.nl_id] = 0
                logits[req_idx, self.think_end_id] = 0
                state["stopped"] = True
                continue
            
            ratio = count / self.max_thinking_tokens
            
            # Gradual bias starting at 95% of budget
            if ratio > 0.95:
                boost = 1 + ratio
                if self.think_end_id < logits.shape[-1]:
                    logits[req_idx, self.think_end_id] = logits[req_idx, self.think_end_id] * boost
                if self.nl_id < logits.shape[-1]:
                    logits[req_idx, self.nl_id] = logits[req_idx, self.nl_id] * boost
            
            # Force end at budget limit
            if count >= self.max_thinking_tokens - 1:
                if count == self.max_thinking_tokens - 1:
                    # Force newline first
                    logits[req_idx] = self.neg_inf
                    if self.nl_id < logits.shape[-1]:
                        logits[req_idx, self.nl_id] = 0
                else:
                    # Then force </think>
                    logits[req_idx] = self.neg_inf
                    if self.think_end_id < logits.shape[-1]:
                        logits[req_idx, self.think_end_id] = 0
                    state["stopped"] = True
            
            state["count"] += 1
        
        return logits
    
    def is_argmax_invariant(self) -> bool:
        """We modify argmax behavior by forcing specific tokens."""
        return False
'''


def setup_processor_module():
    """Write the processor module to a file that vLLM can import."""
    module_path = Path(__file__).parent / "thinking_budget_processor.py"
    module_path.write_text(PROCESSOR_CODE)
    print(f"✓ Created processor module: {module_path}")
    return module_path


def start_vllm_server(model: str, max_thinking_tokens: int, port: int = 8765):
    """Start vLLM server with the thinking budget processor."""
    
    # Ensure the module is importable
    module_dir = str(Path(__file__).parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{module_dir}:{env.get('PYTHONPATH', '')}"
    
    # Pass max_thinking_tokens via environment variable
    env["THINKING_BUDGET_MAX_TOKENS"] = str(max_thinking_tokens)
    
    # Build processor config
    # vLLM expects format: "module_path:ClassName" with colon separator
    # For kwargs, use JSON array with dict: [{"qualname": "path:Class", "kwargs": {...}}]
    processor_spec = f'thinking_budget_processor:ThinkingBudgetProcessor'
    
    # If we need kwargs, use the full JSON format
    # processor_config = f'{{"qualname": "{processor_spec}", "kwargs": {{"max_thinking_tokens": {max_thinking_tokens}}}}}'
    # For now, use simple string format (kwargs via environment or defaults)
    
    cmd = [
        "vllm", "serve", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--dtype", "auto",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.7",
        "--logits-processors", f'["{processor_spec}"]',
    ]
    
    print(f"\n🚀 Starting vLLM server...")
    print(f"   Model: {model}")
    print(f"   Max thinking tokens: {max_thinking_tokens}")
    print(f"   Port: {port}")
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    return process


def wait_for_server(port: int, timeout: int = 300):
    """Wait for vLLM server to be ready."""
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    
    print(f"⏳ Waiting for server to be ready (timeout: {timeout}s)...")
    
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print("✓ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    
    print("✗ Server failed to start within timeout")
    return False


def test_generation(port: int, max_thinking_tokens: int):
    """Test the thinking budget by generating a response."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    
    # A prompt that typically triggers long thinking
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Think step by step."},
        {"role": "user", "content": "What is the square root of 2722? Show your reasoning."},
    ]
    
    payload = {
        "model": "default",  # vLLM uses this for single-model serving
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7,
    }
    
    print(f"\n📝 Testing generation with max_thinking_tokens={max_thinking_tokens}")
    print(f"   Prompt: {messages[-1]['content']}")
    print()
    
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        
        print("=" * 70)
        print("RESPONSE:")
        print("=" * 70)
        print(content)
        print("=" * 70)
        print()
        
        # Analyze the response
        think_start = content.find("<think>")
        think_end = content.find("</think>")
        
        if think_start != -1 and think_end != -1:
            thinking_content = content[think_start + 7:think_end]
            thinking_tokens_approx = len(thinking_content.split())  # Rough word count
            print(f"📊 Analysis:")
            print(f"   Thinking section found: Yes")
            print(f"   Thinking length (approx words): {thinking_tokens_approx}")
            print(f"   Max thinking tokens configured: {max_thinking_tokens}")
            print(f"   Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            
            if max_thinking_tokens > 0 and thinking_tokens_approx > max_thinking_tokens * 1.5:
                print(f"   ⚠️  Thinking may not be properly capped (expected ~{max_thinking_tokens} tokens)")
            else:
                print(f"   ✓ Thinking appears to be within budget")
        elif think_start == -1:
            print(f"📊 Analysis:")
            print(f"   Thinking section found: No (model may not be using <think> tags)")
        else:
            print(f"📊 Analysis:")
            print(f"   Thinking section: Incomplete (no </think> found)")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test vLLM Thinking Budget Processor")
    parser.add_argument(
        "--max-thinking-tokens", "-t",
        type=int,
        default=100,
        help="Maximum thinking tokens (default: 100, use 0 for no thinking)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen3-0.6B",  # Small model for quick testing
        help="Model to use (default: Qwen/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8765,
        help="Port for vLLM server (default: 8765)"
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip starting server (assume it's already running)"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("VLLM THINKING BUDGET PROCESSOR TEST")
    print("=" * 70)
    
    # Setup
    setup_processor_module()
    
    process = None
    try:
        if not args.skip_server:
            process = start_vllm_server(args.model, args.max_thinking_tokens, args.port)
            
            # Stream some server output while waiting
            def stream_output():
                for line in iter(process.stdout.readline, ''):
                    print(f"[vLLM] {line.rstrip()}")
                    if "Uvicorn running" in line or "Application startup complete" in line:
                        break
            
            import threading
            output_thread = threading.Thread(target=stream_output, daemon=True)
            output_thread.start()
            
            if not wait_for_server(args.port):
                print("\n❌ Failed to start server. Check the logs above.")
                return 1
        
        # Run tests
        print("\n" + "=" * 70)
        print("RUNNING TESTS")
        print("=" * 70)
        
        success = test_generation(args.port, args.max_thinking_tokens)
        
        if success:
            print("\n✅ Test completed successfully!")
        else:
            print("\n❌ Test failed!")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    finally:
        if process:
            print("\n🛑 Stopping vLLM server...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    sys.exit(main())

