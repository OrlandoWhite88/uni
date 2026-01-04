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
import math
import logging
import argparse
import random
import time
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
# MODEL LOADING (Simplified for Unsloth)
# ============================================================================

def load_model_and_tokenizer(config: TreeRLConfig, logger: logging.Logger):
    """
    Load model with Unsloth using 4-bit quantization.
    
    This is the CORRECT way to load for a custom training loop:
    1. Load base model with FastLanguageModel.from_pretrained
    2. Add LoRA adapters with FastLanguageModel.get_peft_model
    3. Optionally load weights from existing adapter
    
    Returns:
        model: The model with LoRA adapters ready for training
        tokenizer: The tokenizer
    """
    logger.info("=" * 70)
    logger.info("LOADING MODEL WITH UNSLOTH")
    logger.info("=" * 70)
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"SFT adapter: {config.sft_adapter}")
    logger.info(f"4-bit quantization: {config.load_in_4bit}")
    logger.info(f"Max sequence length: {config.max_seq_length}")
    
    try:
        from unsloth import FastLanguageModel
        
        if config.sft_adapter:
            # OPTION 1: Load existing adapter directly
            # This is the CORRECT way to continue training an existing LoRA
            logger.info(f"Loading model with existing SFT adapter: {config.sft_adapter}")
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.sft_adapter,  # Load adapter directly!
                max_seq_length=config.max_seq_length,
                load_in_4bit=config.load_in_4bit,
                offload_embedding=config.offload_embedding,
                trust_remote_code=True,
            )
            logger.info("Model + SFT adapter loaded successfully")
            
            # The adapter is already attached - just make sure it's trainable
            # Unsloth should have set this up, but let's verify
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
            
            # Get the LoRA rank from loaded adapter for logging
            loaded_rank = "unknown"
            for name, param in model.named_parameters():
                if "lora_A" in name:
                    loaded_rank = param.shape[0]
                    break
            logger.info(f"Loaded adapter LoRA rank: {loaded_rank}")
            
        else:
            # OPTION 2: Fresh LoRA from base model
            logger.info("Loading base model and creating fresh LoRA adapters")
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.base_model,
                max_seq_length=config.max_seq_length,
                load_in_4bit=config.load_in_4bit,
                offload_embedding=config.offload_embedding,
                trust_remote_code=True,
            )
            logger.info("Base model loaded successfully")
            
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
            logger.info(f"Fresh LoRA adapters added (rank={config.lora_rank}, alpha={config.lora_alpha})")
        
        USING_UNSLOTH = True
        
    except ImportError as e:
        logger.warning(f"Unsloth not available ({e}), falling back to standard loading")
        USING_UNSLOTH = False
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig
        
        # Standard loading without Unsloth
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True,
        )
        
        # Add LoRA
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=list(config.lora_target_modules),
            lora_dropout=0.0,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        
        if config.sft_adapter:
            try:
                model.load_adapter(config.sft_adapter, adapter_name="sft")
                model.set_adapter("sft")
            except Exception as e:
                logger.warning(f"Could not load adapter: {e}")
        
        logger.info("Model loaded with standard transformers")
    
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
# LOCAL LLM CLIENT FOR ONLINE ROLLOUTS
# ============================================================================

class LocalLLMClient:
    """
    Local LLM client for rollouts during training.
    
    Key differences from original:
    1. NO FastLanguageModel.for_inference() calls - this breaks training
    2. Simple model.eval() / model.train() switching
    3. Standard HuggingFace generation
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        device: str,
        max_seq_length: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        logger: logging.Logger,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logger = logger
        
        # Try to load system prompt from your API
        try:
            from api.system_prompts_updated import UNIFIED_SYSTEM_PROMPT
            self.system_prompt = UNIFIED_SYSTEM_PROMPT.strip()
        except ImportError:
            self.system_prompt = "You are a helpful assistant."
        
        self._system_prompt_injection: Optional[str] = None
        
        # Compatibility attributes expected by classification_engine.py
        self.log_prompts = False  # Disable verbose prompt logging during training
        self.prompt_logger = logger  # Use same logger if needed
        
        # JSON requirements for structured outputs
        self._json_requirements = (
            "CRITICAL JSON REQUIREMENTS:\n"
            "- Return ONLY valid JSON with no additional text\n"
            "- Do not include any explanation before or after the JSON\n"
            "- If generating an array, ensure it starts with [ and ends with ]\n"
            "- All JSON objects must be properly separated by commas within the array\n"
            "- Do not generate separate JSON objects - they must be inside an array"
        )

    def set_system_prompt_injection(self, prompt: Optional[str]) -> None:
        self._system_prompt_injection = prompt

    def clear_system_prompt_injection(self) -> None:
        self._system_prompt_injection = None

    def _current_system_prompt(self) -> str:
        return (self._system_prompt_injection or self.system_prompt).strip()

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from model response, handling code fences."""
        if not response_text:
            raise ValueError("No response text to parse.")

        text = response_text.strip()

        # Strip code fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Fast path: whole string is JSON
        try:
            json.loads(text)
            return text
        except Exception:
            pass

        # Try to locate array
        start_a = text.find("[")
        end_a = text.rfind("]")
        if start_a != -1 and end_a != -1 and end_a > start_a:
            candidate = text[start_a:end_a + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        # Try to locate object
        start_o = text.find("{")
        end_o = text.rfind("}")
        if start_o != -1 and end_o != -1 and end_o > start_o:
            candidate = text[start_o:end_o + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        raise ValueError(f"Failed to extract valid JSON from response: {response_text[:200]}...")

    @torch.no_grad()
    def _generate_from_messages(self, messages: List[Dict[str, str]], temperature: float) -> str:
        """
        Generate a response from a list of chat messages.
        
        IMPORTANT: We do NOT call FastLanguageModel.for_inference() here!
        That method is for final inference only and breaks the training loop.
        Instead, we just use model.eval() temporarily.
        """
        gen_start = time.time()
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        
        self.logger.debug(f"    [LLM] Generating... input_tokens={input_len}")
        
        # Temporarily switch to eval mode
        was_training = self.model.training
        self.model.eval()
        
        try:
            do_sample = float(temperature) > 0.0
            
            # Standard HuggingFace generation
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        finally:
            # Restore training mode if it was active
            if was_training:
                self.model.train()
        
        # Decode only the newly generated tokens
        output_ids = gen[0][input_len:]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        gen_elapsed = time.time() - gen_start
        num_tokens = len(output_ids)
        tok_per_sec = num_tokens / gen_elapsed if gen_elapsed > 0 else 0
        self.logger.debug(f"    [LLM] Generated {num_tokens} tokens in {gen_elapsed:.1f}s ({tok_per_sec:.1f} tok/s)")
        
        return output_text

    def send_openai_request(
        self,
        prompt: str,
        requires_json: bool = False,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Send a single-turn request."""
        user_content = prompt.strip()
        if requires_json:
            user_content = f"{user_content.rstrip()}\n\n{self._json_requirements}"

        messages = [
            {"role": "system", "content": self._current_system_prompt()},
            {"role": "user", "content": user_content},
        ]

        text = self._generate_from_messages(messages, temperature=temperature)

        if requires_json:
            cleaned = self._extract_json_from_response(text)
            json.loads(cleaned)  # Validate
            return cleaned
        return text

    def send_trajectory_request(
        self,
        messages: List[Dict[str, str]],
        requires_json: bool = False,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Send a multi-turn request."""
        req_messages = [m.copy() for m in messages]
        if requires_json:
            for i in range(len(req_messages) - 1, -1, -1):
                if req_messages[i].get("role") == "user":
                    req_messages[i]["content"] = f"{req_messages[i]['content'].rstrip()}\n\n{self._json_requirements}"
                    break

        text = self._generate_from_messages(req_messages, temperature=temperature)

        if requires_json:
            cleaned = self._extract_json_from_response(text)
            json.loads(cleaned)  # Validate
            return cleaned
        return text

    # Aliases for compatibility
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
    local_llm: LocalLLMClient,
) -> List[Dict]:
    """
    Run beam search rollout for a single ruling and compute TreeRL rewards.
    
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
        
        # Inject local LLM client
        logger.debug(f"  [rollout] Injecting local LLM client...")
        hts_tree.llm_client = local_llm
        hts_tree.client = None
        if hasattr(hts_tree, "classification_engine") and hasattr(hts_tree.classification_engine, "llm"):
            hts_tree.classification_engine.llm = local_llm
        if hasattr(hts_tree, "streaming_engine") and hasattr(hts_tree.streaming_engine, "llm_client"):
            hts_tree.streaming_engine.llm_client = local_llm
        
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
    local_llm: LocalLLMClient,
) -> Dict[str, float]:
    """
    Train for one epoch with online rollouts.
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
        
        # Run online rollout
        samples = run_online_rollout(ruling, config, logger, local_llm)
        
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
            import gc
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
# MAIN TRAINING FUNCTION
# ============================================================================

def train(config: TreeRLConfig):
    """Main training function."""
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("=" * 70)
    logger.info("TREERL GRPO TRAINING (FIXED FOR UNSLOTH)")
    logger.info("=" * 70)
    logger.info(f"Config: {config}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer, using_unsloth = load_model_and_tokenizer(config, logger)
    
    # Note: With Unsloth, model is already on the correct device
    # Only move if not using Unsloth
    if not using_unsloth:
        logger.info(f"Moving model to {config.device}...")
        model.to(config.device)
    logger.info("Model ready")
    
    # Create local LLM client for rollouts
    logger.info("Creating LocalLLMClient for rollouts...")
    local_llm = LocalLLMClient(
        model=model,
        tokenizer=tokenizer,
        device=config.device,
        max_seq_length=config.max_seq_length,
        max_new_tokens=config.rollout_max_new_tokens,
        temperature=config.rollout_temperature,
        top_p=config.rollout_top_p,
        logger=logger,
    )
    logger.info("LocalLLMClient ready")
    
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
        
        # Train epoch
        epoch_metrics = train_epoch(
            model, tokenizer, optimizer,
            rulings, config, logger, epoch, local_llm
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
        
        # Save checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0:
            checkpoint_dir = os.path.join(
                config.output_dir, 
                f"checkpoint-epoch-{epoch + 1}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            logger.info(f"  Checkpoint saved: {checkpoint_dir}")
    
    # Training complete
    total_time = time.time() - training_start
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Final average loss: {all_metrics[-1].get('avg_loss', 0):.4f}")
    
    # Save final model
    final_dir = os.path.join(config.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Final model saved: {final_dir}")
    
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
                       help="SFT LoRA adapter to continue training from (HF Hub or local path)")
    parser.add_argument("--no-sft-adapter", action="store_true",
                       help="Train fresh LoRA from base model (don't load SFT adapter)")
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
    
    args = parser.parse_args()
    
    # Handle SFT adapter logic
    sft_adapter = args.sft_adapter if not args.no_sft_adapter else ""
    
    # Build config
    config = TreeRLConfig(
        base_model=args.base_model,
        sft_adapter=sft_adapter,
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
    )
    
    train(config)


if __name__ == "__main__":
    main()
