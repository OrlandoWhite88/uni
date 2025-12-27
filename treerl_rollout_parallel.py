#!/usr/bin/env python3
"""
TreeRL Rollout Parallel Processor

On-policy rollout script for TreeRL training:
- Runs beam search (k=8) on cross rulings WITHOUT gold injection
- Collects all leaves: final beam paths + pruned partial paths
- Computes fractional leaf rewards via prefix match against gold trace
- Computes TreeRL process supervision: V(s), GA(s), LA(s), R(s)
- Emits LEAF-level training samples with bundled step rewards for GRPO

OUTPUT FORMAT (per TreeRL paper):
- One sample per LEAF (complete trajectory)
- Full message trajectory included
- Step rewards R(s) bundled as an array
- Training loop processes full trajectories, applies R(s) per-step during backprop

Based on TreeRL paper: https://github.com/THUDM/TreeRL
"""

import json
import logging
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from datetime import datetime
import traceback

# Add the api directory to the Python path
script_dir = Path(__file__).parent
api_dir = script_dir / "api"
sys.path.insert(0, str(api_dir))

# CRITICAL: Set TreeRL beam size BEFORE importing classification engine
# Default k=8 for TreeRL training (wider beam = more diverse leaves)
os.environ.setdefault("TREERL_BEAM_SIZE", "8")
os.environ.setdefault("TREERL_CHAPTER_BEAM_SIZE", "8")

# CRITICAL: Disable cross ruling injection (no gold hints during rollout)
os.environ["DISABLE_CROSS_RULING_INJECTION"] = "true"

# Disable training data collection (we handle our own output)
os.environ["COLLECT_TRAINING_DATA"] = "false"

# Default model settings
os.environ.setdefault("OPENROUTER_MODEL", "openai/gpt-4.1-mini")

from api.treerl_gold_trace import build_gold_trace, build_pred_trace_from_path
from api.treerl_rewards import compute_leaf_reward, reward_breakdown
from api.treerl_process_supervision import (
    compute_treerl_rewards,
    emit_leaf_samples,
    summarize_treerl_computation
)
from llm_auto_responder import LLMAutoResponder


class TreeRLRolloutProcessor:
    """
    TreeRL on-policy rollout processor.
    
    Runs beam search on cross rulings, computes TreeRL rewards,
    and emits step-level training samples for GRPO.
    """
    
    def __init__(
        self,
        engine_name: str = "groq",
        beam_size: int = 8,
        debug: bool = False,
        quiet_mode: bool = False,
        max_workers: int = 10,
        max_questions: int = 3,
        output_file: str = "treerl_samples.jsonl",
        leaves_file: str = "treerl_leaves.jsonl"
    ):
        """
        Initialize the TreeRL rollout processor.
        
        Args:
            engine_name: Classification engine to use
            beam_size: Beam width for search (default 8 for TreeRL)
            debug: Enable debug logging
            quiet_mode: Minimal logging
            max_workers: Parallel workers
            max_questions: Max clarification questions per rollout
            output_file: Where to write step-level training samples
            leaves_file: Where to write raw leaves for analysis
        """
        self.engine_name = engine_name
        self.beam_size = beam_size
        self.debug = debug
        self.quiet_mode = quiet_mode
        self.max_workers = max_workers
        self.max_questions = max_questions
        
        # Initialize LLM auto-responder for handling clarification questions
        # Questions are answered but don't count as TreeRL "steps" (only classification decisions do)
        self.auto_responder = LLMAutoResponder(engine_name=engine_name, debug=debug)
        self.output_file = output_file
        self.leaves_file = leaves_file
        
        # Set beam size env vars
        os.environ["TREERL_BEAM_SIZE"] = str(beam_size)
        os.environ["TREERL_CHAPTER_BEAM_SIZE"] = str(beam_size)
        
        # Thread-safe file writing
        self.write_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        
        # Progress tracking
        self.total_processed = 0
        self.successful_count = 0
        self.failed_count = 0
        self.total_leaves = 0
        self.total_samples = 0
        self.start_time = 0
        self.last_progress_time = 0
        
        # Detailed stats for debugging
        self.total_beam_leaves = 0
        self.total_pruned_leaves = 0
        self.perfect_matches = 0  # reward == 1.0
        self.partial_matches = 0  # 0 < reward < 1.0
        self.zero_matches = 0     # reward == 0.0
        self.reward_sum = 0.0
        self.v_root_sum = 0.0
        self.ruling_details = []  # Store per-ruling summaries
        
        self.setup_logging()
        self.load_engine()
        
    def setup_logging(self):
        """Configure logging."""
        if self.debug:
            level = logging.DEBUG
        elif self.quiet_mode:
            level = logging.WARNING
        else:
            level = logging.INFO
            
        file_handler = logging.FileHandler('treerl_rollout.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
        ))
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        for logger_name in ['httpx', 'httpcore', 'urllib3', 'google.auth', 'openai', 'groq']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
            
        self.logger = logging.getLogger(__name__)
        
    def load_engine(self):
        """Load the classification engine."""
        available_engines = {
            "cerebras": "cerebras_tree_engine",
            "groq": "groq_tree_engine",
            "tree": "tree_engine",
            "gemini": "gemini_tree_engine"
        }
        
        if self.engine_name not in available_engines:
            raise ValueError(f"Unknown engine '{self.engine_name}'")
        
        module_name = available_engines[self.engine_name]
        module = __import__(f"api.{module_name}", fromlist=[module_name])
        
        if not hasattr(module, 'HTSTree'):
            raise ImportError(f"Engine '{self.engine_name}' has no HTSTree class")
        
        self.HTSTree = module.HTSTree
        self.logger.info(f"Using {self.engine_name} engine with beam_size={self.beam_size}")
        
    def load_cross_rulings(self, file_path: str) -> List[Dict]:
        """Load cross rulings from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            cross_rulings = json.load(f)
        self.logger.info(f"Loaded {len(cross_rulings)} cross rulings from {file_path}")
        return cross_rulings
    
    def collect_all_leaves(
        self,
        result: Dict,
        state: Dict,
        gold_trace: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Collect all leaves from a rollout: final beam paths + pruned partial paths.
        
        Args:
            result: Classification result
            state: Classification state (contains beam and pruned leaves)
            gold_trace: Gold trace for reward computation
            
        Returns:
            List of leaf dicts with pred_trace, reward, trajectory, etc.
        """
        leaves = []
        
        # 1. Collect final beam paths
        beam = state.get("beam", [])
        for path_data in beam:
            if isinstance(path_data, dict):
                classification_path = path_data.get("classification_path", [])
                trajectory = path_data.get("trajectory", [])
                path_id = path_data.get("path_id", "unknown")
            elif hasattr(path_data, "classification_path"):
                classification_path = path_data.classification_path
                trajectory = path_data.trajectory if hasattr(path_data, "trajectory") else []
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
                "is_complete": path_data.get("is_complete", False) if isinstance(path_data, dict) else getattr(path_data, "is_complete", False)
            })
        
        # 2. Collect pruned partial paths
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
                "is_complete": pruned.get("is_complete", False),
                "pruned_at_iteration": pruned.get("pruned_at_iteration", -1)
            })
        
        return leaves
    
    def process_single_ruling(self, args: Tuple[Dict, int]) -> Dict[str, Any]:
        """
        Process a single cross ruling: run beam search and compute TreeRL rewards.
        
        Args:
            args: (cross_ruling dict, index)
            
        Returns:
            Dict with leaves, step_samples, and metadata
        """
        cross_ruling, index = args
        product_description = cross_ruling.get("short_product_description", "")
        gold_code = cross_ruling.get("hts_code", "")
        
        thread_logger = logging.getLogger(f"{__name__}.worker_{index}")
        thread_logger.info(f"Processing ruling {index + 1}: {product_description[:50]}...")
        
        try:
            # Create HTS tree instance
            hts_tree = self.HTSTree()
            
            # Load HTS data
            hts_data_file = script_dir / "api" / "hts_data.json"
            with open(hts_data_file, "r", encoding="utf-8") as f:
                hts_data = json.load(f)
            hts_tree.build_from_json(hts_data)
            
            # Build gold trace BEFORE classification
            gold_trace = build_gold_trace(gold_code, hts_tree.navigator)
            thread_logger.debug(f"Gold trace has {len(gold_trace)} steps")
            
            # Run classification WITH auto-responder for questions
            # Note: Questions are answered by LLM using cross_ruling context
            # Gold info is used ONLY for: (1) auto-responses (2) reward computation
            # Model doesn't see gold during its actual classification decisions
            if self.max_questions > 0:
                # Use auto-responder for interactive Q&A
                result = self.auto_responder.interactive_classify_with_auto_response(
                    hts_tree=hts_tree,
                    product_description=product_description,
                    cross_ruling=cross_ruling,  # Full context for auto-answers
                    max_questions=self.max_questions
                )
                questions_answered = result.get("total_questions", 0)
                if questions_answered > 0:
                    thread_logger.info(f"Auto-answered {questions_answered} questions during classification")
            else:
                # No questions - pure beam search
                result = hts_tree.start_classification(
                    product=product_description,
                    interactive=False,
                    max_questions=0,
                    use_multi_hypothesis=True,
                    hypothesis_count=self.beam_size
                )
            
            # Get the state (may need deserialization)
            state = result.get("state", {})
            
            # Debug: log state keys to diagnose pruned leaves issue
            thread_logger.debug(f"State keys: {list(state.keys())}")
            pruned_in_state = state.get("_treerl_pruned_leaves", [])
            thread_logger.debug(f"Pruned leaves in state: {len(pruned_in_state)}")
            
            # Collect all leaves (final + pruned)
            leaves = self.collect_all_leaves(result, state, gold_trace)
            thread_logger.info(f"Collected {len(leaves)} trajectories ({sum(1 for l in leaves if l['source'] == 'final_beam')} complete leaves, {sum(1 for l in leaves if l['source'] == 'pruned')} partial paths)")
            
            if not leaves:
                thread_logger.warning("No leaves collected!")
                return {
                    "index": index,
                    "success": False,
                    "error": "No leaves collected",
                    "gold_code": gold_code
                }
            
            # Compute TreeRL process supervision
            step_rewards, v_root = compute_treerl_rewards(leaves)
            
            # Emit LEAF-level training samples (correct format per TreeRL paper)
            # One sample per leaf with bundled step rewards
            leaf_samples = emit_leaf_samples(
                leaves, 
                step_rewards,
                gold_trace=gold_trace,
                gold_code=gold_code,
                ruling_index=index
            )
            
            # Compute summary stats
            summary = summarize_treerl_computation(leaves, step_rewards, v_root)
            
            thread_logger.info(
                f"TreeRL: V(root)={v_root:.3f}, "
                f"mean_R={summary['mean_R']:.3f}, "
                f"leaves={summary['num_leaves']}, "
                f"leaf_samples={len(leaf_samples)}"
            )
            
            # Check if any leaf got perfect reward
            max_reward = max(leaf["reward"] for leaf in leaves)
            best_leaf = next((l for l in leaves if l["reward"] == max_reward), None)
            
            # Compute detailed stats for this ruling
            beam_count = sum(1 for l in leaves if l["source"] == "final_beam")
            pruned_count = sum(1 for l in leaves if l["source"] == "pruned")
            perfect_count = sum(1 for l in leaves if l["reward"] == 1.0)
            partial_count = sum(1 for l in leaves if 0 < l["reward"] < 1.0)
            zero_count = sum(1 for l in leaves if l["reward"] == 0.0)
            
            # Build best path string for comparison
            best_pred_trace = best_leaf.get("pred_trace", []) if best_leaf else []
            best_path_str = " > ".join([f"{s.get('kind', '?')}:{s.get('code', s.get('node_id', '?'))}" for s in best_pred_trace[:4]])
            gold_path_str = " > ".join([f"{s.get('kind', '?')}:{s.get('code', s.get('node_id', '?'))}" for s in gold_trace[:4]])
            
            # Log detailed comparison
            thread_logger.info(
                f"  Gold: {gold_code} ({gold_path_str}...)"
            )
            thread_logger.info(
                f"  Best: reward={max_reward:.2f}, src={best_leaf['source'] if best_leaf else 'N/A'} ({best_path_str}...)"
            )
            thread_logger.info(
                f"  Trajectories: {beam_count} leaves + {pruned_count} partial | "
                f"Perfect: {perfect_count}, Partial: {partial_count}, Zero: {zero_count}"
            )
            
            return {
                "index": index,
                "success": True,
                "gold_code": gold_code,
                "gold_trace": gold_trace,
                "leaves": leaves,
                "leaf_samples": leaf_samples,  # One per leaf with bundled step rewards
                "step_rewards": {str(k): v for k, v in step_rewards.items()},  # Convert keys for JSON
                "v_root": v_root,
                "summary": summary,
                "max_reward": max_reward,
                "best_leaf_source": best_leaf["source"] if best_leaf else None,
                # Detailed stats
                "beam_count": beam_count,
                "pruned_count": pruned_count,
                "perfect_count": perfect_count,
                "partial_count": partial_count,
                "zero_count": zero_count,
                "best_path_str": best_path_str,
                "gold_path_str": gold_path_str
            }
            
        except Exception as e:
            thread_logger.error(f"Error processing ruling {index + 1}: {e}")
            if self.debug:
                thread_logger.error(traceback.format_exc())
            return {
                "index": index,
                "success": False,
                "error": str(e),
                "gold_code": gold_code
            }
    
    def write_results(self, result: Dict[str, Any]):
        """Write results to output files (thread-safe)."""
        with self.write_lock:
            # Write LEAF-level samples (correct TreeRL format)
            # One sample per leaf with full trajectory + bundled step rewards
            leaf_samples = result.get("leaf_samples", [])
            if leaf_samples:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    for sample in leaf_samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # Write leaves summary for analysis (separate file)
            leaves = result.get("leaves", [])
            if leaves:
                with open(self.leaves_file, 'a', encoding='utf-8') as f:
                    for leaf in leaves:
                        leaf_record = {
                            "ruling_index": result.get("index", -1),
                            "gold_code": result.get("gold_code", ""),
                            "path_id": leaf.get("path_id", ""),
                            "reward": leaf.get("reward", 0.0),
                            "source": leaf.get("source", ""),
                            "is_complete": leaf.get("is_complete", False),
                            "pred_trace": leaf.get("pred_trace", []),
                            "classification_path": leaf.get("classification_path", [])
                        }
                        f.write(json.dumps(leaf_record, ensure_ascii=False) + '\n')
    
    def process_rolling(self, cross_rulings: List[Dict], total_count: int) -> Dict:
        """Process cross rulings with rolling parallelization."""
        self.logger.info(f"Starting TreeRL rollout for {total_count} rulings")
        self.logger.info(f"Beam size: {self.beam_size}, Workers: {self.max_workers}")
        
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        
        work_queue = Queue()
        for i, ruling in enumerate(cross_rulings):
            work_queue.put((ruling, i))
        
        # Clear output files
        open(self.output_file, 'w').close()
        open(self.leaves_file, 'w').close()
        
        self.logger.warning(f"🚀 Starting TreeRL rollout with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            active_futures = {}
            
            # Submit initial batch
            for _ in range(min(self.max_workers, work_queue.qsize())):
                try:
                    work_item = work_queue.get_nowait()
                    future = executor.submit(self.process_single_ruling, work_item)
                    active_futures[future] = work_item
                except Empty:
                    break
            
            while active_futures or not work_queue.empty():
                try:
                    completed_futures = []
                    for future in as_completed(active_futures.keys(), timeout=1.0):
                        completed_futures.append(future)
                        break
                    
                    for future in completed_futures:
                        try:
                            result = future.result()
                            active_futures.pop(future)
                            
                            # Write results
                            if result.get("success"):
                                self.write_results(result)
                            
                            # Update progress
                            with self.progress_lock:
                                if result.get("success"):
                                    self.successful_count += 1
                                    self.total_leaves += len(result.get("leaves", []))
                                    self.total_samples += len(result.get("leaf_samples", []))
                                    
                                    # Detailed stats
                                    self.total_beam_leaves += result.get("beam_count", 0)
                                    self.total_pruned_leaves += result.get("pruned_count", 0)
                                    self.perfect_matches += result.get("perfect_count", 0)
                                    self.partial_matches += result.get("partial_count", 0)
                                    self.zero_matches += result.get("zero_count", 0)
                                    self.reward_sum += result.get("max_reward", 0.0)
                                    self.v_root_sum += result.get("v_root", 0.0)
                                    
                                    # Store per-ruling summary
                                    self.ruling_details.append({
                                        "index": result.get("index"),
                                        "gold_code": result.get("gold_code"),
                                        "max_reward": result.get("max_reward"),
                                        "v_root": result.get("v_root"),
                                        "beam": result.get("beam_count"),
                                        "pruned": result.get("pruned_count"),
                                        "perfect": result.get("perfect_count")
                                    })
                                else:
                                    self.failed_count += 1
                                self.total_processed += 1
                                
                                # Progress reporting
                                current_time = time.time()
                                if current_time - self.last_progress_time >= 15.0:
                                    elapsed = current_time - self.start_time
                                    rate = self.total_processed / elapsed if elapsed > 0 else 0
                                    eta = (total_count - self.total_processed) / rate if rate > 0 else 0
                                    
                                    self.logger.warning(
                                        f"⏰ [{datetime.now().strftime('%H:%M:%S')}]: "
                                        f"{self.total_processed}/{total_count} "
                                        f"({self.successful_count} ok, {rate*60:.1f}/min, "
                                        f"ETA: {eta/60:.1f}min, "
                                        f"leaves: {self.total_leaves}, samples: {self.total_samples})"
                                    )
                                    self.last_progress_time = current_time
                            
                            # Start next job
                            if not work_queue.empty():
                                try:
                                    next_item = work_queue.get_nowait()
                                    next_future = executor.submit(self.process_single_ruling, next_item)
                                    active_futures[next_future] = next_item
                                except Empty:
                                    pass
                                    
                        except Exception as e:
                            self.logger.error(f"Worker failed: {e}")
                            active_futures.pop(future, None)
                            with self.progress_lock:
                                self.failed_count += 1
                                self.total_processed += 1
                            
                            if not work_queue.empty():
                                try:
                                    next_item = work_queue.get_nowait()
                                    next_future = executor.submit(self.process_single_ruling, next_item)
                                    active_futures[next_future] = next_item
                                except Empty:
                                    pass
                
                except:
                    pass
        
        return {
            "total_processed": self.total_processed,
            "successful": self.successful_count,
            "failed": self.failed_count,
            "total_leaves": self.total_leaves,
            "total_samples": self.total_samples,
            # Detailed stats
            "total_beam_leaves": self.total_beam_leaves,
            "total_pruned_leaves": self.total_pruned_leaves,
            "perfect_matches": self.perfect_matches,
            "partial_matches": self.partial_matches,
            "zero_matches": self.zero_matches,
            "avg_max_reward": self.reward_sum / self.successful_count if self.successful_count > 0 else 0.0,
            "avg_v_root": self.v_root_sum / self.successful_count if self.successful_count > 0 else 0.0,
            "ruling_details": self.ruling_details
        }
    
    def process_dataset(
        self,
        file_path: str,
        max_count: Optional[int] = None,
        start_index: int = 0
    ) -> Dict:
        """Process a cross rulings dataset."""
        cross_rulings = self.load_cross_rulings(file_path)
        
        if start_index > 0:
            cross_rulings = cross_rulings[start_index:]
        if max_count:
            cross_rulings = cross_rulings[:max_count]
        
        total_count = len(cross_rulings)
        
        self.logger.info(f"Processing {total_count} cross rulings for TreeRL")
        self.logger.info(f"Output: {self.output_file} (step samples)")
        self.logger.info(f"Output: {self.leaves_file} (leaves)")
        
        start_time = time.time()
        results = self.process_rolling(cross_rulings, total_count)
        elapsed = time.time() - start_time
        
        summary = {
            **results,
            "elapsed_time": elapsed,
            "avg_time_per_item": elapsed / total_count if total_count > 0 else 0,
            "output_file": self.output_file,
            "leaves_file": self.leaves_file,
            "beam_size": self.beam_size
        }
        
        self.logger.warning(f"\n{'='*70}")
        self.logger.warning("TREERL ROLLOUT COMPLETE")
        self.logger.warning(f"{'='*70}")
        
        # Basic stats
        self.logger.warning(f"Total processed: {summary['total_processed']}")
        self.logger.warning(f"Successful: {summary['successful']}")
        self.logger.warning(f"Failed: {summary['failed']}")
        
        # Trajectory breakdown
        self.logger.warning(f"\n--- TRAJECTORY BREAKDOWN ---")
        self.logger.warning(f"Total trajectories: {summary['total_leaves']}")
        self.logger.warning(f"  - Complete leaves (final beam): {summary.get('total_beam_leaves', 0)}")
        self.logger.warning(f"  - Partial paths (pruned): {summary.get('total_pruned_leaves', 0)}")
        self.logger.warning(f"Total samples: {summary['total_samples']} (one per trajectory with bundled step rewards)")
        self.logger.warning(f"Avg trajectories per ruling: {summary['total_leaves'] / summary['successful']:.1f}" if summary['successful'] > 0 else "N/A")
        
        # Reward breakdown
        self.logger.warning(f"\n--- REWARD BREAKDOWN ---")
        self.logger.warning(f"Perfect matches (reward=1.0): {summary.get('perfect_matches', 0)}")
        self.logger.warning(f"Partial matches (0<reward<1): {summary.get('partial_matches', 0)}")
        self.logger.warning(f"Zero matches (reward=0.0): {summary.get('zero_matches', 0)}")
        self.logger.warning(f"Avg max reward per ruling: {summary.get('avg_max_reward', 0):.3f}")
        self.logger.warning(f"Avg V(root): {summary.get('avg_v_root', 0):.3f}")
        
        # Per-ruling details
        ruling_details = summary.get('ruling_details', [])
        if ruling_details:
            self.logger.warning(f"\n--- PER-RULING DETAILS ---")
            for rd in ruling_details:
                match_status = "✓ PERFECT" if rd['perfect'] > 0 else ("◐ PARTIAL" if rd['max_reward'] > 0 else "✗ MISS")
                self.logger.warning(
                    f"  [{rd['index']:3d}] {rd['gold_code']:14s} | "
                    f"max_r={rd['max_reward']:.2f} V(root)={rd['v_root']:.3f} | "
                    f"leaves={rd['beam']} partial={rd['pruned']} | {match_status}"
                )
        
        # Timing
        self.logger.warning(f"\n--- TIMING ---")
        self.logger.warning(f"Total time: {summary['elapsed_time']:.2f}s")
        rate = summary['successful'] / summary['elapsed_time'] * 60 if summary['elapsed_time'] > 0 else 0
        self.logger.warning(f"Rate: {rate:.1f} rulings/min")
        self.logger.warning(f"Output: {summary['output_file']}")
        self.logger.warning(f"{'='*70}")
        
        return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TreeRL On-Policy Rollout Processor")
    parser.add_argument("--cross-rulings-file", default="cross_rulings_dataset.json",
                       help="Path to cross rulings JSON file")
    parser.add_argument("--engine", choices=["cerebras", "groq", "tree", "gemini"],
                       default="groq", help="Classification engine")
    parser.add_argument("--beam-size", type=int, default=8,
                       help="Beam width for search (default: 8 for TreeRL)")
    parser.add_argument("--max-count", type=int, default=None,
                       help="Max rulings to process")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Start index for resuming")
    parser.add_argument("--max-workers", type=int, default=10,
                       help="Parallel workers")
    parser.add_argument("--max-questions", type=int, default=5,
                       help="Max clarification questions auto-answered by LLM (0 = non-interactive, default=5)")
    parser.add_argument("--output-file", default="treerl_samples.jsonl",
                       help="Output file for leaf-level samples (one per leaf with bundled step rewards)")
    parser.add_argument("--leaves-file", default="treerl_leaves.jsonl",
                       help="Output file for leaf summaries (for analysis)")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    processor = TreeRLRolloutProcessor(
        engine_name=args.engine,
        beam_size=args.beam_size,
        debug=args.debug,
        quiet_mode=args.quiet,
        max_workers=args.max_workers,
        max_questions=args.max_questions,
        output_file=args.output_file,
        leaves_file=args.leaves_file
    )
    
    try:
        summary = processor.process_dataset(
            args.cross_rulings_file,
            args.max_count,
            args.start_index
        )
        return 0 if summary['successful'] > 0 else 1
        
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

