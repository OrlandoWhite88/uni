#!/usr/bin/env python3
"""
Parallel Cross Rulings Training Dataset Processor

Processes cross rulings dataset with massive parallelization:
- Processes 100+ classifications simultaneously
- Saves results sequentially to prevent data corruption
- Includes retry logic for failed classifications
- Handles large datasets efficiently
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

# CRITICAL: Enable training data collection BEFORE importing training_collector
os.environ["COLLECT_TRAINING_DATA"] = "true"

# CRITICAL: Disable cross ruling injection during training (prevents cheating)
# At training time, the model should learn from the classification path without hints
# At inference time, cross rulings provide helpful context for accuracy
os.environ["DISABLE_CROSS_RULING_INJECTION"] = "true"

# Default OpenRouter model for this processor
os.environ.setdefault("OPENROUTER_MODEL", "openai/gpt-5.2")
# Use OpenRouter's "ignore" field to skip providers (not "deny")
os.environ.setdefault("OPENROUTER_PROVIDER_IGNORE", "")

from api.system_prompts_updated import UNIFIED_SYSTEM_PROMPT
from api.training_data_collector import training_collector
from llm_auto_responder import LLMAutoResponder

class ParallelCrossRulingsProcessor:
    def __init__(self, 
                 engine_name: str = "groq", 
                 debug: bool = False, 
                 quiet_mode: bool = False,
                 use_auto_responder: bool = True,
                 max_workers: int = 100,
                 batch_size: int = 2,
                 max_retries: int = 2,
                 hypothesis_count: int = 2,
                 save_trajectories: bool = False):
        """
        Initialize the parallel processor
        
        Args:
            engine_name: Classification engine to use (groq engine uses Gemini LLM by default)
            debug: Enable debug logging
            quiet_mode: Enable quiet mode (minimal logging)
            use_auto_responder: Use LLM auto-responder for questions
            max_workers: Maximum number of parallel workers
            batch_size: Number of items to process in each batch
            max_retries: Maximum retries for failed classifications (default: 2 = 3 total attempts)
            hypothesis_count: Beam size for classification (default: 2 for training efficiency)
            save_trajectories: Save trajectory files for TreeRL training (default: False)
        """
        self.engine_name = engine_name
        self.debug = debug
        self.quiet_mode = quiet_mode
        self.use_auto_responder = use_auto_responder
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.hypothesis_count = hypothesis_count
        self.save_trajectories = save_trajectories
        self.openrouter_model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b")
        
        # Thread-safe components for progress tracking only
        self.progress_lock = threading.Lock()
        
        # Progress tracking
        self.total_processed = 0
        self.successful_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.start_time = 0
        self.last_progress_time = 0
        
        # Setup logging
        self.setup_logging()
        
        # Load engine module once
        self.load_engine()
        
    def setup_logging(self):
        """Configure logging for parallel processing"""
        if self.debug:
            level = logging.DEBUG
        elif self.quiet_mode:
            level = logging.WARNING  # Only warnings and errors in quiet mode
        else:
            level = logging.INFO
            
        # Configure file logging
        file_handler = logging.FileHandler('cross_rulings_parallel.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Configure console logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Suppress noisy loggers
        for logger_name in ['httpx', 'httpcore', 'urllib3', 'google.auth', 'openai', 'groq']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
            
        self.logger = logging.getLogger(__name__)
        
    def load_engine(self):
        """Load the specified classification engine"""
        available_engines = {
            "cerebras": "cerebras_tree_engine",
            "groq": "groq_tree_engine", 
            "tree": "tree_engine",
            "gemini": "gemini_tree_engine"
        }
        
        if self.engine_name not in available_engines:
            available = ", ".join(available_engines.keys())
            raise ValueError(f"Unknown engine '{self.engine_name}'. Available: {available}")
        
        module_name = available_engines[self.engine_name]
        
        try:
            module = __import__(f"api.{module_name}", fromlist=[module_name])
            if not hasattr(module, 'HTSTree'):
                raise ImportError(f"Engine '{self.engine_name}' does not have an HTSTree class")
            
            self.HTSTree = module.HTSTree
            self.logger.info(f"Using {self.engine_name} engine ({module_name})")
            
        except ImportError as e:
            self.logger.error(f"Error importing {self.engine_name} engine: {e}")
            raise
            
    def load_cross_rulings(self, file_path: str) -> List[Dict]:
        """Load cross rulings from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cross_rulings = json.load(f)
            
            self.logger.info(f"Loaded {len(cross_rulings)} cross rulings from {file_path}")
            return cross_rulings
            
        except Exception as e:
            self.logger.error(f"Error loading cross rulings: {e}")
            raise
    
    def extract_and_save_trajectory(self, result: Dict, product: str, gold_code: str, index: int) -> Optional[List[str]]:
        """
        Extract trajectories from classification result and save to files for TreeRL training.
        
        Args:
            result: Classification result containing beam and state
            product: Product description
            gold_code: Expected/gold HS code for computing is_correct
            index: Cross ruling index for filename uniqueness
        
        Returns:
            List of saved filenames, or None if saving disabled/failed
        """
        if not self.save_trajectories:
            return None
        
        state = result.get("state", {})
        beam = state.get("beam", [])
        
        if not beam or not isinstance(beam, list):
            self.logger.debug(f"No beam found in result for trajectory saving")
            return None
        
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_product = "".join(c if c.isalnum() or c in " -_" else "_" for c in product[:30])
        
        # Save top 3 trajectories (or however many are available)
        for rank, path_data in enumerate(beam[:3], 1):
            # Extract trajectory from path
            if isinstance(path_data, dict):
                trajectory = path_data.get("trajectory", [])
                path_id = path_data.get("path_id", f"path_{rank}")
                final_code = path_data.get("selection", {}).get("tariff") or path_data.get("selection", {}).get("subheading") or path_data.get("selection", {}).get("heading")
                cumulative_conf = path_data.get("cumulative_confidence", 0)
                classification_path = path_data.get("classification_path", [])
                full_path = " > ".join([f"{step.get('code', '')} - {step.get('description', '')}" for step in classification_path])
            elif hasattr(path_data, "trajectory"):
                trajectory = path_data.trajectory
                path_id = path_data.path_id
                final_code = path_data.selection.get("tariff") or path_data.selection.get("subheading") or path_data.selection.get("heading")
                cumulative_conf = path_data.cumulative_confidence
                classification_path = getattr(path_data, 'classification_path', [])
                full_path = " > ".join([f"{step.get('code', '')} - {step.get('description', '')}" for step in classification_path])
            else:
                continue
            
            if not trajectory:
                continue
            
            # Extract path_trace for TreeRL training
            if isinstance(path_data, dict):
                classification_path_data = path_data.get("classification_path", [])
            elif hasattr(path_data, "classification_path"):
                classification_path_data = path_data.classification_path
            else:
                classification_path_data = []
            
            path_trace = [step.get("code", "") for step in classification_path_data if step.get("code")]
            
            # Compute is_correct for TreeRL training
            final_code_normalized = str(final_code).replace(".", "") if final_code else ""
            gold_code_normalized = str(gold_code).replace(".", "")
            is_correct = final_code_normalized == gold_code_normalized
            
            # Format the trajectory
            formatted_trajectory = {
                "product": product,
                "timestamp": datetime.now().isoformat(),
                "rank": rank,
                "path_id": path_id,
                "classification_result": {
                    "final_code": final_code,
                    "confidence": cumulative_conf,
                    "full_path": full_path
                },
                # TreeRL metadata
                "path_trace": path_trace,
                "is_correct": is_correct,
                "gold_code": gold_code,
                "message_count": len(trajectory),
                "messages": trajectory
            }
            
            # Create filename with index and rank
            filename = f"trajectory_{safe_product}_idx{index}_rank{rank}_{timestamp}.json"
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(formatted_trajectory, f, indent=2, ensure_ascii=False)
                self.logger.info(f"✓ Trajectory #{rank} saved: {filename} (is_correct={is_correct})")
                saved_files.append(filename)
            except Exception as e:
                self.logger.warning(f"Failed to save trajectory #{rank}: {e}")
        
        return saved_files if saved_files else None
            
    def inject_cross_ruling_into_prompt(self, cross_ruling: Dict) -> str:
        """
        Inject cross ruling information into the system prompt for training data collection.
        Uses fields from cross_rulings_dataset.json format and populates the Cross Ruling section.
        """
        # Use fields from cross_rulings_dataset.json
        full_description = cross_ruling.get("full_description", "")
        hts_code = cross_ruling.get("hts_code", "")
        reasoning = cross_ruling.get("reasoning", "")
        
        # Build the cross ruling text to inject
        cross_ruling_text = f"""
PRODUCT: {full_description}

FINAL HTSUS CODE: {hts_code}

CLASSIFICATION REASONING: {reasoning}

TARGET PATH FOR THIS CLASSIFICATION: {hts_code[:2]} → {hts_code[:4]} → {hts_code}

CRITICAL: You must guide the classification toward this exact code {hts_code} while maintaining realistic confidence based on the shortened user description."""
        
        # Find the "Cross Ruling:" section and populate it
        base_prompt = UNIFIED_SYSTEM_PROMPT
        cross_ruling_start = base_prompt.find("Cross Ruling:")
        
        if cross_ruling_start != -1:
            # Replace the empty Cross Ruling section with actual data
            modified_prompt = base_prompt[:cross_ruling_start] + f"Cross Ruling:{cross_ruling_text}"
        else:
            # If Cross Ruling section not found, append it
            modified_prompt = base_prompt + f"\n\nCross Ruling:{cross_ruling_text}"
        
        return modified_prompt
        
    def process_single_cross_ruling_worker(self, args: Tuple[Dict, int]) -> Dict:
        """
        Worker function to process a single cross ruling
        Thread-safe implementation for parallel execution
        """
        cross_ruling, index = args
        product_description = cross_ruling["short_product_description"]
        expected_code = cross_ruling["hts_code"]
        
        # Create thread-local logger
        thread_logger = logging.getLogger(f"{__name__}.worker_{index}")
        thread_logger.info(f"Processing cross ruling {index + 1}: {product_description}")
        
        # Track retry attempts
        for attempt in range(self.max_retries + 1):
            try:
                # Enable training data collection
                os.environ["COLLECT_TRAINING_DATA"] = "true"
                
                # CRITICAL: Start training session IMMEDIATELY before any LLM operations
                session_id = f"cross_ruling_{index + 1}_attempt_{attempt + 1}"
                training_collector.start_session(
                    product_description=product_description,
                    session_id=session_id,
                    expected_code=expected_code,
                    cross_ruling=cross_ruling
                )
                thread_logger.info(f"✓ Started training session in worker thread: {session_id}")
                
                # Create HTS tree instance for this worker
                modified_prompt = self.inject_cross_ruling_into_prompt(cross_ruling)
                hts_tree = self.HTSTree()
                
                # Use default provider (Gemini 3 Pro with high reasoning)
                thread_logger.info(
                    f"✓ Using default provider '{hts_tree.llm_client.default_provider}' for training data collection"
                )
                
                # Load HTS data
                hts_data_file = script_dir / "api" / "hts_data.json"
                with open(hts_data_file, "r", encoding="utf-8") as f:
                    hts_data = json.load(f)
                hts_tree.build_from_json(hts_data)
                
                # Set the system prompt injection for this worker's LLM client
                # This ensures thread-safe injection without contaminating other workers
                if hasattr(hts_tree, 'llm_client'):
                    try:
                        hts_tree.llm_client.set_system_prompt_injection(modified_prompt)
                        thread_logger.info(f"Applied system prompt injection (thread-local)")
                    except Exception as e:
                        thread_logger.warning(f"Failed to apply system prompt injection: {e}")
                else:
                    thread_logger.warning("HTSTree has no llm_client, prompt injection may not work")
                
                # Initialize auto-responder for this worker if enabled
                auto_responder = None
                if self.use_auto_responder:
                    auto_responder = LLMAutoResponder(engine_name=self.engine_name, debug=False)
                
                # Start classification
                thread_logger.info(f"Starting classification for: {product_description}")
                result = hts_tree.start_classification(
                    product=product_description,
                    interactive=True,
                    max_questions=5,
                    use_multi_hypothesis=True,
                    hypothesis_count=self.hypothesis_count
                )
                
                # Handle interactive classification
                conversation_log = []
                question_count = 0
                max_questions = 5
                
                while not result.get("final", True) and result.get("clarification_question") and question_count < max_questions:
                    question = result.get("clarification_question")
                    question_count += 1
                    
                    # Extract question text
                    if hasattr(question, 'question_text'):
                        question_text = question.question_text
                    else:
                        question_text = question.get("question_text", "")
                    
                    thread_logger.debug(f"Question {question_count}: {question_text}")
                    
                    if auto_responder:
                        # Use auto-responder
                        response = auto_responder.generate_response(question, cross_ruling, conversation_log)
                        
                        # Log Q&A to training collector and local conversation log
                        training_collector.log_conversation(question_text, response, question_count)
                        
                        qa_pair = {
                            "question": question_text,
                            "answer": response,
                            "question_number": question_count
                        }
                        conversation_log.append(qa_pair)
                        
                        thread_logger.debug(f"Auto-response {question_count}: {response}")
                        
                        # Continue classification
                        result = hts_tree.continue_classification(
                            state=result.get("state", {}),
                            answer=response,
                            interactive=True,
                            max_questions=max_questions
                        )
                    else:
                        # No auto-responder, break
                        thread_logger.warning("No auto-responder available")
                        break
                
                # Clear system prompt injection to avoid cross-thread contamination
                if hasattr(hts_tree, 'llm_client'):
                    try:
                        hts_tree.llm_client.clear_system_prompt_injection()
                        thread_logger.debug("Cleared system prompt injection")
                    except Exception as _e:
                        thread_logger.debug(f"Failed to clear system prompt injection: {_e}")

                # Check final result
                classification_success = False
                if result.get("final", True):
                    final_code = None
                    if "classification" in result:
                        final_code = result["classification"].get("code")
                    else:
                        final_code = result.get("final_code")
                    
                    # Normalize codes for comparison
                    def normalize_code(code):
                        if code:
                            return str(code).replace(".", "")
                        return ""
                    
                    normalized_final = normalize_code(final_code)
                    normalized_expected = normalize_code(expected_code)
                    
                    if normalized_final == normalized_expected:
                        classification_success = True
                        thread_logger.info(f"✓ SUCCESS: Classification matches (attempt {attempt + 1})")
                        
                        # Save trajectories for TreeRL training (if enabled)
                        saved_files = self.extract_and_save_trajectory(
                            result=result,
                            product=product_description,
                            gold_code=expected_code,
                            index=index
                        )
                        if saved_files:
                            thread_logger.info(f"✓ Saved {len(saved_files)} trajectory files for TreeRL")
                        
                        # Explicitly end training session as successful
                        try:
                            # Check if session is still active before ending
                            if hasattr(training_collector.local, 'session'):
                                success = training_collector.should_keep_session(result)
                                training_collector.end_session(result, success)
                                thread_logger.info(f"✓ Ended training session as successful")
                            else:
                                thread_logger.debug("Training session already ended by HTSTree")
                        except Exception as e:
                            thread_logger.error(f"Failed to end training session: {e}")
                        
                        return {
                            "index": index,
                            "success": True,
                            "result": result,
                            "expected_code": expected_code,
                            "final_code": final_code,
                            "conversation_log": conversation_log,
                            "attempts": attempt + 1
                        }
                    else:
                        thread_logger.warning(f"✗ MISMATCH: {final_code} != {expected_code} (attempt {attempt + 1})")
                        
                        # If this is not the last attempt, retry
                        if attempt < self.max_retries:
                            thread_logger.info(f"Retrying classification (attempt {attempt + 2} of {self.max_retries + 1})")
                            # End failed training session before retry
                            try:
                                if hasattr(training_collector.local, 'session'):
                                    training_collector.end_session(result, False)
                                    thread_logger.info(f"✓ Ended failed training session before retry")
                            except Exception as e:
                                thread_logger.error(f"Failed to end training session: {e}")
                            continue
                        else:
                            # Final attempt failed, skip
                            thread_logger.warning(f"Skipping after {self.max_retries + 1} attempts")
                            # End failed training session
                            try:
                                if hasattr(training_collector.local, 'session'):
                                    training_collector.end_session(result, False)
                                    thread_logger.info(f"✓ Ended failed training session (final attempt)")
                            except Exception as e:
                                thread_logger.error(f"Failed to end training session: {e}")
                            
                            return {
                                "index": index,
                                "success": False,
                                "skipped": True,
                                "result": result,
                                "expected_code": expected_code,
                                "final_code": final_code,
                                "conversation_log": conversation_log,
                                "attempts": attempt + 1
                            }
                
            except Exception as e:
                thread_logger.error(f"Error in attempt {attempt + 1}: {e}")
                if self.debug:
                    thread_logger.error(traceback.format_exc())
                
                # Clear injection if set
                try:
                    if 'hts_tree' in locals() and hasattr(hts_tree, 'llm_client'):
                        hts_tree.llm_client.clear_system_prompt_injection()
                except Exception:
                    pass

                # End session with failure only if still active
                try:
                    if hasattr(training_collector, 'local') and hasattr(training_collector.local, 'session'):
                        training_collector.end_session({}, False)
                except Exception:
                    pass
                
                # If not the last attempt, retry
                if attempt < self.max_retries:
                    thread_logger.info(f"Retrying after error (attempt {attempt + 2})")
                    time.sleep(1)  # Brief delay before retry
                    continue
                else:
                    return {
                        "index": index,
                        "success": False,
                        "error": str(e),
                        "expected_code": expected_code,
                        "attempts": attempt + 1
                    }
        
        # Should not reach here, but just in case
        return {
            "index": index,
            "success": False,
            "error": "Unexpected exit from retry loop",
            "expected_code": expected_code
        }
    
    def process_rolling(self, cross_rulings: List[Dict], total_dataset_count: int) -> Dict:
        """
        Process cross rulings with rolling/continuous processing.
        Maintains constant worker count - as soon as one finishes, start the next.
        """
        self.logger.info(f"Starting rolling processing of {total_dataset_count} cross rulings")
        self.logger.info(f"Maintaining {self.max_workers} concurrent workers")
        
        # Initialize timing
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        
        # Create a queue of work items
        work_queue = Queue()
        for i, cross_ruling in enumerate(cross_rulings):
            work_queue.put((cross_ruling, i))
        
        # Use WARNING level so these appear in quiet mode
        self.logger.warning(f"🚀 Starting rolling processing with {self.max_workers} workers")
        self.logger.warning(f"⏳ Progress updates every 15 seconds")
        
        # Rolling processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Track active futures
            active_futures = {}
            
            # Submit initial batch of workers up to max_workers
            for _ in range(min(self.max_workers, work_queue.qsize())):
                try:
                    work_item = work_queue.get_nowait()
                    future = executor.submit(self.process_single_cross_ruling_worker, work_item)
                    active_futures[future] = work_item
                except Empty:
                    break
            
            # Rolling processing loop
            while active_futures or not work_queue.empty():
                try:
                    # Wait for at least one future to complete (timeout to allow progress updates)
                    completed_futures = []
                    for future in as_completed(active_futures.keys(), timeout=1.0):
                        completed_futures.append(future)
                        break
                    
                    # Process completed futures
                    for future in completed_futures:
                        try:
                            result = future.result()
                            work_item = active_futures.pop(future)
                            
                            # Update progress counters
                            with self.progress_lock:
                                if result.get("success"):
                                    self.successful_count += 1
                                elif result.get("skipped"):
                                    self.skipped_count += 1
                                else:
                                    self.failed_count += 1
                                self.total_processed += 1
                                
                                # Time-based progress reporting (every 15 seconds)
                                current_time = time.time()
                                if current_time - self.last_progress_time >= 15.0:  # 15 seconds
                                    elapsed = current_time - self.start_time
                                    rate = self.total_processed / elapsed if elapsed > 0 else 0
                                    eta = (total_dataset_count - self.total_processed) / rate if rate > 0 else 0
                                    success_rate = (self.successful_count / self.total_processed * 100) if self.total_processed > 0 else 0
                                    active_workers = len(active_futures)
                                    
                                    # Use WARNING level so progress appears in quiet mode
                                    self.logger.warning(
                                        f"⏰ ROLLING [{datetime.now().strftime('%H:%M:%S')}]: "
                                        f"{self.total_processed}/{total_dataset_count} "
                                        f"({success_rate:.1f}% success, {rate:.1f}/min, "
                                        f"ETA: {eta/60:.1f}min, Active: {active_workers})"
                                    )
                                    self.last_progress_time = current_time
                            
                            # Immediately start next job if available
                            if not work_queue.empty():
                                try:
                                    next_work_item = work_queue.get_nowait()
                                    next_future = executor.submit(self.process_single_cross_ruling_worker, next_work_item)
                                    active_futures[next_future] = next_work_item
                                except Empty:
                                    pass  # Queue became empty between check and get
                                    
                        except Exception as e:
                            self.logger.error(f"Worker failed: {e}")
                            work_item = active_futures.pop(future, None)
                            with self.progress_lock:
                                self.failed_count += 1
                                self.total_processed += 1
                            
                            # Start replacement worker if work available
                            if not work_queue.empty():
                                try:
                                    next_work_item = work_queue.get_nowait()
                                    next_future = executor.submit(self.process_single_cross_ruling_worker, next_work_item)
                                    active_futures[next_future] = next_work_item
                                except Empty:
                                    pass
                
                except:
                    # Timeout occurred, no futures completed - just continue for progress updates
                    pass
        
        return {
            "total_processed": self.total_processed,
            "successful": self.successful_count,
            "skipped": self.skipped_count,
            "failed": self.failed_count
        }
    
    def process_cross_rulings_dataset(self, file_path: str, max_count: Optional[int] = None, start_index: int = 0) -> Dict:
        """
        Process the entire cross rulings dataset with rolling/continuous processing
        
        Args:
            file_path: Path to the cross rulings JSON file
            max_count: Maximum number of rulings to process (None = all)
            start_index: Index to start processing from (0-based, for resuming)
        """
        cross_rulings = self.load_cross_rulings(file_path)
        
        # Apply start_index first (for resuming)
        if start_index > 0:
            self.logger.info(f"Resuming from cross ruling index {start_index}")
            cross_rulings = cross_rulings[start_index:]
        
        # Then apply max_count if specified
        if max_count:
            cross_rulings = cross_rulings[:max_count]
        
        total_count = len(cross_rulings)
        
        self.logger.info(f"Starting rolling processing of {total_count} cross rulings")
        self.logger.info(f"Training data will be written to: {training_collector.output_file}")
        self.logger.info(f"Max workers: {self.max_workers} (rolling/continuous)")
        self.logger.info(f"Beam size (hypothesis_count): {self.hypothesis_count}")
        
        # Reset counters
        self.total_processed = 0
        self.successful_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        
        # Use rolling processing for maximum efficiency
        start_time = time.time()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"ROLLING PROCESSING: {total_count} items")
        self.logger.info(f"{'='*60}")
        
        rolling_result = self.process_rolling(cross_rulings, total_count)
        
        # Calculate final statistics
        elapsed_time = time.time() - start_time
        success_rate = (self.successful_count / total_count * 100) if total_count > 0 else 0
        
        summary = {
            "total_processed": rolling_result["total_processed"],
            "successful": rolling_result["successful"],
            "skipped": rolling_result["skipped"],
            "failed": rolling_result["failed"],
            "success_rate": success_rate,
            "elapsed_time": elapsed_time,
            "avg_time_per_item": elapsed_time / total_count if total_count > 0 else 0,
            "output_file": training_collector.output_file
        }
        
        # Use WARNING level so final results appear in quiet mode
        self.logger.warning(f"\n{'='*60}")
        self.logger.warning("ROLLING PROCESSING COMPLETE")
        self.logger.warning(f"{'='*60}")
        self.logger.warning(f"Total processed: {summary['total_processed']}")
        self.logger.warning(f"Successful: {summary['successful']}")
        self.logger.warning(f"Skipped (after retries): {summary['skipped']}")
        self.logger.warning(f"Failed: {summary['failed']}")
        self.logger.warning(f"Success rate: {summary['success_rate']:.1f}%")
        self.logger.warning(f"Total time: {summary['elapsed_time']:.2f} seconds")
        self.logger.warning(f"Avg time per item: {summary['avg_time_per_item']:.2f} seconds")
        self.logger.warning(f"Training data saved to: {summary['output_file']}")
        
        return summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel cross rulings dataset processor")
    parser.add_argument("--cross-rulings-file", default="cross_rulings_dataset.json", 
                       help="Path to cross rulings JSON file (default: cross_rulings_dataset.json for training)")
    parser.add_argument("--engine", choices=["cerebras", "groq", "tree", "gemini"], default="groq",
                       help="Classification engine to use")
    parser.add_argument("--max-count", type=int, default=100,
                       help="Maximum number of cross rulings to process (default: 100)")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Index to start processing from (0-based, for resuming). Default: 0")
    parser.add_argument("--max-workers", type=int, default=25, 
                       help="Maximum number of parallel workers (default: 25)")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for processing (default: 2)")
    parser.add_argument("--max-retries", type=int, default=2,
                       help="Maximum retries for failed classifications (default: 2)")
    parser.add_argument("--hypothesis-count", type=int, default=2,
                       help="Beam size / hypothesis count (default: 2 for training efficiency)")
    parser.add_argument("--save-trajectories", action="store_true",
                       help="Save trajectory files for TreeRL training (includes is_correct and path_trace)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet-mode", action="store_true", help="Enable quiet mode (minimal logging)")
    parser.add_argument("--no-auto-responder", action="store_true", 
                       help="Disable auto-responder (manual mode)")
    
    args = parser.parse_args()
    
    # Enable training data collection
    os.environ["COLLECT_TRAINING_DATA"] = "true"
    
    processor = ParallelCrossRulingsProcessor(
        engine_name=args.engine,
        debug=args.debug,
        quiet_mode=args.quiet_mode,
        use_auto_responder=not args.no_auto_responder,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        hypothesis_count=args.hypothesis_count,
        save_trajectories=args.save_trajectories
    )
    
    try:
        summary = processor.process_cross_rulings_dataset(
            args.cross_rulings_file, 
            args.max_count,
            args.start_index
        )
        
        return 0 if summary['success_rate'] > 0 else 1
        
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
