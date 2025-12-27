import json
import logging
import os
import re
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from api.cross_rulings_injector import get_injector

class TrainingDataCollector:
    """
    Thread-safe training data collector that gathers data during processing
    and writes at session end to prevent concurrent write issues.
    
    Supports parallel LLM requests within a single session by using:
    - Shared sessions dictionary for cross-thread access
    - Thread-local storage for default session lookup
    - Lock protection for session modifications
    """
    
    def __init__(self):
        self.enabled = os.environ.get("COLLECT_TRAINING_DATA", "false").lower() == "true"
        self.output_file = os.environ.get("TRAINING_DATA_FILE", "training_data.jsonl")
        self.failed_file = os.environ.get("FAILED_DATA_FILE", "failed_classifications.jsonl")
        self.raw_state_file = os.environ.get("RAW_STATE_FILE", "raw_states.jsonl")
        
        # Thread-local storage for session data (backward compatibility)
        self.local = threading.local()
        
        # Shared sessions dictionary for cross-thread access during parallel processing
        self._shared_sessions: Dict[str, Dict] = {}
        self._sessions_lock = threading.Lock()
        
        # File writing lock for thread-safe concurrent writes
        self.write_lock = threading.Lock()
        
        if self.enabled:
            logging.info(f"Training data collection enabled - output: {self.output_file}")
            logging.info(f"Failed classifications output: {self.failed_file}")
            logging.info(f"Raw state output enabled - output: {self.raw_state_file}")
    
    def start_session(self, product_description: str, session_id: str = None, expected_code: str = None, cross_ruling: Dict = None):
        """Start a new training data collection session (accessible from any thread)."""
        if not self.enabled:
            return
        
        actual_session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        session_data = {
            "session_id": actual_session_id,
            "product": product_description,
            "expected_code": expected_code,
            "cross_ruling": cross_ruling,
            "timestamp": datetime.now().isoformat(),
            "requests": [],
            "conversation_log": [],
            "final_result": None,
            "success": None,
            "training_buffer": []  # Buffer training examples until session ends
        }
        
        # Store in thread-local for backward compatibility
        self.local.session = session_data
        
        # Also store in shared sessions for cross-thread access
        with self._sessions_lock:
            self._shared_sessions[actual_session_id] = session_data
        
        logging.info(f"Started training data session: {actual_session_id}")
        return actual_session_id
    
    def get_current_session_id(self) -> Optional[str]:
        """Get the current session ID (for passing to parallel workers)."""
        if hasattr(self.local, 'session') and self.local.session:
            return self.local.session.get("session_id")
        return None
    
    def set_thread_session(self, session_id: str):
        """Set the session for the current thread (used by parallel workers)."""
        with self._sessions_lock:
            if session_id in self._shared_sessions:
                self.local.session = self._shared_sessions[session_id]
    
    def log_request(
        self,
        task_type: str,
        prompt_json: Dict,
        response: str,
        metadata: Dict = None,
        system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        trajectory_messages: Optional[List[Dict[str, str]]] = None,
    ):
        """Log an LLM request/response pair - creates training data with system prompt.
        
        Args:
            task_type: Type of task (rank_candidates, generate_question, etc.)
            prompt_json: The prompt data for this turn
            response: The assistant's response
            metadata: Optional metadata (not included in training output)
            system_prompt: Optional system prompt override
            session_id: Optional session ID for cross-thread logging. If not provided,
                       uses the thread-local session.
            trajectory_messages: Optional full trajectory (system + all prior turns + current user).
                                When provided, creates multi-turn training data instead of single-turn.
                                The response is appended as the final assistant message.
        """
        if not self.enabled:
            logging.debug("Training data collection disabled")
            return
            
        # Validate inputs to prevent skipping warnings
        if not prompt_json or not task_type:
            logging.warning(f"⚠️ Skipping training data log: prompt_json={bool(prompt_json)}, task_type={task_type}")
            return
        
        # Try to get session: first from explicit session_id, then thread-local
        session = None
        if session_id:
            with self._sessions_lock:
                session = self._shared_sessions.get(session_id)
        
        if session is None and hasattr(self.local, 'session'):
            session = self.local.session
            
        if session is None:
            logging.warning("No training session active - cannot log request")
            return
        
        # TRAJECTORY MODE: Use full conversation history if provided
        if trajectory_messages:
            # Deep copy the trajectory to avoid mutation
            messages = [msg.copy() for msg in trajectory_messages]
            
            # Append the assistant response as the final message
            messages.append({
                "role": "assistant",
                "content": response
            })
            
            turn_count = sum(1 for m in messages if m["role"] == "user")
            logging.info(f"✓ Using trajectory mode: {turn_count} user turns in conversation")
        else:
            # SINGLE-TURN MODE: Build system + user + assistant from scratch
            # Import system prompt builder
            from .prompt_builder import build_system_prompt
            
            messages = []
            
            # System message - use task-specific prompt if no custom prompt provided
            if system_prompt:
                system_content = system_prompt.strip()
            else:
                # Build a task-specific prompt (without notes for training data portability)
                system_content = build_system_prompt(task_type).strip()
            messages.append({
                "role": "system",
                "content": system_content
            })
            
            # User message - use the actual unified state JSON prompt (exact input)
            try:
                user_content = json.dumps(prompt_json, ensure_ascii=False)
            except Exception:
                # Fallback in case prompt_json is not serializable
                user_content = str(prompt_json)
            
            messages.append({
                "role": "user", 
                "content": user_content
            })
            
            # Assistant message (clean response without metadata)
            messages.append({
                "role": "assistant",
                "content": response
            })
        
        # Create training example with messages
        training_example = {
            "messages": messages
        }
        
        grading_metadata = self._build_grading_metadata(task_type, prompt_json, response)
        if grading_metadata:
            training_example["grading"] = grading_metadata
        
        # Note: Removed all metadata (reference_answer, session_id, task_type, etc.) 
        # for clean fine-tuning format. Only messages array is needed.
        
        # Request metadata is intentionally NOT included in training examples
        # to avoid contaminating training data with model/runtime details.
        # (Previously allowed small metadata; now disabled by design.)
        
        # Buffer training example instead of writing immediately (thread-safe)
        try:
            with self._sessions_lock:
                # Add to session buffer - will only be written if session succeeds
                session["training_buffer"].append(training_example)
                
                # Also store in session for counting
                request_entry = {
                    "task_type": task_type,
                    "timestamp": datetime.now().isoformat()
                }
                session["requests"].append(request_entry)
                request_count = len(session['requests'])
            
            is_trajectory = "trajectory" if trajectory_messages else "single-turn"
            logging.info(f"✓ Buffered {is_trajectory} training request: {task_type} (total requests: {request_count})")
            
        except Exception as e:
            logging.error(f"❌ Failed to buffer training data: {e}")
    
    def log_conversation(self, question: str, answer: str, question_number: int):
        """Log conversation Q&A pair to thread-local storage."""
        if not self.enabled or not hasattr(self.local, 'session'):
            return
            
        qa_pair = {
            "question": question,
            "answer": answer,
            "question_number": question_number,
            "timestamp": datetime.now().isoformat()
        }
        
        self.local.session["conversation_log"].append(qa_pair)
        logging.debug(f"Logged conversation pair {question_number}")
    
    def end_session(self, final_result: Dict, success: bool, session_id: Optional[str] = None):
        """End the current session - write buffered training data only if successful.
        
        Args:
            session_id: Optional session ID. If not provided, uses thread-local session.
        """
        if not self.enabled:
            logging.info("Training data collection disabled - end_session called but skipping")
            return
        
        # Get session from explicit ID or thread-local
        session = None
        if session_id:
            with self._sessions_lock:
                session = self._shared_sessions.get(session_id)
        
        if session is None and hasattr(self.local, 'session'):
            session = self.local.session
            
        if session is None:
            logging.warning("No training session active in end_session - no data to write")
            return
        
        actual_session_id = session.get("session_id", "unknown")
        
        with self._sessions_lock:
            session["final_result"] = final_result
            session["success"] = success
            buffered_count = len(session.get("training_buffer", []))
            request_count = len(session.get("requests", []))
        
        if success:
            # SUCCESSFUL SESSION: Write all buffered training data to file
            try:
                with self.write_lock:
                    with open(self.output_file, 'a', encoding='utf-8') as f:
                        for training_example in session["training_buffer"]:
                            f.write(json.dumps(training_example) + '\n')
                
                logging.info(f"✅ SUCCESS: Written {buffered_count} training examples from session {actual_session_id} to {self.output_file}")
                
            except Exception as e:
                logging.error(f"❌ Failed to write successful training data: {e}")
                
        else:
            # FAILED SESSION: Discard buffered training data
            logging.info(f"❌ FAILED SESSION: Discarded {buffered_count} training examples from session {actual_session_id} (not written to file)")
        
        logging.info(f"Ending session {actual_session_id} with success={success}, requests={request_count}")
        
        # Clean up from shared sessions
        with self._sessions_lock:
            if actual_session_id in self._shared_sessions:
                del self._shared_sessions[actual_session_id]
        
        # Clear thread-local session
        if hasattr(self.local, 'session'):
            delattr(self.local, 'session')
    
    def _convert_to_individual_calls_format(self, session: Dict) -> List[Dict]:
        """
        Convert collected session data to individual LLM call format for OpenAI reinforcement fine-tuning.
        Each request/response becomes a separate JSONL line.
        
        Args:
            session: Session data with requests and conversation log
            
        Returns:
            List of individual training examples in OpenAI format
        """
        training_lines = []
        
        # Process each LLM request/response pair
        for i, request_entry in enumerate(session["requests"]):
            prompt_data = request_entry.get("prompt", {})
            response = request_entry.get("response", "")
            task_type = request_entry.get("task_type", "unknown")
            
            # Create OpenAI format for this individual call
            messages = []
            
            # Developer message (replaces system for o1+ models) - use injected system prompt if available
            if "system" in prompt_data:
                developer_content = prompt_data["system"]
            else:
                developer_content = "You are an expert customs classification specialist with deep knowledge of HTS codes and international trade regulations."
            
            messages.append({
                "role": "developer",
                "content": developer_content
            })
            
            # User message (use user from prompt if available, else construct from task)
            if "user" in prompt_data:
                user_content = prompt_data["user"]
            else:
                # Construct user message from task data
                task_data = prompt_data.get("data", {})
                product_text = task_data.get("product_text", session["product"])
                user_content = f"Task: {task_type}\nProduct: {product_text}"
                
                # Add relevant task context based on task type
                if task_type == "rank_candidates":
                    user_content += f"\nRank and score the top {task_data.get('select_count', 3)} candidate classifications."
                elif task_type == "generate_question":
                    user_content += "\nGenerate a clarification question to help with classification."
                elif task_type == "process_answer":
                    question_text = task_data.get("question_text", "")
                    answer_text = task_data.get("answer_text", "")
                    user_content += f"\nQuestion: {question_text}\nAnswer: {answer_text}\nProcess this answer to continue classification."
            
            messages.append({
                "role": "user", 
                "content": user_content
            })
            
            # Assistant message (the response)
            messages.append({
                "role": "assistant",
                "content": response
            })
            
            # Create training example in OpenAI format
            training_example = {
                "messages": messages
            }
            
            # Add reference answer (ground truth) if available
            if session.get("expected_code"):
                training_example["reference_answer"] = {
                    "expected_hts_code": session["expected_code"],
                    "task_type": task_type
                }
            
            # Add other arbitrary data that can be referenced during grading
            training_example["session_id"] = session["session_id"]
            training_example["request_index"] = i
            training_example["task_type"] = task_type
            training_example["timestamp"] = request_entry.get("timestamp", session["timestamp"])
            training_example["product_description"] = session["product"]
            training_example["session_success"] = session.get("success", False)
            
            # Do NOT include request-level metadata in training examples
            # (avoid storing model/runtime details)
            
            training_lines.append(training_example)
        
        return training_lines
    
    def _convert_to_fireworks_format(self, session: Dict) -> Dict:
        """
        Convert collected session data to Fireworks RFT format.
        
        Args:
            session: Session data with requests and conversation log
            
        Returns:
            Dict in Fireworks format with messages and ground_truth
        """
        messages = []
        
        # Extract system prompt from first request
        system_content = "You are a tariff classification expert. Classify products into HTS codes."
        if session["requests"]:
            first_request = session["requests"][0]
            if "prompt" in first_request and "system" in first_request["prompt"]:
                system_content = first_request["prompt"]["system"]
        
        messages.append({
            "role": "system", 
            "content": system_content
        })
        
        # Build user message with product description
        product_description = session["product"]
        user_content = f"Please classify the following product into the appropriate HTS code: {product_description}"
        
        # Add conversation context if available
        if session["conversation_log"]:
            conversation_text = "\n\nDuring classification, the following questions were asked and answered:\n"
            for qa in session["conversation_log"]:
                conversation_text += f"Q: {qa['question']}\n"
                conversation_text += f"A: {qa['answer']}\n"
            user_content += conversation_text
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Build assistant response from final result
        assistant_content = "Based on my analysis, "
        final_result = session.get("final_result", {})
        
        if final_result:
            # Extract classification details
            classification = final_result.get("classification", {})
            final_code = classification.get("code") or final_result.get("final_code", "")
            confidence = classification.get("confidence", 0)
            reasoning = classification.get("reasoning", "")
            
            if final_code:
                assistant_content += f"I classify this product as HTS code {final_code}"
                if confidence:
                    assistant_content += f" with {confidence:.2f} confidence"
                assistant_content += "."
                
                if reasoning:
                    assistant_content += f"\n\nReasoning: {reasoning}"
            else:
                assistant_content += "I was unable to determine a specific HTS code for this product."
        else:
            assistant_content += "the classification process was incomplete."
        
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })
        
        # Build Fireworks format
        fireworks_data = {
            "messages": messages,
            "session_id": session["session_id"],
            "timestamp": session["timestamp"],
            "success": session["success"],
            "product_description": session["product"]
        }
        
        # Add ground truth if available
        expected_code = session.get("expected_code")
        if expected_code:
            fireworks_data["ground_truth"] = expected_code
        
        # Add metadata
        if session["conversation_log"]:
            fireworks_data["question_count"] = len(session["conversation_log"])
        
        if session["requests"]:
            fireworks_data["request_count"] = len(session["requests"])
        
        return fireworks_data
    
    def should_keep_session(self, final_result: Dict) -> bool:
        """
        Determine if a training session should be kept based on quality criteria.
        """
        if not final_result:
            return False
        
        # Check if we have a valid classification result
        has_classification = "classification" in final_result
        has_final_code = bool(final_result.get("classification", {}).get("code"))
        
        # If we have a classification with a code, keep the session
        # (removed confidence requirement as successful code matching is more important)
        return has_classification and has_final_code

    def _build_grading_metadata(self, task_type: str, prompt_json: Dict, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Build grading metadata for reinforcement fine-tuning when expected_code is available.
        """
        session = getattr(self.local, "session", None)
        if not session:
            return None
        
        expected_code = self._normalize_code(session.get("expected_code"))
        if not expected_code:
            return None
        
        data = prompt_json.get("data", {})
        metadata: Dict[str, Any] = {
            "task_type": task_type,
            "expected_code": expected_code
        }
        
        if task_type == "select_chapters":
            metadata["target_chapter"] = expected_code[:2]
            chapter_meta = self._extract_target_confidences(task_type, response_text, metadata, data)
            if chapter_meta:
                metadata.update(chapter_meta)
            return metadata
        
        if task_type == "rank_candidates":
            target_index = self._find_matching_candidate(
                data.get("classification_tree", {}).get("children", []),
                expected_code,
                key_name="index"
            )
            if target_index is None:
                return None
            metadata["correct_option_index"] = target_index
            rank_meta = self._extract_target_confidences(task_type, response_text, metadata, data)
            if rank_meta:
                metadata.update(rank_meta)
            return metadata
        
        return None

    def _find_matching_candidate(self, candidates: List[Dict[str, Any]], expected_code: str, key_name: str) -> Optional[int]:
        """
        Find the candidate index whose code matches the expected code prefix.
        Also handles [GROUP] nodes by elimination - if no other candidate matches,
        and there's a GROUP node, the GROUP is likely the correct path.
        
        Matching priority:
        1. Exact prefix match (longest matching code wins)
        2. First GROUP node as fallback if no direct match
        """
        if not candidates:
            return None
        
        best_match_index = None
        best_match_length = 0
        group_indices: List[int] = []
        
        for candidate in candidates:
            code = candidate.get("code", "")
            is_group = code == "[GROUP]" or candidate.get("is_group")
            
            if is_group:
                # Collect all GROUP nodes (use first one as fallback)
                index_value = candidate.get(key_name)
                if isinstance(index_value, (int, float)):
                    group_indices.append(int(index_value))
            else:
                # Try direct code prefix matching - prefer longest match
                candidate_code = self._normalize_code(code)
                if candidate_code and expected_code.startswith(candidate_code):
                    if len(candidate_code) > best_match_length:
                        best_match_length = len(candidate_code)
                        index_value = candidate.get(key_name)
                        if isinstance(index_value, (int, float)):
                            best_match_index = int(index_value)
        
        # If we found a direct match, use it
        if best_match_index is not None:
            return best_match_index
        
        # If no direct match but there are GROUP nodes, the expected code likely
        # goes through a GROUP (since it didn't match any specific sibling codes)
        # Use the first GROUP node encountered
        if group_indices:
            return group_indices[0]
        
        return None

    def _extract_target_confidences(
        self,
        task_type: str,
        response_text: str,
        metadata: Dict[str, Any],
        prompt_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract reference confidences from the assistant response for grading.
        """
        if not response_text:
            return None
        
        try:
            parsed = json.loads(response_text)
        except Exception:
            return None
        
        if task_type == "select_chapters":
            chapters = parsed if isinstance(parsed, list) else parsed.get("chapters", [])
            target_chapter = metadata.get("target_chapter")
            if not chapters or not target_chapter:
                return None
            
            # Extract top_selection from response
            top_selection = parsed.get("top_selection") if isinstance(parsed, dict) else None
            top_selection_normalized = self._normalize_code(top_selection) if top_selection else None
            
            normalized = [self._normalize_code(ch.get("chapter")) for ch in chapters]
            for rank, chapter_code in enumerate(normalized, start=1):
                if chapter_code == target_chapter:
                    return {
                        "target_chapter_rank": rank,
                        "target_is_top": top_selection_normalized == target_chapter if top_selection_normalized else None,
                        "model_top_selection": top_selection
                    }
            return None
        
        if task_type == "rank_candidates":
            target_index = metadata.get("correct_option_index")
            if target_index is None or not isinstance(parsed, dict):
                return None
            
            # Extract selections from new unified format
            selections = []
            for key in ["primary_selection", "alternative_1", "alternative_2"]:
                sel = parsed.get(key)
                if sel and isinstance(sel, dict):
                    selections.append(sel)
            
            # Find target in selections
            target_entry = None
            target_rank = None
            for rank, sel in enumerate(selections, start=1):
                try:
                    option_index = int(sel.get("option_index"))
                except (TypeError, ValueError):
                    continue
                if option_index == target_index:
                    target_entry = sel
                    target_rank = rank
                    break
            
            # Check if target is primary_selection
            primary = parsed.get("primary_selection", {})
            primary_index = primary.get("option_index") if isinstance(primary, dict) else None
            target_is_primary = primary_index == target_index if primary_index is not None else None
            
            # Use 0.60 threshold as specified in prompts
            threshold = 0.60
            info_conf = None
            path_conf = None
            
            if target_entry:
                info_conf = target_entry.get("information_context_score")
                path_conf = target_entry.get("path_score")
                try:
                    info_conf = float(info_conf)
                except (TypeError, ValueError):
                    info_conf = None
                try:
                    path_conf = float(path_conf)
                except (TypeError, ValueError):
                    path_conf = None
            
            model_should_proceed = parsed.get("should_proceed")
            
            return {
                "target_selected": target_entry is not None,
                "target_rank": target_rank,
                "target_is_primary": target_is_primary,
                "target_information_confidence": info_conf,
                "target_path_confidence": path_conf,
                "confidence_threshold": threshold,
                "should_proceed": (info_conf is not None and info_conf >= threshold),
                "model_should_proceed": model_should_proceed
            }
        
        return None

    @staticmethod
    def _normalize_code(code_value: Optional[str]) -> str:
        if not code_value or not isinstance(code_value, str):
            return ""
        if code_value.startswith("["):
            return ""
        return re.sub(r"\D", "", code_value)

def build_state(task: str, **kwargs) -> dict:
    """Unified schema builder - ensures all required fields are properly populated"""
    
    # Universal fields (always present)
    state = {
        "task": task,
        "data": {
            "product_text": kwargs.get("product_text", "")
        }
    }
    
    # Context fields (when relevant)
    if task != "select_chapters":
        state["data"]["path_so_far"] = kwargs.get("path_so_far", "")
        decision_history = kwargs.get("decision_history")
        if decision_history:
            state["data"]["decision_history"] = decision_history
    
    # Classification tree for tasks that need it
    if task in ["rank_candidates", "generate_question", "process_answer"]:
        classification_tree = kwargs.get("classification_tree")
        # Ensure we always have a proper classification_tree structure
        if classification_tree is not None:
            # Check if cross ruling injection is disabled (training mode = no cheating)
            disable_injection = os.environ.get("DISABLE_CROSS_RULING_INJECTION", "false").lower() == "true"
            
            if disable_injection:
                # Training mode: use raw classification tree without cross ruling hints
                state["data"]["classification_tree"] = classification_tree
            else:
                # Inference mode: inject cross rulings for better accuracy
                injector = get_injector()
                product_text = kwargs.get("product_text", "")
                state["data"]["classification_tree"] = injector.inject_into_classification_tree(
                    classification_tree, 
                    product_text
                )
        else:
            state["data"]["classification_tree"] = {}
    
    # Task-specific fields
    if task == "rank_candidates":
        state["data"]["select_count"] = kwargs.get("select_count", 3)
        state["data"]["confidence_threshold"] = kwargs.get("confidence_threshold", 0.60)
    
    elif task == "process_answer":
        state["data"]["question_text"] = kwargs.get("question_text", "")
        state["data"]["answer_text"] = kwargs.get("answer_text", "")
    
    elif task == "generate_question":
        state["data"]["conversation_history"] = kwargs.get("conversation_history", [])
        state["data"]["stage"] = kwargs.get("stage", "")
    
    elif task == "select_chapters":
        chapters = kwargs.get("chapters")
        if chapters is not None:
            state["data"]["chapters"] = chapters
        state["data"]["count"] = kwargs.get("count", 3)
        diagnosis = kwargs.get("diagnosis")
        if diagnosis:
            state["data"]["diagnosis"] = diagnosis
    
    return state

# Global instance
training_collector = TrainingDataCollector()
