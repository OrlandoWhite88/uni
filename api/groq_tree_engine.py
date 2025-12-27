import json
import logging
import os
from typing import Dict, List, Any, Optional, Union
from .models import HTSNode, ClassificationPath, ClarificationQuestion, TreeUtils
from .tree_navigator import TreeNavigator
from .llm_client import LLMClient
from .classification_engine import (
    ClassificationEngine,
    CLASSIFICATION_BEAM_SIZE,
    CONFIDENCE_THRESHOLD,
)
from .streaming_engine import StreamingEngine
from .serialization_utils import HTSSerializer, SerializationError
from .training_data_collector import training_collector

class HTSTree:
    """
    Main orchestrator for HTS classification.
    Now a thin facade that coordinates between the separated engines.
    Maintains backward compatibility with the original API.
    """
    
    def __init__(self):
        # Initialize all the separated components
        self.navigator = TreeNavigator()
        self.llm_client = LLMClient()
        self.classification_engine = ClassificationEngine(self.navigator, self.llm_client)
        self.streaming_engine = StreamingEngine(self.navigator, self.llm_client, self.classification_engine)
        
        # Keep track of steps for backward compatibility
        self.steps: List[Dict[str, Any]] = []
        
        # Expose navigator properties for backward compatibility
        self.root = self.navigator.root
        self.chapters = self.navigator.chapters
        self.code_index = self.navigator.code_index
        self.node_index = self.navigator.node_index
        self.next_node_id = self.navigator.next_node_id
        self.chapters_map = self.navigator.chapters_map

        # Expose LLM client properties for backward compatibility
        self.client = self.llm_client.client
        self.log_prompts = self.llm_client.log_prompts
        if hasattr(self.llm_client, 'prompt_logger'):
            self.prompt_logger = self.llm_client.prompt_logger

        # Expose classification engine properties for backward compatibility
        self.max_questions_per_level = self.classification_engine.max_questions_per_level
        self.path_workers = self.classification_engine.path_workers
        self.calibrate_workers = self.classification_engine.calibrate_workers

    # ----------------------
    # Tree Building and Navigation (delegated to navigator)
    # ----------------------
    
    def build_from_json(self, json_data: Union[str, List[Dict[str, Any]]]) -> None:
        """Build the HTS hierarchy from JSON data."""
        self.navigator.build_from_json(json_data)
        # Update exposed properties
        self.root = self.navigator.root
        self.chapters = self.navigator.chapters
        self.code_index = self.navigator.code_index
        self.node_index = self.navigator.node_index
        self.next_node_id = self.navigator.next_node_id

    def get_node_by_id(self, node_id: int) -> Optional[HTSNode]:
        """Get a node by its ID."""
        return self.navigator.get_node_by_id(node_id)
    
    def get_children(self, node: HTSNode) -> List[HTSNode]:
        """Get the immediate children of a node."""
        return self.navigator.get_children(node)
        
    def get_chapter_nodes(self, chapter: str) -> List[HTSNode]:
        """Get all nodes for a given chapter."""
        return self.navigator.get_chapter_nodes(chapter)
    
    def get_chapter_headings(self, chapter: str) -> List[HTSNode]:
        """Get all heading nodes for a given chapter."""
        return self.navigator.get_chapter_headings(chapter)
    
    def create_chapter_parent(self, chapter: str) -> HTSNode:
        """Create a pseudo-node that has all chapter heading nodes as children."""
        return self.navigator.create_chapter_parent(chapter)
    
    def find_node_by_prefix(self, prefix: str) -> Optional[HTSNode]:
        """Find a node that matches the given prefix."""
        return self.navigator.find_node_by_prefix(prefix)

    # ----------------------
    # LLM Communication (delegated to llm_client)
    # ----------------------
    
    def send_groq_request(self, prompt, requires_json=False, temperature=0):
        """Send a request to the Groq model and return the response."""
        return self.llm_client.send_groq_request(prompt, requires_json, temperature)
    
    def send_vertex_ai_request(self, prompt, requires_json=False, temperature=0):
        """Alias for send_groq_request for backward compatibility."""
        return self.llm_client.send_vertex_ai_request(prompt, requires_json, temperature)

    def _generate_classification_knowledge(self, product_description: str) -> str:
        """Generate classification knowledge using Google Vertex AI with web search."""
        return self.llm_client.generate_classification_knowledge(product_description)

    async def _generate_classification_knowledge_async(self, product_description: str) -> str:
        """Generate classification knowledge using Google Vertex AI with web search (async version)."""
        return await self.llm_client.generate_classification_knowledge_async(product_description)

    # ----------------------
    # Core Classification Logic (delegated to classification_engine)
    # ----------------------

    def determine_top_chapters(self, product_description: str, k: int = 3, diagnosis: Optional[str] = None, state: Dict = None) -> List[tuple]:
        """Ask the LLM to pick the top K most likely chapters with confidence scores."""
        return self.classification_engine.determine_top_chapters(product_description, k, diagnosis, state)

    def initialize_classification_paths(self, product_description: str, k: int = 3, diagnosis: Optional[str] = None, state: Dict = None) -> List[ClassificationPath]:
        """Initialize multiple classification paths based on top K chapters."""
        return self.classification_engine.initialize_classification_paths(product_description, k, diagnosis, state)

    def generate_all_next_candidates(self, paths: List[ClassificationPath], product_description: str, state: Dict = None) -> List[tuple]:
        """Generate all possible next steps from all active paths."""
        return self.classification_engine.generate_all_next_candidates(paths, product_description, state)

    def score_candidates(
        self,
        product_description: str,
        parent_node: HTSNode,
        candidates: List[HTSNode],
        path: ClassificationPath,
        state: Dict = None
    ) -> List[tuple]:
        """Score candidate nodes using Select-Then-Calibrate approach."""
        return self.classification_engine.score_candidates(product_description, parent_node, candidates, path, state)

    def advance_beam(self, current_beam: List[ClassificationPath], product_description: str, k: int, state: Dict = None) -> List[ClassificationPath]:
        """Advances the beam by finding the top K overall next steps."""
        return self.classification_engine.advance_beam(current_beam, product_description, k, state)

    def check_termination_conditions(self, beam: List[ClassificationPath]) -> tuple:
        """Check if termination conditions are met."""
        return self.classification_engine.check_termination_conditions(beam)

    def generate_clarification_question(self, product_description: str, node: HTSNode, stage: str, state: Dict, path: ClassificationPath = None) -> ClarificationQuestion:
        """Generate a user-friendly clarification question.
        
        Args:
            path: Optional ClassificationPath for trajectory mode support
        """
        return self.classification_engine.generate_clarification_question(product_description, node, stage, state, path)

    def process_answer(self, original_query: str, question: ClarificationQuestion, answer: str, options: List[Dict[str, Any]], state: Dict, path: ClassificationPath = None) -> tuple:
        """Process the user's answer to update the product description and select the best matching option.
        
        Args:
            path: Optional ClassificationPath for trajectory mode support
        """
        return self.classification_engine.process_answer(original_query, question, answer, options, state, path)

    def explain_classification(self, original_query: str, enriched_query: str, full_path: str, conversation: List[Dict[str, Any]], state: Dict = None) -> str:
        """Generate an explanation of the classification."""
        return self.classification_engine.explain_classification(original_query, enriched_query, full_path, conversation, state)

    # ----------------------
    # Main Classification Workflows
    # ----------------------

    def start_classification(self, product: str, interactive: bool = True, max_questions: int = 3, use_multi_hypothesis: bool = True, hypothesis_count: int = 3, classification_knowledge: Optional[str] = None) -> Dict:
        """
        Begin classification. Returns either a clarification_question or final classification.
        Now includes training data collection.
        """
        # Note: Training data session is started by the calling code (e.g., parallel processor)
        # to ensure proper thread-local session management
        
        # Research generation disabled
        formatted_knowledge = ""
        
        # Create a fresh state dictionary with all necessary fields
        state = {
            "product": product,
            "original_query": product,
            "current_query": product,
            "questions_asked": 0,
            "selection": {},
            "current_node": None,
            "classification_path": [],
            "steps": [],
            "conversation": [],
            "pending_question": None,
            "pending_stage": None,
            "max_questions": max_questions,
            "visited_nodes": [],
            "history": [],
            "product_attributes": {},
            "recent_questions": [],
            "global_retry_count": 0,
            "classification_diagnosis": None,
            "use_multi_hypothesis": use_multi_hypothesis,
            "hypothesis_count": hypothesis_count,
            "paths": [],
            "beam": [],
            "paths_considered": [],
            "active_question_path": None,
            "used_multi_hypothesis": False,
            "classification_knowledge": formatted_knowledge
        }

        try:
            result = self.process_classification(state, interactive, max_questions)
            
            # End training session and determine if we should keep it
            if result.get("final", False):
                success = training_collector.should_keep_session(result)
                training_collector.end_session(result, success)
            
            # Serialize the final state if needed
            if "state" in result:
                result["state"] = HTSSerializer.serialize_state(result["state"])
            
            return result
            
        except Exception as e:
            logging.error(f"Error in start_classification: {e}")
            # End training session as failed
            training_collector.end_session({"error": str(e)}, False)
            raise

    def continue_classification(self, state: Dict, answer: str, interactive: bool = True, max_questions: int = 3) -> Dict:
        """
        Continue classification after receiving an answer to a clarification question.
        """
        logging.info("=== STARTING continue_classification ===")
        
        try:
            # First, deserialize the state to ensure we have proper objects
            state = HTSSerializer.deserialize_state(state, self)
            
            # Restore beam from JSON if needed
            if state.get("beam"):
                logging.info(f"Restoring beam with {len(state['beam'])} paths")
                restored_beam = []
                
                for i, path_data in enumerate(state["beam"]):
                    if isinstance(path_data, dict):
                        try:
                            path = ClassificationPath.from_dict(path_data, self)
                            restored_beam.append(path)
                            logging.info(f"[OK] Restored path {i+1}: {path.path_id}")
                        except Exception as e:
                            logging.error(f"Failed to restore path {i+1}: {e}")
                    elif hasattr(path_data, 'path_id'):
                        restored_beam.append(path_data)
                        logging.info(f"[OK] Restored path {i+1}: {path_data.path_id}")
                
                if not restored_beam:
                    logging.error("Beam restoration failed - restarting")
                    state["beam"] = []
                    state["global_retry_count"] = state.get("global_retry_count", 0) + 1
                    return self.process_classification(state, interactive, max_questions)
                
                state["beam"] = restored_beam
                logging.info(f"[OK] Successfully restored beam with {len(restored_beam)} paths")
            
            # Validate pending question
            if not state.get("pending_question"):
                logging.warning("No pending question - resuming process")
                return self.process_classification(state, interactive, max_questions)
            
            # Process the answer to enrich the product description
            state["questions_asked"] += 1
            pending_question_dict = state.get("pending_question", {})
            question_text = pending_question_dict.get("question_text", "")
            
            # Log Q&A
            state["conversation"].append({"question": question_text, "answer": answer})
            if "history" not in state:
                state["history"] = []
            state["history"].append({"question": question_text, "answer": answer})
            
            # Create question object for processing
            question_obj = ClarificationQuestion()
            question_obj.question_text = question_text
            question_obj.question_type = pending_question_dict.get("question_type", "text")
            question_obj.options = pending_question_dict.get("options", [])
            question_obj.metadata = pending_question_dict.get("metadata", {})
            options_metadata = question_obj.metadata.get("options", [])
            
            # Process answer to update product description
            logging.info("Processing answer to enrich product description...")
            # Get the active path for trajectory mode (top path in beam)
            active_path = state["beam"][0] if state.get("beam") else None
            updated_query, _ = self.process_answer(state["current_query"], question_obj, answer, options_metadata, state, active_path)
            state["current_query"] = updated_query
            logging.info(f"Product description enriched: \"{state['current_query']}\"")
            
            # Clear pending question
            state["pending_question"] = None
            state["pending_stage"] = None
            
            # Resume main processing loop
            logging.info("Resuming main classification with enriched description")
            result = self.process_classification(state, interactive, max_questions)
            
            # If classification reached final in continue path, end training session
            if result.get("final", False):
                try:
                    success = training_collector.should_keep_session(result)
                    training_collector.end_session(result, success)
                except Exception as _e:
                    logging.error(f"Failed to end training session in continue_classification: {_e}")
            
            # Serialize the final state if needed
            if "state" in result:
                result["state"] = HTSSerializer.serialize_state(result["state"])
            
            logging.info("=== ENDING continue_classification ===")
            return result
            
        except Exception as e:
            logging.error(f"Error in continue_classification: {e}")
            raise

    def process_classification(self, state: Dict, interactive: bool, max_questions: int) -> Dict:
        """
        Core classification logic with multi-hypothesis support.
        """
        # --- Initialization Step (only runs once at the very beginning) ---
        if not state.get("beam"):
            logging.info("--- Initializing New Classification ---")
            state["global_retry_count"] = state.get("global_retry_count", 0) + 1
            if state["global_retry_count"] >= 5:
                return {"error": "Classification failed after multiple attempts.", "final": False, "state": state}

            hypothesis_count = state.get("hypothesis_count", 3)
            diagnosis = state.get("classification_diagnosis")
            
            paths = self.initialize_classification_paths(state["current_query"], hypothesis_count, diagnosis, state)
            if not paths:
                return {"error": "Could not initialize classification paths.", "final": False, "state": state}
            
            state["beam"] = paths
            state["used_multi_hypothesis"] = True
            state["paths_considered"] = [{"chapter": p.chapter, "reasoning": p.reasoning_log[0] if p.reasoning_log else ""} for p in paths]
            logging.info(f"Initialized beam with {len(paths)} paths")

        beam = state.get("beam", [])
        k = CLASSIFICATION_BEAM_SIZE
        
        # Validate beam
        if not beam:
            logging.error("Empty beam in process_classification")
            return {"error": "Empty classification beam.", "final": False, "state": state}
        
        max_iterations = 25
        iteration = state.get("iteration_count", 0)

        while iteration < max_iterations:
            iteration += 1
            state["iteration_count"] = iteration
            logging.info(f"--- Beam Search Iteration {iteration} (Beam Size: {len(beam)}) ---")
            
            # Clear any stale cached candidates from previous iterations
            state.pop("_cached_scored_candidates", None)
            
            # 1. Check for termination
            should_terminate, best_path = self.check_termination_conditions(beam)
            if should_terminate:
                if best_path:
                    logging.info(f"Termination condition met. Returning best path: {best_path.path_id}")
                    return self._convert_path_to_result(best_path, state)
                else:
                    logging.warning("Termination met, but no best path found. All paths pruned.")
                    return {"error": "Classification failed - all paths were eliminated.", "final": False, "state": state}

            # 2. Check if a question is needed for the top path
            top_path = beam[0]
            if top_path.is_active and interactive and state["questions_asked"] < max_questions:
                node_for_next_step = top_path.current_node or self.create_chapter_parent(top_path.chapter)
                options = self.get_children(node_for_next_step)
                
                if options:
                    scored_candidates = self.score_candidates(state["current_query"], node_for_next_step, options, top_path, state)
                    if scored_candidates:
                        # Cache scored candidates to avoid duplicate LLM calls in beam advancement
                        # This prevents the same path/node from being scored twice
                        state["_cached_scored_candidates"] = {
                            "path_id": top_path.path_id,
                            "node_id": node_for_next_step.node_id if hasattr(node_for_next_step, 'node_id') else id(node_for_next_step),
                            "candidates": scored_candidates
                        }
                        
                        # Find best candidate by decision confidence (for question threshold)
                        best_candidate = max(scored_candidates, key=lambda item: item[1])
                        _, best_decision_confidence, best_path_confidence, _ = best_candidate
                        
                        # Log all candidates with both confidences
                        logging.info(f"--- Candidate Scores for {top_path.path_id} ---")
                        for node, dec_conf, path_conf, reasoning in scored_candidates:
                            marker = " [BEST]" if (node, dec_conf, path_conf, reasoning) == best_candidate else ""
                            logging.info(
                                f"  {node.htsno or '[GROUP]'}: "
                                f"decision_conf={dec_conf:.3f}, path_conf={path_conf:.3f}{marker}"
                            )
                        logging.info("--- End Candidate Scores ---")
                        
                        if best_decision_confidence < CONFIDENCE_THRESHOLD:
                            logging.info(
                                f"Top path ({top_path.path_id}) has low decision confidence "
                                f"({best_decision_confidence:.3f}) but path confidence "
                                f"({best_path_confidence:.3f}) for next step. Generating question."
                            )
                            stage = TreeUtils.determine_next_stage(node_for_next_step)
                            question = self.generate_clarification_question(state["current_query"], node_for_next_step, stage, state, top_path)
                            
                            # Save state and return the question to the user
                            state["pending_question"] = question.to_dict()
                            state["pending_stage"] = stage
                            state["beam"] = [p.to_dict() for p in beam]
                            
                            return {
                                "final": False, 
                                "clarification_question": question, 
                                "state": state,
                                "debug_info": {"next_step_confidence_trigger": best_decision_confidence}
                            }
                        else:
                            logging.info(
                                f"Top path ({top_path.path_id}) has sufficient decision confidence "
                                f"({best_decision_confidence:.3f}) and path confidence "
                                f"({best_path_confidence:.3f}). Proceeding without question."
                            )

            # 3. If no question was asked, advance the beam
            logging.info(f"Confidence sufficient or questioning disabled. Advancing the beam.")
            new_beam = self.advance_beam(beam, state["current_query"], k, state)
            
            beam = new_beam
            state["beam"] = beam
            
            if not beam:
                logging.error("Beam became empty after advancement.")
                return {"error": "Classification failed - all paths were eliminated during beam advancement.", "final": False, "state": state}

        # If loop finishes due to max iterations, return the best path
        if not beam:
            return {"error": "Classification failed after maximum iterations.", "final": False, "state": state}
             
        best_path = max(beam, key=lambda p: p.log_score)
        logging.warning(f"Max iterations reached. Returning best available path: {best_path.path_id}")
        return self._convert_path_to_result(best_path, state)

    def _convert_path_to_result(self, path: ClassificationPath, state: Dict) -> Dict:
        """Convert a ClassificationPath to a result dictionary."""
        # Get the final code and full path
        final_code = path.get_final_code() or ""
        full_path = path.get_full_path_string()
        
        # Debug: Log TreeRL pruned leaves count
        pruned_count = len(state.get("_treerl_pruned_leaves", []))
        logging.info(f"TreeRL: Returning result with {pruned_count} accumulated pruned leaves in state")
        
        # Create the classification result
        result = {
            "final": True,
            "classification": {
                "code": final_code,
                "path": full_path,
                "confidence": path.cumulative_confidence,
                "log_score": path.log_score,
                "chapter": path.chapter,
                "steps": path.steps,
                "reasoning": path.reasoning_log
            },
            "state": state
        }
        
        # Add explanation if we have conversation history
        if state.get("conversation"):
            explanation = self.explain_classification(
                state.get("original_query", ""),
                state.get("current_query", ""),
                full_path,
                state.get("conversation", []),
                state
            )
            result["explanation"] = explanation
        
        return result

    # ----------------------
    # Streaming Support (delegated to streaming_engine)
    # ----------------------

    async def run_classification_with_events(self, product: str, interactive: bool = True, max_questions: int = 3, 
                                           use_multi_hypothesis: bool = True, hypothesis_count: int = 3):
        """Run classification while yielding Server-Sent Events for real-time updates."""
        async for event in self.streaming_engine.run_classification_with_events(product, interactive, max_questions, use_multi_hypothesis, hypothesis_count):
            yield event

    async def continue_classification_with_events(self, state: Dict, answer: str, interactive: bool = True, max_questions: int = 3):
        """Continue streaming classification after receiving an answer to a clarification question."""
        async for event in self.streaming_engine.continue_classification_with_events(state, answer, interactive, max_questions):
            yield event

    # ----------------------
    # Diagnosis Methods (placeholder for future implementation)
    # ----------------------
    
    def _diagnose_classification_issue(self, product_description: str, chapter: str, issue_type: str) -> str:
        """Generate a diagnosis for classification issues."""
        # This would be implemented if needed for diagnosis functionality
        return f"Classification issue diagnosed for {product_description} in chapter {chapter}: {issue_type}"

    # ----------------------
    # Helper Methods for Backward Compatibility
    # ----------------------

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from response text (delegated to LLM client)."""
        return self.llm_client._extract_json_from_response(response_text)

    def _get_vertex_ai_client(self):
        """Get authenticated Vertex AI client (delegated to LLM client)."""
        return self.llm_client._get_vertex_ai_client()

    def _setup_prompt_logger(self):
        """Set up prompt logger (delegated to LLM client)."""
        if hasattr(self.llm_client, '_setup_prompt_logger'):
            self.llm_client._setup_prompt_logger()

    def _format_history_for_prompt(self, history_entries):
        """Convert history entries to formatted string for prompt."""
        return self.classification_engine._format_history_for_prompt(history_entries)

    def _has_similar_question(self, question_text, history_entries, similarity_threshold=0.6):
        """Check if a similar question exists in history."""
        return self.classification_engine._has_similar_question(question_text, history_entries, similarity_threshold)

    def _format_classification_knowledge(self, raw_knowledge: str) -> str:
        """Format the classification knowledge with explicit guidance."""
        return self.classification_engine._format_classification_knowledge(raw_knowledge)

    def _evaluate_path_with_answer(self, path: ClassificationPath, updated_description: str, 
                                 answer: str, question: ClarificationQuestion, state: Dict = None) -> float:
        """Evaluate how well a path aligns with the user's answer."""
        # This would need to be implemented if the original had this method
        return 0.5  # Default neutral confidence

    def _process_beam_search_classification(self, state: Dict, interactive: bool, max_questions: int) -> Dict:
        """Process classification using beam search (alias for process_classification)."""
        return self.process_classification(state, interactive, max_questions)
