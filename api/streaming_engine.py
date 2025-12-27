import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, AsyncGenerator
from .models import ClassificationPath
from .tree_navigator import TreeNavigator
from .llm_client import LLMClient
from .classification_engine import ClassificationEngine, CLASSIFICATION_BEAM_SIZE, CONFIDENCE_THRESHOLD, LLM_TEMPERATURE

class StreamingEngine:
    """
    Handles Server-Sent Events (SSE) streaming for real-time classification updates.
    Separated from the main engine for modularity and async support.
    """
    
    def __init__(self, navigator: TreeNavigator, llm_client: LLMClient, classification_engine: ClassificationEngine):
        self.navigator = navigator
        self.llm = llm_client
        self.engine = classification_engine

    async def run_classification_with_events(self, product: str, interactive: bool = True, max_questions: int = 3, 
                                           use_multi_hypothesis: bool = True, hypothesis_count: int = 3) -> AsyncGenerator[Dict, None]:
        """
        Run classification while yielding Server-Sent Events for real-time updates.
        Implements full streaming beam search with event yielding.
        
        Yields events for:
        - Chapter selection
        - Path initialization
        - Beam advancement
        - Candidate scoring
        - Question generation
        - Final classification
        """
        # Yield start event
        yield {
            "type": "classification_start",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "product": product,
                "interactive": interactive,
                "max_questions": max_questions,
                "use_multi_hypothesis": use_multi_hypothesis,
                "hypothesis_count": hypothesis_count
            }
        }
        
        # No longer generating classification knowledge - removed research logic
        formatted_knowledge = ""
        
        # Create enhanced state that tracks streaming
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
            "streaming": True,  # Flag to indicate streaming mode
            "iteration_count": 0,  # Initialize iteration count
            "classification_knowledge": formatted_knowledge  # Store the formatted knowledge
        }
        
        try:
            # Initialize classification paths with streaming events
            yield {
                "type": "initialization_start",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "status": "Determining top chapters",
                    "hypothesis_count": hypothesis_count
                }
            }
            
            # Get top chapters with events
            diagnosis = state.get("classification_diagnosis")
            top_chapters = self.engine.determine_top_chapters(state["current_query"], hypothesis_count, diagnosis, state)
            
            if not top_chapters:
                yield {
                    "type": "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "error": "No chapters found",
                        "message": "Could not determine appropriate chapters for this product"
                    }
                }
                return
            
            yield {
                "type": "chapter_selection",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "chapters": [
                        {
                            "chapter": chapter,
                            "confidence": confidence,
                            "reasoning": reasoning,
                            "description": self.navigator.chapters_map.get(int(chapter), "Unknown chapter")
                        }
                        for chapter, confidence, reasoning in top_chapters
                    ]
                }
            }
            
            # Initialize paths from chapters
            paths = []
            for i, (chapter, confidence, reasoning) in enumerate(top_chapters, 1):
                chapter_desc = self.navigator.chapters_map.get(int(chapter), "Unknown chapter")
                path = ClassificationPath(f"path_{i}", chapter, confidence, chapter_desc)
                path.reasoning_log.append(f"Initial chapter selection: {chapter} - {chapter_desc} (conf: {confidence:.3f}) - {reasoning}")
                paths.append(path)
            
            state["beam"] = paths
            state["used_multi_hypothesis"] = True
            state["paths_considered"] = [{"chapter": p.chapter, "reasoning": p.reasoning_log[0] if p.reasoning_log else ""} for p in paths]
            
            # Now run the beam search with event streaming
            beam = state["beam"]
            k = CLASSIFICATION_BEAM_SIZE
            max_iterations = 25
            iteration = state.get("iteration_count", 0)
            
            while iteration < max_iterations:
                iteration += 1
                state["iteration_count"] = iteration
                
                # Clear any stale cached candidates from previous iterations
                state.pop("_cached_scored_candidates", None)
                
                yield {
                    "type": "iteration_start",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "iteration": iteration,
                        "beam_size": len(beam)
                    }
                }
                
                # Check for termination
                should_terminate, best_path = self.engine.check_termination_conditions(beam)
                if should_terminate:
                    if best_path:
                        # Generate explanation
                        explanation = ""
                        if state.get("conversation"):
                            explanation = self.engine.explain_classification(
                                state.get("original_query", ""),
                                state.get("current_query", ""),
                                best_path.get_full_path_string(),
                                state.get("conversation", []),
                                state
                            )
                        
                        yield {
                            "type": "classification_complete",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "data": {
                                "final_code": best_path.get_final_code(),
                                "full_path": best_path.get_full_path_string(),
                                "confidence": best_path.cumulative_confidence,
                                "log_score": best_path.log_score,
                                "reasoning": best_path.reasoning_log,
                                "explanation": explanation
                            }
                        }
                        return
                    else:
                        yield {
                            "type": "error",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "data": {
                                "error": "No valid classification path found",
                                "message": "All paths were pruned"
                            }
                        }
                        return
                
                # Check if question needed for the top path
                top_path = beam[0]
                if top_path.is_active and interactive and state["questions_asked"] < max_questions:
                    node_for_next_step = top_path.current_node or self.navigator.create_chapter_parent(top_path.chapter)
                    options = self.navigator.get_children(node_for_next_step)
                    
                    if options:
                        yield {
                            "type": "candidate_scoring_start",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "data": {
                                "path_id": top_path.path_id,
                                "current_node": node_for_next_step.htsno or "[GROUP]",
                                "options_count": len(options)
                            }
                        }
                        
                        scored_candidates = self.engine.score_candidates(state["current_query"], node_for_next_step, options, top_path, state)
                        
                        if scored_candidates:
                            _, best_decision_confidence, _, _ = max(scored_candidates, key=lambda item: item[1])
                            
                            yield {
                                "type": "candidate_scoring_complete",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "data": {
                                    "candidates": [
                                        {
                                            "code": node.htsno or "[GROUP]",
                                            "description": node.description,
                                            "decision_confidence": decision_confidence,
                                            "path_confidence": path_confidence,
                                            "reasoning": reasoning
                                        }
                                        for node, decision_confidence, path_confidence, reasoning in scored_candidates
                                    ],
                                    "best_decision_confidence": best_decision_confidence
                                }
                            }
                            
                            if best_decision_confidence < CONFIDENCE_THRESHOLD:
                                from .models import TreeUtils
                                stage = TreeUtils.determine_next_stage(node_for_next_step)
                                # Pass the top_path for trajectory mode support
                                question = self.engine.generate_clarification_question(state["current_query"], node_for_next_step, stage, state, top_path)
                                
                                # Save state before returning question
                                state["pending_question"] = question.to_dict()
                                state["pending_stage"] = stage
                                # Serialize the beam for state preservation
                                state["beam"] = [p.to_dict() for p in beam]
                                
                                yield {
                                    "type": "question_generated",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "data": {
                                        "question": question.to_dict(),
                                        "confidence_trigger": best_decision_confidence,
                                        "stage": stage,
                                        "state": state  # Include state for continuation
                                    }
                                }
                                return  # Stop here and wait for user response
                
                # Advance beam
                yield {
                    "type": "beam_advancement_start",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "current_beam_size": len(beam)
                    }
                }
                
                new_beam = self.engine.advance_beam(beam, state["current_query"], k, state)
                
                yield {
                    "type": "beam_leaderboard",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "beam": [
                            {
                                "position": i + 1,
                                "path_id": path.path_id,
                                "chapter": path.chapter,
                                "current_path": path.get_full_path_string(),
                                "log_score": path.log_score,
                                "cumulative_confidence": path.cumulative_confidence,
                                "is_active": path.is_active,
                                "is_complete": path.is_complete
                            }
                            for i, path in enumerate(new_beam)
                        ]
                    }
                }
                
                beam = new_beam
                state["beam"] = beam
                
                if not beam:
                    yield {
                        "type": "error",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "error": "Beam became empty",
                            "message": "All paths were eliminated during advancement"
                        }
                    }
                    return
            
            # Max iterations reached
            if beam:
                best_path = max(beam, key=lambda p: p.log_score)
                yield {
                    "type": "classification_complete",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "final_code": best_path.get_final_code(),
                        "full_path": best_path.get_full_path_string(),
                        "confidence": best_path.cumulative_confidence,
                        "log_score": best_path.log_score,
                        "reasoning": best_path.reasoning_log,
                        "note": "Max iterations reached"
                    }
                }
            else:
                yield {
                    "type": "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "error": "Classification failed",
                        "message": "Maximum iterations reached with no valid paths"
                    }
                }
                
        except Exception as e:
            logging.error(f"Error in run_classification_with_events: {e}", exc_info=True)
            yield {
                "type": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "error": str(e),
                    "message": "Classification stream error"
                }
            }

    async def continue_classification_with_events(self, state: Dict, answer: str, interactive: bool = True, max_questions: int = 3) -> AsyncGenerator[Dict, None]:
        """
        Continue streaming classification after receiving an answer to a clarification question.
        This version processes the answer and continues beam search with full event streaming.
        
        Args:
            state: The classification state (including serialized beam)
            answer: User's answer to the clarification question
            interactive: Whether to ask clarification questions
            max_questions: Maximum number of questions to ask
            
        Yields:
            SSE events for the continuation process
        """
        import hashlib
        
        logging.info("=== STREAMING CONTINUE: Starting continue_classification_with_events ===")
        logging.info(f"Answer received: '{answer}'")
        logging.info(f"Questions asked so far: {state.get('questions_asked', 0)}")
        logging.info(f"Beam size in state: {len(state.get('beam', []))}")
        
        # Create detailed checksum for state comparison
        try:
            beam_data = state.get('beam', [])
            import json
            state_str = json.dumps(beam_data, sort_keys=True)
            state_checksum = hashlib.md5(state_str.encode()).hexdigest()[:8]
            logging.info(f"State checksum (beam): {state_checksum}")
        except Exception as e:
            logging.error(f"Failed to create state checksum: {e}")
            state_checksum = "ERROR"
        
        # Yield continuation start event
        yield {
            "type": "continuation_start",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "answer": answer,
                "questions_asked": state.get("questions_asked", 0),
                "max_questions": max_questions,
                "beam_size": len(state.get("beam", [])),
                "state_checksum": state_checksum  # For debugging
            }
        }
        
        try:
            # Validate state has a pending question
            if not state.get("pending_question"):
                logging.error("No pending question in state!")
                yield {
                    "type": "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "error": "No pending question",
                        "message": "Cannot continue without a pending question"
                    }
                }
                return
            
            # Yield answer processing event
            yield {
                "type": "answer_processing",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "question": state.get("pending_question", {}).get("question_text", ""),
                    "answer": answer,
                    "stage": state.get("pending_stage", "unknown")
                }
            }
            
            # Enhanced beam restoration with corruption detection
            if state.get("beam"):
                logging.info(f"Restoring beam with {len(state['beam'])} paths")
                restored_beam = []
                restoration_errors = []
                
                for i, path_data in enumerate(state["beam"]):
                    if isinstance(path_data, dict):
                        try:
                            path = ClassificationPath.from_dict(path_data, self.navigator)
                            restored_beam.append(path)
                            logging.info(f"[OK] Restored path {i+1}: {path.path_id}")
                        except Exception as e:
                            error_msg = f"Failed to restore path {i+1}: {e}"
                            logging.error(error_msg)
                            restoration_errors.append(error_msg)
                    elif hasattr(path_data, 'path_id'):
                        # Already a ClassificationPath object
                        restored_beam.append(path_data)
                        logging.info(f"[OK] Path {i+1} already restored: {path_data.path_id}")
                    else:
                        error_msg = f"Path {i+1} invalid type: {type(path_data)}"
                        logging.error(f"CORRUPTION DETECTED: {error_msg}")
                        restoration_errors.append(error_msg)
                
                if not restored_beam:
                    logging.error("CRITICAL ERROR: Complete beam restoration failure")
                    yield {
                        "type": "error",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "error": "Beam restoration failed",
                            "message": "Could not restore classification paths from state",
                            "restoration_errors": restoration_errors
                        }
                    }
                    return
                
                state["beam"] = restored_beam
                logging.info(f"[OK] Successfully restored beam with {len(restored_beam)} paths")
                
                # Process the answer
                state["questions_asked"] += 1
                pending_question_dict = state.get("pending_question", {})
                question_text = pending_question_dict.get("question_text", "")
                
                # Log Q&A
                state["conversation"].append({"question": question_text, "answer": answer})
                if "history" not in state:
                    state["history"] = []
                state["history"].append({"question": question_text, "answer": answer})
                
                # Create question object for processing
                from .models import ClarificationQuestion
                question_obj = ClarificationQuestion()
                question_obj.question_text = question_text
                question_obj.question_type = pending_question_dict.get("question_type", "text")
                question_obj.options = pending_question_dict.get("options", [])
                question_obj.metadata = pending_question_dict.get("metadata", {})
                options_metadata = question_obj.metadata.get("options", [])
                
                # Get the active path for trajectory mode (top path in beam)
                active_path = state["beam"][0] if state.get("beam") else None
                
                # Process answer to update product description
                updated_query, _ = self.engine.process_answer(state["current_query"], question_obj, answer, options_metadata, state, active_path)
                state["current_query"] = updated_query
                logging.info(f"Product description enriched: \"{state['current_query']}\"")
                
                # Yield the enriched description event
                yield {
                    "type": "description_enriched",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "original_description": state.get("original_query", ""),
                        "previous_description": question_obj.metadata.get("original_query", state.get("original_query", "")),
                        "updated_description": updated_query,
                        "extracted_attributes": state.get("product_attributes", {}),
                        "question_asked": question_text,
                        "answer_provided": answer
                    }
                }
                
                # Clear pending question
                state["pending_question"] = None
                state["pending_stage"] = None
                
                # Continue with beam search iterations WITH EVENT STREAMING
                beam = state.get("beam", [])
                k = CLASSIFICATION_BEAM_SIZE
                max_iterations = 25
                iteration = state.get("iteration_count", 0)
                
                while iteration < max_iterations:
                    iteration += 1
                    state["iteration_count"] = iteration
                    
                    # Clear any stale cached candidates from previous iterations
                    state.pop("_cached_scored_candidates", None)
                    
                    yield {
                        "type": "iteration_start",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "iteration": iteration,
                            "beam_size": len(beam)
                        }
                    }
                    
                    # Check for termination
                    should_terminate, best_path = self.engine.check_termination_conditions(beam)
                    if should_terminate:
                        if best_path:
                            # Generate explanation
                            explanation = ""
                            if state.get("conversation"):
                                explanation = self.engine.explain_classification(
                                    state.get("original_query", ""),
                                    state.get("current_query", ""),
                                    best_path.get_full_path_string(),
                                    state.get("conversation", []),
                                    state
                                )
                            
                            yield {
                                "type": "classification_complete",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "data": {
                                    "final_code": best_path.get_final_code(),
                                    "full_path": best_path.get_full_path_string(),
                                    "confidence": best_path.cumulative_confidence,
                                    "log_score": best_path.log_score,
                                    "reasoning": best_path.reasoning_log,
                                    "explanation": explanation
                                }
                            }
                            return
                        else:
                            yield {
                                "type": "error",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "data": {
                                    "error": "No valid classification path found",
                                    "message": "All paths were pruned"
                                }
                            }
                            return
                    
                    # Check if question needed
                    top_path = beam[0]
                    if top_path.is_active and interactive and state["questions_asked"] < max_questions:
                        node_for_next_step = top_path.current_node or self.navigator.create_chapter_parent(top_path.chapter)
                        options = self.navigator.get_children(node_for_next_step)
                        
                        if options:
                            yield {
                                "type": "candidate_scoring_start",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "data": {
                                    "path_id": top_path.path_id,
                                    "current_node": node_for_next_step.htsno or "[GROUP]",
                                    "options_count": len(options)
                                }
                            }
                            
                            scored_candidates = self.engine.score_candidates(state["current_query"], node_for_next_step, options, top_path, state)
                            
                            if scored_candidates:
                                # Cache scored candidates to avoid duplicate LLM calls in beam advancement
                                # This prevents the same path/node from being scored twice
                                state["_cached_scored_candidates"] = {
                                    "path_id": top_path.path_id,
                                    "node_id": node_for_next_step.node_id if hasattr(node_for_next_step, 'node_id') else id(node_for_next_step),
                                    "candidates": scored_candidates
                                }
                                
                                _, best_decision_confidence, _, _ = max(scored_candidates, key=lambda item: item[1])
                                
                                yield {
                                    "type": "candidate_scoring_complete",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "data": {
                                        "candidates": [
                                            {
                                                "code": node.htsno or "[GROUP]",
                                                "description": node.description,
                                                "decision_confidence": decision_confidence,
                                                "path_confidence": path_confidence,
                                                "reasoning": reasoning
                                            }
                                            for node, decision_confidence, path_confidence, reasoning in scored_candidates
                                        ],
                                        "best_decision_confidence": best_decision_confidence
                                    }
                                }
                                
                                if best_decision_confidence < CONFIDENCE_THRESHOLD:
                                    from .models import TreeUtils
                                    stage = TreeUtils.determine_next_stage(node_for_next_step)
                                    # Pass the top_path for trajectory mode support
                                    question = self.engine.generate_clarification_question(state["current_query"], node_for_next_step, stage, state, top_path)
                                    
                                    # Save state before returning question
                                    state["pending_question"] = question.to_dict()
                                    state["pending_stage"] = stage
                                    # Serialize the beam for state preservation
                                    state["beam"] = [p.to_dict() for p in beam]
                                    
                                    yield {
                                        "type": "question_generated",
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "data": {
                                            "question": question.to_dict(),
                                            "confidence_trigger": best_decision_confidence,
                                            "stage": stage,
                                            "state": state
                                        }
                                    }
                                    return
                    
                    # Advance beam
                    yield {
                        "type": "beam_advancement_start",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "current_beam_size": len(beam)
                        }
                    }
                    
                    new_beam = self.engine.advance_beam(beam, state["current_query"], k, state)
                    
                    yield {
                        "type": "beam_leaderboard",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "beam": [
                                {
                                    "position": i + 1,
                                    "path_id": path.path_id,
                                    "chapter": path.chapter,
                                    "current_path": path.get_full_path_string(),
                                    "log_score": path.log_score,
                                    "cumulative_confidence": path.cumulative_confidence,
                                    "is_active": path.is_active,
                                    "is_complete": path.is_complete
                                }
                                for i, path in enumerate(new_beam)
                            ]
                        }
                    }
                    
                    beam = new_beam
                    state["beam"] = beam
                    
                    if not beam:
                        yield {
                            "type": "error",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "data": {
                                "error": "Beam became empty",
                                "message": "All paths were eliminated during advancement"
                            }
                        }
                        return
                
                # Max iterations reached
                if beam:
                    best_path = max(beam, key=lambda p: p.log_score)
                    yield {
                        "type": "classification_complete",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "final_code": best_path.get_final_code(),
                            "full_path": best_path.get_full_path_string(),
                            "confidence": best_path.cumulative_confidence,
                            "log_score": best_path.log_score,
                            "reasoning": best_path.reasoning_log,
                            "note": "Max iterations reached"
                        }
                    }
                else:
                    yield {
                        "type": "error",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "error": "Classification failed",
                            "message": "Maximum iterations reached with no valid paths"
                        }
                    }
                
        except Exception as e:
            logging.error(f"Error in continue_classification_with_events: {e}", exc_info=True)
            yield {
                "type": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "error": str(e),
                    "message": "Classification continuation stream error"
                }
            }
        finally:
            logging.info("=== STREAMING CONTINUE: Ending continue_classification_with_events ===")
