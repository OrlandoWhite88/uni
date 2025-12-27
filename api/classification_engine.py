import json
import logging
import math
import re
import time
import os
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from .models import HTSNode, ClassificationPath, ClarificationQuestion, TreeUtils
from .tree_navigator import TreeNavigator
from .llm_client import LLMClient
from .training_data_collector import training_collector, build_state
from .notes_loader import (
    load_relevant_notes_for_chapters,
    load_chapter_context,
    extract_chapter_from_node,
    extract_chapter_from_path,
)
from .prompt_builder import (
    build_system_prompt,
    build_chapter_selection_prompt_stage1,
    build_chapter_selection_prompt_stage2,
    build_candidate_ranking_prompt,
    build_question_generation_prompt,
    build_answer_processing_prompt,
)
from .system_prompts_updated import UNIFIED_SYSTEM_PROMPT


class RetryableLLMError(RuntimeError):
    """Raised when an LLM/API call fails and the entire classification should be retried."""

# ----------------------
# Configuration Constants
# ----------------------

# Beam search configuration - adjust these values to experiment
# These can be overridden via environment variables for TreeRL training (k=8 recommended)
CHAPTER_BEAM_SIZE = int(os.environ.get("TREERL_CHAPTER_BEAM_SIZE", "6"))  # Number of chapters to request from LLM
CLASSIFICATION_BEAM_SIZE = int(os.environ.get("TREERL_BEAM_SIZE", "3"))  # Beam size for heading/subheading/tariff

# Confidence threshold configuration
CONFIDENCE_THRESHOLD = 0.60 # Confidence threshold for asking clarification questions

# LLM Temperature configuration
LLM_TEMPERATURE = 0  # Set to 0 for deterministic, reproducible classifications

# Debug logging configuration - set LOG_PROMPT_DEBUG=true to enable full prompt logging
LOG_PROMPT_DEBUG = os.environ.get("LOG_PROMPT_DEBUG", "false").lower() == "true"

# Trajectory mode configuration - set TRAJECTORY_MODE=true to use multi-turn chat trajectories per path
TRAJECTORY_MODE = os.environ.get("TRAJECTORY_MODE", "true").lower() == "true"


def _unwrap_json_response(data: Any) -> Dict:
    """
    Unwrap JSON response that might be wrapped in a list or use alternative format.
    
    Handles two formats:
    1. Standard: {"primary_selection": {...}, "alternative_1": {...}, ...}
    2. List format: [{"option_index": 1, ...}, {"option_index": 2, ...}]
       -> Converted to standard format with inferred keys
    """
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            # Check if this is a list of selections (LLM returning array instead of named dict)
            # If first item has 'option_index' but no 'primary_selection', it's the list format
            if "option_index" in data[0] and "primary_selection" not in data[0]:
                logging.debug(f"Converting list-of-selections format: [{len(data)} items] -> named dict")
                # Convert list to named format
                selection_keys = ["primary_selection", "alternative_1", "alternative_2"]
                result = {}
                for i, item in enumerate(data[:3]):  # Take up to 3
                    if i < len(selection_keys):
                        result[selection_keys[i]] = item
                return result
            else:
                # Original behavior - take first dict (e.g., Gemini wrapping)
                logging.debug(f"Unwrapped list response: [{len(data)} items] -> first dict")
                return data[0]
        logging.warning(f"Received empty or invalid list response, returning empty dict")
        return {}
    if isinstance(data, dict):
        return data
    logging.warning(f"Unexpected JSON response type: {type(data).__name__}, returning empty dict")
    return {}


class ClassificationEngine:
    """
    Handles all classification logic including beam search, candidate scoring,
    question generation, and answer processing.
    Combined engine with configuration, core logic, and question handling.
    """
    
    def __init__(self, tree_navigator: TreeNavigator, llm_client: LLMClient):
        self.navigator = tree_navigator
        self.llm = llm_client
        self.max_questions_per_level = 3
        
        # Concurrency settings
        self.path_workers = int(os.environ.get("PATH_WORKERS", "10"))  # Number of concurrent path workers
        self.calibrate_workers = int(os.environ.get("CALIBRATE_WORKERS", "10"))  # Number of concurrent calibration workers

    # ----------------------
    # Multi-Hypothesis Path Exploration Methods
    # ----------------------

    def determine_top_chapters(self, product_description: str, k: int = CHAPTER_BEAM_SIZE, diagnosis: Optional[str] = None, state: Dict = None) -> List[Tuple[str, float, str]]:
        """
        Two-stage chapter selection with notes consultation.
        
        Stage 1: Select initial candidates WITHOUT notes (fast, broad selection)
        Stage 2: Re-evaluate with specific chapter notes for selected chapters (refined selection)
        
        When TRAJECTORY_MODE is enabled, captures the LLM calls for later inclusion in path trajectories.
        """
        # Initialize chapter selection trajectory storage in state
        if state is not None and TRAJECTORY_MODE:
            state["_chapter_selection_trajectory"] = []
        
        # Stage 1: Initial selection WITHOUT notes
        logging.info("=== STAGE 1: Initial chapter selection (no notes) ===")
        stage1_k = 6  # Always request top 6 chapters in stage 1
        stage1_chapters = self._select_chapters_stage1(
            product_description, stage1_k, diagnosis, state
        )
        
        if not stage1_chapters:
            logging.error("Stage 1 chapter selection failed, returning empty list")
            return []
        
        logging.info(f"Stage 1 selected {len(stage1_chapters)} chapters: {[ch[0] for ch in stage1_chapters]}")
        
        # Stage 2: Re-evaluate with section + chapter notes for the 6 selected chapters
        logging.info("=== STAGE 2: Refined selection with section + chapter notes ===")
        selected_chapter_nums = [int(ch[0]) for ch in stage1_chapters]
        relevant_notes = load_relevant_notes_for_chapters(selected_chapter_nums)
        if LOG_PROMPT_DEBUG:
            logging.info(f"📚 STAGE 2 NOTES INJECTION: Loaded notes for {len(selected_chapter_nums)} chapters ({len(relevant_notes)} chars)")
            if relevant_notes:
                labels = re.findall(r'=== (SECTION \d+|CHAPTER \d+) NOTES ===', relevant_notes)
                logging.info(f"   Contains: {', '.join(labels)}")
            self._log_prompt_to_file("STAGE_2_NOTES", relevant_notes)
        
        stage2_k = 3  # Always select top 3 chapters in stage 2
        final_chapters = self._select_chapters_stage2(
            product_description, stage2_k, stage1_chapters, relevant_notes, diagnosis, state
        )
        
        if final_chapters:
            logging.info(f"Stage 2 final selection: {[ch[0] for ch in final_chapters]}")
            return final_chapters
        
        # Fallback to stage 1 results if stage 2 fails
        logging.warning("Stage 2 failed, using Stage 1 results")
        return stage1_chapters[:stage2_k]

    def _select_chapters_stage1(self, product_description: str, k: int, diagnosis: Optional[str] = None, state: Dict = None) -> List[Tuple[str, float, str]]:
        """
        Stage 1: Initial chapter selection WITHOUT notes.
        Fast, broad selection based on product description and chapter titles only.
        """
        # Create unified state for chapters selection
        prompt_json = build_state(
            "select_chapters",
            product_text=product_description
        )
        
        # Add chapters data to the state for this specific task
        prompt_json["data"]["chapters"] = self.navigator.chapters_map
        prompt_json["data"]["count"] = k
        if diagnosis:
            prompt_json["data"]["diagnosis"] = diagnosis

        # Build system prompt WITHOUT notes (empty string)
        system_instruction = build_chapter_selection_prompt_stage1("")
        if LOG_PROMPT_DEBUG:
            logging.info(f"📝 STAGE 1 SYSTEM PROMPT: Total length = {len(system_instruction)} chars (no notes)")
            self._log_prompt_to_file("STAGE_1_FULL_PROMPT", system_instruction)

        if self.llm.log_prompts:
            self.llm.prompt_logger.info(f"==== JSON CHAPTERS PROMPT ====\n{json.dumps(prompt_json, indent=2)}\n==== END PROMPT ====")
        
        logging.info(f"Requesting top {k} chapters using JSON prompt")
        
        def _extract_chapter_entries(obj: Any, depth: int = 0) -> Optional[List[Dict[str, Any]]]:
            """Recursively locate a list of chapter dictionaries in arbitrary JSON payloads."""
            if obj is None or depth > 5:
                return None
            if isinstance(obj, list):
                if obj and all(isinstance(item, dict) for item in obj):
                    if any(
                        ("chapter" in item)
                        or ("code" in item)
                        or ("chapter_code" in item)
                        for item in obj
                    ):
                        return obj
                for item in obj:
                    nested = _extract_chapter_entries(item, depth + 1)
                    if nested:
                        return nested
                return None
            if isinstance(obj, dict):
                values = list(obj.values())
                if values and all(isinstance(value, dict) for value in values):
                    if any(
                        ("chapter" in value)
                        or ("code" in value)
                        or ("chapter_code" in value)
                        for value in values
                    ):
                        return values
                for value in values:
                    nested = _extract_chapter_entries(value, depth + 1)
                    if nested:
                        return nested
                return None
            return None

        def _get_first_value(data: Dict[str, Any], keys: List[str]) -> Optional[Any]:
            for key in keys:
                if key in data and data[key] is not None:
                    return data[key]
            return None
        
        # Try multiple times with different approaches
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Build the user message for trajectory capture
                user_message = f"TASK: select_chapters_stage1\n\nJSON INPUT:\n{json.dumps(prompt_json, indent=2)}"
                full_prompt = f"{system_instruction}\n\n{user_message}"
                
                # Use OpenAI o3 if training data collection is enabled, otherwise use Vertex AI
                # NOTE: Don't pass task_type/prompt_json when TRAJECTORY_MODE is on - 
                # these turns will be captured in the trajectory instead of logged as single-turn
                if TRAJECTORY_MODE:
                    chapters_response = self.llm.send_openai_request(
                        prompt=full_prompt,
                        temperature=LLM_TEMPERATURE,
                        requires_json=True
                    )
                else:
                    chapters_response = self.llm.send_openai_request(
                        prompt=full_prompt,
                        temperature=LLM_TEMPERATURE,
                        requires_json=True,
                        task_type="select_chapters",
                        prompt_json=prompt_json
                    )
                
                # Capture trajectory if enabled
                if state is not None and TRAJECTORY_MODE and "_chapter_selection_trajectory" in state:
                    state["_chapter_selection_trajectory"].append({
                        "role": "user",
                        "content": user_message
                    })
                    state["_chapter_selection_trajectory"].append({
                        "role": "assistant", 
                        "content": chapters_response
                    })
                

                if self.llm.log_prompts:
                    self.llm.prompt_logger.info(f"==== TOP CHAPTERS RESPONSE (Attempt {attempt + 1}) ====\n{chapters_response}\n==== END RESPONSE ====")

                # Clean the response - remove any text before/after JSON
                chapters_response = chapters_response.strip()
                
                # Find JSON array in response
                json_start = chapters_response.find('[')
                json_end = chapters_response.rfind(']') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_content = chapters_response[json_start:json_end]
                else:
                    # Fallback: try to parse entire response
                    json_content = chapters_response
                
                # Parse JSON - handle multiple response formats flexibly
                try:
                    chapters_data = json.loads(json_content)
                except json.JSONDecodeError as parse_error:
                    # ENHANCED FALLBACK: Try to extract the failed_generation from Groq error
                    logging.info("JSON parsing failed, checking for Groq failed_generation pattern")
                    
                    if 'failed_generation' in chapters_response or 'json_validate_failed' in chapters_response:
                        logging.info("Detected Groq JSON validation failure - extracting content")
                        try:
                            # Try multiple patterns to extract the failed JSON
                            failed_json = None
                            
                            # Pattern 1: Extract from failed_generation field
                            failed_match = re.search(r"'failed_generation': '([^']+)'", chapters_response)
                            if failed_match:
                                failed_json = failed_match.group(1).replace('\\n', '\n')
                                logging.info("Extracted JSON from failed_generation field")
                            
                            # Pattern 2: Look for the actual JSON objects in the error message
                            if not failed_json:
                                # Look for JSON-like content in the error
                                json_pattern = r'\{\s*"chapter"[^}]+\}(?:\s*,\s*\{\s*"chapter"[^}]+\})*'
                                json_match = re.search(json_pattern, chapters_response, re.DOTALL)
                                if json_match:
                                    failed_json = json_match.group(0)
                                    logging.info("Extracted JSON from error message pattern")
                            
                            if failed_json:
                                # Clean up the extracted JSON
                                failed_json = failed_json.strip()
                                
                                # Wrap in array brackets if not already wrapped
                                if not failed_json.startswith('['):
                                    failed_json = f"[{failed_json}]"
                                
                                # Now wrap in the expected object structure
                                wrapped_json = f'{{"chapters": {failed_json}}}'
                                
                                # Try to parse the wrapped JSON
                                chapters_data = json.loads(wrapped_json)
                                logging.info("Successfully recovered from Groq JSON validation failure")
                            else:
                                raise parse_error
                                
                        except Exception as fallback_error:
                            logging.error(f"Fallback extraction failed: {fallback_error}")
                            raise parse_error
                    else:
                        raise parse_error
                
                # Handle different response formats
                chapters_list = None
                
                if isinstance(chapters_data, list):
                    # Direct array format: [{"chapter": "03", ...}, ...]
                    chapters_list = chapters_data
                    logging.info(f"Parsed direct array format with {len(chapters_list)} items")
                    
                elif isinstance(chapters_data, dict):
                    # Object format: {"chapters": [...]} or other nested formats
                    logging.info(f"Received object format, extracting array...")
                    
                    # Try common key names (chapters is now the primary expected key)
                    possible_keys = [
                        "chapters",
                        f"top{k}Chapters",
                        "top3Chapters",
                        "results",
                        "data",
                        "chapter_selection",
                        "chapterSelection",
                        "chapter_candidates",
                        "chapterCandidates",
                        "options",
                        "items",
                        "choices",
                    ]
                    for key in possible_keys:
                        if key not in chapters_data:
                            continue
                        value = chapters_data[key]
                        if isinstance(value, list):
                            chapters_list = value
                            logging.info(f"Extracted chapters from key '{key}'")
                            break
                        if isinstance(value, dict):
                            extracted = _extract_chapter_entries(value)
                            if extracted:
                                chapters_list = extracted
                                logging.info(f"Extracted chapters from nested dict key '{key}'")
                                break
                    
                    if chapters_list is None:
                        extracted = _extract_chapter_entries(chapters_data)
                        if extracted:
                            chapters_list = extracted
                            logging.info("Extracted chapters from nested dict structure")
                
                if chapters_list is None:
                    logging.error(f"Could not extract chapters list from response format: {type(chapters_data)}")
                    if attempt < max_attempts - 1:
                        logging.info(f"Retrying attempt {attempt + 2}...")
                        continue
                    else:
                        return []
                
                # Extract and validate results
                results = []
                for i, item in enumerate(chapters_list):
                    if not isinstance(item, dict):
                        logging.warning(f"Item {i} is not a dict (type={type(item)}): {item}")
                        continue
                        
                    chapter = _get_first_value(
                        item,
                        [
                            "chapter",
                            "chapter_code",
                            "chapterCode",
                            "chapter_number",
                            "chapterNumber",
                            "code",
                        ],
                    ) or ""
                    path_conf_raw = _get_first_value(
                        item,
                        [
                            "path_score",
                            "pathScore",
                            "path_confidence",
                            "pathConfidence",
                            "chapter_confidence",
                            "chapterConfidence",
                            "selection_confidence",
                            "score",
                            "confidence",
                        ],
                    )
                    info_conf_raw = _get_first_value(
                        item,
                        [
                            "information_context_score",
                            "informationContextScore",
                            "information_context_confidence",
                            "informationContextConfidence",
                            "information_confidence",
                            "informationConfidence",
                            "info_confidence",
                            "context_confidence",
                            "support_confidence",
                            "confidence",
                        ],
                    )
                    reasoning = _get_first_value(
                        item,
                        ["reasoning", "explanation", "justification", "analysis", "notes"],
                    ) or ""
                    
                    if path_conf_raw is None and info_conf_raw is not None:
                        path_conf_raw = info_conf_raw
                    if info_conf_raw is None and path_conf_raw is not None:
                        info_conf_raw = path_conf_raw
                    
                    if path_conf_raw is None or info_conf_raw is None:
                        logging.warning(
                            "Missing confidence fields in chapter response: %s", item
                        )
                        continue
                    
                    try:
                        path_confidence = float(path_conf_raw)
                        info_confidence = float(info_conf_raw)
                    except (ValueError, TypeError):
                        logging.warning(
                            "Invalid confidence values (info=%s, path=%s) in item: %s",
                            info_conf_raw,
                            path_conf_raw,
                            item,
                        )
                        continue
                    
                    # Validate chapter format (must be 2-digit string)
                    if not isinstance(chapter, str):
                        chapter = str(chapter)
                    
                    # Handle both "03" and "3" formats
                    if chapter.isdigit():
                        chapter = f"{int(chapter):02d}"
                    
                    # Final validation
                    if (
                        re.match(r'^\d{2}$', chapter)
                        and 0.0 <= path_confidence <= 1.0
                        and 0.0 <= info_confidence <= 1.0
                    ):
                        results.append((chapter, path_confidence, reasoning))
                        logging.debug(
                            f"Valid chapter: {chapter} "
                            f"(path_confidence: {path_confidence:.3f}, "
                            f"info_confidence: {info_confidence:.3f})"
                        )
                    else:
                        logging.warning(
                            "Invalid chapter data - chapter: '%s', "
                            "info_confidence: %.3f, path_confidence: %.3f",
                            chapter,
                            info_confidence,
                            path_confidence,
                        )
                
                # Check if we got the expected number of results
                if len(results) >= k:
                    # Take exactly k results
                    final_results = results[:k]
                    logging.info(f"Successfully extracted {len(final_results)} chapters: {[(r[0], r[1]) for r in final_results]}")
                    
                    # Validate confidence sum (should be approximately 1.0)
                    total_confidence = sum(r[1] for r in final_results)
                    if total_confidence < 0.5 or total_confidence > 1.5:
                        logging.debug(f"Total confidence {total_confidence:.3f} outside expected range")
                    
                    return final_results
                    
                elif len(results) > 0:
                    # We got some results but not enough
                    logging.warning(f"Expected {k} chapters, got {len(results)}. Using what we have.")
                    return results
                else:
                    # No valid results
                    logging.error(f"No valid chapters extracted from response")
                    if attempt < max_attempts - 1:
                        logging.info(f"Retrying attempt {attempt + 2}...")
                        continue
                    else:
                        return []
                        
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                logging.error(f"Response content: {chapters_response}")
                if attempt < max_attempts - 1:
                    logging.info(f"Retrying attempt {attempt + 2}...")
                    continue
                else:
                    return []
                    
            except Exception as e:
                logging.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    logging.info(f"Retrying attempt {attempt + 2}...")
                    continue
                else:
                    return []
        
        # If all attempts failed
        logging.error(f"All {max_attempts} attempts failed to determine top chapters")
        return []

    def _select_chapters_stage2(
        self, 
        product_description: str, 
        k: int, 
        stage1_chapters: List[Tuple[str, float, str]], 
        relevant_notes: str,
        diagnosis: Optional[str] = None, 
        state: Dict = None
    ) -> List[Tuple[str, float, str]]:
        """
        Stage 2: Re-evaluate chapters with specific section + chapter notes.
        Only evaluates the chapters selected in Stage 1.
        """
        if not stage1_chapters:
            return []
        
        # Create unified state for chapters selection
        prompt_json = build_state(
            "select_chapters",
            product_text=product_description
        )
        
        # Only include the chapters from Stage 1 (not all chapters)
        stage1_chapter_nums = [int(ch[0]) for ch in stage1_chapters]
        filtered_chapters = {
            ch_num: desc 
            for ch_num, desc in self.navigator.chapters_map.items() 
            if ch_num in stage1_chapter_nums
        }
        
        prompt_json["data"]["chapters"] = filtered_chapters
        prompt_json["data"]["count"] = k
        prompt_json["data"]["stage"] = 2
        prompt_json["data"]["stage1_selection"] = [
            {"chapter": ch, "initial_score": score, "reasoning": reason}
            for ch, score, reason in stage1_chapters
        ]
        if diagnosis:
            prompt_json["data"]["diagnosis"] = diagnosis

        # Build system prompt with relevant notes for selected chapters
        system_instruction = build_chapter_selection_prompt_stage2(relevant_notes)
        if LOG_PROMPT_DEBUG:
            logging.info(f"📝 STAGE 2 SYSTEM PROMPT: Total length = {len(system_instruction)} chars")
            self._log_prompt_to_file("STAGE_2_FULL_PROMPT", system_instruction)

        if self.llm.log_prompts:
            self.llm.prompt_logger.info(f"==== STAGE 2 CHAPTERS PROMPT ====\n{json.dumps(prompt_json, indent=2)}\n==== END PROMPT ====")
        
        logging.info(f"Stage 2: Re-evaluating {len(stage1_chapters)} chapters with full notes context")
        
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                # Build the user message for trajectory capture - include section + chapter notes for context
                if relevant_notes:
                    user_message = f"TASK: select_chapters_stage2\n\nRELEVANT NOTES (SECTION + CHAPTER):\n{relevant_notes}\n\nJSON INPUT:\n{json.dumps(prompt_json, indent=2)}"
                else:
                    user_message = f"TASK: select_chapters_stage2\n\nJSON INPUT:\n{json.dumps(prompt_json, indent=2)}"
                full_prompt = f"{system_instruction}\n\n{user_message}"
                
                # NOTE: Don't pass task_type/prompt_json when TRAJECTORY_MODE is on -
                # these turns will be captured in the trajectory instead of logged as single-turn
                if TRAJECTORY_MODE:
                    chapters_response = self.llm.send_openai_request(
                        prompt=full_prompt,
                        temperature=LLM_TEMPERATURE,
                        requires_json=True
                    )
                else:
                    chapters_response = self.llm.send_openai_request(
                        prompt=full_prompt,
                        temperature=LLM_TEMPERATURE,
                        requires_json=True,
                        task_type="select_chapters_stage2",
                        prompt_json=prompt_json
                    )
                
                # Capture trajectory if enabled
                if state is not None and TRAJECTORY_MODE and "_chapter_selection_trajectory" in state:
                    state["_chapter_selection_trajectory"].append({
                        "role": "user",
                        "content": user_message
                    })
                    state["_chapter_selection_trajectory"].append({
                        "role": "assistant", 
                        "content": chapters_response
                    })
                
                if self.llm.log_prompts:
                    self.llm.prompt_logger.info(f"==== STAGE 2 RESPONSE (Attempt {attempt + 1}) ====\n{chapters_response}\n==== END RESPONSE ====")
                
                # Parse response - same logic as stage 1
                chapters_response = chapters_response.strip()
                
                # Find JSON in response
                json_start = chapters_response.find('[')
                json_end = chapters_response.rfind(']') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_content = chapters_response[json_start:json_end]
                else:
                    # Try to find object format
                    json_start = chapters_response.find('{')
                    json_end = chapters_response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_content = chapters_response[json_start:json_end]
                    else:
                        json_content = chapters_response
                
                chapters_data = json.loads(json_content)
                
                # Handle different response formats
                chapters_list = None
                if isinstance(chapters_data, list):
                    chapters_list = chapters_data
                elif isinstance(chapters_data, dict):
                    for key in ["chapters", "results", "data", "selections"]:
                        if key in chapters_data and isinstance(chapters_data[key], list):
                            chapters_list = chapters_data[key]
                            break
                
                if chapters_list is None:
                    logging.error("Could not extract chapters list from Stage 2 response")
                    continue
                
                # Extract results
                results = []
                for item in chapters_list:
                    if not isinstance(item, dict):
                        continue
                    
                    chapter = str(item.get("chapter", ""))
                    path_conf = item.get("path_confidence") or item.get("path_score") or item.get("confidence", 0)
                    reasoning = item.get("reasoning", "")
                    
                    try:
                        path_confidence = float(path_conf)
                    except (ValueError, TypeError):
                        continue
                    
                    if chapter.isdigit():
                        chapter = f"{int(chapter):02d}"
                    
                    if re.match(r'^\d{2}$', chapter) and 0.0 <= path_confidence <= 1.0:
                        results.append((chapter, path_confidence, reasoning))
                
                if len(results) >= k:
                    # Sort by confidence and take top k
                    results.sort(key=lambda x: x[1], reverse=True)
                    return results[:k]
                elif results:
                    logging.warning(f"Stage 2: Expected {k} chapters, got {len(results)}")
                    results.sort(key=lambda x: x[1], reverse=True)
                    return results
                    
            except json.JSONDecodeError as e:
                logging.error(f"Stage 2 JSON error on attempt {attempt + 1}: {e}")
            except Exception as e:
                logging.error(f"Stage 2 error on attempt {attempt + 1}: {e}")
        
        # Fallback: return stage 1 results
        logging.warning("Stage 2 failed, returning Stage 1 results")
        return stage1_chapters[:k]

    def initialize_classification_paths(self, product_description: str, k: int = CHAPTER_BEAM_SIZE, diagnosis: Optional[str] = None, state: Dict = None) -> List[ClassificationPath]:
        """
        Initialize multiple classification paths based on top K chapters.
        
        When TRAJECTORY_MODE is enabled, each path is initialized with its own
        chat trajectory containing the unified system prompt and initial context.
        
        Args:
            product_description: The product being classified
            k: Number of paths to initialize
            diagnosis: Optional diagnosis from previous attempts
            state: The current state dictionary
            
        Returns:
            List of ClassificationPath objects
        """
        # Get top chapters
        top_chapters = self.determine_top_chapters(product_description, k, diagnosis, state)
        
        if not top_chapters:
            logging.warning("No valid chapters returned")
            return []
        
        # Create classification paths
        paths = []
        for i, (chapter, confidence, reasoning) in enumerate(top_chapters, 1):
            chapter_desc = self.navigator.chapters_map.get(int(chapter), "Unknown chapter")
            path = ClassificationPath(f"path_{i}", chapter, confidence, chapter_desc)
            path.reasoning_log.append(f"Initial chapter selection: {chapter} - {chapter_desc} (conf: {confidence:.3f}) - {reasoning}")
            
            # Initialize trajectory if trajectory mode is enabled
            # The trajectory will contain:
            # 1. The system prompt
            # 2. Chapter selection LLM calls (shared across all paths)
            # 3. Actual per-path LLM request/response pairs
            if TRAJECTORY_MODE:
                path.initialize_trajectory(UNIFIED_SYSTEM_PROMPT)
                
                # Prepend chapter selection calls to trajectory (these happened before paths were created)
                if state is not None and "_chapter_selection_trajectory" in state:
                    chapter_selection_turns = state["_chapter_selection_trajectory"]
                    # Insert chapter selection turns after the system prompt
                    path.trajectory = path.trajectory[:1] + chapter_selection_turns + path.trajectory[1:]
                    logging.info(f"Initialized trajectory for path {i} with system prompt + {len(chapter_selection_turns)} chapter selection messages")
                else:
                    logging.info(f"Initialized trajectory for path {i} with system prompt")
            
            paths.append(path)
            logging.info(f"Initialized path {i}: Chapter {chapter} with confidence {confidence:.3f}")
        
        # Clean up temporary chapter selection trajectory from state (already copied to each path)
        if state is not None and "_chapter_selection_trajectory" in state:
            del state["_chapter_selection_trajectory"]
        
        return paths

    def generate_all_next_candidates(self, paths: List[ClassificationPath], product_description: str, state: Dict = None) -> List[Tuple[ClassificationPath, HTSNode, float, float, str]]:
        """
        Generate all possible next steps from all active paths.
        Now parallelized to process multiple paths concurrently.
        
        Args:
            paths: List of active classification paths
            product_description: The product being classified
            state: The current state dictionary
            
        Returns:
            List of tuples: (parent_path, candidate_node, decision_confidence, path_confidence, reasoning)
        """
        all_candidates = []
        active_paths = [p for p in paths if p.is_active and not p.is_complete]
        
        if not active_paths:
            return all_candidates
        
        # Prepare function for parallel execution - capture state in closure
        def process_path(path):
            path_candidates = []
            
            # Determine current node
            current_node = path.current_node
            if current_node is None:
                # Need to select heading from chapter
                chapter_parent = self.navigator.create_chapter_parent(path.chapter)
                current_node = chapter_parent
            
            # Get children
            children = self.navigator.get_children(current_node)
            if not children:
                # No children, this path is complete
                path.mark_complete()
                return []
            
            # Check for cached candidates from question-checking phase
            # This avoids duplicate LLM calls when the same path/node was already scored
            cached = state.get("_cached_scored_candidates") if state else None
            current_node_id = current_node.node_id if hasattr(current_node, 'node_id') else id(current_node)
            
            if cached and cached.get("path_id") == path.path_id and cached.get("node_id") == current_node_id:
                # Reuse cached candidates - prevents duplicate LLM calls
                logging.info(f"Using cached scores for path {path.path_id} at node {current_node_id}")
                candidates = cached["candidates"]
                # Clear the cache after use to prevent stale data
                del state["_cached_scored_candidates"]
            else:
                # Score each child as a potential next step - use captured state from outer scope
                candidates = self.score_candidates(product_description, current_node, children, path, state)
            
            # Add each candidate with its parent path
            for child_node, decision_conf, path_conf, reasoning in candidates:
                path_candidates.append((path, child_node, decision_conf, path_conf, reasoning))
                
            return path_candidates
        
        # Process paths - disable parallelization if training data collection is enabled
        # to preserve thread-local training sessions
        if training_collector.enabled:
            # Sequential processing to preserve training session
            for path in active_paths:
                try:
                    path_results = process_path(path)
                    all_candidates.extend(path_results)
                except RetryableLLMError:
                    logging.error("Retryable LLM error while processing path; propagating for higher-level retry.")
                    raise
                except Exception as e:
                    logging.error(f"Error processing path: {e}")
        else:
            # Process paths in parallel
            with ThreadPoolExecutor(max_workers=self.path_workers) as executor:
                futures = [executor.submit(process_path, path) for path in active_paths]
                
                for future in as_completed(futures):
                    try:
                        path_results = future.result()
                        all_candidates.extend(path_results)
                    except RetryableLLMError:
                        logging.error("Retryable LLM error in parallel path processing; propagating for retry.")
                        raise
                    except Exception as e:
                        logging.error(f"Error processing path in parallel: {e}")
        
        return all_candidates

    def score_candidates(
        self,
        product_description: str,
        parent_node: HTSNode,
        candidates: List[HTSNode],
        path: ClassificationPath,
        state: Dict = None
    ) -> List[Tuple[HTSNode, float, float, str]]:
        """
        Score candidate nodes using unified Policy-Driven ranking.
        
        The model selects AND scores in a single atomic decision:
        - primary_selection: Most likely correct path
        - alternative_1: Viable alternative
        - alternative_2: Fallback option
        
        The beam uses path_score directly from the model's selections.
        This is the Policy driving the beam, not an external scorer.
        """
        if not candidates:
            return []
        
        # Use unified ranking for all cases
        return self._rank_candidates(
            product_description,
            parent_node,
            candidates,
            path,
            top_k=min(3, len(candidates)),
            state=state
        )

    def _rank_candidates(
        self,
        product_description: str,
        parent_node: HTSNode,
        candidates: List[HTSNode],
        path: ClassificationPath,
        top_k: int = 3,
        state: Dict = None
    ) -> List[Tuple[HTSNode, float, float, str]]:
        """
        UNIFIED: Select AND score candidates in a single LLM call.
        
        The model outputs:
        - primary_selection: Most likely correct path (highest path_score)
        - alternative_1: Viable alternative
        - alternative_2: Fallback option
        
        Each includes both information_context_score and path_score.
        The beam uses path_score directly from the model's selections.
        
        This is Policy-Driven beam search: the model's selection IS the beam direction.
        """
        current_path = TreeUtils.get_classification_path(parent_node)
        
        # Extract current chapter - only load notes if NOT in trajectory mode
        current_chapter = extract_chapter_from_node(parent_node)
        chapter_notes = ""
        if not (TRAJECTORY_MODE and path._trajectory_initialized):
            if current_chapter:
                chapter_notes = load_chapter_context(current_chapter)
                if LOG_PROMPT_DEBUG:
                    logging.info(f"📚 CANDIDATE RANKING NOTES: Chapter {current_chapter} notes loaded ({len(chapter_notes)} chars)")
        
        # Format candidates for JSON
        candidates_list = []
        for i, node in enumerate(candidates, 1):
            candidate_entry = {
                "index": i,
                "code": node.htsno or "[GROUP]",
                "description": node.description,
                "is_group": not bool(node.htsno),
                "node_id": node.node_id,
            }
            candidates_list.append(candidate_entry)
        
        # Create classification tree structure
        classification_tree = {"children": candidates_list}
        
        # Create unified state for candidate ranking
        # In trajectory mode, skip decision_history - it's already in the chat context
        if TRAJECTORY_MODE and path._trajectory_initialized:
            prompt_json = build_state(
                "rank_candidates",
                product_text=product_description,
                path_so_far=current_path,
                classification_tree=classification_tree,
                select_count=top_k
            )
        else:
            prompt_json = build_state(
                "rank_candidates",
                product_text=product_description,
                path_so_far=current_path,
                classification_tree=classification_tree,
                decision_history=self._build_decision_history(path),
                select_count=top_k
            )
        
        # Only add chapter notes in stateless mode
        if chapter_notes and not (TRAJECTORY_MODE and path._trajectory_initialized):
            prompt_json["data"]["chapter_notes"] = chapter_notes

        # Build the user message content
        user_message_content = f"TASK: rank_candidates\n\nJSON INPUT:\n{json.dumps(prompt_json, indent=2)}"

        if self.llm.log_prompts:
            self.llm.prompt_logger.info(f"==== JSON RANKING PROMPT ====\n{json.dumps(prompt_json, indent=2)}\n==== END PROMPT ====")
        
        scored_candidates: List[Tuple[HTSNode, float, float, str]] = []
        
        for attempt in range(3):
            try:
                # Use trajectory mode if enabled and path has trajectory
                if TRAJECTORY_MODE and path._trajectory_initialized:
                    messages = path.get_trajectory_for_request(user_message_content)
                    
                    response = self.llm.send_trajectory_request(
                        messages=messages,
                        temperature=LLM_TEMPERATURE,
                        requires_json=True,
                        task_type="rank_candidates",
                        prompt_json=prompt_json
                    )
                else:
                    # Fallback to stateless mode
                    system_instruction = build_candidate_ranking_prompt(chapter_notes)
                    if LOG_PROMPT_DEBUG:
                        logging.info(f"📝 CANDIDATE RANKING PROMPT: Total length = {len(system_instruction)} chars")
                        self._log_prompt_to_file(f"CANDIDATE_RANKING_CH{current_chapter}_PROMPT", system_instruction)
                    
                    full_prompt = f"{system_instruction}\n\n{user_message_content}"
                    
                    response = self.llm.send_openai_request(
                        prompt=full_prompt,
                        temperature=LLM_TEMPERATURE,
                        requires_json=True,
                        task_type="rank_candidates",
                        prompt_json=prompt_json
                    )
                
                if self.llm.log_prompts:
                    self.llm.prompt_logger.info(f"==== RANKING RESPONSE (Attempt {attempt+1}) ====\n{response}\n==== END RESPONSE ====")
                
                # Parse JSON response
                rank_data = _unwrap_json_response(json.loads(response))
                
                # Extract the three selections
                selection_keys = ["primary_selection", "alternative_1", "alternative_2"]
                seen_indices = set()
                
                for sel_key in selection_keys:
                    selection = rank_data.get(sel_key)
                    if not selection or not isinstance(selection, dict):
                        # Only warn if we don't have enough selections yet AND there are candidates remaining
                        if len(scored_candidates) < len(candidates):
                            logging.debug(f"Missing or invalid {sel_key} in ranking response (may be expected if fewer than 3 candidates)")
                        continue
                    
                    try:
                        option_index = int(selection.get("option_index", 0))
                        if option_index < 1 or option_index > len(candidates):
                            logging.warning(f"Invalid option_index {option_index} in {sel_key}")
                            continue
                        
                        # Skip duplicates
                        if option_index in seen_indices:
                            logging.debug(f"Duplicate option_index {option_index} in {sel_key}, skipping")
                            continue
                        seen_indices.add(option_index)
                        
                        info_score = float(selection.get("information_context_score", 0.5))
                        path_score = float(selection.get("path_score", 0.5))
                        reasoning = selection.get("reasoning", f"Selected as {sel_key}")
                        
                        # Validate score ranges
                        info_score = max(0.0, min(1.0, info_score))
                        path_score = max(0.0, min(1.0, path_score))
                        
                        node = candidates[option_index - 1]
                        scored_candidates.append((node, info_score, path_score, reasoning))
                        
                    except (ValueError, TypeError) as e:
                        # Expected when model returns null/missing fields for selections beyond available candidates
                        logging.debug(f"Skipping {sel_key} - incomplete data: {e}")
                        continue
                
                # Check if we got enough valid selections
                if len(scored_candidates) >= min(top_k, len(candidates)):
                    # Append to trajectory after successful parse
                    if TRAJECTORY_MODE and path._trajectory_initialized:
                        path.append_to_trajectory(user_message_content, response)
                    break
                else:
                    logging.warning(f"Only got {len(scored_candidates)} valid selections, need {min(top_k, len(candidates))}. Retrying...")
                    scored_candidates.clear()
                    
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error in ranking stage on attempt {attempt+1}: {e}")
            except RetryableLLMError:
                raise
            except Exception as e:
                logging.error(f"Error in ranking stage on attempt {attempt+1}: {e}")
            
            if attempt < 2:
                time.sleep(1)
        
        # Fallback: if we still don't have enough, fill with sequential candidates
        if len(scored_candidates) < min(top_k, len(candidates)):
            logging.warning(f"Ranking failed after retries. Using fallback sequential selection.")
            seen_node_ids = {node.node_id for node, _, _, _ in scored_candidates}
            for i, node in enumerate(candidates):
                if len(scored_candidates) >= top_k:
                    break
                if node.node_id not in seen_node_ids:
                    # Assign decreasing fallback scores
                    fallback_score = 0.3 - (i * 0.05)
                    scored_candidates.append((node, 0.5, max(0.1, fallback_score), f"Fallback selection {i+1}"))
                    seen_node_ids.add(node.node_id)
        
        # Sort by path_score descending (primary should be first)
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        logging.info("--- Ranking Summary ---")
        for i, (node, info_conf, path_conf, reasoning) in enumerate(scored_candidates):
            rank_label = ["PRIMARY", "ALT_1", "ALT_2"][i] if i < 3 else f"EXTRA_{i}"
            logging.info(
                f"  [{rank_label}] {node.htsno or '[GROUP]'}: "
                f"info_score={info_conf:.3f}, path_score={path_conf:.3f}"
            )
        logging.info("--- End Ranking Summary ---")
        
        return scored_candidates[:top_k]
    
    def advance_beam(self, current_beam: List[ClassificationPath], product_description: str, k: int, state: Dict = None) -> List[ClassificationPath]:
        """
        Corrected version: Advances the beam by finding the top K overall next steps.
        This version is simpler, safer, and adheres to beam search principles.
        
        TreeRL Enhancement: Also captures pruned hypotheses for reward computation.
        Pruned paths are stored in state["_treerl_pruned_leaves"] for later processing.
        """
        all_new_hypotheses = []

        # 1. Generate all possible next steps from all active paths in the current beam
        all_candidates_info = self.generate_all_next_candidates(current_beam, product_description, state)

        # 2. Create the full set of new potential hypotheses by combining parent paths with their scored children
        for parent_path, candidate_node, decision_confidence, path_confidence, reasoning in all_candidates_info:
            # ALWAYS CLONE to prevent state corruption. This is a critical fix.
            new_path = parent_path.clone()
            
            # Determine stage and get options for logging
            stage = TreeUtils.determine_next_stage(parent_path.current_node) if parent_path.current_node else "heading"
            current_node_for_options = parent_path.current_node or self.navigator.create_chapter_parent(parent_path.chapter)
            children = self.navigator.get_children(current_node_for_options)
            options_metadata = TreeUtils.create_options_metadata(children)

            # Add the new step (this updates the log_score and cumulative_confidence)
            new_path.add_step(stage, candidate_node, decision_confidence, path_confidence, reasoning, options_metadata)
            
            # Check if the new path has reached a terminal node
            is_terminal = not self.navigator.get_children(candidate_node)
            if is_terminal:
                new_path.mark_complete()
                
                # Record final classification event in trajectory for RL training
                if TRAJECTORY_MODE and new_path._trajectory_initialized:
                    new_path.append_event_to_trajectory("classification_complete", {
                        "final_code": candidate_node.htsno or f"[GROUP:{candidate_node.node_id}]",
                        "final_description": candidate_node.description,
                        "cumulative_confidence": round(new_path.cumulative_confidence, 4),
                        "path": new_path.get_full_path_string()
                    })
            
            all_new_hypotheses.append(new_path)
        
        # Also, carry over any already completed paths from the previous beam so they can continue to compete.
        for path in current_beam:
            if path.is_complete:
                all_new_hypotheses.append(path)

        # 3. PRUNING STEP: Sort the ENTIRE pool of generated hypotheses and simply take the top K.
        # This is the global leaderboard. We sort by log_score.
        all_new_hypotheses.sort(key=lambda p: p.log_score, reverse=True)
        
        new_beam = all_new_hypotheses[:k]
        pruned_hypotheses = all_new_hypotheses[k:]
        
        # --- TreeRL: Capture pruned hypotheses for reward computation ---
        # These are partial paths that were dropped due to beam width limits
        # They still get rewards based on how far they matched the gold trace
        if state is not None and pruned_hypotheses:
            if "_treerl_pruned_leaves" not in state:
                state["_treerl_pruned_leaves"] = []
                logging.info(f"TreeRL: Initialized _treerl_pruned_leaves in state")
            
            for pruned_path in pruned_hypotheses:
                # Extract the trace info needed for reward computation
                pruned_info = {
                    "path_id": pruned_path.path_id,
                    "classification_path": [step.copy() for step in pruned_path.classification_path],
                    "trajectory": [msg.copy() for msg in pruned_path.trajectory] if pruned_path.trajectory else [],
                    "log_score": pruned_path.log_score,
                    "cumulative_confidence": pruned_path.cumulative_confidence,
                    "is_complete": pruned_path.is_complete,
                    "pruned_at_iteration": state.get("iteration_count", 0),
                    "chapter": pruned_path.chapter
                }
                state["_treerl_pruned_leaves"].append(pruned_info)
            
            logging.info(f"TreeRL: Captured {len(pruned_hypotheses)} pruned paths (total accumulated: {len(state['_treerl_pruned_leaves'])})")
        elif state is None:
            logging.warning(f"TreeRL: state is None, cannot capture {len(pruned_hypotheses)} pruned paths!")
        
        # --- ENHANCED LOGGING ---
        logging.info(f"--- BEAM ADVANCEMENT COMPLETE ---")
        logging.info(f"Generated {len(all_new_hypotheses)} total candidate paths.")
        logging.info(f"Selected top {len(new_beam)} to form the new beam (pruned {len(pruned_hypotheses)}):")
        for i, path in enumerate(new_beam):
            status = "COMPLETE" if path.is_complete else "ACTIVE"
            last_dec_conf = getattr(path, 'last_decision_confidence', 'N/A')
            last_dec_conf_str = f"{last_dec_conf:.3f}" if isinstance(last_dec_conf, float) else str(last_dec_conf)
            logging.info(
                f"  [Beam Pos {i+1}] Path: {path.get_full_path_string()} "
                f"| Log Score: {path.log_score:.4f} | "
                f"Cumulative Path Conf: {path.cumulative_confidence:.3f} | "
                f"Last Decision Conf: {last_dec_conf_str} | "
                f"Status: {status}"
            )
        logging.info(f"---------------------------------")
        # --- END LOGGING ---

        return new_beam

    def check_termination_conditions(self, beam: List[ClassificationPath]) -> Tuple[bool, Optional[ClassificationPath]]:
        """
        Check if termination conditions are met.
        
        Args:
            beam: Current beam of paths
            
        Returns:
            Tuple of (should_terminate, best_path)
        """
        if not beam:
            return True, None
        
        # Check if all paths are complete or pruned
        active_paths = [p for p in beam if p.is_active]
        if not active_paths:
            # Return the best completed path
            completed_paths = [p for p in beam if p.is_complete]
            if completed_paths:
                best_path = max(completed_paths, key=lambda p: p.log_score)
                return True, best_path
            else:
                return True, None
        
        # Check confident completion condition
        if beam[0].is_complete and len(beam) > 1:
            # Check if top path is significantly better than second path
            # A difference of 1.0 in log space represents ~2.7x probability difference
            if beam[0].log_score - beam[1].log_score > 1.0:
                logging.info(f"Confident completion: {beam[0].path_id} with log_score {beam[0].log_score:.3f} is significantly better than second place ({beam[1].log_score:.3f})")
                return True, beam[0]
        
        # Check if all paths in the beam are complete
        if all(p.is_complete for p in beam):
            # Return the path with the highest log_score
            best_path = max(beam, key=lambda p: p.log_score)
            logging.info(f"All paths complete, selecting best path: {best_path.path_id} with log_score {best_path.log_score:.3f}")
            return True, best_path
        
        return False, None

    # ----------------------
    # Question Generation Methods
    # ----------------------

    def generate_clarification_question(self, product_description: str, node: HTSNode, stage: str, state: Dict, path: ClassificationPath = None) -> ClarificationQuestion:
        """
        Generate a user-friendly clarification question to help choose between immediate children.
        Now uses JSON prompts for cleaner training data.
        Supports trajectory mode when path is provided.
        """
        if not node:
            logging.error("Cannot generate question from a null node")
            question = ClarificationQuestion()
            question.question_text = "Could you provide more details about your product?"
            question.metadata = {"stage": stage}
            return question
            
        # Get the immediate options to choose from
        options = self.navigator.get_children(node)
        
        if not options:
            logging.warning(f"No question options found for node {node.node_id}: {node.description}")
            question = ClarificationQuestion()
            question.question_text = "Could you provide more details about your product for classification?"
            question.metadata = {"stage": stage}
            return question
        
        # NOTE: Question generation does NOT inject chapter notes
        # Only chapter selection and candidate selection/scoring get notes injection
        
        # Format options for JSON
        options_list = []
        for option in options:
            options_list.append({
                "code": option.htsno or "[GROUP]",
                "description": option.description,
                "is_group": not bool(option.htsno)
            })
        
        # Create unified state for question generation
        prompt_json = build_state(
            "generate_question",
            product_text=product_description,
            path_so_far=TreeUtils.get_classification_path(node),
            conversation_history=state.get("history", [])
        )
        
        # Add classification_tree for this task
        prompt_json["data"]["classification_tree"] = {"children": options_list}
        prompt_json["data"]["stage"] = stage

        # Build the user message content
        user_message_content = f"TASK: generate_question\n\nJSON INPUT:\n{json.dumps(prompt_json, indent=2)}"

        if self.llm.log_prompts:
            self.llm.prompt_logger.info(f"==== JSON QUESTION GENERATION PROMPT ====\n{json.dumps(prompt_json, indent=2)}\n==== END PROMPT ====")
            
        try:
            logging.info(f"Generating clarification question for stage: {stage}")
            
            # Use trajectory mode if enabled and path has trajectory
            if TRAJECTORY_MODE and path and path._trajectory_initialized:
                messages = path.get_trajectory_for_request(user_message_content)
                
                content = self.llm.send_trajectory_request(
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    requires_json=True,
                    task_type="generate_question",
                    prompt_json=prompt_json
                )
                
                # Append to trajectory after successful response
                path.append_to_trajectory(user_message_content, content)
            else:
                # Fallback to stateless mode
                system_instruction = build_question_generation_prompt("")
                full_prompt = f"{system_instruction}\n\n{user_message_content}"
                
                content = self.llm.send_openai_request(
                    prompt=full_prompt,
                    temperature=LLM_TEMPERATURE,
                    requires_json=True,
                    task_type="generate_question",
                    prompt_json=prompt_json
                )

            if self.llm.log_prompts:
                self.llm.prompt_logger.info(f"==== QUESTION GENERATION RESPONSE ====\n{content}\n==== END RESPONSE ====")
            
            question_data = json.loads(content)
            
            # Handle models that wrap the object in a list
            if isinstance(question_data, list):
                if not question_data:
                    raise ValueError("Question generation response returned an empty list.")
                question_data = question_data[0]
            
            if not isinstance(question_data, dict):
                raise ValueError(f"Question generation response must be a JSON object, got {type(question_data).__name__}")
            
            question = ClarificationQuestion()
            question.question_type = question_data.get("question_type", "text")
            question.question_text = question_data.get("question_text", "")
            question.options = question_data.get("options", [])

            # Validate and retry once if malformed
            needs_retry = (not question.question_text) or (question.question_type == "multiple_choice" and not question.options)
            if needs_retry:
                logging.warning("Malformed question generated (empty text or options). Retrying once...")
                try:
                    time.sleep(1)
                    content_retry = self.llm.send_openai_request(
                        prompt=full_prompt,
                        temperature=LLM_TEMPERATURE,
                        requires_json=True,
                        task_type="generate_question",
                        prompt_json=prompt_json
                    )
                    question_data = json.loads(content_retry)
                    
                    # Handle models that wrap the object in a list
                    if isinstance(question_data, list):
                        if not question_data:
                            raise ValueError("Question generation retry response returned an empty list.")
                        question_data = question_data[0]
                    
                    if not isinstance(question_data, dict):
                        raise ValueError(f"Question generation retry response must be a JSON object, got {type(question_data).__name__}")
                    
                    question.question_type = question_data.get("question_type", question.question_type)
                    question.question_text = question_data.get("question_text", question.question_text)
                    question.options = question_data.get("options", question.options)
                except Exception as e_retry:
                    logging.error(f"Retry failed for question generation: {e_retry}")
            
            # Store options data in metadata for later use
            question.metadata = {
                "stage": stage,
                "options": TreeUtils.create_options_metadata(options),
                "node_id": node.node_id
            }
            
            # Check if this question is similar to one we've already asked
            if self._has_similar_question(question.question_text, state.get("history", [])):
                # Generate a fallback question that's more specific
                fallback_question = f"Can you provide more specific details about the {product_description} that would help distinguish between these options? For example, information about its processing, material, or intended use."
                question.question_text = fallback_question
                question.question_type = "text"
                question.options = []
            
            if not question.question_text:
                question.question_text = f"Can you provide more details about the product?"
                
            # Add to recent questions to avoid repetition (in state)
            recent_questions = state.get("recent_questions", [])
            recent_questions.append(question.question_text)
            if len(recent_questions) > 5:
                recent_questions.pop(0)
            state["recent_questions"] = recent_questions
                
            return question
        except Exception as e:
            logging.error(f"Error generating clarification question: {e}")
            question = ClarificationQuestion()
            question.question_text = "Could you provide more details about your product?"
            question.metadata = {"stage": stage}
            return question

    def process_answer(self, original_query: str, question: ClarificationQuestion, answer: str, options: List[Dict[str, Any]], state: Dict, path: ClassificationPath = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Process the user's answer to update the product description and select the best matching option.
        Now uses JSON prompts for cleaner training data.
        Supports trajectory mode when path is provided.
        Returns a tuple of (updated_description, best_match).
        """
        # Get history from state instead of instance
        answer_text = answer
        
        # Handle multiple-choice answers
        if question.question_type == "multiple_choice" and question.options:
            try:
                if answer.strip().isdigit() and 1 <= int(answer.strip()) <= len(question.options):
                    option_index = int(answer.strip()) - 1
                    answer_text = question.options[option_index]["text"]
            except (ValueError, IndexError):
                pass
        
        # NOTE: Answer processing does NOT inject chapter notes
        # Only chapter selection and candidate selection/scoring get notes injection
        
        # Format options for JSON
        options_list = []
        for i, opt in enumerate(options, 1):
            options_list.append({
                "index": i,
                "code": opt.get("code", "[GROUP]"),
                "description": opt.get("description", ""),
                "is_group": opt.get("is_group", False),
                "node_id": opt.get("node_id")
            })
        
        # Create unified state for answer processing
        prompt_json = build_state(
            "process_answer",
            product_text=original_query,
            question_text=question.question_text,
            answer_text=answer_text
        )
        
        # Add classification_tree for this task
        prompt_json["data"]["classification_tree"] = {"children": options_list}

        # Build the user message content
        user_message_content = f"TASK: process_answer\n\nJSON INPUT:\n{json.dumps(prompt_json, indent=2)}"

        if self.llm.log_prompts:
            self.llm.prompt_logger.info(f"==== JSON ANSWER PROCESSING PROMPT ====\n{json.dumps(prompt_json, indent=2)}\n==== END PROMPT ====")
        try:
            logging.info("Processing user's answer")
            
            # Use trajectory mode if enabled and path has trajectory
            if TRAJECTORY_MODE and path and path._trajectory_initialized:
                # Send process_answer request - the prompt already contains question and answer context
                messages = path.get_trajectory_for_request(user_message_content)
                
                content = self.llm.send_trajectory_request(
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    requires_json=True,
                    task_type="process_answer",
                    prompt_json=prompt_json
                )
                
                # Append to trajectory after successful response
                path.append_to_trajectory(user_message_content, content)
            else:
                # Fallback to stateless mode
                system_instruction = build_answer_processing_prompt("")
                full_prompt = f"{system_instruction}\n\n{user_message_content}"
                
                content = self.llm.send_openai_request(
                    prompt=full_prompt,
                    temperature=LLM_TEMPERATURE,
                    requires_json=True,
                    task_type="process_answer",
                    prompt_json=prompt_json
                )

            if self.llm.log_prompts:
                logging.info(f"==== ANSWER PROCESSING RESPONSE ====\n{content}\n==== END RESPONSE ====")

            # Parse JSON response (unwrap if Gemini returns a list)
            result = _unwrap_json_response(json.loads(content))
            
            # Extract and store attributes in state
            extracted_attributes = result.get("extracted_attributes", {})
            if "product_attributes" not in state:
                state["product_attributes"] = {}
            state["product_attributes"].update(extracted_attributes)
            
            updated_description = result.get("updated_description", original_query)
            selected_option = result.get("selected_option")
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "")
            
            # Validate and retry once if malformed
            invalid_selection = (selected_option is not None and (not isinstance(selected_option, (int, float)) or selected_option < 1 or selected_option > len(options)))
            invalid_update = not updated_description
            if invalid_selection or invalid_update:
                logging.warning(f"Answer processing validation failed (selected_option={selected_option}, updated_desc_present={bool(updated_description)}). Retrying once...")
                try:
                    time.sleep(1)
                    content_retry = self.llm.send_openai_request(
                        prompt=full_prompt,
                        temperature=LLM_TEMPERATURE,
                        requires_json=True,
                        task_type="process_answer",
                        prompt_json=prompt_json
                    )
                    result = _unwrap_json_response(json.loads(content_retry))
                    extracted_attributes = result.get("extracted_attributes", {})
                    if "product_attributes" not in state:
                        state["product_attributes"] = {}
                    state["product_attributes"].update(extracted_attributes)
                    updated_description = result.get("updated_description", original_query)
                    selected_option = result.get("selected_option")
                    confidence = result.get("confidence", 0.0)
                    reasoning = result.get("reasoning", "")
                except Exception as e_retry:
                    logging.error(f"Retry failed for answer processing: {e_retry}")
            
            best_match = None
            if selected_option is not None and isinstance(selected_option, (int, float)) and 1 <= selected_option <= len(options):
                option_index = int(selected_option) - 1
                selected_opt = options[option_index]
                best_match = {
                    "index": selected_option,
                    "node_id": selected_opt.get("node_id"),
                    "code": selected_opt.get("code", "[GROUP]"),
                    "description": selected_opt.get("description", ""),
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "is_group": selected_opt.get("is_group", False)
                }

            # Store the reasoning in state for potential diagnosis
            if selected_option is None and state:
                state["no_match_reason"] = reasoning

            return updated_description, best_match
        except Exception as e:
            logging.error(f"Error processing answer: {e}")
            return original_query, None

    def explain_classification(self, original_query: str, enriched_query: str, full_path: str, conversation: List[Dict[str, Any]], state: Dict = None) -> str:
        """Generate an explanation of the classification."""
        conversation_text = ""
        for i, qa in enumerate(conversation):
            conversation_text += f"Q{i+1}: {qa['question']}\n"
            conversation_text += f"A{i+1}: {qa['answer']}\n\n"

        # Include extracted product attributes in the explanation from state
        product_attributes_text = ""
        product_attributes = state.get("product_attributes", {}) if state else {}
        if product_attributes:
            product_attributes_text = "KEY PRODUCT ATTRIBUTES:\n"
            for attr, value in product_attributes.items():
                product_attributes_text += f"- {attr}: {value}\n"
            product_attributes_text += "\n"

        # Include any classification diagnosis if available
        diagnosis_text = ""
        if state and "classification_diagnosis" in state:
            diagnosis_text = f"""
CLASSIFICATION CHALLENGES:
The system initially had difficulty classifying this product. The following diagnosis was provided:
{state['classification_diagnosis']}

"""

        # Include retry information if available
        retry_text = ""
        if state and state.get("global_retry_count", 0) > 1:
            retry_text = f"Note: This classification required {state.get('global_retry_count')} attempts to find the correct category.\n\n"

        # Include multi-hypothesis information if available
        multi_hypothesis_text = ""
        if state and state.get("used_multi_hypothesis"):
            paths_info = state.get("paths_considered", [])
            if paths_info:
                multi_hypothesis_text = f"""
MULTI-HYPOTHESIS ANALYSIS:
The system considered {len(paths_info)} possible classification paths:
"""
                for i, path_info in enumerate(paths_info, 1):
                    multi_hypothesis_text += f"Path {i}: {path_info.get('chapter', 'Unknown')} - {path_info.get('reasoning', 'No reasoning')}\n"
                multi_hypothesis_text += "\n"

        # Get classification knowledge from state
        classification_section = state.get("classification_knowledge", "") if state else ""

        prompt = f"""As a customs classification expert, explain how this product was classified.

PRODUCT INFORMATION:
- ORIGINAL DESCRIPTION: {original_query}
- ENRICHED DESCRIPTION: {enriched_query}
{product_attributes_text}
{classification_section}
{diagnosis_text}
{multi_hypothesis_text}
FINAL CLASSIFICATION:
{full_path}

CONVERSATION THAT LED TO THIS CLASSIFICATION:
{conversation_text}

{retry_text}
TASK:
Explain step-by-step why this classification is correct.
Focus on how each specific product characteristic led to classification decisions at each level.
Make your explanation clear, logical, and accessible to someone without customs expertise.
"""
        try:
            if self.llm.log_prompts:
                self.llm.prompt_logger.info(f"==== EXPLANATION PROMPT ====\n{prompt}\n==== END PROMPT ====")
            logging.info("Generating classification explanation")

            # Use Vertex AI for generating explanation
            explanation = self.llm.send_vertex_ai_request(
                prompt=prompt,
                temperature=LLM_TEMPERATURE,  # Medium-low temperature for factual explanation
                task_type="explain_classification",
                prompt_json={"task": "explain_classification", "data": {"product_text": original_query, "classification_path": full_path}}
            )

            if self.llm.log_prompts:
                self.llm.prompt_logger.info(f"==== EXPLANATION RESPONSE ====\n{explanation}\n==== END RESPONSE ====")

            logging.info("Explanation generated successfully")
            return explanation
        except Exception as e:
            logging.error(f"Failed to generate explanation: {e}")
            return "Could not generate explanation due to an error."

    # ----------------------
    # Helper Methods
    # ----------------------

    def _log_prompt_to_file(self, label: str, content: str):
        """Log full prompt content to a dedicated file for debugging. Only runs if LOG_PROMPT_DEBUG=true."""
        if not LOG_PROMPT_DEBUG:
            return
        try:
            with open("prompt_debug.log", "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"[{label}] - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n")
                f.write(content)
                f.write(f"\n{'='*80}\n\n")
            logging.info(f"   📄 Full content logged to prompt_debug.log")
        except Exception as e:
            logging.warning(f"Failed to log prompt to file: {e}")

    def _format_history_for_prompt(self, history_entries):
        """Convert history entries to formatted string for prompt."""
        s = ""
        for entry in history_entries:
            s += f"Q: {entry['question']}\nA: {entry['answer']}\n"
        return s

    def _has_similar_question(self, question_text, history_entries, similarity_threshold=0.6):
        """Check if a similar question exists in history."""
        question_words = set(re.findall(r'\b\w+\b', question_text.lower()))
        if not question_words:
            return False
            
        for entry in history_entries:
            asked_question = entry.get('question', '')
            asked_words = set(re.findall(r'\b\w+\b', asked_question.lower()))
            if not asked_words:
                continue
                
            # Calculate Jaccard similarity
            intersection = len(question_words.intersection(asked_words))
            union = len(question_words.union(asked_words))
            similarity = intersection / union if union > 0 else 0
            
            if similarity > similarity_threshold:
                return True
                
        return False

    def _format_classification_knowledge(self, raw_knowledge: str) -> str:
        """
        Format the classification knowledge with explicit guidance about its importance and accuracy.
        
        Args:
            raw_knowledge: The raw classification knowledge from the LLM
            
        Returns:
            Formatted knowledge with explanatory context
        """
        if not raw_knowledge or raw_knowledge.strip() == "":
            return ""
        
        return f"""
CRITICAL CLASSIFICATION GUIDANCE
═══════════════════════════════════════════════════════════════════════════════

This information is VERY ACCURATE and AUTHORITATIVE

The following classification guidance was generated by a specialized expert system using:
• Comprehensive web search of official customs sources
• Current trade practices and regulations
• Industry-standard classification interpretations
• Official government customs guidance

CLASSIFICATION DIRECTIVE:
• This guidance provides the EXTREMELY LIKELY CORRECT 6-digit HS code for this product
• Your classification decisions MUST ALIGN with this expert guidance
• The provided 6-digit code should be your PRIMARY TARGET
• Focus on asking clarifying questions ONLY for final digit differentiation (8th, 10th digits)
• Do NOT deviate from this guidance without extremely compelling contradictory evidence

{raw_knowledge}


Again This is guidance is CRITICAL and should be followed closely when making classifications.
If you have any doubts or need further clarification, consult this guidance before proceeding.
═══════════════════════════════════════════════════════════════════════════════
"""

    def _build_decision_history(self, path: Optional[ClassificationPath]) -> List[Dict[str, Any]]:
        """
        Format the path steps into a compact history payload for downstream prompts.
        """
        if not path or not path.steps:
            return []
        
        history = []
        for step in path.steps:
            history.append({
                "step": step.get("step"),
                "stage": step.get("stage"),
                "code": step.get("selected_code"),
                "description": step.get("selected_description"),
                "decision_confidence": step.get("decision_confidence"),
                "path_confidence": step.get("path_confidence"),
                "cumulative_confidence": step.get("cumulative_confidence"),
                "node_id": step.get("node_id") or step.get("selected_node_id"),
            })
        return history
