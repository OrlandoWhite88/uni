#!/usr/bin/env python3
"""
Training Data Quality Reviewer

Reviews training data in batches of 5 (same task type) for consistent scoring.
The reviewer re-scores each example to ensure calibration across the batch.

Key insight: Individual classifications lack calibration reference points.
By reviewing batches of the same task type together, we ensure consistent scoring.
"""

import json
import logging
import os
import sys
import time
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Add the api directory to the Python path
script_dir = Path(__file__).parent
api_dir = script_dir / "api"
sys.path.insert(0, str(api_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# TASK-SPECIFIC OUTPUT SCHEMAS
# =============================================================================

TASK_OUTPUT_SCHEMAS = {
    "score_candidate_batch": {
        "has_scores": True,
        "score_fields": ["information_context_score", "path_score"],
        "structure": "scores array with option_number, information_context_score, path_score, reasoning"
    },
    "select_chapters": {
        "has_scores": True,
        "score_fields": ["information_context_score", "path_score"],
        "structure": "chapters array with chapter, information_context_score, path_score, reasoning"
    },
    "select_candidates": {
        "has_scores": True,
        "score_fields": ["information_context_scores", "path_scores"],
        "structure": "selected_indices array, information_context_scores array, path_scores array"
    },
    "generate_question": {
        "has_scores": False,
        "structure": "question_type, question_text, options array"
    },
    "process_answer": {
        "has_scores": False,
        "structure": "updated_description, extracted_attributes"
    }
}


# =============================================================================
# REVIEWER SYSTEM PROMPT - Task-specific
# =============================================================================

def get_reviewer_system_prompt(task_type: str) -> str:
    """Get the reviewer system prompt for a specific task type."""
    
    schema = TASK_OUTPUT_SCHEMAS.get(task_type, {})
    
    base_prompt = """You are a CALIBRATION REVIEWER for HS Code classification training data.

You will see 5 training examples of the SAME task type. Your job is to RE-SCORE each one 
to ensure CONSISTENT calibration across all 5, following the scoring schema EXACTLY.

## SCORING SCHEMA

### TRAINING MODE RULES (NON-NEGOTIABLE):
- PATH_SCORE for ruling-path options: 0.85-1.00 (MUST be highest)
- PATH_SCORE for non-ruling options: Maximum 0.60 (HARD CEILING)
- MINIMUM GAP: 25 points between ruling-path and best non-ruling
- INFORMATION_CONTEXT_SCORE: Honest based on prompt quality ONLY

### INFORMATION_CONTEXT_SCORE BANDS (±0.05):
0.90-1.00 EXPLICIT: ALL criteria EXPLICITLY stated
0.80-0.89 COMPLETE: All key criteria present or unambiguously inferrable
0.70-0.79 STRONG: Key criteria present, 1-2 minor assumptions
0.60-0.69 SUFFICIENT: Enough to decide [PROCEED THRESHOLD]
0.50-0.59 PARTIAL: ONE determinative criterion unclear
0.40-0.49 WEAK: 2-3 criteria unclear
0.30-0.39 MINIMAL: General category identifiable
0.20-0.29 SPARSE: Basic nature unclear
0.10-0.19 VAGUE: Too ambiguous

### PATH_SCORE BANDS (±0.05):
0.90-1.00 CERTAIN: "This IS the correct path" (90%+)
0.80-0.89 CONFIDENT: "Very likely correct" (80-89%)
0.70-0.79 PROBABLE: "Probably correct" (70-79%)
0.60-0.69 PLAUSIBLE: "Reasonable path" (60-69%)
0.50-0.59 NEUTRAL: "Could go either way"
0.40-0.49 DOUBTFUL: "Probably not"
0.30-0.39 UNLIKELY: "Doubt it"
0.20-0.29 IMPROBABLE: "Almost certainly not"

## CALIBRATION PROCESS

1. Read all 5 examples
2. Identify the ruling-path option in each (from the Cross Ruling in system prompt)
3. Re-score EACH example so that:
   - Ruling-path options: 0.85-1.00
   - Non-ruling options: ≤0.60
   - Gap: ≥25 points
   - Info scores match the evidence band
4. Ensure scores are CONSISTENT across similar product descriptions

## OUTPUT

Output a JSON array with 5 COMPLETE assistant responses (one per example).
Each response must be valid JSON matching the original task's output schema.
"""
    
    if task_type == "score_candidate_batch":
        base_prompt += """
## TASK: score_candidate_batch

Output schema for each example:
{
  "thinking": "...",
  "scores": [
    {"option_number": 1, "information_context_score": 0.XX, "path_score": 0.XX, "reasoning": "..."},
    ...
  ]
}
"""
    elif task_type == "select_chapters":
        base_prompt += """
## TASK: select_chapters

Output schema for each example:
{
  "thinking": "...",
  "chapters": [
    {"chapter": "XX", "information_context_score": 0.XX, "path_score": 0.XX, "reasoning": "..."},
    ...
  ]
}
"""
    elif task_type == "select_candidates":
        base_prompt += """
## TASK: select_candidates

Output schema for each example:
{
  "thinking": "...",
  "selected_indices": [primary_idx, alternative_idx, safety_idx],
  "information_context_scores": [0.XX, 0.XX, 0.XX],
  "path_scores": [0.XX, 0.XX, 0.XX],
  "reasoning": "..."
}
"""
    
    base_prompt += """
## FINAL OUTPUT FORMAT

{
  "corrected_responses": [
    { ... complete response for example 1 ... },
    { ... complete response for example 2 ... },
    { ... complete response for example 3 ... },
    { ... complete response for example 4 ... },
    { ... complete response for example 5 ... }
  ]
}

Output ONLY the JSON. No explanations."""

    return base_prompt


def get_batch_user_prompt(examples: List[Tuple[int, Dict]]) -> str:
    """Format the batch of examples for the user prompt."""
    
    prompt = f"Re-score these {len(examples)} examples for consistent calibration:\n\n"
    
    for i, (orig_idx, example) in enumerate(examples):
        messages = example.get("messages", [])
        
        user_msg = ""
        assistant_msg = ""
        system_excerpt = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                # Extract just the Cross Ruling part if present
                if "Cross Ruling:" in content:
                    ruling_start = content.find("Cross Ruling:")
                    system_excerpt = content[ruling_start:ruling_start+500]
                else:
                    system_excerpt = content[-300:]  # Last part of system prompt
            elif role == "user":
                user_msg = content
            elif role == "assistant":
                assistant_msg = content
        
        prompt += f"""
### EXAMPLE {i + 1} (dataset index: {orig_idx})

**Cross Ruling Context:**
{system_excerpt}...

**User Input:**
```json
{user_msg[:2500]}
```

**Original Response:**
```json
{assistant_msg[:3500]}
```

---
"""
    
    return prompt


# =============================================================================
# Training Data Reviewer Class
# =============================================================================

class TrainingDataReviewer:
    """Reviews training data in batches for consistent scoring calibration."""
    
    def __init__(
        self,
        input_file: str = "training_data.jsonl",
        output_file: str = "calibrated_training_data.jsonl",
        batch_size: int = 5,
        max_workers: int = 5,
        provider: str = "gemini",
        debug: bool = False
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.provider = provider
        self.debug = debug
        
        # Statistics
        self.total_processed = 0
        self.successful_count = 0
        self.failed_count = 0
        self.task_counts: Dict[str, int] = defaultdict(int)
        
        # Initialize LLM client
        self._init_llm_client()
        
    def _init_llm_client(self):
        """Initialize the LLM client."""
        try:
            from api.llm_client import LLMClient
            self.llm_client = LLMClient()
            logger.info(f"Initialized LLM client, will use provider: {self.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def load_and_group_by_task(self) -> Dict[str, List[Tuple[int, Dict]]]:
        """Load training examples and group by task type."""
        task_groups: Dict[str, List[Tuple[int, Dict]]] = defaultdict(list)
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    
                    # Extract task type from user message
                    task_type = "unknown"
                    for msg in example.get("messages", []):
                        if msg.get("role") == "user":
                            try:
                                user_data = json.loads(msg.get("content", "{}"))
                                task_type = user_data.get("task", "unknown")
                            except:
                                pass
                            break
                    
                    task_groups[task_type].append((idx, example))
                    
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line {idx}")
        
        # Log distribution
        logger.info(f"Loaded {sum(len(v) for v in task_groups.values())} examples:")
        for task, examples in sorted(task_groups.items(), key=lambda x: -len(x[1])):
            logger.info(f"  {task}: {len(examples)}")
        
        return task_groups
    
    def review_batch(self, task_type: str, batch: List[Tuple[int, Dict]]) -> List[Optional[Dict]]:
        """
        Review a batch of same-task-type examples and return corrected responses.
        
        Returns list of corrected assistant response dicts (or None if failed).
        """
        if task_type not in TASK_OUTPUT_SCHEMAS:
            logger.warning(f"Unknown task type {task_type}, skipping batch")
            return [None] * len(batch)
        
        schema = TASK_OUTPUT_SCHEMAS[task_type]
        if not schema.get("has_scores"):
            # Non-scoring tasks pass through unchanged
            return [None] * len(batch)
        
        system_prompt = get_reviewer_system_prompt(task_type)
        user_prompt = get_batch_user_prompt(batch)
        
        try:
            response = self.llm_client.send_openai_request(
                prompt=user_prompt,
                requires_json=True,
                temperature=0.1,
                task_type="calibrate_training_batch",
                provider_override=self.provider
            )
            
            # Parse response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON
                json_match = re.search(r'\{[\s\S]*"corrected_responses"[\s\S]*\}', response)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    logger.error(f"Could not parse reviewer response as JSON")
                    return [None] * len(batch)
            
            corrected = result.get("corrected_responses", [])
            
            # Validate we got the right number
            if len(corrected) != len(batch):
                logger.warning(f"Got {len(corrected)} responses for {len(batch)} examples")
                # Pad or truncate
                while len(corrected) < len(batch):
                    corrected.append(None)
                corrected = corrected[:len(batch)]
            
            return corrected
            
        except Exception as e:
            logger.error(f"Batch review failed: {e}")
            return [None] * len(batch)
    
    def process_all(self, max_per_task: Optional[int] = None):
        """
        Process all training examples, grouping by task type.
        
        Args:
            max_per_task: Maximum examples to process per task type (None = all)
        """
        task_groups = self.load_and_group_by_task()
        
        # Only process scoring tasks
        scoring_tasks = [t for t in task_groups.keys() 
                        if t in TASK_OUTPUT_SCHEMAS and TASK_OUTPUT_SCHEMAS[t].get("has_scores")]
        
        logger.info(f"Processing {len(scoring_tasks)} scoring task types: {scoring_tasks}")
        
        start_time = time.time()
        
        with open(self.output_file, 'w', encoding='utf-8') as out_f:
            
            for task_type in scoring_tasks:
                examples = task_groups[task_type]
                
                if max_per_task:
                    examples = examples[:max_per_task]
                
                logger.info(f"\nProcessing {len(examples)} examples of task '{task_type}'")
                
                # Create batches
                batches = []
                for i in range(0, len(examples), self.batch_size):
                    batch = examples[i:i + self.batch_size]
                    batches.append(batch)
                
                # Process batches
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_batch = {
                        executor.submit(self.review_batch, task_type, batch): (batch_idx, batch)
                        for batch_idx, batch in enumerate(batches)
                    }
                    
                    for future in as_completed(future_to_batch):
                        batch_idx, batch = future_to_batch[future]
                        try:
                            corrected_responses = future.result()
                            
                            for i, (orig_idx, example) in enumerate(batch):
                                corrected = corrected_responses[i] if i < len(corrected_responses) else None
                                
                                if corrected:
                                    # Replace assistant message with corrected version
                                    new_messages = []
                                    for msg in example.get("messages", []):
                                        if msg.get("role") == "assistant":
                                            new_messages.append({
                                                "role": "assistant",
                                                "content": json.dumps(corrected, ensure_ascii=False)
                                            })
                                        else:
                                            new_messages.append(msg)
                                    
                                    calibrated_example = {"messages": new_messages}
                                    out_f.write(json.dumps(calibrated_example) + '\n')
                                    self.successful_count += 1
                                else:
                                    # Keep original if correction failed
                                    out_f.write(json.dumps(example) + '\n')
                                    self.failed_count += 1
                                
                                self.total_processed += 1
                                self.task_counts[task_type] += 1
                        
                        except Exception as e:
                            logger.error(f"Batch {batch_idx} failed: {e}")
                            # Write originals on failure
                            for orig_idx, example in batch:
                                out_f.write(json.dumps(example) + '\n')
                                self.failed_count += 1
                                self.total_processed += 1
                
                logger.info(f"  Completed {task_type}: {self.task_counts[task_type]} processed")
            
            # Also write non-scoring tasks unchanged
            non_scoring_tasks = [t for t in task_groups.keys() if t not in scoring_tasks]
            for task_type in non_scoring_tasks:
                for orig_idx, example in task_groups[task_type]:
                    out_f.write(json.dumps(example) + '\n')
                    self.total_processed += 1
                logger.info(f"  Passed through {len(task_groups[task_type])} '{task_type}' examples unchanged")
        
        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("CALIBRATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total processed: {self.total_processed}")
        logger.info(f"Successfully calibrated: {self.successful_count}")
        logger.info(f"Kept original (calibration failed): {self.failed_count}")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"Output: {self.output_file}")


# =============================================================================
# Quick Analysis
# =============================================================================

def quick_analyze(input_file: str, sample_size: int = 1000):
    """Quick analysis of training data distribution and scores."""
    
    task_counts = defaultdict(int)
    score_data = defaultdict(lambda: {"info": [], "path": []})
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                
                task_type = "unknown"
                for msg in example.get("messages", []):
                    if msg.get("role") == "user":
                        try:
                            user_data = json.loads(msg.get("content", "{}"))
                            task_type = user_data.get("task", "unknown")
                        except:
                            pass
                    
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        # Extract scores
                        info_scores = re.findall(r'"information_context_score":\s*([0-9.]+)', content)
                        path_scores = re.findall(r'"path_score":\s*([0-9.]+)', content)
                        
                        for s in info_scores:
                            try:
                                score_data[task_type]["info"].append(float(s))
                            except:
                                pass
                        for s in path_scores:
                            try:
                                score_data[task_type]["path"].append(float(s))
                            except:
                                pass
                
                task_counts[task_type] += 1
                
            except:
                pass
    
    print("\n" + "=" * 60)
    print("TRAINING DATA ANALYSIS")
    print("=" * 60)
    
    total = sum(task_counts.values())
    print(f"\nTotal examples analyzed: {total}")
    
    print("\nTask distribution:")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task}: {count} ({count/total*100:.1f}%)")
    
    print("\nScore statistics by task:")
    for task in sorted(score_data.keys()):
        data = score_data[task]
        if data["info"]:
            info_avg = sum(data["info"]) / len(data["info"])
            info_min = min(data["info"])
            info_max = max(data["info"])
            print(f"\n  {task}:")
            print(f"    info_context: avg={info_avg:.2f}, min={info_min:.2f}, max={info_max:.2f}, n={len(data['info'])}")
        if data["path"]:
            path_avg = sum(data["path"]) / len(data["path"])
            path_min = min(data["path"])
            path_max = max(data["path"])
            print(f"    path_score:   avg={path_avg:.2f}, min={path_min:.2f}, max={path_max:.2f}, n={len(data['path'])}")
            
            # Check for gap issues
            # This is approximate - we'd need to know which is ruling-path
            high_scores = [s for s in data["path"] if s > 0.60]
            if high_scores:
                print(f"    path > 0.60:  {len(high_scores)} ({len(high_scores)/len(data['path'])*100:.1f}%)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Training Data Calibration Reviewer")
    
    parser.add_argument("--input", default="training_data.jsonl",
                       help="Input training data file")
    parser.add_argument("--output", default="calibrated_training_data.jsonl",
                       help="Output calibrated data file")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Examples per batch (same task type)")
    parser.add_argument("--max-workers", type=int, default=5,
                       help="Parallel workers")
    parser.add_argument("--max-per-task", type=int, default=None,
                       help="Max examples per task type (for testing)")
    parser.add_argument("--provider", default="gemini",
                       help="LLM provider (default: gemini)")
    parser.add_argument("--debug", action="store_true")
    
    parser.add_argument("--quick-analyze", action="store_true",
                       help="Quick analysis only (no LLM)")
    parser.add_argument("--sample-size", type=int, default=1000,
                       help="Sample size for analysis")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.quick_analyze:
        quick_analyze(args.input, args.sample_size)
    else:
        reviewer = TrainingDataReviewer(
            input_file=args.input,
            output_file=args.output,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            provider=args.provider,
            debug=args.debug
        )
        reviewer.process_all(max_per_task=args.max_per_task)


if __name__ == "__main__":
    main()
