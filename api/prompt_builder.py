"""
Prompt Builder
==============
Modular prompt construction for HS code classification.
Builds task-specific prompts with optional notes injection.
"""

from typing import Optional
from .system_prompts_updated import (
    CORE_PROMPT,
    SELECT_CHAPTERS_TASK,
    RANK_CANDIDATES_TASK,
    GENERATE_QUESTION_TASK,
    PROCESS_ANSWER_TASK,
)

# Task type to prompt mapping
TASK_PROMPTS = {
    "select_chapters": SELECT_CHAPTERS_TASK,
    "rank_candidates": RANK_CANDIDATES_TASK,
    "generate_question": GENERATE_QUESTION_TASK,
    "process_answer": PROCESS_ANSWER_TASK,
}


def build_system_prompt(
    task_type: str,
    chapter_notes: str = "",
    section_notes: str = "",
    additional_context: str = "",
    training_mode: bool = False,
    cross_ruling: Optional[str] = None,
    include_core: bool = True,
) -> str:
    """
    Build a task-specific system prompt with optional notes injection.
    
    Args:
        task_type: One of "select_chapters", "rank_candidates", 
                   "generate_question", "process_answer"
        chapter_notes: Chapter-specific notes to inject
        section_notes: Section-specific notes to inject
        additional_context: Any additional context to include
        training_mode: Whether this is for training data generation
        cross_ruling: CROSS ruling text for training mode
        include_core: Include the global CORE_PROMPT when True (default)
        
    Returns:
        Complete system prompt string
    """
    prompt_parts = []

    if include_core:
        prompt_parts.append(CORE_PROMPT.strip())
    
    # Add notes section if any notes provided
    notes_content = []
    if section_notes:
        notes_content.append(section_notes.strip())
    if chapter_notes:
        notes_content.append(chapter_notes.strip())
    
    if notes_content:
        notes_section = (
            "=" * 60 + "\n"
            "RELEVANT SECTION AND CHAPTER NOTES\n"
            "These notes are LEGALLY BINDING and must be applied.\n"
            + "=" * 60 + "\n"
            + "\n\n".join(notes_content)
            + "\n" + "=" * 60
        ).strip()
        prompt_parts.append(notes_section)
    
    # Add additional context if provided
    if additional_context:
        prompt_parts.append(f"ADDITIONAL CONTEXT:\n{additional_context.strip()}")
    
    # Add training mode section if applicable
    if training_mode:
        training_section = (
            "TRAINING MODE ACTIVE\n"
            "====================\n"
            "You are generating training data. The CROSS ruling provides the ground truth.\n"
            "- Your classification MUST arrive at the exact code in the ruling\n"
            "- Score ruling-path options with path_score >= 0.85\n"
            "- Score non-ruling options with path_score <= 0.60\n"
            "- information_context_score should reflect prompt quality only, not ruling knowledge"
        )
        prompt_parts.append(training_section.strip())
        
        if cross_ruling:
            prompt_parts.append(f"CROSS RULING (GROUND TRUTH):\n{cross_ruling.strip()}")
    
    # Add task-specific prompt
    task_prompt = TASK_PROMPTS.get(task_type, "")
    task_section = task_prompt.strip() if task_prompt else f"TASK: {task_type}"
    prompt_parts.append(task_section)
    
    return "\n\n".join(part.strip() for part in prompt_parts if part.strip())


def build_chapter_selection_prompt_stage1(all_section_notes: str = "") -> str:
    """
    Build prompt for Stage 1 chapter selection.
    Includes ALL section notes for broad context.
    
    Args:
        all_section_notes: Concatenated notes from all sections
        
    Returns:
        System prompt for stage 1 chapter selection
    """
    return build_system_prompt(
        task_type="select_chapters",
        section_notes=all_section_notes,
        additional_context="STAGE 1: Initial chapter selection with section-level guidance.",
        include_core=False,
    )


def build_chapter_selection_prompt_stage2(relevant_notes: str = "") -> str:
    """
    Build prompt for Stage 2 chapter selection.
    Includes only notes for initially selected chapters.
    
    Args:
        relevant_notes: Section + chapter notes for selected chapters only
        
    Returns:
        System prompt for stage 2 chapter selection
    """
    # For stage 2, we pass the combined notes as section_notes
    # since they're already formatted with section and chapter labels
    return build_system_prompt(
        task_type="select_chapters",
        section_notes=relevant_notes,
        additional_context=(
            "STAGE 2: Refined chapter selection.\n"
            "You are re-evaluating the initially selected chapters with FULL context.\n"
            "Apply the section and chapter notes carefully - they contain exclusions and definitions that affect classification."
        ),
        include_core=False,
    )


def build_candidate_ranking_prompt(chapter_notes: str = "") -> str:
    """
    Build prompt for unified candidate ranking (select + score in one step).
    
    Args:
        chapter_notes: Combined section + chapter notes for current chapter
        
    Returns:
        System prompt for candidate ranking
    """
    return build_system_prompt(
        task_type="rank_candidates",
        section_notes=chapter_notes,
        include_core=False,
    )


def build_question_generation_prompt(chapter_notes: str = "") -> str:
    """
    Build prompt for clarification question generation.
    
    Args:
        chapter_notes: Combined section + chapter notes for current chapter
        
    Returns:
        System prompt for question generation
    """
    return build_system_prompt(
        task_type="generate_question",
        section_notes=chapter_notes,
        include_core=False,
    )


def build_answer_processing_prompt(chapter_notes: str = "") -> str:
    """
    Build prompt for processing user answers.
    
    Args:
        chapter_notes: Combined section + chapter notes for current chapter
        
    Returns:
        System prompt for answer processing
    """
    return build_system_prompt(
        task_type="process_answer",
        section_notes=chapter_notes,
        include_core=False,
    )


# Convenience function for getting just the task prompt
def get_task_prompt(task_type: str) -> str:
    """
    Get just the task-specific prompt without the core prompt.
    
    Args:
        task_type: Task type identifier
        
    Returns:
        Task-specific prompt string
    """
    return TASK_PROMPTS.get(task_type, "")

