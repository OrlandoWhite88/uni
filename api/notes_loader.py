"""
Notes Loader Utility
====================
Dynamically loads section and chapter notes from the notes/ folder.
Used for injecting relevant classification context into prompts.
"""

import importlib
import logging
from typing import Dict, List, Optional
from functools import lru_cache

# Section to Chapter mapping (HTS structure)
SECTION_CHAPTER_MAP = {
    1: list(range(1, 6)),      # Section I: Chapters 1-5 (Live Animals; Animal Products)
    2: list(range(6, 15)),     # Section II: Chapters 6-14 (Vegetable Products)
    3: [15],                   # Section III: Chapter 15 (Fats and Oils)
    4: list(range(16, 25)),    # Section IV: Chapters 16-24 (Prepared Foodstuffs)
    5: list(range(25, 28)),    # Section V: Chapters 25-27 (Mineral Products)
    6: list(range(28, 39)),    # Section VI: Chapters 28-38 (Chemical Products)
    7: list(range(39, 41)),    # Section VII: Chapters 39-40 (Plastics and Rubber)
    8: list(range(41, 44)),    # Section VIII: Chapters 41-43 (Leather, Furskins)
    9: list(range(44, 47)),    # Section IX: Chapters 44-46 (Wood, Cork, Basketware)
    10: list(range(47, 50)),   # Section X: Chapters 47-49 (Pulp, Paper)
    11: list(range(50, 64)),   # Section XI: Chapters 50-63 (Textiles)
    12: list(range(64, 68)),   # Section XII: Chapters 64-67 (Footwear, Headgear)
    13: list(range(68, 71)),   # Section XIII: Chapters 68-70 (Stone, Ceramic, Glass)
    14: [71],                  # Section XIV: Chapter 71 (Precious Metals, Jewelry)
    15: list(range(72, 84)),   # Section XV: Chapters 72-83 (Base Metals)
    16: list(range(84, 86)),   # Section XVI: Chapters 84-85 (Machinery, Electrical)
    17: list(range(86, 90)),   # Section XVII: Chapters 86-89 (Vehicles, Aircraft, Vessels)
    18: list(range(90, 93)),   # Section XVIII: Chapters 90-92 (Instruments)
    19: [93],                  # Section XIX: Chapter 93 (Arms and Ammunition)
    20: list(range(94, 97)),   # Section XX: Chapters 94-96 (Miscellaneous)
    21: [97],                  # Section XXI: Chapter 97 (Works of Art)
    22: [98, 99],              # Section XXII: Chapters 98-99 (Special Classification)
}

# Reverse mapping: chapter -> section
CHAPTER_TO_SECTION: Dict[int, int] = {}
for section, chapters in SECTION_CHAPTER_MAP.items():
    for chapter in chapters:
        CHAPTER_TO_SECTION[chapter] = section


@lru_cache(maxsize=50)
def load_section_notes(section_number: int) -> str:
    """
    Load section notes from notes/00_section_XX.py
    
    Args:
        section_number: Section number (1-22)
        
    Returns:
        Section notes string, or empty string if not found
    """
    try:
        module_name = f"notes.00_section_{section_number:02d}"
        module = importlib.import_module(module_name)
        
        # Try different variable naming conventions
        var_names = [
            f"SECTION_{section_number:02d}_NOTES",
            f"SECTION_{section_number}_NOTES",
            "SECTION_NOTES",
            "NOTES"
        ]
        
        for var_name in var_names:
            if hasattr(module, var_name):
                notes = getattr(module, var_name)
                if notes:
                    logging.debug(f"Loaded section {section_number} notes from {var_name}")
                    return notes.strip()
        
        logging.warning(f"Section {section_number} module found but no notes variable")
        return ""
        
    except ImportError as e:
        logging.debug(f"Section {section_number} notes not found: {e}")
        return ""
    except Exception as e:
        logging.error(f"Error loading section {section_number} notes: {e}")
        return ""


@lru_cache(maxsize=100)
def load_chapter_notes(chapter_number: int) -> str:
    """
    Load chapter notes from notes/chapter_XX.py
    
    Args:
        chapter_number: Chapter number (1-99)
        
    Returns:
        Chapter notes string, or empty string if not found
    """
    try:
        module_name = f"notes.chapter_{chapter_number:02d}"
        module = importlib.import_module(module_name)
        
        # Try different variable naming conventions
        var_names = [
            f"CHAPTER_{chapter_number:02d}_NOTES",
            f"CHAPTER_{chapter_number}_NOTES",
            "CHAPTER_NOTES",
            "NOTES"
        ]
        
        for var_name in var_names:
            if hasattr(module, var_name):
                notes = getattr(module, var_name)
                if notes:
                    logging.debug(f"Loaded chapter {chapter_number} notes from {var_name}")
                    return notes.strip()
        
        logging.warning(f"Chapter {chapter_number} module found but no notes variable")
        return ""
        
    except ImportError as e:
        logging.debug(f"Chapter {chapter_number} notes not found: {e}")
        return ""
    except Exception as e:
        logging.error(f"Error loading chapter {chapter_number} notes: {e}")
        return ""


def get_section_for_chapter(chapter: int) -> int:
    """
    Get the section number for a given chapter.
    
    Args:
        chapter: Chapter number (1-99)
        
    Returns:
        Section number (1-22), or 0 if not found
    """
    return CHAPTER_TO_SECTION.get(chapter, 0)


def get_chapters_for_section(section: int) -> List[int]:
    """
    Get the list of chapters in a given section.
    
    Args:
        section: Section number (1-22)
        
    Returns:
        List of chapter numbers in that section
    """
    return SECTION_CHAPTER_MAP.get(section, [])


def load_all_section_notes() -> str:
    """
    Load and concatenate all section notes.
    Used for Stage 1 chapter selection.
    
    Returns:
        Concatenated string of all section notes
    """
    all_notes = []
    
    for section_num in range(1, 23):
        notes = load_section_notes(section_num)
        if notes:
            all_notes.append(notes)
    
    if all_notes:
        return "\n\n".join(all_notes)
    
    return ""


def load_chapter_notes_only(chapters: List[int]) -> str:
    """
    Load ONLY chapter notes for specific chapters (no section notes).
    Used for Stage 2 chapter selection.
    
    Args:
        chapters: List of chapter numbers to load notes for
        
    Returns:
        Concatenated string of chapter notes only
    """
    notes_parts = []
    
    for chapter in chapters:
        chapter_num = int(chapter) if isinstance(chapter, str) else chapter
        
        # Load chapter notes only
        chapter_notes = load_chapter_notes(chapter_num)
        if chapter_notes:
            notes_parts.append(f"=== CHAPTER {chapter_num:02d} NOTES ===\n{chapter_notes}")
    
    if notes_parts:
        return "\n\n".join(notes_parts)
    
    return ""


def load_relevant_notes_for_chapters(chapters: List[int]) -> str:
    """
    Load section and chapter notes relevant to specific chapters.
    Used during classification within chapters (not for chapter selection).
    
    Args:
        chapters: List of chapter numbers to load notes for
        
    Returns:
        Concatenated string of relevant section and chapter notes
    """
    notes_parts = []
    loaded_sections = set()
    
    for chapter in chapters:
        chapter_num = int(chapter) if isinstance(chapter, str) else chapter
        
        # Load section notes (only once per section)
        section = get_section_for_chapter(chapter_num)
        if section and section not in loaded_sections:
            section_notes = load_section_notes(section)
            if section_notes:
                notes_parts.append(f"=== SECTION {section} NOTES ===\n{section_notes}")
                loaded_sections.add(section)
        
        # Load chapter notes
        chapter_notes = load_chapter_notes(chapter_num)
        if chapter_notes:
            notes_parts.append(f"=== CHAPTER {chapter_num} NOTES ===\n{chapter_notes}")
    
    if notes_parts:
        return "\n\n".join(notes_parts)
    
    return ""


def load_chapter_context(chapter: int) -> str:
    """
    Load both section and chapter notes for a specific chapter.
    Used during classification within a chapter.
    
    Args:
        chapter: Chapter number
        
    Returns:
        Combined section and chapter notes
    """
    chapter_num = int(chapter) if isinstance(chapter, str) else chapter
    
    notes_parts = []
    
    # Load section notes
    section = get_section_for_chapter(chapter_num)
    if section:
        section_notes = load_section_notes(section)
        if section_notes:
            notes_parts.append(f"=== SECTION {section} NOTES ===\n{section_notes}")
    
    # Load chapter notes
    chapter_notes = load_chapter_notes(chapter_num)
    if chapter_notes:
        notes_parts.append(f"=== CHAPTER {chapter_num} NOTES ===\n{chapter_notes}")
    
    if notes_parts:
        return "\n\n".join(notes_parts)
    
    return ""


def extract_chapter_from_path(path_string: str) -> Optional[int]:
    """
    Extract chapter number from a classification path string.
    
    Args:
        path_string: Path like "Chapter 03 > 0302 > 0302.11" or "03.02.11"
        
    Returns:
        Chapter number as int, or None if not found
    """
    if not path_string:
        return None
    
    import re
    
    # Try "Chapter XX" pattern
    chapter_match = re.search(r'Chapter\s*(\d{1,2})', path_string, re.IGNORECASE)
    if chapter_match:
        return int(chapter_match.group(1))
    
    # Try HTS code pattern (first 2 digits)
    code_match = re.search(r'(\d{2})[\.\d]*', path_string)
    if code_match:
        return int(code_match.group(1))
    
    return None


def extract_chapter_from_node(node) -> Optional[int]:
    """
    Extract chapter number from an HTS node.
    
    Args:
        node: HTSNode object with htsno or description
        
    Returns:
        Chapter number as int, or None if not found
    """
    if not node:
        return None
    
    # Try htsno first
    if hasattr(node, 'htsno') and node.htsno:
        try:
            # First 2 digits of HTS code
            return int(node.htsno[:2])
        except (ValueError, IndexError):
            pass
    
    # Try description
    if hasattr(node, 'description'):
        return extract_chapter_from_path(node.description)
    
    return None


# Clear cache utility (useful for testing)
def clear_notes_cache():
    """Clear all cached notes."""
    load_section_notes.cache_clear()
    load_chapter_notes.cache_clear()
    logging.info("Notes cache cleared")

