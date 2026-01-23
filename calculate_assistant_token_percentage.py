#!/usr/bin/env python3
"""
Calculate the percentage of assistant tokens in a trajectory JSON file.
"""

import json
import sys
from pathlib import Path


def approximate_tokens(text):
    """
    Approximate token count. Common approximations:
    - ~4 characters per token (for English text)
    - Or ~0.75 words per token
    Using character-based approximation as it's simpler.
    """
    if not text:
        return 0
    # Rough approximation: 1 token ≈ 4 characters
    return len(text) / 4


def extract_task_type(user_content):
    """Extract task type from user message content."""
    if not isinstance(user_content, str):
        return None
    
    # Look for "TASK: " pattern
    if 'TASK: ' in user_content:
        task_line = [line for line in user_content.split('\n') if 'TASK: ' in line]
        if task_line:
            task_part = task_line[0].split('TASK: ')[1].strip()
            # Remove any trailing content after task name
            task_name = task_part.split()[0] if task_part.split() else None
            return task_name
    
    # Also check JSON input if present
    if 'task":' in user_content or '"task"' in user_content:
        import re
        # Try to extract from JSON
        match = re.search(r'"task"\s*:\s*"([^"]+)"', user_content)
        if match:
            return match.group(1)
    
    return None


def calculate_assistant_percentage(trajectory_file):
    """Calculate percentage of assistant tokens by task type."""
    
    # Load JSON file
    with open(trajectory_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract messages
    messages = data.get('messages', [])
    
    total_tokens = 0
    assistant_tokens = 0
    
    # Track tokens by task type
    task_tokens = {}
    
    # Process messages to identify task types
    for i, msg in enumerate(messages):
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        # Handle both string and list content
        if isinstance(content, list):
            # If content is a list, join all items
            content = ' '.join(str(item) for item in content)
        elif not isinstance(content, str):
            content = str(content)
        
        msg_tokens = approximate_tokens(content)
        total_tokens += msg_tokens
        
        if role == 'assistant':
            assistant_tokens += msg_tokens
            
            # Find the preceding user message to get task type
            task_type = None
            for j in range(i - 1, -1, -1):
                prev_msg = messages[j]
                if prev_msg.get('role') == 'user':
                    task_type = extract_task_type(prev_msg.get('content', ''))
                    break
            
            # Normalize task type names
            if task_type:
                # Normalize variations
                if 'select_chapters' in task_type.lower():
                    if 'stage2' in task_type.lower() or 'stage_2' in task_type.lower():
                        task_type = 'select_chapters_stage2'
                    else:
                        task_type = 'select_chapters_stage1'
                elif 'rank_candidates' in task_type.lower():
                    task_type = 'rank_candidates'
                elif 'generate_question' in task_type.lower():
                    task_type = 'generate_question'
                elif 'process_answer' in task_type.lower():
                    task_type = 'process_answer'
                else:
                    task_type = task_type.lower()
            else:
                task_type = 'unknown'
            
            task_tokens[task_type] = task_tokens.get(task_type, 0) + msg_tokens
    
    # Calculate percentages
    if assistant_tokens == 0:
        return {
            'total_tokens': total_tokens,
            'assistant_tokens': 0,
            'percentage': 0.0,
            'task_breakdown': {},
            'grouped_percentages': {}
        }
    
    # Calculate percentage of total tokens
    total_percentage = (assistant_tokens / total_tokens) * 100
    
    # Calculate task breakdown (as percentage of assistant tokens)
    task_breakdown = {
        task: {
            'tokens': tokens,
            'percentage_of_assistant': (tokens / assistant_tokens) * 100,
            'percentage_of_total': (tokens / total_tokens) * 100
        }
        for task, tokens in task_tokens.items()
    }
    
    # Group tasks as requested
    question_generation_tokens = task_tokens.get('generate_question', 0)
    process_answer_tokens = task_tokens.get('process_answer', 0)
    non_selection_tokens = question_generation_tokens + process_answer_tokens
    
    chapter_selection_tokens = (task_tokens.get('select_chapters_stage1', 0) + 
                                task_tokens.get('select_chapters_stage2', 0))
    rank_candidates_tokens = task_tokens.get('rank_candidates', 0)
    selection_tokens = chapter_selection_tokens + rank_candidates_tokens
    
    grouped_percentages = {
        'non_selection_tasks': {
            'tasks': ['generate_question', 'process_answer'],
            'tokens': non_selection_tokens,
            'percentage_of_assistant': (non_selection_tokens / assistant_tokens) * 100,
            'percentage_of_total': (non_selection_tokens / total_tokens) * 100
        },
        'selection_tasks': {
            'tasks': ['select_chapters_stage1', 'select_chapters_stage2', 'rank_candidates'],
            'tokens': selection_tokens,
            'percentage_of_assistant': (selection_tokens / assistant_tokens) * 100,
            'percentage_of_total': (selection_tokens / total_tokens) * 100
        }
    }
    
    return {
        'total_tokens': total_tokens,
        'assistant_tokens': assistant_tokens,
        'percentage': total_percentage,
        'total_messages': len(messages),
        'assistant_messages': sum(1 for msg in messages if msg.get('role') == 'assistant'),
        'task_breakdown': task_breakdown,
        'grouped_percentages': grouped_percentages
    }


if __name__ == '__main__':
    # Get file paths from command line arguments
    if len(sys.argv) > 1:
        file_paths = sys.argv[1:]
    else:
        # Default files if none provided
        file_paths = ['trajectory_audi r8 v10_rank1_20251221_224727.json']
    
    all_results = []
    
    for file_path in file_paths:
        try:
            results = calculate_assistant_percentage(file_path)
            results['file_path'] = file_path
            all_results.append(results)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            continue
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file '{file_path}': {e}")
            continue
        except Exception as e:
            print(f"Error processing '{file_path}': {e}")
            continue
    
    # Print individual file results
    for results in all_results:
        file_path = results['file_path']
        print(f"\n{'=' * 80}")
        print(f"Trajectory Analysis: {file_path}")
        print("=" * 80)
        print(f"Total messages: {results['total_messages']}")
        print(f"Assistant messages: {results['assistant_messages']}")
        print(f"Total tokens (approx): {results['total_tokens']:.1f}")
        print(f"Assistant tokens (approx): {results['assistant_tokens']:.1f}")
        print(f"Assistant token percentage: {results['percentage']:.2f}%")
        print()
        
        # Grouped breakdown
        print("GROUPED BREAKDOWN:")
        print("-" * 80)
        grouped = results['grouped_percentages']
        
        non_sel = grouped['non_selection_tasks']
        print(f"Non-selection tasks (Question generation + Process answer):")
        print(f"  Tokens: {non_sel['tokens']:.1f}")
        print(f"  % of assistant tokens: {non_sel['percentage_of_assistant']:.2f}%")
        print(f"  % of total tokens: {non_sel['percentage_of_total']:.2f}%")
        print()
        
        sel = grouped['selection_tasks']
        print(f"Selection tasks (Chapter selection + Rank candidates):")
        print(f"  Tokens: {sel['tokens']:.1f}")
        print(f"  % of assistant tokens: {sel['percentage_of_assistant']:.2f}%")
        print(f"  % of total tokens: {sel['percentage_of_total']:.2f}%")
        print()
        
        # Calculate chapter selection + rank_candidates percentage
        chapter_tokens = (results['task_breakdown'].get('select_chapters_stage1', {}).get('tokens', 0) +
                         results['task_breakdown'].get('select_chapters_stage2', {}).get('tokens', 0))
        rank_tokens = results['task_breakdown'].get('rank_candidates', {}).get('tokens', 0)
        chapter_rank_tokens = chapter_tokens + rank_tokens
        
        if results['assistant_tokens'] > 0:
            chapter_rank_pct = (chapter_rank_tokens / results['assistant_tokens']) * 100
        else:
            chapter_rank_pct = 0.0
        
        print("CHAPTER SELECTION + RANK_CANDIDATES:")
        print("-" * 80)
        print(f"  Chapter selection tokens: {chapter_tokens:.1f}")
        print(f"  Rank candidates tokens: {rank_tokens:.1f}")
        print(f"  Total chapter selection + rank_candidates tokens: {chapter_rank_tokens:.1f}")
        print(f"  % of assistant tokens: {chapter_rank_pct:.2f}%")
        print()
        
        # Detailed task breakdown
        print("DETAILED TASK BREAKDOWN:")
        print("-" * 80)
        for task, stats in sorted(results['task_breakdown'].items(), 
                                  key=lambda x: x[1]['tokens'], reverse=True):
            print(f"{task}:")
            print(f"  Tokens: {stats['tokens']:.1f}")
            print(f"  % of assistant tokens: {stats['percentage_of_assistant']:.2f}%")
            print(f"  % of total tokens: {stats['percentage_of_total']:.2f}%")
    
    # Summary across all files
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print("SUMMARY ACROSS ALL FILES")
        print("=" * 80)
        
        total_assistant_tokens = sum(r['assistant_tokens'] for r in all_results)
        total_chapter_rank_tokens = 0
        
        for results in all_results:
            chapter_tokens = (results['task_breakdown'].get('select_chapters_stage1', {}).get('tokens', 0) +
                             results['task_breakdown'].get('select_chapters_stage2', {}).get('tokens', 0))
            rank_tokens = results['task_breakdown'].get('rank_candidates', {}).get('tokens', 0)
            total_chapter_rank_tokens += (chapter_tokens + rank_tokens)
        
        if total_assistant_tokens > 0:
            overall_pct = (total_chapter_rank_tokens / total_assistant_tokens) * 100
        else:
            overall_pct = 0.0
        
        print(f"Total assistant tokens across all files: {total_assistant_tokens:.1f}")
        print(f"Total chapter selection + rank_candidates tokens: {total_chapter_rank_tokens:.1f}")
        print(f"Overall % of assistant tokens: {overall_pct:.2f}%")

