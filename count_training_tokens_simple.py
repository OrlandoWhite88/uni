#!/usr/bin/env python3
"""
Simple script to count approximate tokens in training_data.jsonl
Uses word-based estimation when tiktoken is not available
"""

import json
import sys
import re
from pathlib import Path

def estimate_tokens_simple(text: str) -> int:
    """
    Simple token estimation without tiktoken
    Uses: words + punctuation + special characters
    Rough approximation: 1 token ≈ 0.75 words
    """
    if not text:
        return 0
    
    # Count words (split by whitespace)
    words = len(text.split())
    
    # Count punctuation and special characters
    punctuation_count = len(re.findall(r'[^\w\s]', text))
    
    # Rough token estimate
    estimated_tokens = int((words * 1.3) + (punctuation_count * 0.5))
    
    return max(1, estimated_tokens)  # At least 1 token

def count_training_tokens(file_path: str):
    """Count approximate tokens in training data JSONL file"""
    
    if not Path(file_path).exists():
        print(f"Error: File {file_path} not found")
        return
    
    print(f"Counting tokens in: {file_path}")
    print("=" * 50)
    print("NOTE: Using simple word-based estimation")
    print("Install 'tiktoken' for precise OpenAI token counts")
    print("=" * 50)
    
    total_tokens = 0
    total_lines = 0
    total_messages = 0
    user_tokens = 0
    assistant_tokens = 0
    total_chars = 0
    total_words = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    
                    line_tokens = 0
                    for message in messages:
                        role = message.get("role", "")
                        content = message.get("content", "")
                        
                        # Count tokens, chars, words
                        content_tokens = estimate_tokens_simple(content)
                        
                        # Add small overhead for role and JSON structure
                        message_tokens = content_tokens + 4
                        
                        line_tokens += message_tokens
                        total_messages += 1
                        total_chars += len(content)
                        total_words += len(content.split())
                        
                        # Track by role
                        if role == "user":
                            user_tokens += message_tokens
                        elif role == "assistant":
                            assistant_tokens += message_tokens
                    
                    total_tokens += line_tokens
                    total_lines += 1
                    
                    # Progress indicator for large files
                    if line_num % 1000 == 0:
                        print(f"Processed {line_num:,} lines... ({total_tokens:,} tokens so far)")
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
                
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Results
    print("\n" + "=" * 60)
    print("TOKEN COUNT RESULTS (ESTIMATED)")
    print("=" * 60)
    print(f"Total lines processed: {total_lines:,}")
    print(f"Total messages: {total_messages:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total words: {total_words:,}")
    print(f"Estimated tokens: {total_tokens:,}")
    print()
    print("BREAKDOWN BY ROLE:")
    if total_tokens > 0:
        print(f"User tokens: {user_tokens:,} ({user_tokens/total_tokens*100:.1f}%)")
        print(f"Assistant tokens: {assistant_tokens:,} ({assistant_tokens/total_tokens*100:.1f}%)")
    print()
    print("STATISTICS:")
    if total_lines > 0:
        print(f"Average tokens per line: {total_tokens/total_lines:.1f}")
