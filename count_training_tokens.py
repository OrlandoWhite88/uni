#!/usr/bin/env python3
"""
Quick script to count tokens in training_data.jsonl
Uses tiktoken for accurate OpenAI token counting
"""

import json
import sys
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken not installed. Install with: pip install tiktoken")
    sys.exit(1)

def count_tokens_in_text(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using specified encoding"""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error encoding text: {e}")
        return 0

def count_training_tokens(file_path: str):
    """Count tokens in training data JSONL file"""
    
    if not Path(file_path).exists():
        print(f"Error: File {file_path} not found")
        return
    
    print(f"Counting tokens in: {file_path}")
    print("=" * 50)
    
    total_tokens = 0
    total_lines = 0
    total_messages = 0
    user_tokens = 0
    assistant_tokens = 0
    
    # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo)
    encoding_name = "cl100k_base"
    
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
                        
                        # Count tokens in content
                        content_tokens = count_tokens_in_text(content, encoding_name)
                        
                        # Add small overhead for role and structure (roughly 4 tokens per message)
                        message_tokens = content_tokens + 4
                        
                        line_tokens += message_tokens
                        total_messages += 1
                        
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
    print("\n" + "=" * 50)
    print("TOKEN COUNT RESULTS")
    print("=" * 50)
    print(f"Total lines processed: {total_lines:,}")
    print(f"Total messages: {total_messages:,}")
    print(f"Total tokens: {total_tokens:,}")
    print()
    print("BREAKDOWN BY ROLE:")
    print(f"User tokens: {user_tokens:,} ({user_tokens/total_tokens*100:.1f}%)")
    print(f"Assistant tokens: {assistant_tokens:,} ({assistant_tokens/total_tokens*100:.1f}%)")
    print()
    print("STATISTICS:")
    print(f"Average tokens per line: {total_tokens/total_lines:.1f}")
    print(f"Average tokens per message: {total_tokens/total_messages:.1f}")
    print(f"Average messages per line: {total_messages/total_lines:.1f}")
    print()
    print("COST ESTIMATES (rough):")
    # GPT-3.5-turbo fine-tuning costs (as of 2024)
    training_cost_per_1k = 0.008  # $0.008 per 1K tokens
    estimated_cost = (total_tokens / 1000) * training_cost_per_1k
    print(f"Est. fine-tuning cost (GPT-3.5): ${estimated_cost:.2f}")
    
    # Also show in different token units
    print()
    print("TOKEN VOLUME:")
    print(f"Tokens in thousands: {total_tokens/1000:.1f}K")
    print(f"Tokens in millions: {total_tokens/1000000:.2f}M")

def main():
    file_path = "training_data.jsonl"
    
    # Allow custom file path as argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    count_training_tokens(file_path)

if __name__ == "__main__":
    main()
