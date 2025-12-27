#!/usr/bin/env python3
"""
Script to create a derived JSONL dataset for Reinforcement Fine-Tuning (RFT).

This script processes training_data.jsonl and creates a new format where:
- Original messages are preserved
- Complete assistant JSON response becomes the "targets" field
- confidence_threshold is extracted from user data when present
"""

import json
import sys
from typing import Dict, Any, Optional

def extract_confidence_threshold(user_content: str) -> Optional[float]:
    """Extract confidence_threshold from user content JSON."""
    try:
        data = json.loads(user_content)
        if isinstance(data, dict) and 'data' in data:
            return data['data'].get('confidence_threshold')
    except (json.JSONDecodeError, KeyError):
        pass
    return None

def process_training_example(line: str) -> Optional[Dict[str, Any]]:
    """Process a single training example line."""
    try:
        example = json.loads(line.strip())
        if 'messages' not in example:
            return None
            
        messages = example['messages']
        if len(messages) < 2:
            return None
            
        # Find user and assistant messages
        user_msg = None
        assistant_msg = None
        
        for msg in messages:
            if msg.get('role') == 'user':
                user_msg = msg
            elif msg.get('role') == 'assistant':
                assistant_msg = msg
                
        if not user_msg or not assistant_msg:
            return None
            
        # Parse assistant response as the targets
        try:
            targets = json.loads(assistant_msg['content'])
        except json.JSONDecodeError:
            print(f"Warning: Could not parse assistant response as JSON: {assistant_msg['content'][:100]}...")
            return None
            
        # Extract confidence_threshold from user content
        confidence_threshold = extract_confidence_threshold(user_msg['content'])
        
        # Build the new format
        result = {
            "messages": messages,
            "targets": targets
        }
        
        # Only add confidence_threshold if it exists
        if confidence_threshold is not None:
            result["confidence_threshold"] = confidence_threshold
            
        return result
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error processing line: {e}")
        return None

def main():
    input_file = "training_data.jsonl"
    output_file = "rft_training_data.jsonl"
    
    processed_count = 0
    error_count = 0
    
    print(f"Processing {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                if not line.strip():
                    continue
                    
                result = process_training_example(line)
                if result:
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    processed_count += 1
                else:
                    error_count += 1
                    
                if line_num % 100 == 0:
                    print(f"Processed {line_num} lines...")
                    
    except FileNotFoundError:
        print(f"Error: {input_file} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    print(f"\nCompleted!")
    print(f"Successfully processed: {processed_count} examples")
    print(f"Errors/skipped: {error_count} examples")
    print(f"Output written to: {output_file}")

if __name__ == "__main__":
    main()
