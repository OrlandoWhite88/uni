import json

# Check first few lines of the output file
with open("rft_training_data_with_system.jsonl", 'r') as f:
    for i, line in enumerate(f):
        if i >= 2:  # Check first 2 examples
            break
            
        data = json.loads(line.strip())
        messages = data["messages"]
        
        print(f"=== Example {i+1} ===")
        print(f"Total messages: {len(messages)}")
        print(f"First message role: {messages[0]['role']}")
        print(f"First message content (first 100 chars): {messages[0]['content'][:100]}...")
        print(f"Second message role: {messages[1]['role']}")
        print(f"Second message content (first 100 chars): {messages[1]['content'][:100]}...")
        print()
