import json

# The schema information to append
SCHEMA_APPENDIX = """

OUTPUT FORMAT: You must respond with valid JSON only. Use the appropriate schema based on the task:

## 1. __Chapter Selection__ (`select_chapters`)

```json
{
  "chapters": [
    {"chapter": "XX", "confidence": 0.XX, "reasoning": "brief explanation"},
    {"chapter": "XX", "confidence": 0.XX, "reasoning": "brief explanation"}
  ]
}
```

## 2. __Candidate Selection__ (`select_candidates`)

```json
{
  "selected_indices": [number1, number2, number3],
  "reasoning": "Brief explanation of selection criteria"
}
```

## 3. __Candidate Scoring__ (`score_candidate`)

```json
{
  "option_number": 1,
  "confidence": 0.85,
  "reasoning": "Brief explanation of the score"
}
```

## 4. __Question Generation__ (`generate_question`)

```json
{
  "question_type": "multiple_choice",
  "question_text": "Balanced question incorporating product context and classification distinction",
  "options": [
    {"text": "Option 1", "value": "value1"},
    {"text": "Option 2", "value": "value2"}
  ]
}
```

## 5. __Answer Processing__ (`process_answer`)

```json
{
  "updated_description": "Enhanced product description",
  "extracted_attributes": {
    "attribute1": "value1",
    "attribute2": "value2"
  }
}
```"""

# Read the input file and process each line
input_file = "rft_training_data.jsonl"
output_file = "rft_training_data_with_schema.jsonl"

processed_count = 0

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
            
        # Parse the JSON object
        data = json.loads(line)
        
        # Update the system message (first message) by appending schema info
        if data["messages"][0]["role"] == "system":
            data["messages"][0]["content"] += SCHEMA_APPENDIX
        
        # Write the modified JSON object to output file
        outfile.write(json.dumps(data) + '\n')
        processed_count += 1

print(f"Processed {processed_count} training examples")
print(f"Updated system messages with JSON schema information")
print(f"Output written to: {output_file}")
