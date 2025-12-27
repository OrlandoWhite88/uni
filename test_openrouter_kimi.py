#!/usr/bin/env python3
"""
Test OpenRouter connection with Kimi K2 Thinking model
"""
import os
import json
import requests

# Set the model
os.environ["OPENROUTER_MODEL"] = "moonshotai/kimi-k2-thinking"
# Use "ignore" instead of "deny" per OpenRouter docs
os.environ["OPENROUTER_PROVIDER_IGNORE"] = "DeepInfra"

def test_openrouter_direct():
    """Test OpenRouter API directly"""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not set")
        return
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    payload = {
        "model": "moonshotai/kimi-k2-thinking",
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer in JSON format: {\"answer\": \"...\"}"}
        ],
        "temperature": 0.0,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},
        "extra_body": {"reasoning": {"enabled": True}}
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/hscode",
        "X-Title": "HSCode Test"
    }
    
    print("Testing OpenRouter with Kimi K2 Thinking...")
    print(f"Model: {payload['model']}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:1000]}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✓ Success!")
            print(f"Model used: {data.get('model')}")
            if 'choices' in data and len(data['choices']) > 0:
                message = data['choices'][0].get('message', {})
                print(f"Content: {message.get('content', 'N/A')[:200]}")
                print(f"Reasoning: {message.get('reasoning', 'N/A')}")
        else:
            print("\n❌ Failed!")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")

def test_via_llm_client():
    """Test via our LLM client"""
    from api.llm_client import LLMClient
    
    print("\n" + "="*60)
    print("Testing via LLMClient...")
    print("="*60)
    
    client = LLMClient()
    client.default_provider = "openrouter"
    
    try:
        response = client.send_openai_request(
            prompt='What is 2+2? Answer in JSON format: {"answer": "..."}',
            requires_json=True,
            temperature=0.0,
            provider_override="openrouter"
        )
        print(f"\n✓ Success via LLMClient!")
        print(f"Response: {response}")
    except Exception as e:
        print(f"\n❌ Failed via LLMClient: {e}")

if __name__ == "__main__":
    test_openrouter_direct()
    test_via_llm_client()

