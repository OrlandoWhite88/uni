#!/bin/bash

# Test Suite for Session-Based HTS Classification API
# Run this to validate all endpoints are working correctly

BASE_URL="http://localhost:8000"
echo "🧪 Testing Session-Based HTS Classification API"
echo "================================================"

# Test 1: API Health Check
echo "1. API Health Check"
echo "-------------------"
curl -s "$BASE_URL/" | python3 -m json.tool
echo -e "\n"

# Test 2: Session Statistics (Initial)
echo "2. Initial Session Statistics"
echo "-----------------------------"
curl -s "$BASE_URL/sessions/stats" | python3 -m json.tool
echo -e "\n"

# Test 3: Create Session with Groq Engine (Working)
echo "3. Create Session (Groq Engine)"
echo "-------------------------------"
SESSION_RESPONSE=$(curl -s -X POST "$BASE_URL/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "product": "laptop computer", 
    "interactive": true, 
    "max_questions": 2,
    "engine": "groq",
    "use_multi_hypothesis": true,
    "hypothesis_count": 3
  }')

echo "$SESSION_RESPONSE" | python3 -m json.tool
SESSION_ID=$(echo "$SESSION_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['session_id'])")
echo "📝 Session ID: $SESSION_ID"
echo -e "\n"

# Test 4: Create Session with Vertex Engine (May Fail)
echo "4. Create Session (Vertex Engine - May Fail)"
echo "--------------------------------------------"
VERTEX_SESSION_RESPONSE=$(curl -s -X POST "$BASE_URL/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "product": "steel knife", 
    "interactive": false, 
    "max_questions": 0,
    "engine": "vertex",
    "use_multi_hypothesis": true,
    "hypothesis_count": 3
  }')

echo "$VERTEX_SESSION_RESPONSE" | python3 -m json.tool
VERTEX_SESSION_ID=$(echo "$VERTEX_SESSION_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['session_id'])" 2>/dev/null || echo "FAILED")
echo "📝 Vertex Session ID: $VERTEX_SESSION_ID"
echo -e "\n"

# Test 5: Session Statistics (After Creation)
echo "5. Session Statistics (After Creation)"
echo "--------------------------------------"
curl -s "$BASE_URL/sessions/stats" | python3 -m json.tool
echo -e "\n"

# Test 6: Get Session Result (Before Running)
echo "6. Get Session Result (Before Classification)"
echo "---------------------------------------------"
curl -s "$BASE_URL/sessions/$SESSION_ID/result" | python3 -m json.tool
echo -e "\n"

# Test 7: Stream Classification (Non-Interactive)
echo "7. Test Classification Stream (10 seconds max)"
echo "----------------------------------------------"
echo "Starting classification stream for session: $SESSION_ID"
echo "Press Ctrl+C if it takes too long..."

# Create a non-interactive session for quick testing
QUICK_SESSION_RESPONSE=$(curl -s -X POST "$BASE_URL/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "product": "plastic water bottle", 
    "interactive": false, 
    "max_questions": 0,
    "engine": "groq",
    "use_multi_hypothesis": false,
    "hypothesis_count": 1
  }')

QUICK_SESSION_ID=$(echo "$QUICK_SESSION_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['session_id'])")
echo "📝 Quick Test Session ID: $QUICK_SESSION_ID"

# Stream for max 15 seconds
timeout 15s curl -N -s "$BASE_URL/sessions/$QUICK_SESSION_ID/stream" || echo -e "\n[Stream timed out or completed]"
echo -e "\n"

# Test 8: Get Final Session Result
echo "8. Get Final Session Result"
echo "---------------------------"
sleep 2  # Give it a moment to complete
curl -s "$BASE_URL/sessions/$QUICK_SESSION_ID/result" | python3 -m json.tool
echo -e "\n"

# Test 9: Delete Session
echo "9. Delete Session"
echo "----------------"
curl -s -X DELETE "$BASE_URL/sessions/$QUICK_SESSION_ID" | python3 -m json.tool
echo -e "\n"

# Test 10: Try to Get Deleted Session (Should Fail)
echo "10. Try to Access Deleted Session (Should Fail)"
echo "-----------------------------------------------"
curl -s "$BASE_URL/sessions/$QUICK_SESSION_ID/result" | python3 -m json.tool
echo -e "\n"

# Test 11: Final Session Statistics
echo "11. Final Session Statistics"
echo "---------------------------"
curl -s "$BASE_URL/sessions/stats" | python3 -m json.tool
echo -e "\n"

# Test 12: List Session Files on Disk
echo "12. Session Files on Disk"
echo "-------------------------"
ls -la sessions/ 2>/dev/null || echo "Sessions directory not accessible from here"
echo -e "\n"

echo "🎉 Test Suite Complete!"
echo "======================"
echo "Key Points:"
echo "• Use 'groq' engine for working classifications"
echo "• Use 'vertex' engine only if you have OPENAI_API_KEY set"
echo "• Sessions auto-expire after 30 minutes"
echo "• All state is managed server-side, clients only see session_id"
echo "• Streaming events contain no internal state"
