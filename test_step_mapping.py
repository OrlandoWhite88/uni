#!/usr/bin/env python3
"""
Test script for robust step-to-turn mapping in TreeRL training.

This verifies that the new mapping functions correctly:
1. Parse assistant JSON to extract selected code/node_id
2. Match against pred_trace to find the correct step index
3. Handle various edge cases (groups, chapters, missing data)
"""

import json
import sys
import re
from typing import Dict, List, Any, Optional, Tuple

# Import the mapping functions we're testing
# We'll copy them here for standalone testing, but in production they're in nemotron_train.py


def _normalize_code(code: str) -> str:
    """Strip dots and non-digit characters from an HTS code."""
    if not code:
        return ""
    return re.sub(r"\D", "", code)


def _extract_selected_identifier_from_turn(
    user_msg: str,
    assistant_msg: str
) -> Optional[Dict[str, Any]]:
    """
    Extract the selected code/node_id from an assistant turn by parsing JSON.
    """
    try:
        assistant_json = json.loads(assistant_msg)
    except (json.JSONDecodeError, TypeError):
        return None
    
    # SELECT_CHAPTERS: uses top_selection with chapter code directly
    if "top_selection" in assistant_json:
        chapter_code = assistant_json.get("top_selection", "")
        if chapter_code:
            return {
                "kind": "chapter",
                "code": _normalize_code(str(chapter_code))
            }
        return None
    
    # RANK_CANDIDATES: uses option_index to lookup from user message
    if "primary_selection" in assistant_json:
        primary = assistant_json["primary_selection"]
        if not isinstance(primary, dict):
            return None
        option_index = primary.get("option_index")
        if option_index is None:
            return None
        
        try:
            option_index = int(option_index)
        except (ValueError, TypeError):
            return None
        
        # Parse user message to get classification_tree.children
        try:
            json_start = user_msg.find("JSON INPUT:")
            if json_start == -1:
                return None
            json_str = user_msg[json_start + len("JSON INPUT:"):].strip()
            user_json = json.loads(json_str)
            
            children = user_json.get("data", {}).get("classification_tree", {}).get("children", [])
            if not children or not (1 <= option_index <= len(children)):
                return None
            
            selected_child = children[option_index - 1]
            
            if selected_child.get("is_group"):
                return {
                    "kind": "group",
                    "node_id": selected_child.get("node_id")
                }
            else:
                return {
                    "kind": "code",
                    "code": _normalize_code(selected_child.get("code", ""))
                }
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
    
    return None


def _match_identifier_to_trace(
    identifier: Dict[str, Any],
    pred_trace: List[Dict[str, Any]]
) -> Optional[int]:
    """Find the step index in pred_trace that matches the given identifier."""
    if not identifier or not pred_trace:
        return None
    
    kind = identifier.get("kind")
    
    for step_idx, step in enumerate(pred_trace):
        step_kind = step.get("kind", "")
        
        if kind == "group":
            if step_kind == "group" and step.get("node_id") == identifier.get("node_id"):
                return step_idx
        elif kind in ("chapter", "code"):
            if step_kind in ("chapter", "code"):
                if step.get("code", "") == identifier.get("code", ""):
                    return step_idx
    
    return None


def _build_assistant_step_mapping(
    messages: List[Dict[str, str]],
    pred_trace: List[Dict[str, Any]]
) -> List[int]:
    """Build a mapping from assistant turn index to step index in pred_trace."""
    assistant_step_map = []
    last_user_msg = ""
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            last_user_msg = content
        elif role == "assistant":
            identifier = _extract_selected_identifier_from_turn(last_user_msg, content)
            
            if identifier:
                step_idx = _match_identifier_to_trace(identifier, pred_trace)
                assistant_step_map.append(step_idx if step_idx is not None else -1)
            else:
                assistant_step_map.append(-1)
    
    return assistant_step_map


# ============================================================================
# Test Cases
# ============================================================================

def test_select_chapters():
    """Test chapter selection mapping."""
    print("\n=== Test 1: select_chapters ===")
    
    # User message (simplified - just needs task type)
    user_msg = "TASK: select_chapters\n\nJSON INPUT:\n{}"
    
    # Assistant response with top_selection
    assistant_msg = json.dumps({
        "thinking": "The product is a motor vehicle...",
        "top_selection": "87",
        "chapters": [
            {"chapter": "87", "information_context_score": 0.95, "path_score": 0.98}
        ]
    })
    
    # pred_trace for this path
    pred_trace = [
        {"kind": "chapter", "code": "87", "node_id": 123},
        {"kind": "code", "code": "8703", "node_id": 28689},
        {"kind": "code", "code": "87039001", "node_id": 28805}
    ]
    
    identifier = _extract_selected_identifier_from_turn(user_msg, assistant_msg)
    print(f"  Extracted identifier: {identifier}")
    
    assert identifier is not None, "Should extract identifier"
    assert identifier["kind"] == "chapter", f"Expected chapter, got {identifier['kind']}"
    assert identifier["code"] == "87", f"Expected '87', got {identifier['code']}"
    
    step_idx = _match_identifier_to_trace(identifier, pred_trace)
    print(f"  Matched to step index: {step_idx}")
    
    assert step_idx == 0, f"Expected step 0 (chapter), got {step_idx}"
    print("  ✅ PASSED")


def test_rank_candidates_code():
    """Test rank_candidates with regular code selection."""
    print("\n=== Test 2: rank_candidates (code selection) ===")
    
    # User message with classification_tree children
    user_json = {
        "task": "rank_candidates",
        "data": {
            "product_text": "audi r8 v10",
            "path_so_far": "87 - Vehicles...",
            "classification_tree": {
                "children": [
                    {"index": 1, "code": "8701", "description": "Tractors", "is_group": False, "node_id": 28612},
                    {"index": 2, "code": "8702", "description": "Motor vehicles for 10+ persons", "is_group": False, "node_id": 28673},
                    {"index": 3, "code": "8703", "description": "Motor cars...", "is_group": False, "node_id": 28689},
                    {"index": 4, "code": "8704", "description": "Motor vehicles for goods", "is_group": False, "node_id": 28806}
                ]
            }
        }
    }
    user_msg = f"TASK: rank_candidates\n\nJSON INPUT:\n{json.dumps(user_json, indent=2)}"
    
    # Assistant selects option_index 3 (8703)
    assistant_msg = json.dumps({
        "primary_selection": {
            "option_index": 3,
            "code": "8703",
            "description": "Motor cars...",
            "information_context_score": 0.95,
            "path_score": 0.95
        },
        "alternative_1": {"option_index": 2},
        "alternative_2": {"option_index": 4}
    })
    
    pred_trace = [
        {"kind": "chapter", "code": "87", "node_id": 123},
        {"kind": "code", "code": "8703", "node_id": 28689},
        {"kind": "code", "code": "87039001", "node_id": 28805}
    ]
    
    identifier = _extract_selected_identifier_from_turn(user_msg, assistant_msg)
    print(f"  Extracted identifier: {identifier}")
    
    assert identifier is not None, "Should extract identifier"
    assert identifier["kind"] == "code", f"Expected code, got {identifier['kind']}"
    assert identifier["code"] == "8703", f"Expected '8703', got {identifier['code']}"
    
    step_idx = _match_identifier_to_trace(identifier, pred_trace)
    print(f"  Matched to step index: {step_idx}")
    
    assert step_idx == 1, f"Expected step 1 (8703), got {step_idx}"
    print("  ✅ PASSED")


def test_rank_candidates_group():
    """Test rank_candidates with GROUP node selection."""
    print("\n=== Test 3: rank_candidates (GROUP selection) ===")
    
    # User message with GROUP nodes
    user_json = {
        "task": "rank_candidates",
        "data": {
            "product_text": "Audi R8 V10, diesel engine",
            "path_so_far": "87 > 8703",
            "classification_tree": {
                "children": [
                    {"index": 1, "code": "8703.10", "description": "Snow vehicles", "is_group": False, "node_id": 28690},
                    {"index": 2, "code": "[GROUP]", "description": "Other vehicles, with spark-ignition", "is_group": True, "node_id": 28695},
                    {"index": 3, "code": "[GROUP]", "description": "Other vehicles, with compression-ignition (diesel)", "is_group": True, "node_id": 28725},
                    {"index": 4, "code": "8703.40.00", "description": "Hybrid vehicles", "is_group": False, "node_id": 28736}
                ]
            }
        }
    }
    user_msg = f"TASK: rank_candidates\n\nJSON INPUT:\n{json.dumps(user_json, indent=2)}"
    
    # Assistant selects option_index 3 (diesel GROUP)
    assistant_msg = json.dumps({
        "primary_selection": {
            "option_index": 3,
            "code": "[GROUP]",
            "description": "Other vehicles, with compression-ignition (diesel)",
            "information_context_score": 0.95,
            "path_score": 0.95
        }
    })
    
    pred_trace = [
        {"kind": "chapter", "code": "87", "node_id": 123},
        {"kind": "code", "code": "8703", "node_id": 28689},
        {"kind": "group", "node_id": 28725, "description": "Other vehicles, with compression-ignition"},
        {"kind": "code", "code": "87033190", "node_id": 28730}
    ]
    
    identifier = _extract_selected_identifier_from_turn(user_msg, assistant_msg)
    print(f"  Extracted identifier: {identifier}")
    
    assert identifier is not None, "Should extract identifier"
    assert identifier["kind"] == "group", f"Expected group, got {identifier['kind']}"
    assert identifier["node_id"] == 28725, f"Expected node_id 28725, got {identifier['node_id']}"
    
    step_idx = _match_identifier_to_trace(identifier, pred_trace)
    print(f"  Matched to step index: {step_idx}")
    
    assert step_idx == 2, f"Expected step 2 (diesel group), got {step_idx}"
    print("  ✅ PASSED")


def test_full_trajectory():
    """Test mapping a full multi-turn trajectory."""
    print("\n=== Test 4: Full trajectory mapping ===")
    
    # Construct a realistic trajectory
    messages = [
        {"role": "system", "content": "You are a customs classification expert."},
        
        # Turn 1: Chapter selection
        {"role": "user", "content": "TASK: select_chapters\n\nJSON INPUT:\n{}"},
        {"role": "assistant", "content": json.dumps({
            "top_selection": "87",
            "chapters": [{"chapter": "87", "path_score": 0.98}]
        })},
        
        # Turn 2: Heading selection (8703)
        {"role": "user", "content": f"TASK: rank_candidates\n\nJSON INPUT:\n{json.dumps({'task': 'rank_candidates', 'data': {'classification_tree': {'children': [{'index': 1, 'code': '8701', 'is_group': False, 'node_id': 28612}, {'index': 2, 'code': '8702', 'is_group': False, 'node_id': 28673}, {'index': 3, 'code': '8703', 'is_group': False, 'node_id': 28689}]}}})}"},
        {"role": "assistant", "content": json.dumps({
            "primary_selection": {"option_index": 3, "code": "8703"}
        })},
        
        # Turn 3: Subheading selection (GROUP - diesel)
        {"role": "user", "content": f"TASK: rank_candidates\n\nJSON INPUT:\n{json.dumps({'task': 'rank_candidates', 'data': {'classification_tree': {'children': [{'index': 1, 'code': '8703.10', 'is_group': False, 'node_id': 28690}, {'index': 2, 'code': '[GROUP]', 'is_group': True, 'node_id': 28695}, {'index': 3, 'code': '[GROUP]', 'is_group': True, 'node_id': 28725}]}}})}"},
        {"role": "assistant", "content": json.dumps({
            "primary_selection": {"option_index": 3, "code": "[GROUP]"}
        })},
        
        # Turn 4: Final code selection
        {"role": "user", "content": f"TASK: rank_candidates\n\nJSON INPUT:\n{json.dumps({'task': 'rank_candidates', 'data': {'classification_tree': {'children': [{'index': 1, 'code': '8703.31.00', 'is_group': False, 'node_id': 28726}, {'index': 2, 'code': '8703.32.00', 'is_group': False, 'node_id': 28728}, {'index': 3, 'code': '8703.33.00', 'is_group': False, 'node_id': 28730}]}}})}"},
        {"role": "assistant", "content": json.dumps({
            "primary_selection": {"option_index": 2, "code": "8703.32.00"}
        })},
    ]
    
    pred_trace = [
        {"kind": "chapter", "code": "87", "node_id": 123},
        {"kind": "code", "code": "8703", "node_id": 28689},
        {"kind": "group", "node_id": 28725, "description": "diesel group"},
        {"kind": "code", "code": "87033200", "node_id": 28728}
    ]
    
    mapping = _build_assistant_step_mapping(messages, pred_trace)
    print(f"  Mapping: {mapping}")
    
    # Expected: [0, 1, 2, 3] - each assistant turn maps to its corresponding step
    assert len(mapping) == 4, f"Expected 4 mappings, got {len(mapping)}"
    assert mapping[0] == 0, f"Turn 0 should map to step 0 (chapter 87), got {mapping[0]}"
    assert mapping[1] == 1, f"Turn 1 should map to step 1 (8703), got {mapping[1]}"
    assert mapping[2] == 2, f"Turn 2 should map to step 2 (diesel group), got {mapping[2]}"
    assert mapping[3] == 3, f"Turn 3 should map to step 3 (8703.32.00), got {mapping[3]}"
    
    print("  ✅ PASSED")


def test_edge_case_malformed_json():
    """Test handling of malformed assistant JSON."""
    print("\n=== Test 5: Malformed JSON handling ===")
    
    user_msg = "TASK: rank_candidates\n\nJSON INPUT:\n{}"
    assistant_msg = "This is not valid JSON { broken"
    
    identifier = _extract_selected_identifier_from_turn(user_msg, assistant_msg)
    print(f"  Result for malformed JSON: {identifier}")
    
    assert identifier is None, "Should return None for malformed JSON"
    print("  ✅ PASSED")


def test_edge_case_missing_option_index():
    """Test handling of missing option_index."""
    print("\n=== Test 6: Missing option_index handling ===")
    
    user_msg = "TASK: rank_candidates\n\nJSON INPUT:\n{}"
    assistant_msg = json.dumps({
        "primary_selection": {
            "code": "8703",  # No option_index!
            "path_score": 0.95
        }
    })
    
    identifier = _extract_selected_identifier_from_turn(user_msg, assistant_msg)
    print(f"  Result for missing option_index: {identifier}")
    
    assert identifier is None, "Should return None when option_index is missing"
    print("  ✅ PASSED")


def test_edge_case_invalid_option_index():
    """Test handling of out-of-bounds option_index."""
    print("\n=== Test 7: Invalid option_index handling ===")
    
    user_json = {
        "task": "rank_candidates",
        "data": {
            "classification_tree": {
                "children": [
                    {"index": 1, "code": "8701", "is_group": False, "node_id": 28612},
                    {"index": 2, "code": "8702", "is_group": False, "node_id": 28673}
                ]
            }
        }
    }
    user_msg = f"TASK: rank_candidates\n\nJSON INPUT:\n{json.dumps(user_json)}"
    
    # option_index 5 is out of bounds (only 2 children)
    assistant_msg = json.dumps({
        "primary_selection": {"option_index": 5, "code": "invalid"}
    })
    
    identifier = _extract_selected_identifier_from_turn(user_msg, assistant_msg)
    print(f"  Result for invalid option_index: {identifier}")
    
    assert identifier is None, "Should return None for out-of-bounds option_index"
    print("  ✅ PASSED")


def test_normalize_code():
    """Test code normalization."""
    print("\n=== Test 8: Code normalization ===")
    
    test_cases = [
        ("8703.32.00", "87033200"),
        ("87", "87"),
        ("8703", "8703"),
        ("8703.90.01.00", "8703900100"),
        ("[GROUP]", ""),
        ("", ""),
        (None, ""),
    ]
    
    for input_code, expected in test_cases:
        result = _normalize_code(input_code or "")
        print(f"  '{input_code}' -> '{result}' (expected: '{expected}')")
        assert result == expected, f"Expected '{expected}', got '{result}'"
    
    print("  ✅ PASSED")


def test_audi_r8_from_user_data():
    """Test with actual data from user's trajectory file."""
    print("\n=== Test 9: Audi R8 real data ===")
    
    # This matches the user's actual trajectory_audi r8 v10_rank2_20251221_224727.json
    
    # Chapter selection response (lines 28-31)
    chapter_assistant = json.dumps({
        "thinking": "The product description 'audi r8 v10' unambiguously identifies a motor vehicle...",
        "top_selection": "87",
        "chapters": [
            {"chapter": "87", "information_context_score": 0.95, "path_score": 0.98}
        ]
    })
    
    # Heading selection (lines 40-47) - user provides 16 candidates, assistant picks option 3
    heading_user_children = [
        {"index": 1, "code": "8701", "description": "Tractors", "is_group": False, "node_id": 28612},
        {"index": 2, "code": "8702", "description": "Motor vehicles for 10+ persons", "is_group": False, "node_id": 28673},
        {"index": 3, "code": "8703", "description": "Motor cars...", "is_group": False, "node_id": 28689},
    ]
    heading_user = f"TASK: rank_candidates\n\nJSON INPUT:\n{json.dumps({'data': {'classification_tree': {'children': heading_user_children}}})}"
    heading_assistant = json.dumps({
        "primary_selection": {
            "option_index": 3,
            "code": "8703",
            "description": "Motor cars and other motor vehicles...",
            "information_context_score": 0.95,
            "path_score": 0.95
        }
    })
    
    # Subheading GROUP selection (lines 64-71) - assistant picks option 3 which is a GROUP
    subheading_user_children = [
        {"index": 1, "code": "8703.10", "description": "Snow vehicles", "is_group": False, "node_id": 28690},
        {"index": 2, "code": "[GROUP]", "description": "spark-ignition", "is_group": True, "node_id": 28695},
        {"index": 3, "code": "[GROUP]", "description": "compression-ignition (diesel)", "is_group": True, "node_id": 28725},
    ]
    subheading_user = f"TASK: rank_candidates\n\nJSON INPUT:\n{json.dumps({'data': {'classification_tree': {'children': subheading_user_children}}})}"
    subheading_assistant = json.dumps({
        "primary_selection": {
            "option_index": 3,
            "code": "[GROUP]",
            "description": "Other vehicles, with compression-ignition (diesel)",
            "information_context_score": 0.95,
            "path_score": 0.95
        }
    })
    
    # Test 1: Chapter selection
    identifier = _extract_selected_identifier_from_turn("TASK: select_chapters\n\nJSON INPUT:\n{}", chapter_assistant)
    assert identifier == {"kind": "chapter", "code": "87"}, f"Chapter: {identifier}"
    print("  Chapter selection: ✅")
    
    # Test 2: Heading selection
    identifier = _extract_selected_identifier_from_turn(heading_user, heading_assistant)
    assert identifier == {"kind": "code", "code": "8703"}, f"Heading: {identifier}"
    print("  Heading selection (8703): ✅")
    
    # Test 3: GROUP selection - this is the critical test!
    identifier = _extract_selected_identifier_from_turn(subheading_user, subheading_assistant)
    assert identifier == {"kind": "group", "node_id": 28725}, f"GROUP: {identifier}"
    print("  GROUP selection (diesel node_id=28725): ✅")
    
    print("  ✅ ALL PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("TreeRL Robust Step-to-Turn Mapping Tests")
    print("=" * 60)
    
    try:
        test_select_chapters()
        test_rank_candidates_code()
        test_rank_candidates_group()
        test_full_trajectory()
        test_edge_case_malformed_json()
        test_edge_case_missing_option_index()
        test_edge_case_invalid_option_index()
        test_normalize_code()
        test_audi_r8_from_user_data()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

