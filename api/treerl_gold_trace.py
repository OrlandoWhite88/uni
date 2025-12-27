"""
TreeRL Gold Trace Builder

Builds the gold trace (including group node IDs) by:
1. Resolving gold_code to the exact HTS node in the tree
2. Walking from that node to root via parent pointers
3. Prepending the chapter step

The gold trace is used for computing prefix-match rewards.
"""

import re
from typing import Dict, List, Any, Optional
from .models import HTSNode


def normalize_code(code: str) -> str:
    """Strip dots and non-digit characters from an HTS code."""
    if not code:
        return ""
    return re.sub(r"\D", "", code)


def build_trace_step(node: HTSNode, is_chapter: bool = False) -> Dict[str, Any]:
    """
    Build a single trace step from an HTSNode.
    
    Returns:
        Dict with:
        - kind: "chapter", "group", or "code"
        - code: digits-only code (for chapter/code kinds)
        - node_id: the node's ID (for all kinds, but especially important for groups)
    """
    if is_chapter:
        return {
            "kind": "chapter",
            "code": normalize_code(node.htsno) if node.htsno else "",
            "node_id": node.node_id
        }
    
    if node.is_group_node():
        return {
            "kind": "group",
            "node_id": node.node_id,
            "description": node.description[:80] if node.description else ""
        }
    else:
        return {
            "kind": "code",
            "code": normalize_code(node.htsno),
            "node_id": node.node_id
        }


def build_gold_trace(gold_code: str, tree_navigator) -> List[Dict[str, Any]]:
    """
    Build the gold trace by finding the gold node and walking to root.
    
    Args:
        gold_code: The expected/gold HTS code (e.g., "8703.33.01.45")
        tree_navigator: TreeNavigator instance with code_index and node_index
        
    Returns:
        List of trace steps from chapter to terminal node, including any
        intermediate group nodes. Each step has kind, code (if applicable),
        and node_id.
        
    Example output for "8703.33.01.45":
        [
            {"kind": "chapter", "code": "87", "node_id": -1},
            {"kind": "code", "code": "8703", "node_id": 12345},
            {"kind": "group", "node_id": 54321, "description": "Vehicles with..."},
            {"kind": "code", "code": "87033301", "node_id": 67890},
            {"kind": "group", "node_id": 11111, "description": "New:"},
            {"kind": "code", "code": "8703330145", "node_id": 22222}
        ]
    """
    normalized_gold = normalize_code(gold_code)
    if not normalized_gold:
        return []
    
    # Try to find the exact gold node
    gold_node = None
    
    # First try exact match in code_index
    if gold_code in tree_navigator.code_index:
        gold_node = tree_navigator.code_index[gold_code]
    else:
        # Try normalized version or with different dot patterns
        for stored_code, node in tree_navigator.code_index.items():
            if normalize_code(stored_code) == normalized_gold:
                gold_node = node
                break
    
    if gold_node is None:
        # Fallback: find the longest prefix match
        best_match = None
        best_match_len = 0
        for stored_code, node in tree_navigator.code_index.items():
            stored_normalized = normalize_code(stored_code)
            if normalized_gold.startswith(stored_normalized) and len(stored_normalized) > best_match_len:
                best_match = node
                best_match_len = len(stored_normalized)
        gold_node = best_match
    
    if gold_node is None:
        # Can't find gold node at all - return just chapter
        chapter = normalized_gold[:2] if len(normalized_gold) >= 2 else normalized_gold
        return [{"kind": "chapter", "code": chapter, "node_id": -1}]
    
    # Walk from gold_node to root, collecting the path
    path_to_root = []
    current = gold_node
    
    while current is not None and current.node_id != 0:  # Stop at ROOT (node_id=0)
        path_to_root.append(current)
        current = current.parent
    
    # Reverse to get root-to-leaf order
    path_to_root.reverse()
    
    # Build trace steps
    trace = []
    chapter_added = False
    chapter_code = normalized_gold[:2]
    
    for i, node in enumerate(path_to_root):
        # Add chapter step at the start (before first real node)
        if not chapter_added:
            trace.append({
                "kind": "chapter",
                "code": chapter_code,
                "node_id": -1  # Virtual chapter node
            })
            chapter_added = True
        
        # Add the actual node step
        if node.is_group_node():
            trace.append({
                "kind": "group",
                "node_id": node.node_id,
                "description": node.description[:80] if node.description else ""
            })
        else:
            trace.append({
                "kind": "code",
                "code": normalize_code(node.htsno),
                "node_id": node.node_id
            })
    
    # If we didn't add anything (empty path), at least add chapter
    if not trace:
        trace.append({
            "kind": "chapter",
            "code": chapter_code,
            "node_id": -1
        })
    
    return trace


def build_pred_trace_from_path(classification_path: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a prediction trace from a ClassificationPath's classification_path list.
    
    Args:
        classification_path: List of step dicts from ClassificationPath.classification_path
            Each step has: type, code, description, is_group, node_id, confidence, etc.
            
    Returns:
        List of trace steps in the same format as build_gold_trace output.
    """
    trace = []
    
    for step in classification_path:
        step_type = step.get("type", "")
        code = step.get("code", "")
        node_id = step.get("node_id")
        is_group = step.get("is_group", False)
        
        if step_type == "chapter":
            trace.append({
                "kind": "chapter",
                "code": normalize_code(code),
                "node_id": node_id if node_id is not None else -1
            })
        elif is_group or code == "[GROUP]":
            trace.append({
                "kind": "group",
                "node_id": node_id,
                "description": step.get("description", "")[:80]
            })
        else:
            trace.append({
                "kind": "code",
                "code": normalize_code(code),
                "node_id": node_id
            })
    
    return trace


def trace_to_key(trace: List[Dict[str, Any]]) -> tuple:
    """
    Convert a trace to a hashable key for use in prefix tree building.
    
    For chapters and codes, uses the code.
    For groups, uses the node_id (since groups don't have codes).
    """
    key_parts = []
    for step in trace:
        if step["kind"] == "group":
            key_parts.append(("group", step["node_id"]))
        else:
            key_parts.append((step["kind"], step.get("code", "")))
    return tuple(key_parts)

