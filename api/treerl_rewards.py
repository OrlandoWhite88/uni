"""
TreeRL Reward Computation

Computes fractional leaf rewards based on longest-prefix match between
predicted trace and gold trace.

Reward formula:
    r_leaf = matched_prefix_len / len(gold_trace)

Where:
- matched_prefix_len is the number of trace steps that match before first mismatch
- Both chapter steps, code steps, AND group steps contribute equally to the reward
- A mismatch at any step (including groups) stops further credit

This gives incremental credit for:
- correct chapter (e.g., "87")
- correct heading (e.g., "8703")  
- correct intermediate group nodes (by node_id)
- progressively more specific codes
"""

from typing import Dict, List, Any, Optional
from .treerl_gold_trace import normalize_code


def steps_match(pred_step: Dict[str, Any], gold_step: Dict[str, Any]) -> bool:
    """
    Check if a predicted step matches a gold step.
    
    Matching rules:
    - chapter: pred.code == gold.code
    - code: pred.code == gold.code (both normalized)
    - group: pred.node_id == gold.node_id (must be exact same group node)
    """
    pred_kind = pred_step.get("kind", "")
    gold_kind = gold_step.get("kind", "")
    
    # Kinds must match
    if pred_kind != gold_kind:
        return False
    
    if pred_kind == "chapter":
        return pred_step.get("code", "") == gold_step.get("code", "")
    
    elif pred_kind == "code":
        return pred_step.get("code", "") == gold_step.get("code", "")
    
    elif pred_kind == "group":
        # Groups must match by node_id (same exact group in tree)
        return pred_step.get("node_id") == gold_step.get("node_id")
    
    return False


def compute_prefix_match_length(pred_trace: List[Dict[str, Any]], 
                                 gold_trace: List[Dict[str, Any]]) -> int:
    """
    Compute the length of the longest matching prefix between pred and gold traces.
    
    Stops at the first mismatch. Groups count as steps.
    
    Args:
        pred_trace: Predicted trace from classification
        gold_trace: Gold trace built from gold_code
        
    Returns:
        Number of consecutive matching steps from the start
    """
    matched = 0
    
    for i in range(min(len(pred_trace), len(gold_trace))):
        if steps_match(pred_trace[i], gold_trace[i]):
            matched += 1
        else:
            break
    
    return matched


def compute_leaf_reward(pred_trace: List[Dict[str, Any]], 
                        gold_trace: List[Dict[str, Any]]) -> float:
    """
    Compute the fractional leaf reward for a prediction.
    
    Formula: r_leaf = matched_prefix_len / len(gold_trace)
    
    Args:
        pred_trace: Predicted trace from classification path
        gold_trace: Gold trace built from gold_code
        
    Returns:
        Reward in [0, 1]. 1.0 means perfect match, 0.0 means no match.
    """
    if not gold_trace:
        return 0.0
    
    matched = compute_prefix_match_length(pred_trace, gold_trace)
    return matched / len(gold_trace)


def compute_leaf_reward_from_path(classification_path: List[Dict[str, Any]],
                                   gold_trace: List[Dict[str, Any]]) -> float:
    """
    Convenience function to compute reward directly from classification_path.
    
    Args:
        classification_path: The classification_path list from a ClassificationPath object
        gold_trace: Gold trace built from gold_code
        
    Returns:
        Fractional reward in [0, 1]
    """
    from .treerl_gold_trace import build_pred_trace_from_path
    pred_trace = build_pred_trace_from_path(classification_path)
    return compute_leaf_reward(pred_trace, gold_trace)


def reward_breakdown(pred_trace: List[Dict[str, Any]], 
                     gold_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detailed breakdown of reward computation for debugging/logging.
    
    Returns:
        Dict with matched_steps, total_gold_steps, reward, and per-step match info
    """
    matched = 0
    step_matches = []
    
    for i in range(min(len(pred_trace), len(gold_trace))):
        is_match = steps_match(pred_trace[i], gold_trace[i])
        step_matches.append({
            "index": i,
            "pred": pred_trace[i],
            "gold": gold_trace[i],
            "matched": is_match
        })
        if is_match:
            matched += 1
        else:
            break
    
    # Mark remaining gold steps as unmatched
    for i in range(len(step_matches), len(gold_trace)):
        step_matches.append({
            "index": i,
            "pred": None,
            "gold": gold_trace[i],
            "matched": False
        })
    
    reward = matched / len(gold_trace) if gold_trace else 0.0
    
    return {
        "matched_steps": matched,
        "total_gold_steps": len(gold_trace),
        "total_pred_steps": len(pred_trace),
        "reward": reward,
        "step_matches": step_matches
    }

