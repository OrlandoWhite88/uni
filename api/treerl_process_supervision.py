"""
TreeRL Process Supervision

Implements TreeRL's process supervision signals from the paper:
- V(s) = mean reward of all leaves that pass through node s
- GA(s) = V(s) - V(root) = Global Advantage
- LA(s) = V(s) - V(parent(s)) = Local Advantage  
- R(s) = (GA(s) + LA(s)) / sqrt(|L(s)|) = Final step reward with reweighting

The reweighting by sqrt(|L(s)|) prevents early steps (which appear in many leaves)
from dominating the gradient.

OUTPUT FORMAT (per the TreeRL paper):
- One sample per LEAF (complete trajectory)
- Each sample contains the full message trajectory
- Step rewards R(s) are bundled as an array within each sample
- Training loop processes full trajectories and applies R(s) per-step during backprop

This module:
1. Builds a prefix tree from collected leaves
2. Computes values for each prefix node
3. Computes advantages (GA/LA) and final rewards R(s)
4. Emits LEAF-level training samples with bundled step rewards
"""

import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from .treerl_gold_trace import trace_to_key, build_pred_trace_from_path


logger = logging.getLogger(__name__)


class PrefixTreeNode:
    """A node in the prefix tree for computing TreeRL values."""
    
    def __init__(self, key: tuple = ()):
        self.key = key  # The trace prefix as a tuple
        self.children: Dict[Any, 'PrefixTreeNode'] = {}
        self.leaf_rewards: List[float] = []  # Rewards of leaves at exactly this node
        self.leaf_ids: List[str] = []  # IDs of leaves at exactly this node
        
        # Computed values
        self.value: float = 0.0  # V(s) = mean reward of descendant leaves
        self.descendant_count: int = 0  # |L(s)|
        self.ga: float = 0.0  # Global advantage
        self.la: float = 0.0  # Local advantage
        self.R: float = 0.0  # Final step reward


def build_prefix_tree(
    leaves: List[Dict[str, Any]],
    reward_key: str = "reward",
) -> PrefixTreeNode:
    """
    Build a prefix tree from collected leaves.
    
    Args:
        leaves: List of leaf dicts, each with:
            - pred_trace: List of trace steps
            - reward: float leaf reward
            - path_id: str identifier
            
    Returns:
        Root node of the prefix tree
    """
    root = PrefixTreeNode(key=())
    
    for leaf in leaves:
        pred_trace = leaf.get("pred_trace", [])
        reward = leaf.get(reward_key)
        if reward is None:
            reward = leaf.get("reward", 0.0)
        path_id = leaf.get("path_id", "unknown")
        
        # Convert trace to key format
        trace_key = trace_to_key(pred_trace)
        
        # Insert into tree, creating intermediate nodes as needed
        current = root
        for i in range(len(trace_key)):
            prefix_key = trace_key[:i+1]
            step_key = trace_key[i]
            
            if step_key not in current.children:
                current.children[step_key] = PrefixTreeNode(key=prefix_key)
            
            current = current.children[step_key]
        
        # Mark this as a leaf node with its reward
        current.leaf_rewards.append(reward)
        current.leaf_ids.append(path_id)
    
    return root


def compute_tree_values(root: PrefixTreeNode) -> float:
    """
    Compute V(s) for all nodes via post-order traversal.
    
    V(s) = mean of all leaf rewards in subtree rooted at s
    
    Args:
        root: Root of the prefix tree
        
    Returns:
        V(root) for use in GA computation
    """
    def _compute_subtree(node: PrefixTreeNode) -> Tuple[float, int]:
        """Returns (sum of rewards, count of leaves) for subtree."""
        total_reward = sum(node.leaf_rewards)
        total_count = len(node.leaf_rewards)
        
        for child in node.children.values():
            child_reward, child_count = _compute_subtree(child)
            total_reward += child_reward
            total_count += child_count
        
        node.descendant_count = total_count
        node.value = total_reward / total_count if total_count > 0 else 0.0
        
        return total_reward, total_count
    
    _compute_subtree(root)
    return root.value


def compute_advantages(root: PrefixTreeNode):
    """
    Compute GA(s) and LA(s) for all nodes.
    
    GA(s) = V(s) - V(root)
    LA(s) = V(s) - V(parent(s))
    R(s) = (GA(s) + LA(s)) / sqrt(|L(s)|)
    
    Args:
        root: Root of prefix tree (must have values computed already)
    """
    v_root = root.value
    
    def _compute_node_advantages(node: PrefixTreeNode, parent_value: float):
        # Compute advantages
        node.ga = node.value - v_root
        node.la = node.value - parent_value
        
        # Compute final step reward with reweighting
        # The sqrt(|L(s)|) prevents early steps from dominating
        if node.descendant_count > 0:
            node.R = (node.ga + node.la) / math.sqrt(node.descendant_count)
        else:
            node.R = 0.0
        
        # Recurse to children
        for child in node.children.values():
            _compute_node_advantages(child, node.value)
    
    # Root has no parent, so LA(root) = 0, GA(root) = 0
    root.ga = 0.0
    root.la = 0.0
    root.R = 0.0
    
    # Process children with root as parent
    for child in root.children.values():
        _compute_node_advantages(child, root.value)


def collect_step_rewards(root: PrefixTreeNode) -> Dict[tuple, Dict[str, float]]:
    """
    Collect R(s) values for all prefixes in the tree.
    
    Returns:
        Dict mapping prefix key -> {value, ga, la, R, descendant_count}
    """
    rewards = {}
    
    def _collect(node: PrefixTreeNode):
        rewards[node.key] = {
            "value": node.value,
            "ga": node.ga,
            "la": node.la,
            "R": node.R,
            "descendant_count": node.descendant_count
        }
        for child in node.children.values():
            _collect(child)
    
    _collect(root)
    return rewards


def compute_treerl_rewards(
    leaves: List[Dict[str, Any]],
    reward_key: str = "reward",
) -> Tuple[Dict[tuple, Dict[str, float]], float]:
    """
    Main entry point: compute all TreeRL rewards from leaves.
    
    Args:
        leaves: List of leaf dicts with pred_trace, reward, path_id
        
    Returns:
        Tuple of:
        - Dict mapping prefix key -> reward info
        - V(root) baseline
    """
    if not leaves:
        return {}, 0.0
    
    # Build prefix tree
    root = build_prefix_tree(leaves, reward_key=reward_key)
    
    # Compute values
    v_root = compute_tree_values(root)
    
    # Compute advantages
    compute_advantages(root)
    
    # Collect all step rewards
    step_rewards = collect_step_rewards(root)
    
    return step_rewards, v_root


def emit_leaf_samples(
    leaves: List[Dict[str, Any]],
    step_rewards: Dict[tuple, Dict[str, float]],
    gold_trace: Optional[List] = None,
    gold_code: Optional[str] = None,
    ruling_index: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Emit LEAF-level training samples from leaves with bundled step rewards.
    
    This is the correct format per the TreeRL paper:
    - ONE sample per leaf (complete trajectory)
    - Full message trajectory included
    - Step rewards R(s) bundled as an array
    - Training loop processes full trajectories, applies R(s) per-step during backprop
    
    Each sample contains:
    - messages: The FULL trajectory (all turns)
    - leaf_reward: The final reward for this complete path
    - step_rewards: Array of {step, trace_prefix, R, V, GA, LA} for each step
    - pred_trace: The predicted trace for this leaf
    - gold_trace: The gold trace (if provided)
    - source: "final_beam" or "pruned"
    - path_id: Identifier for this path
    
    Args:
        leaves: List of leaf dicts with trajectory, pred_trace, reward, etc.
        step_rewards: Output from compute_treerl_rewards
        gold_trace: Optional gold trace for reference
        gold_code: Optional gold HTS code
        ruling_index: Optional ruling index
        
    Returns:
        List of leaf-level training samples (one per leaf)
    """
    samples = []
    
    for leaf in leaves:
        trajectory = leaf.get("trajectory", [])
        pred_trace = leaf.get("pred_trace", [])
        path_id = leaf.get("path_id", "unknown")
        leaf_reward = leaf.get("reward", 0.0)
        source = leaf.get("source", "unknown")
        classification_path = leaf.get("classification_path", [])
        
        if not trajectory:
            continue
        
        trace_key = trace_to_key(pred_trace)
        
        # Build step rewards array - one entry per step in the trace
        step_rewards_list = []
        for step_idx in range(len(pred_trace)):
            prefix_key = trace_key[:step_idx + 1]
            reward_info = step_rewards.get(prefix_key, {})
            
            step_rewards_list.append({
                "step": step_idx,
                "trace_prefix": list(pred_trace[:step_idx + 1]),
                "R": reward_info.get("R", 0.0),
                "V": reward_info.get("value", 0.0),
                "GA": reward_info.get("ga", 0.0),
                "LA": reward_info.get("la", 0.0),
                "descendant_count": reward_info.get("descendant_count", 0)
            })
        
        sample = {
            # Full trajectory - this is what gets trained on
            "messages": trajectory,
            
            # Leaf-level reward (fractional prefix match)
            "leaf_reward": leaf_reward,
            
            # GDPO advantage for optional loss scaling (computed externally)
            "gdpo_advantage": leaf.get("gdpo_advantage", 0.0),
            
            # Reward components for debugging/analysis
            "reward_components": leaf.get("reward_components", {}),
            
            # Step-level rewards - training loop applies these per-step during backprop
            "step_rewards": step_rewards_list,
            
            # Trace information
            "pred_trace": pred_trace,
            "gold_trace": gold_trace,
            
            # Metadata
            "source": source,  # "final_beam" or "pruned"
            "path_id": path_id,
            "classification_path": classification_path,
            
            # Gold reference
            "gold_code": gold_code,
            "ruling_index": ruling_index
        }
        samples.append(sample)
    
    return samples


# Keep backward compatibility alias
def emit_step_level_samples(
    leaves: List[Dict[str, Any]],
    step_rewards: Dict[tuple, Dict[str, float]]
) -> List[Dict[str, Any]]:
    """
    DEPRECATED: Use emit_leaf_samples instead.
    
    This was the old format that emitted one sample per step.
    The correct TreeRL format is one sample per leaf with bundled step rewards.
    """
    logger.warning("emit_step_level_samples is deprecated. Use emit_leaf_samples instead.")
    return emit_leaf_samples(leaves, step_rewards)


def summarize_treerl_computation(
    leaves: List[Dict[str, Any]],
    step_rewards: Dict[tuple, Dict[str, float]],
    v_root: float
) -> Dict[str, Any]:
    """
    Generate a summary of the TreeRL computation for logging.
    """
    leaf_rewards = [leaf.get("reward", 0.0) for leaf in leaves]
    
    # Collect all R values
    all_R = [info["R"] for info in step_rewards.values() if info.get("R") is not None]
    
    return {
        "num_leaves": len(leaves),
        "v_root": v_root,
        "mean_leaf_reward": sum(leaf_rewards) / len(leaf_rewards) if leaf_rewards else 0.0,
        "min_leaf_reward": min(leaf_rewards) if leaf_rewards else 0.0,
        "max_leaf_reward": max(leaf_rewards) if leaf_rewards else 0.0,
        "num_prefix_nodes": len(step_rewards),
        "mean_R": sum(all_R) / len(all_R) if all_R else 0.0,
        "min_R": min(all_R) if all_R else 0.0,
        "max_R": max(all_R) if all_R else 0.0
    }

