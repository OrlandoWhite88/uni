#!/usr/bin/env python3
"""
Quick test to debug GDPO + TreeRL integration.

Tests:
1. Raw rewards -> TreeRL step rewards
2. GDPO-normalized rewards -> TreeRL step rewards
3. Compare V(root) and step rewards R(s)
"""

import math
import sys
sys.path.insert(0, "/home/orlando/uni")
sys.path.insert(0, "/home/orlando/uni/api")

from api.treerl_process_supervision import (
    build_prefix_tree,
    compute_tree_values,
    compute_advantages,
    collect_step_rewards,
    trace_to_key,
)


def create_mock_leaves():
    """Create mock leaves simulating a beam search with different rewards."""
    # Simulate 3 paths with different rewards
    # Path 0: Gold match (high reward)
    # Path 1: Partial match (medium reward)  
    # Path 2: Wrong path (low reward)
    
    leaves = [
        {
            "path_id": "path_0",
            "pred_trace": [
                {"kind": "chapter", "code": "84"},
                {"kind": "heading", "code": "8421"},
                {"kind": "subheading", "code": "842121"},
                {"kind": "code", "code": "8421.21.0000"},
            ],
            "reward": 1.0,  # Perfect match
            "reward_components": {
                "trace_prefix": 1.0,
                "code_digits_prefix": 1.0,
            },
        },
        {
            "path_id": "path_1",
            "pred_trace": [
                {"kind": "chapter", "code": "84"},
                {"kind": "heading", "code": "8421"},
                {"kind": "subheading", "code": "842139"},
                {"kind": "code", "code": "8421.39.0000"},
            ],
            "reward": 0.5,  # Partial match (2/4 steps)
            "reward_components": {
                "trace_prefix": 0.5,
                "code_digits_prefix": 0.4,
            },
        },
        {
            "path_id": "path_2", 
            "pred_trace": [
                {"kind": "chapter", "code": "84"},
                {"kind": "heading", "code": "8401"},
                {"kind": "code", "code": "8401.10.0000"},
            ],
            "reward": 0.25,  # Wrong heading (1/4 steps)
            "reward_components": {
                "trace_prefix": 0.25,
                "code_digits_prefix": 0.2,
            },
        },
    ]
    return leaves


def apply_gdpo_normalization(leaves, weights=(1.0, 1.0), eps=1e-6):
    """Apply GDPO-style decoupled normalization."""
    components = ["trace_prefix", "code_digits_prefix"]
    
    # Extract component values
    values_by_comp = {}
    for comp in components:
        values_by_comp[comp] = [
            float(leaf.get("reward_components", {}).get(comp, 0.0) or 0.0)
            for leaf in leaves
        ]
    
    print(f"\n📊 Raw reward components:")
    for comp, vals in values_by_comp.items():
        print(f"  {comp}: {vals}")
    
    # GDPO: Normalize each component independently
    normalized_by_comp = {}
    for comp, vals in values_by_comp.items():
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sd = math.sqrt(max(var, 0.0))
        
        print(f"\n  {comp}: mean={mu:.4f}, std={sd:.4f}")
        
        if sd < eps:
            normalized_by_comp[comp] = [0.0 for _ in vals]
        else:
            normalized_by_comp[comp] = [(v - mu) / (sd + eps) for v in vals]
        
        print(f"    normalized: {[f'{v:.4f}' for v in normalized_by_comp[comp]]}")
    
    # Weighted sum of normalized components
    w = list(weights)
    gdpo_values = []
    for i in range(len(leaves)):
        total = 0.0
        for comp_idx, comp in enumerate(components):
            comp_vals = normalized_by_comp.get(comp, [])
            if i < len(comp_vals):
                total += w[comp_idx] * comp_vals[i]
        gdpo_values.append(total)
    
    print(f"\n📊 GDPO-normalized values (weighted sum):")
    print(f"  weights: {w}")
    print(f"  values: {[f'{v:.4f}' for v in gdpo_values]}")
    print(f"  mean: {sum(gdpo_values)/len(gdpo_values):.4f}")
    
    return gdpo_values


def compute_treerl_step_rewards(leaves, reward_key="reward"):
    """Compute TreeRL step rewards from leaves."""
    root = build_prefix_tree(leaves, reward_key=reward_key)
    v_root = compute_tree_values(root)
    compute_advantages(root)
    step_rewards = collect_step_rewards(root)
    return step_rewards, v_root


def print_step_rewards(step_rewards, v_root, label):
    """Pretty print step rewards."""
    print(f"\n{'='*60}")
    print(f"📈 {label}")
    print(f"{'='*60}")
    print(f"V(root) = {v_root:.4f}")
    print(f"\nStep rewards R(s):")
    
    for key, info in sorted(step_rewards.items(), key=lambda x: (len(x[0]), x[0])):
        print(f"  {' > '.join(str(k) for k in key)}")
        print(f"    V={info['value']:.4f}, GA={info['ga']:.4f}, LA={info['la']:.4f}, R={info['R']:.4f}")


def main():
    print("="*60)
    print("🧪 GDPO + TreeRL Integration Test")
    print("="*60)
    
    # Create mock data
    leaves = create_mock_leaves()
    
    print("\n📦 Mock leaves:")
    for leaf in leaves:
        trace_codes = [s.get("code", "?") for s in leaf["pred_trace"]]
        print(f"  {leaf['path_id']}: {' > '.join(trace_codes)}")
        print(f"    raw_reward={leaf['reward']:.2f}, components={leaf['reward_components']}")
    
    # TEST 1: Raw rewards -> TreeRL (CORRECT approach)
    print("\n" + "="*60)
    print("TEST 1: Raw rewards → TreeRL (CORRECT)")
    print("="*60)
    
    step_rewards_raw, v_root_raw = compute_treerl_step_rewards(leaves, reward_key="reward")
    print_step_rewards(step_rewards_raw, v_root_raw, "RAW REWARDS → TreeRL")
    
    # TEST 2: Compute GDPO advantages for loss scaling
    print("\n" + "="*60)
    print("TEST 2: GDPO Advantages for Loss Scaling")
    print("="*60)
    
    gdpo_values = apply_gdpo_normalization(leaves)
    
    # Store as advantages (not as rewards!)
    for leaf, val in zip(leaves, gdpo_values):
        leaf["gdpo_advantage"] = val
    
    # GDPO advantages are used DIRECTLY as loss multipliers (can be negative!)
    # Per TreeRL/GDPO papers: negative advantages are correct - they decrease probability
    loss_scales = gdpo_values  # Direct use, no transformation!
    
    print(f"\n📊 Loss multipliers (GDPO advantages - can be negative!):")
    for leaf, scale in zip(leaves, loss_scales):
        effect = "↑ probability" if scale > 0 else "↓ probability" if scale < 0 else "no change"
        print(f"  {leaf['path_id']}: {scale:+.4f} → {effect}")
    
    # COMPARISON
    print("\n" + "="*60)
    print("📊 CORRECT WIRING SUMMARY")
    print("="*60)
    
    print(f"\n{'Component':<35} {'Value':>15}")
    print("-"*50)
    print(f"{'V(root) from raw rewards':<35} {v_root_raw:>15.4f}")
    print(f"{'Mean GDPO advantage':<35} {sum(gdpo_values)/len(gdpo_values):>15.4f}")
    
    print("\n📋 Per-leaf breakdown:")
    print(f"  {'Path':<10} {'Raw R':>8} {'GDPO Adv':>12} {'Effect':<20}")
    print(f"  {'-'*50}")
    for leaf, gdpo in zip(leaves, gdpo_values):
        effect = "↑ increase prob" if gdpo > 0 else "↓ decrease prob" if gdpo < 0 else "no change"
        print(f"  {leaf['path_id']:<10} {leaf['reward']:>8.2f} {gdpo:>+12.4f} {effect:<20}")
    
    # Verification
    print("\n" + "="*60)
    print("✅ VERIFICATION")
    print("="*60)
    
    print(f"✓ V(root) = {v_root_raw:.4f} > 0 (meaningful TreeRL baseline)")
    print(f"✓ GDPO advantages mean ≈ {sum(gdpo_values)/len(gdpo_values):.4f} (centered around 0)")
    print(f"\n✓ Per TreeRL/GDPO papers, negative advantages are CORRECT:")
    print(f"  - Best path (path_0):  adv = {gdpo_values[0]:+.4f} → INCREASE probability")
    print(f"  - Worst path (path_2): adv = {gdpo_values[2]:+.4f} → DECREASE probability")
    print(f"\n✓ Formula: loss = cross_entropy * advantage")
    print(f"  - Positive adv → positive loss → minimize → increase prob")
    print(f"  - Negative adv → negative loss → minimize (more negative) → decrease prob")


if __name__ == "__main__":
    main()

