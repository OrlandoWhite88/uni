#!/usr/bin/env python3
"""
Test that negative advantages work correctly in policy gradient.

The policy gradient formula is:
    ∇J = E[A * ∇log π]

We want to MAXIMIZE J, so we MINIMIZE loss = -J.

With cross_entropy = -log π, the loss becomes:
    loss = A * cross_entropy

Let's verify:
- If A > 0 (good): minimize loss → decrease CE → increase probability ✓
- If A < 0 (bad): minimize loss (make more negative) → increase CE → decrease probability ✓
"""

import torch
import torch.nn.functional as F


def test_gradient_direction():
    """Verify gradient direction for positive and negative advantages."""
    
    print("="*60)
    print("Testing gradient direction with advantages")
    print("="*60)
    
    # Simple setup: 3 tokens, vocab size 10
    vocab_size = 10
    
    # Create a simple "model output" with requires_grad
    logits = torch.randn(3, vocab_size, requires_grad=True)
    
    # Target tokens
    targets = torch.tensor([2, 5, 7])
    
    # Compute base cross entropy (per-token)
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    print(f"\nBase cross_entropy per token: {ce_loss.detach().numpy()}")
    print(f"Mean CE: {ce_loss.mean().item():.4f}")
    
    # Test 1: Positive advantage
    print("\n" + "-"*40)
    print("TEST 1: Positive advantage (A = +2.0)")
    print("-"*40)
    
    logits1 = logits.clone().detach().requires_grad_(True)
    ce1 = F.cross_entropy(logits1, targets, reduction='none')
    loss1 = (ce1 * 2.0).mean()  # Positive advantage
    loss1.backward()
    
    print(f"Loss = CE * (+2.0) = {loss1.item():.4f}")
    print(f"Gradient norm: {logits1.grad.norm().item():.4f}")
    
    # The gradient should point in direction to DECREASE probability
    # (because we're minimizing loss, which increases probability for positive A)
    
    # Test 2: Negative advantage
    print("\n" + "-"*40)
    print("TEST 2: Negative advantage (A = -2.0)")
    print("-"*40)
    
    logits2 = logits.clone().detach().requires_grad_(True)
    ce2 = F.cross_entropy(logits2, targets, reduction='none')
    loss2 = (ce2 * (-2.0)).mean()  # Negative advantage
    loss2.backward()
    
    print(f"Loss = CE * (-2.0) = {loss2.item():.4f}")
    print(f"Gradient norm: {logits2.grad.norm().item():.4f}")
    
    # Test 3: Verify gradient directions are OPPOSITE
    print("\n" + "-"*40)
    print("TEST 3: Verify gradient directions")
    print("-"*40)
    
    # Cosine similarity between gradients
    cos_sim = F.cosine_similarity(
        logits1.grad.flatten().unsqueeze(0),
        logits2.grad.flatten().unsqueeze(0)
    ).item()
    
    print(f"Cosine similarity between gradients: {cos_sim:.4f}")
    
    if cos_sim < -0.99:
        print("✅ CORRECT: Gradients are opposite (cos ≈ -1)")
        print("   Positive advantage → increase probability")
        print("   Negative advantage → decrease probability")
    else:
        print("❌ UNEXPECTED: Gradients should be opposite!")
    
    # Test 4: Simulate one optimization step
    print("\n" + "-"*40)
    print("TEST 4: Simulate optimization step")
    print("-"*40)
    
    lr = 0.1
    
    # For positive advantage
    logits_pos = logits.clone().detach()
    logits_pos = logits_pos - lr * logits1.grad  # Gradient descent
    probs_before = F.softmax(logits, dim=-1)[range(3), targets]
    probs_after_pos = F.softmax(logits_pos, dim=-1)[range(3), targets]
    
    print(f"Positive advantage (A=+2):")
    print(f"  Probs before: {probs_before.detach().numpy()}")
    print(f"  Probs after:  {probs_after_pos.numpy()}")
    print(f"  Change: {(probs_after_pos - probs_before.detach()).numpy()}")
    
    if (probs_after_pos > probs_before.detach()).all():
        print("  ✅ Probabilities INCREASED (correct for positive advantage)")
    
    # For negative advantage
    logits_neg = logits.clone().detach()
    logits_neg = logits_neg - lr * logits2.grad  # Gradient descent
    probs_after_neg = F.softmax(logits_neg, dim=-1)[range(3), targets]
    
    print(f"\nNegative advantage (A=-2):")
    print(f"  Probs before: {probs_before.detach().numpy()}")
    print(f"  Probs after:  {probs_after_neg.numpy()}")
    print(f"  Change: {(probs_after_neg - probs_before.detach()).numpy()}")
    
    if (probs_after_neg < probs_before.detach()).all():
        print("  ✅ Probabilities DECREASED (correct for negative advantage)")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
Negative advantages are CORRECT and work as expected:
- loss = advantage * cross_entropy
- When advantage < 0, loss < 0
- Minimizing negative loss → make more negative → increase CE → decrease probability

This matches TreeRL paper's R(s) which can be negative.
No need for exp() transformation - use advantages directly!
""")


def test_treerl_rewards():
    """Test TreeRL-style step rewards."""
    
    print("\n" + "="*60)
    print("TreeRL Step Rewards R(s) Test")
    print("="*60)
    
    # Simulate a tree with different paths
    # Path 1: Good path (correct answer)
    # Path 2: Bad path (wrong answer)
    
    v_root = 0.5  # 50% of paths are correct
    
    # Good path steps
    v_good = [0.5, 0.7, 0.9, 1.0]  # Values increasing toward correct
    
    # Bad path steps  
    v_bad = [0.5, 0.3, 0.1, 0.0]  # Values decreasing toward wrong
    
    print("\nGood path (leads to correct answer):")
    for i, v in enumerate(v_good):
        ga = v - v_root
        la = v - v_good[i-1] if i > 0 else 0
        r = ga + la
        print(f"  Step {i}: V={v:.1f}, GA={ga:+.2f}, LA={la:+.2f}, R(s)={r:+.2f}")
    
    print("\nBad path (leads to wrong answer):")
    for i, v in enumerate(v_bad):
        ga = v - v_root
        la = v - v_bad[i-1] if i > 0 else 0
        r = ga + la
        print(f"  Step {i}: V={v:.1f}, GA={ga:+.2f}, LA={la:+.2f}, R(s)={r:+.2f}")
    
    print("""
Notice:
- Good path has positive R(s) → increase probability ✓
- Bad path has negative R(s) → decrease probability ✓
- R(s) can definitely be negative - this is by design!
""")


if __name__ == "__main__":
    test_gradient_direction()
    test_treerl_rewards()


