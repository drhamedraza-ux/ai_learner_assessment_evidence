#!/usr/bin/env python3
"""Does each model's prediction for target position t+1 use the response at position t?

Every model in the suite is trained to predict `correct[:, 1:]` from history. The
standard knowledge-tracing alignment is that the prediction for target j uses
interactions 0..j-1. This probe perturbs the response at exactly one position and
measures which output positions move. It needs no training and no data: the
dependency structure is a property of the forward pass.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/claude/work/P02")
import pcdt_p2 as P

torch.manual_seed(0)
CLS = P.build_model_classes()
B, L, NS, H = 4, 12, 7, 16
g = torch.Generator().manual_seed(1)

skill = torch.randint(1, NS, (B, L), generator=g)
correct = torch.randint(0, 2, (B, L), generator=g).float()
behav = torch.randn(B, L, len(P.BEHAV), generator=g)
dt = torch.rand(B, L, generator=g) * 5000
mask = torch.ones(B, L)


def batch(c):
    return dict(skill=skill, correct=c, behav=behav, dt=dt, mask=mask)


specs = P.model_specs(NS, P.Config(dataset="junyi"), CLS)
print(f"{'model':12s} | for target position j, the highest observed position it depends on")
print("-" * 92)
verdict = {}
for name, build in specs.items():
    torch.manual_seed(7)
    m = build(H).eval()
    with torch.no_grad():
        base = m(batch(correct))
    lags = []
    for t in range(L - 1):
        c2 = correct.clone()
        c2[:, t] = 1.0 - c2[:, t]            # flip the response at position t only
        with torch.no_grad():
            alt = m(batch(c2))
        moved = (alt - base).abs().max(dim=0).values > 1e-6   # per output position
        idx = torch.nonzero(moved).flatten().tolist()
        # output index u corresponds to target position u+1
        first = min(idx) if idx else None
        lags.append((t, first))
    # for each target j, the largest observed position that influences it
    dep = {}
    for t, first in lags:
        if first is None:
            continue
        for u in range(first, L - 1):
            dep[u + 1] = max(dep.get(u + 1, -1), t)
    sample = {j: dep.get(j, None) for j in range(1, L)}
    line = " ".join(f"{j}:{'-' if v is None else v}" for j, v in sample.items())
    ok = all(v == j - 1 for j, v in sample.items() if v is not None)
    verdict[name] = ok
    print(f"{name:12s} | {line}   {'OK' if ok else '<-- LAGS'}")

print()
print("Expected for standard KT alignment: target j depends on observed position j-1.")
bad = [k for k, v in verdict.items() if not v]
print(f"\nModels whose prediction for target j does NOT use interaction j-1: {bad or 'none'}")

# Quantify for the offender: which positions are affected at all
for name in bad:
    torch.manual_seed(7)
    m = specs[name](H).eval()
    with torch.no_grad():
        base = m(batch(correct))
    print(f"\n--- {name}: output index -> set of observed positions it depends on ---")
    dep = {u: [] for u in range(L - 1)}
    for t in range(L - 1):
        c2 = correct.clone()
        c2[:, t] = 1.0 - c2[:, t]
        with torch.no_grad():
            alt = m(batch(c2))
        moved = (alt - base).abs().max(dim=0).values > 1e-6
        for u in torch.nonzero(moved).flatten().tolist():
            dep[u].append(t)
    for u in range(L - 1):
        print(f"  out[{u}] (target position {u+1}) depends on observed positions {dep[u]}")
