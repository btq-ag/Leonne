# Generalized Fisher-Yates Shuffle

> **GFY** · A group-theoretic extension of the classic Fisher-Yates algorithm  
> capable of sampling permutations under arbitrary **compact (Lie) groups**,  
> with orbit-size–weighted probabilities and non-degenerate edge-set support.

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Key Concepts](#key-concepts)  
3. [Installation](#installation)  
4. [Quick Start](#quick-start)  
5. [Algorithm Walk-through](#algorithm-walk-through)  
   5.1 [Classic Fisher-Yates](#classic-fisher-yates)  
   5.2 [Permutation-Orbit GFY](#permutation-orbit-gfy)  
   5.3 [Symmetric ND GFY](#symmetric-nd-gfy)  
   5.4 [Consensus GFY](#consensus-gfy)  
6. [Edge-Set Shuffles](#edge-set-shuffles)  
7. [Probability Diagnostics](#probability-diagnostics)  
8. [API Reference](#api-reference)  
9. [License](#license)

---

## Introduction
Traditional Fisher-Yates yields *uniform* random permutations—appropriate when the
shuffling symmetry group is the **full symmetric group** `S_n`.  
In distributed consensus, however, we often act with **quotient groups** where
degenerate states (orbits) carry different statistical weight.

**GFY** generalises Fisher-Yates to:

| Variant | Group `G` | Probability rule | Use case |
| ------- | --------- | ---------------- | -------- |
| `FisherYates` | trivial (`S_n` uniform) | 1/|S_n| | Baseline shuffle |
| `permutationOrbitFisherYates` | *Permutation subgroup* | 1/|orbit| | Weighted sampling of subgroup action |
| `symmetricNDFisherYates` | `S_n / Stab(i)` | Non-degenerate (`ND`) | Removes fixed-point degeneracy |
| `consensusFisherYates` | **Consensus group**<br>(`S_n` mod duplicates) | 1/|orbit\stab| | Node allocation in TCN |

All variants retain **O(n)** runtime, mirroring the efficiency of the original
algorithm.

---

## Key Concepts
```text
Input set  A = [a₀ … a_{n−1}]
Group G ⊆ S_n  acts on A by permutation
Orbit      Orb_G(x) = { g·x | g∈G }
Stabiliser Stab_G(x) = { g | g·x = x }
```
GFY replaces the *uniform* random index selection at step `i` by a
**group-orbit–weighted** choice:

\[
\Pr(j=i \text{ swap}) = \frac{1}{|\text{Orb}_G(a_i) ∖ \text{Stab}_G(a_i)|}.
\]

Multiplying the inverse orbit sizes yields the final permutation probability,
allowing exact diagnostic checks.

---

## Installation
```bash
git clone https://github.com/btq-ag/GeneralizedPermutations.git
cd GeneralizedPermutations
pip install -r requirements.txt  # numpy, sympy, matplotlib
```

---

## Quick Start
```python
from generalizedPermutations import (
    FisherYates,
    permutationOrbitFisherYates,
    symmetricNDFisherYates,
    consensusFisherYates,
    groupDistribution
)

data = [0,1,2,3,4]

# Classic uniform shuffle
print(FisherYates(data.copy(), extraInfo=False))

# Permutation-orbit shuffle (subgroup ⊂ S_n)
permuted, p = permutationOrbitFisherYates(data.copy(), extraInfo=False)
print("Permutation:", permuted, "prob:", p)

# Non-degenerate symmetric shuffle
print(symmetricNDFisherYates(data.copy(), extraInfo=False)[0])

# Consensus shuffle removes duplicates
lst = [1,1,2,2,3]
print(consensusFisherYates(lst.copy(), extraInfo=False)[0])

# Visualise empirical vs expected distribution
groupDistribution(consensusFisherYates, lst, iterationSize=50000)
```

---

## Algorithm Walk-through

### Classic Fisher-Yates
```python
def FisherYates(arr):
    for i in reversed(range(1,len(arr))):
        j = np.random.randint(0, i+1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr
```

### Permutation-Orbit GFY
```python
def permutationOrbitFisherYates(arr):
    P = PermutationGroup([Permutation(arr)])
    inv_orbit = np.zeros(len(arr))
    for i in reversed(range(len(arr))):
        orb  = list(P.orbit(arr[i]))
        j = np.random.choice(list(set(arr[:i+1]) & set(orb)))
        inv_orbit[i] = 1/len(orb)
        k = arr.index(j)
        arr[i], arr[k] = arr[k], arr[i]
    return arr, np.prod(inv_orbit)
```

### Symmetric ND GFY
Removes stabiliser orbit to avoid swapping with equal elements:
```python
stab = SymmetricGroup(n).stabilizer(last)
overlap = orbit − stab
```

### Consensus GFY
Implements the *consensus group* `S_n / duplicates`,
producing uniform permutations of **multisets** (weighted by inverse orbit size).

---

## Edge-Set Shuffles
For bipartite graphs `G=(U,V,E)` we shuffle the **u-column** while ensuring
non-degenerate edges:

```python
def edgeFisherYates(edgeSet):
    u,v = edgeSet[:,0].tolist(), edgeSet[:,1].tolist()
    for i in reversed(range(len(u)-1)):
        t = [k for k in range(i) if (u[k]!=u[i] and v[k]!=v[i])]
        j = random.choice(t)
        u[i], u[j] = u[j], u[i]
    return np.column_stack((u,v)), u
```

---

## Probability Diagnostics
`groupDistribution` and `groupOrbitDistribution` produce scatter plots of:

* Empirical permutation frequency  
* Expected average (1/|Perms|)  
* Orbit-predicted probability (for weighted variants)  

```python
groupOrbitDistribution(
    permutationOrbitFisherYates,
    [0,1,2,3],
    iterationSize=100_000
)
```

---

## API Reference

| Function | Description |
| -------- | ----------- |
| **`FisherYates(lst, extraInfo)`** | Classic shuffle. |
| **`permutationOrbitFisherYates(lst, extraInfo)`** | Subgroup-orbit version; returns `(perm, prob)`. |
| **`symmetricNDFisherYates(lst, extraInfo)`** | Non-degenerate symmetric shuffle. |
| **`consensusFisherYates(lst, extraInfo)`** | Multiset-aware consensus shuffle. |
| **`consensusShuffle(edgeSet, extraInfo)`** | Edge-set variant for bipartite graphs. |
| **`groupDistribution(fn, lst, N)`** | Empirical frequency plot for unweighted shuffles. |
| **`groupOrbitDistribution(fn, lst, N)`** | Diagnostics for orbit-weighted variants. |
| **`edgeDistribution(fn, E, N)`** | Distribution over edge-set permutations. |

All `extraInfo` flags print step-by-step traces for educational purposes.

---

