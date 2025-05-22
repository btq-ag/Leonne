# Graph Generator Toolkit

> **GGT** – A compact Python implementation that builds *valid bipartite graphs*
> from arbitrary degree sequences and explores the probability landscape of
> edge-set permutations.  
> Part of the **Topological Consensus Networks (TCN)** stack.

---

## Contents
1. [Why Edge Assignment?](#why-edge-assignment)  
2. [Feature Highlights](#feature-highlights)  
3. [Quick Install](#quick-install)  
4. [Hello, Bipartite World](#hello-bipartite-world)  
5. [Core Concepts](#core-concepts)  
   - Degree Sequences  
   - Feasibility Checks  
   - Greedy Construction  
6. [API Walk-Through](#api-walk-through)  
   - `sequenceVerifier`  
   - `edgeAssignment`  
   - `consensusShuffle` _(bonus)_  
7. [Worked Examples](#worked-examples)  
8. [Advanced Topics](#advanced-topics)  
   - Degenerate Edge Sets  
   - Probability Diagnostics  
9. [Design Choices](#design-choices)  
10. [Caveats & Pitfalls](#caveats--pitfalls)  
11. [Roadmap](#roadmap)  
12. [License](#license)

---

## Why Edge Assignment?

In TCN we repeatedly need **balanced bipartite graphs** to map:

```
Nodes  ─┬─> Consensus Sets
        └─> Edge-set View of Shuffles
```

A single integer sequence `[d₀,…,d_{u+v−1}]` uniquely specifies such a graph **iff**
two constraints hold (conservation + boundedness).  
`graphGenerator.py` automates:

1. *Validation* – check constraints in `O(u·v)`  
2. *Construction* – produce one concrete edge set  
3. *Exploration* – shuffle edges under non-degenerate Fisher-Yates  
4. *Diagnostics* – plot empirical vs expected permutation frequencies

All in ~200 lines of NumPy/SymPy.

---

## Feature Highlights

| 🔧  | Description |
| --- | ----------- |
| **Fast verifier** | Two-pass check of Gale–Ryser inequalities. |
| **Greedy builder** | Deterministic construction of an edge set meeting the sequence. |
| **Degeneracy removal** | `removeDegeneracy` drops duplicate edge sets for clean sampling. |
| **Permutation analytics** | `probabilityDistribution()` plots frequency & residuals. |
| **Network visualizations** | Static and animated visualizations of bipartite graphs. |
| **Consensus animations** | Animated visualizations of consensus shuffle process. |
| **Tiny dependency footprint** | Only `numpy`, `sympy`, `matplotlib`, `networkx` (for visualizations). |

---

## Quick Install

```bash
git clone https://github.com/btq-ag/GraphGenerator.git
cd GraphGenerator
pip install numpy sympy matplotlib networkx
```

---

## Hello, Bipartite World

```python
from graphGenerator import sequenceVerifier, edgeAssignment

# degree sequence [U|V]  ->  U has 3 vertices, V has 3
deg_seq = [2,2,2, 2,3,1]   # length 6

u_size, v_size = 3, 3

if sequenceVerifier(deg_seq, u_size, v_size):
    edges = edgeAssignment(deg_seq, u_size, v_size)
    print("Initial edge set:", edges)
else:
    print("Sequence not realisable!")
```

_Output_

```
Partially satisfiable sequence.
Final edge set:  [[0, 0], [0, 1], [1, 1], [2, 0], [2, 2], [1, 2]]
```

---

## Core Concepts

### Degree Sequences

A **degree sequence** encodes desired vertex degrees:

```
inputSequence = [u₀,u₁,…,u_{u−1}, v₀,…,v_{v−1}]
```

- First `u_size` numbers → target degrees of `U` vertices  
- Remaining `v_size` numbers → `V` degrees  

### Feasibility Checks  `sequenceVerifier()`

```python
def sequenceVerifier(seq, uSize, vSize):
    # 1. Conservation ∑u = ∑v
    # 2. Boundedness  ∑_i≤k u_i ≤ Σ_j min(v_j,k+1)
```

Breaks immediately on failure; otherwise returns `True`.

### Greedy Construction  `edgeAssignment()`

```
for each u in U:
    for each v in V:
        if u_deg>0 and v_deg>0:
            add edge (u,v)
            u_deg-- ; v_deg--
```

Produces a **lexicographically ordered** edge list – handy for deterministic
testing.

---

## API Walk-Through

### `sequenceVerifier(sequence, uSize, vSize, extraInfo=True) → bool`

*Prints step-by-step inequalities when `extraInfo`.*

```python
sequenceVerifier([1,1,1,1,1,1], 3, 3)
# ✅ valid (perfect matching)
sequenceVerifier([6,2,1,3,3,3], 3, 3)
# ❌ fails boundedness
```

---

### `edgeAssignment(sequence, uSize, vSize, extraInfo=True) → List[List[int]]`

Returns a list of `[u,v]` pairs.

```python
edges = edgeAssignment([2,2,1, 1,2,2], 3, 3, extraInfo=False)
# [[0,0],[0,1],[1,1],[2,2],[1,2]]
```

---

### Bonus: `consensusShuffle(edgeSet, extraInfo=True)`

Non-degenerate Fisher-Yates shuffle of the **u-column** while keeping degrees.

```python
shuffled, uperm = consensusShuffle(np.array(edges), extraInfo=False)
```

---

## Worked Examples

### 1 · Minimal Perfect Matching

```python
seq = [1,1,1, 1,1,1]
edges = edgeAssignment(seq, 3, 3)
# [[0,0],[1,1],[2,2]]
```

### 2 · Large Random Sequence

```python
import numpy as np
u,v = 5,4
seq = np.random.randint(1,4,size=u+v).tolist()
seq[-v:] = seq[-v:] + [sum(seq[:u]) - sum(seq[-v:])]  # fix conservation
if sequenceVerifier(seq,u,v):
    E = edgeAssignment(seq,u,v,extraInfo=False)
    print(len(E),"edges built")
```

### 3 · Distribution Plot

```python
from edgeAssignment import probabilityDistribution
probabilityDistribution(edges, 'demoSet', iterationSize=20000)
```

*(opens matplotlib window)*

---

## Advanced Topics

### Degenerate Edge Sets

`removeDegeneracy(edgeSetList)` converts inner lists to frozensets and drops
duplicates – mandatory before probability tallies.

### Probability Diagnostics

```python
possibleSets = probabilityDistribution(
        inputEdgeSet=[[1,2],[2,1],[2,3],[3,2],[4,4]],
        edgeName="5-edge demo",
        iterationSize=50_000,
        extraInfo=False
)
```

Top subplot → frequencies; bottom → log-variance residuals.

---

## Network Visualizations

The toolkit now includes comprehensive network visualization capabilities through `networkVisualizer.py`:

### Static Visualizations
- **Bipartite Graph Structure** - Clear visualization of U and V node sets and their connections
- **Permutation Stability Heat Maps** - Analyze which edges persist across shuffles
- **Edge Permutation Orbits** - Track how the graph structure changes over repeated shuffles

### Animations
- **Consensus Shuffle Animation** - Watch the network evolve under consensus shuffling
- **Network Evolution Animation** - Visualize the dynamics of network connection patterns

Run the visualizer directly to generate all visualizations:

```bash
python networkVisualizer.py
```

Or import and use specific visualization functions:

```python
from networkVisualizer import create_bipartite_graph_visualization, create_network_evolution_animation

# Generate a visualization of a bipartite graph
create_bipartite_graph_visualization(edge_set, u_size, v_size, "my_network")

# Create an animation of network evolution under consensus shuffling
create_network_evolution_animation(degree_sequence, u_size, v_size, 30, "network_evolution")
```

All visualizations are saved directly to the Network Graph Generator folder.

---

## Design Choices

| Area | Decision | Rationale |
| ---- | -------- | --------- |
| **Greedy build** | row-major scan | deterministic, easy to reason about. |
| **Validation** | explicit loops instead of vector math | keep code newbie-friendly. |
| **SymPy permutations** | `multiset_permutations` | handles duplicate labels gracefully. |
| **Matplotlib diagnostics** | 2-panel scatter | mirrors TCN paper’s figures. |

---

## Caveats & Pitfalls

1. **Dense matrices** – algorithm keeps degree arrays in memory; for
   `|U|,|V| > 10³` consider streaming construction.  
2. **EdgeOrdering** – `edgeAssignment` sorts columns decreasingly; if you require
   alternative ordering adjust the last `return`.  
3. **Performance of `probabilityDistribution`** grows factorially with unique
   permutations; use small sets when plotting.

---

## Roadmap

- [ ] **Sparse builder** using heap priority queues  
- [ ] Parallelised probability diagnostics  
- [ ] Native support for half-integer degree sequences (multi-graph)  
- [ ] Jupyter notebook with interactive sliders  

---

