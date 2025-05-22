# Algorithms Modules

A compact Python toolbox for simulating *trust-driven*, *quantum-enhanced* and
*group-theoretic* protocols that underpin **Topological Consensus Networks**.
Everything lives in a single folder; each `*.py` file is executable and
self-documenting.

| Algorithm | File | Purpose |
|-----------|------|---------|
| **Quantum Network Partitioner** | `quantum_trust_partitioner.py` | Re-distribute nodes across consensus networks using quantum-enhanced trust matrices. |
| **Quantum Topological Visualizer** | `quantum_topological_visualizer.py` | Visualize topological features with quantum enhancements. |
| **Quantum Blockchain Simulation** | `quantumBlockchainVisualizer.py` | Visualize blockchain networks with quantum enhancements like QKD, QRNG, and entropy addition. |
| **Quantum Generalized Permutations** | `quantumGeneralizedPermutations.py` | Orbit-weighted permutations enhanced with quantum randomness. |
| **Quantum Graph Generator** | `quantumGraphGenerator.py` | Construct quantum-inspired network graphs. |

> **TL;DR**  
> Clone the repo, `pip install numpy sympy matplotlib`, then run any script:
> `python networkPartitioner.py`.

---

## 1 · Classical Network Partitioner

```python
from quantum_trust_partitioner import quantum_network_partitioner
import numpy as np

# four toy networks
nets = [[0,1,2], [3,4], [5,6,7], [8,9]]
trust = np.random.rand(10,10); np.fill_diagonal(trust, 0)

node_sec = {i: 0.25+0.05*i for i in range(10)}
net_sec  = {idx: max(sorted([node_sec[n] for n in net])[len(net)//2:])
            for idx, net in enumerate(nets)}

new_nets, *_ = quantum_network_partitioner(nets, trust, node_sec, net_sec)
print("→ repartitioned:", new_nets)
```

**What happens?**

*Jump phase* moves nodes to networks with lower average distrust (if the
destination passes its own security threshold).  
*Abandon phase* isolates nodes whose internal trust dips below their personal
threshold. The function returns the updated partition in `O(|V|+|E|)` time.

---

## 2 · Quantum Network Partitioner

```python
from quantumNetworkPartitioner import quantumNetworkPartitioner

nets  = [[0,1,2], [3,4], [5]]
q_sec = {0:0.3,1:0.4,2:0.3,3:0.6,4:0.5,5:0.2}

final, qTrust = quantumNetworkPartitioner(
        nets,
        complianceThresh=0.5,   # QKD bit-overlap threshold
        bitLength=16,           # QRiNG string length
        nodeSecurity=q_sec,
        extraInfo=False)

print(final)
```

**Key points**

* Genuine randomness via `qringRandomBits()` (stubbed; plug your QRNG SDK).  
* Trust edges arise only when the *QKD compliance test* passes.  
* Hybrid classical-quantum mode toggled by λ in `qringRandomInt()`.

---

## 3 · Quantum Blockchain Simulation

```python
from quantumBlockchainVisualizer import create_quantum_blockchain_visualization

# Create a quantum blockchain visualization with 5 blocks, each containing 12 nodes
create_quantum_blockchain_visualization(
    n_blocks=5, 
    n_nodes_per_block=12, 
    n_frames=30,
    quantum_ratio=0.6,  # 60% of nodes are quantum-enabled
    show_compliance=True  # Show the QKD compliance graph
)

# Generate analysis plots comparing quantum and classical approaches
from quantumBlockchainVisualizer import create_quantum_vs_classical_entropy_plot, create_quantum_security_comparison

create_quantum_vs_classical_entropy_plot()
create_quantum_security_comparison()
```

**Quantum Enhancements**

* **Quantum Random Number Generation (QRNG)**: True random number generation for unpredictable network connections.
* **Quantum Key Distribution (QKD)**: Secure communication channels between blockchain nodes with eavesdropping detection.
* **Entropy Addition**: Combines classical and quantum randomness for maximum entropy even when sources are compromised.
* **Compliance Graphs**: Visualizes which quantum communication channels are validated as secure.
* **Quantum Network Topologies**: Enhanced network models incorporating quantum effects like tunneling and entanglement.

---

## 4 · Generalised Fisher-Yates Shuffle

```python
from generalizedShuffle import consensusFisherYates

items = [1,1,2,2,3]          # multiset
perm, prob = consensusFisherYates(items, extraInfo=False)
print("Permutation:", perm, "orbit-prob:", prob)
```

**Highlights**

| Variant | Call | When to use |
|---------|------|-------------|
| Classic (uniform) | `FisherYates(lst)` | Ordinary random perms. |
| Subgroup orbit | `permutationOrbitFisherYates(lst)` | Weighted by |orbit|. |
| Non-degenerate `S_n` | `symmetricNDFisherYates(lst)` | Reject fixed points. |
| **Consensus multiset** | `consensusFisherYates(lst)` | Remove duplicate-node degeneracy in TCN. |

Diagnostics: `groupDistribution()` and `groupOrbitDistribution()` plot empirical
vs expected frequencies.

---

## 5 · Edge-Set Assignment

```python
from edgeAssignment import sequenceVerifier, edgeAssignment

deg = [2,2,2, 2,3,1]   # |U|=|V|=3
if sequenceVerifier(deg,3,3):
    edges = edgeAssignment(deg,3,3,extraInfo=False)
    print("Edge set:", edges)
```

The script implements the Gale–Ryser test (“conservation” & “boundedness”) then
performs a deterministic, row-major construction.

---

### 4.1 Consensus Edge Shuffle

```python
from edgeAssignment import consensusShuffle, edgeDistribution
import numpy as np

E = np.array([[1,2],[2,1],[2,3],[3,2],[4,4]])
shuffled, uperm = consensusShuffle(E, extraInfo=False)
print(shuffled)                       # new bipartite graph

# optional diagnostics
edgeDistribution(consensusShuffle, E, iterationSize=20_000)
```

Swaps only *non-degenerate* pairs, guaranteeing degree preservation while
randomising the left column.

---

## 5 · Putting It All Together

```python
# 1. build an initial graph from degree sequence
edges = edgeAssignment([2,2,1, 1,2,2], 3, 3, extraInfo=False)

# 2. shuffle edges under consensus Fisher-Yates
shuffled_edges, _ = consensusShuffle(np.array(edges), extraInfo=False)

# 3. map shuffled edges into network IDs and feed to (quantum) partitioner
nets = [[0,1,2], [3,4], [5]]
quantumNetworkPartitioner(nets, nodeSecurity={...})
```

---

## Design Choices

* **Single-file scripts** for zero-overhead experimentation.  
* `extraInfo=True` prints tutorial-level traces; keep `False` in production.  
* No external state — every script can be run standalone for quick sanity checks.

---

## Caveats & Pitfalls

1. `probabilityDistribution()` grows factorially with unique permutations;
   use small demo sets.  
2. QRiNG functions are stubbed; attach your hardware SDK.  
3. Sparse graphs >10³ vertices may need custom builder logic.

---

## Roadmap

- Unit-test suite (`pytest`) covering all edge cases.  
- Cython acceleration for giant degree sequences.  
- Web-based Playground (Streamlit) to visualise partitioning in real time.  

---
