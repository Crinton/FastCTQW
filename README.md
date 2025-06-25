# FastCTQW

FastCTQW is a high-performance, GPU-accelerated Python package for simulating Continuous-Time Quantum Walks (CTQW). It delivers unparalleled speed among GPU-accelerated CTQW simulators, enabling efficient research and development.

The package offers a user-friendly Python interface, allowing researchers and developers to quickly leverage its high performance within a Python environment. FastCTQW supports Python 3.10 or newer.

## Features

- **User-Friendly Interface**: Provides an intuitive Python API for rapid development and validation of CTQW algorithms in virtual environments.
- **High Performance**: Achieves the fastest CTQW simulations, completing a simulation on a dense graph with 10,000 nodes in under 20 seconds on a personal PC equipped with an NVIDIA GPU (8GB VRAM).
- **Numerical Stability**: Ensures robust numerical stability, maintaining infidelity on the order of $10^{-12}$ in single-precision simulations for graphs with thousands of nodes, surpassing the stability of `scipy.linalg.expm`.

## Installation

FastCTQW can be installed via PyPI or by cloning the repository and installing locally. Using a virtual environment is recommended.

### PyPI Installation
```bash
pip install FastCTQW
```

### Local Installation
```bash
git clone https://github.com/Crinton/FastCTQW.git
cd FastCTQW
pip install .
```

## Dependencies

- NumPy >= 1.20
- NetworkX >= 3.0
- SciPy >= 1.14
- Matplotlib >= 3.5.0
- Seaborn >= 0.11.0
- CUDA >= 11.4
- NVIDIA GPU with compute capability >= 7.0
- C++17 standard

### Recommended Virtual Environment Setup
```bash
conda create -n env_FastCTQW python=3.13
conda activate env_FastCTQW
```

## Usage

### Importing Required Libraries
```python
import numpy as np
import networkx as nx
from fastCTQW.State import State
from fastCTQW.Ctqw import Ctqw
```

### Creating a Ctqw Object from a Graph
```python
N = 10
G = nx.random_regular_graph(d=4, n=N)
init_state = State.getUniformSuperposition(n=N)

qwalker = Ctqw.from_networkx_graph(G, init_state, device="cuda", dtype=np.complex64)
```

### Running the Simulation and Retrieving Results
```python
qwalker.runWalk(time=4.5)
final_state = qwalker.getFinalState()
state = final_state.getState()
print(state)
```