<div align="center">

# 🚁 AeroMorph

### Unified Perception-Driven Morphological Adaptation Framework for Multi-Modal Aerial Robots

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Status-Active%20Research-orange.svg)]()
[![CI](https://github.com/hmshujaatzaheer/AeroMorph/actions/workflows/ci.yml/badge.svg)](https://github.com/hmshujaatzaheer/AeroMorph/actions)

**Bridging the Gap Between Morphing Mechanisms and Autonomous Decision Intelligence**

[Overview](#-overview) • [Key Contributions](#-key-contributions) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Citation](#-citation)

</div>

---

## 🔬 Research Context

**The Problem:** Morphing aerial robots can change their physical shape during flight, but current systems lack the autonomous decision intelligence to determine *when*, *where*, *how*, and *with whom* to morph optimally. Existing approaches rely on pre-programmed triggers or human commands, creating a critical gap between morphing *mechanisms* and morphing *decisions*.

**Our Solution:** AeroMorph introduces a unified framework that treats morphological adaptation as an autonomous, perception-driven decision process. By formalizing the **Morphological Decision Space** and developing perception-to-action algorithms, we enable aerial robots that don't just *can* morph—they *decide when to* morph.

---

## 🎯 Overview

AeroMorph addresses a fundamental limitation in morphing robotics: the absence of autonomous decision-making for shape reconfiguration. While recent advances have demonstrated impressive morphing mechanisms—variable stiffness structures, aerial-terrestrial transitions, and adaptive locomotion—these systems operate reactively rather than predictively.

This framework introduces:

| Component | Description |
|-----------|-------------|
| **Morphological Decision Space (𝒟ₘ)** | Unified formalism integrating perception, spatial feasibility, energy constraints, and swarm coordination |
| **P2MA Algorithm** | Perception-to-Morphology-Action pipeline enabling autonomous morphing decisions |
| **Spatial Feasibility Analysis** | Collision-free guarantee for time-varying robot geometry during transformation |
| **EA-SMP Protocol** | Energy-Aware Swarm Morphing Protocol for coordinated multi-robot reconfiguration |

---

## 🌟 Key Contributions

### 1. Morphological Decision Space Formalism

We formalize morphing decisions through a unified configuration space:

```
𝒟ₘ = 𝒫 × 𝒮 × ℰ × 𝒞
```

Where:
- **𝒫** — Perception state space (environmental sensing)
- **𝒮** — Spatial feasibility space (collision-free transformation)
- **ℰ** — Energy constraint space (battery-aware decisions)
- **𝒞** — Coordination state space (swarm synchronization)

### 2. Perception-to-Morphology-Action (P2MA) Algorithm

A four-stage autonomous decision pipeline:

```
Sensor Data → Perception Fusion → Morphing Need Assessment → Spatial Verification → Energy-Constrained Execution
```

### 3. Energy-Aware Swarm Morphing Protocol (EA-SMP)

Distributed consensus protocol enabling synchronized morphing across multi-robot teams:

```
α̇ᵢ(t) = Σⱼ aᵢⱼ(αⱼ(t) - αᵢ(t)) + bᵢ(α_leader(t) - αᵢ(t))
```

### 4. Theoretical Safety Guarantees

Formal proofs ensuring collision-free morphological transformation and swarm synchronization convergence.

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Setup (PowerShell)

```powershell
# Clone the repository
git clone https://github.com/hmshujaatzaheer/AeroMorph.git

# Navigate to directory
cd AeroMorph

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install the package
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"

# Verify installation
python -c "import aeromorph; print(f'AeroMorph v{aeromorph.__version__} installed successfully')"
```

### Quick Install (PowerShell)

```powershell
git clone https://github.com/hmshujaatzaheer/AeroMorph.git; cd AeroMorph; pip install -e .
```

---

## 🚀 Quick Start

### Basic P2MA Decision

```python
import numpy as np
from aeromorph import MorphDecisionEngine, PerceptionState

# Initialize decision engine
engine = MorphDecisionEngine(config_space_dim=6, energy_budget=1000.0)

# Current morphing configuration
current_config = np.array([0.7, 0.3, 0.5, 0.5, 0.5, 0.4])

# Simulate perception (narrow passage detected)
perception = PerceptionState(
    lidar_points=np.random.randn(100, 3) * 0.5,  # Close obstacles
    timestamp=0.0
)

# Get autonomous morphing decision
decision = engine.decide(current_config=current_config, perception=perception)

print(f"Action: {decision.action.value}")
print(f"Target Config: {decision.target_config}")
print(f"Energy Cost: {decision.energy_cost:.2f} J")
```

### Swarm Coordination

```python
import numpy as np
from aeromorph import SwarmMorphProtocol, ExtendedRobotState
from aeromorph.core.state import Pose

# Initialize 5-robot swarm
protocol = SwarmMorphProtocol(num_robots=5, communication_range=15.0)

# Create swarm states
swarm_states = {
    i: ExtendedRobotState(
        pose=Pose(position=np.array([i*3.0, 0.0, 10.0])),
        morph_config=np.random.uniform(0.3, 0.7, 6),
        energy_state=4000.0 + np.random.uniform(-500, 500)
    ) for i in range(5)
}

# Coordinate synchronized morphing
target_config = np.array([0.3, 0.6, 0.5, 0.4, 0.5, 0.8])
result = protocol.coordinate_morph(swarm_states=swarm_states, target_config=target_config)

print(f"Leader: Robot {result.leader_id}")
print(f"Sync Time: {result.morph_timestamp:.3f}s")
print(f"Participating: {result.participating}")
```

### Energy-Aware Mission Planning

```python
import numpy as np
from aeromorph import EnergyAwarePlanner, ExtendedRobotState
from aeromorph.core.state import Pose

# Initialize planner
planner = EnergyAwarePlanner(morphing_budget_fraction=0.15)

# Define mission
start = ExtendedRobotState(
    pose=Pose(position=np.array([0.0, 0.0, 10.0])),
    morph_config=np.array([0.7, 0.3, 0.5, 0.5, 0.5, 0.4]),
    energy_state=5000.0
)

goal = ExtendedRobotState(
    pose=Pose(position=np.array([100.0, 50.0, 15.0])),
    morph_config=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
)

# Plan mission with optimal morphing schedule
plan = planner.plan(start_state=start, goal_state=goal, battery_capacity=5000.0)

print(f"Feasible: {plan.feasible}")
print(f"Morphing Events: {len(plan.morph_schedule)}")
print(f"Total Morph Energy: {plan.total_morph_energy:.1f} J")
```

---

## 🏗️ Architecture

```
AeroMorph/
├── aeromorph/
│   ├── core/                    # Core data structures
│   │   ├── decision_space.py    # Morphological Decision Space 𝒟ₘ
│   │   ├── config_space.py      # Morphing configuration manifold ℳ
│   │   ├── state.py             # Extended robot state representations
│   │   └── types.py             # Type definitions and enums
│   │
│   ├── algorithms/              # Decision algorithms
│   │   ├── p2ma.py              # Perception-to-Morphology-Action
│   │   ├── spatial_feasibility.py  # Collision-free verification
│   │   └── unified_optimizer.py # Joint trajectory-morphing optimization
│   │
│   ├── swarm/                   # Multi-robot coordination
│   │   └── ea_smp.py            # Energy-Aware Swarm Morphing Protocol
│   │
│   ├── planning/                # Mission planning
│   │   └── energy_aware.py      # Energy-constrained planning
│   │
│   └── utils/                   # Utilities
│       ├── geometry.py          # Geometric computations
│       └── visualization.py     # Plotting functions
│
├── examples/                    # Usage demonstrations
│   ├── basic_p2ma.py           # P2MA algorithm demo
│   ├── swarm_coordination.py   # Swarm morphing demo
│   └── energy_planning.py      # Mission planning demo
│
├── tests/                       # Unit tests
├── docs/                        # Documentation and proposal
└── setup.py                     # Package installation
```

---

## 📊 Research Validation Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Mission Success | >30% improvement | vs. fixed-morphology baselines |
| Energy Efficiency | >20% gain | through optimized morphing frequency |
| Swarm Synchronization | <100ms error | across 10-robot teams |
| Safety | Zero collisions | during autonomous morphing |

---

## 🧪 Running Tests (PowerShell)

```powershell
# Navigate to repository
cd AeroMorph

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install test dependencies
pip install pytest pytest-cov

# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_p2ma.py -v

# Run with coverage report
python -m pytest tests/ --cov=aeromorph --cov-report=html
```

---

## 🔧 Development (PowerShell)

```powershell
# Clone and setup for development
git clone https://github.com/hmshujaatzaheer/AeroMorph.git
cd AeroMorph
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Run linting
python -m flake8 aeromorph --select=E9,F63,F7,F82

# Format code (optional)
pip install black
python -m black aeromorph/

# Run examples
python examples/basic_p2ma.py
python examples/swarm_coordination.py
python examples/energy_planning.py
```

---

## 📚 Theoretical Foundation

### Extended Robot State

```
x_ext = [x_pose ∈ SE(3), x_vel ∈ ℝ⁶, α ∈ ℳ, E_state ∈ ℝ⁺]ᵀ
```

### Morphing Utility Function

```
U_morph(α, pₜ) = w₁·f_clearance(α, d) + w₂·f_aero(α, w) - w₃·E_morph(α)
```

### Unified Optimization

```
min     J = J_trajectory + J_morphing + J_swarm + J_energy
s.t.    x_{k+1} = f(x_k, u_k, α_k)     (Coupled dynamics)
        g(x_k, α_k, O) ≥ 0              (Collision avoidance)
        α_k ∈ ℳ_feasible                (Reachable configs)
        Σ_k E_morph,k ≤ E_budget        (Energy constraint)
```

---

## 🔗 Related Work

This research builds upon and addresses gaps in recent advances:

- **Morphing Mechanisms:** GOAT robot (Polzin et al., 2025), programmable lattices (Guan et al., 2025)
- **Aerial Transitions:** ATMO morphobot (Mandralis et al., 2025)
- **Control Strategies:** NMPC-based posture manipulation (Pandya, 2025)

AeroMorph addresses the critical gap: autonomous decision intelligence for morphing.

---

## 📖 Citation

If you use AeroMorph in your research, please cite:

```bibtex
@misc{zaheer2026aeromorph,
  author       = {Zaheer, H M Shujaat},
  title        = {AeroMorph: Unified Perception-Driven Morphological Adaptation 
                  Framework for Multi-Modal Aerial Robots},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/hmshujaatzaheer/AeroMorph}},
  note         = {Open-source research framework for autonomous morphing decisions}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**H M Shujaat Zaheer**

- Research Focus: Morphing Robotics, Autonomous Systems, AI/ML
- Email: shujabis@gmail.com
- GitHub: [@hmshujaatzaheer](https://github.com/hmshujaatzaheer)

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```powershell
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/AeroMorph.git
cd AeroMorph
git checkout -b feature/your-feature
# Make changes...
git add .
git commit -m "Add your feature"
git push origin feature/your-feature
# Create Pull Request on GitHub
```

---

<div align="center">

**AeroMorph** — Enabling Aerial Robots That Decide When to Morph

⭐ Star this repository if you find it useful for your research!

</div>
