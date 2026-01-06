# AeroMorph: Unified Perception-Driven Morphological Adaptation Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-PhD%20Proposal-purple.svg)](docs/proposal.pdf)
[![Status](https://img.shields.io/badge/Status-Research%20Framework-orange.svg)](#status)

> **Integrating Environmental Awareness, Predictive Planning, and Swarm Coordination for Autonomous Shape-Changing Flight**

<p align="center">
  <img src="docs/images/aeromorph_architecture.png" alt="AeroMorph Architecture" width="800"/>
</p>

## 🎯 Overview

**AeroMorph** is a unified research framework for autonomous morphing decisions in aerial robots. Unlike existing systems that pre-program morphing triggers, AeroMorph enables robots to **autonomously decide when, where, how, and with whom to morph** based on real-time perception, spatial constraints, energy budgets, and swarm coordination.

### The Problem We Solve

| Current Paradigm | AeroMorph Paradigm |
|------------------|-------------------|
| Design morphing mechanisms | **Perceive** environment |
| Pre-program triggers | **Decide** autonomously |
| Execute open-loop | **Verify** spatial feasibility |
| React after contact | **Adapt** predictively |

### Key Innovation: Morphological Decision Space

We formalize morphing decisions through the **Morphological Decision Space**:

```
𝒟_M = 𝒫 × 𝒮 × ℰ × 𝒞
```

Where:
- **𝒫** (Perception Space): Multi-modal sensor fusion for environmental understanding
- **𝒮** (Spatial Space): Collision-free morphing feasibility verification
- **ℰ** (Energy Space): Mission-level energy-aware morphing optimization
- **𝒞** (Coordination Space): Swarm synchronization for multi-robot morphing

## ✨ Features

### 1. Perception-to-Morphology-Action (P2MA) Algorithm
- Multi-modal sensor fusion (LiDAR, Camera, IMU)
- Autonomous morphing trigger based on environmental utility
- Real-time decision-making pipeline

### 2. Spatial Feasibility Analysis
- Collision-free morphing path verification
- Time-varying robot geometry handling
- Self-collision detection during transformation

### 3. Energy-Aware Mission Planning
- Mission-level morphing budget optimization
- Energy cost modeling for morphing actions
- Battery-constrained morphing decisions

### 4. Swarm Morphing Protocol (EA-SMP)
- Distributed consensus for coordinated morphing
- Energy-balanced leader election
- Synchronized transformation across multi-robot teams

### 5. Unified Optimization Framework
- Joint trajectory-morphing-swarm optimization
- Model Predictive Control with morphing dynamics
- Stability guarantees during shape transitions

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- PyTorch 2.0 or higher
- ROS2 Humble (optional, for hardware integration)

### From Source (Recommended for Research)

```bash
# Clone the repository
git clone https://github.com/hmshujaatzaheer/AeroMorph.git
cd AeroMorph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Quick Install

```bash
pip install aeromorph
```

## 🚀 Quick Start

### Basic Usage: P2MA Decision Making

```python
from aeromorph import MorphDecisionEngine, PerceptionState

# Initialize the decision engine
engine = MorphDecisionEngine(
    config_space_dim=6,  # 6-DOF morphing configuration
    energy_budget=1000.0  # Joules
)

# Create perception state from sensors
perception = PerceptionState(
    lidar_points=lidar_data,
    camera_depth=depth_image,
    imu_data=imu_reading,
    wind_estimate=wind_vector
)

# Get autonomous morphing decision
decision = engine.decide(
    current_config=current_morphology,
    perception=perception,
    trajectory=planned_path
)

if decision.action == "MORPH":
    print(f"Morphing to: {decision.target_config}")
    print(f"Energy cost: {decision.energy_cost:.2f} J")
    print(f"Feasibility: {decision.spatial_feasibility}")
else:
    print("Holding current configuration")
```

### Spatial Feasibility Verification

```python
from aeromorph.algorithms import SpatialFeasibilityChecker

# Initialize checker with environment
checker = SpatialFeasibilityChecker(
    safety_margin=0.1,  # meters
    interpolation_steps=20
)

# Verify morphing is collision-free
result = checker.verify(
    start_config=alpha_start,
    target_config=alpha_target,
    environment=occupancy_grid
)

if result.feasible:
    print("Morphing path is collision-free")
    print(f"Trajectory: {result.trajectory}")
else:
    print(f"Collision detected at: {result.collision_point}")
```

### Swarm Coordination

```python
from aeromorph.swarm import SwarmMorphProtocol

# Initialize swarm protocol
protocol = SwarmMorphProtocol(
    num_robots=5,
    communication_range=10.0  # meters
)

# Coordinate swarm morphing
sync_result = protocol.coordinate_morph(
    swarm_states=robot_states,
    target_config=swarm_target,
    energy_states=battery_levels
)

print(f"Leader: Robot {sync_result.leader_id}")
print(f"Sync time: {sync_result.morph_timestamp}")
print(f"Participating robots: {sync_result.participating}")
```

### Energy-Aware Mission Planning

```python
from aeromorph.planning import EnergyAwarePlanner

# Initialize planner
planner = EnergyAwarePlanner(
    morphing_budget_fraction=0.15,  # 15% of energy for morphing
    prediction_horizon=50
)

# Plan mission with morphing schedule
plan = planner.plan(
    start_state=current_state,
    goal_state=mission_goal,
    environment=env_map,
    battery_capacity=5000.0  # Joules
)

print(f"Planned morphing events: {len(plan.morph_schedule)}")
print(f"Total morphing energy: {plan.total_morph_energy:.2f} J")
print(f"Mission feasibility: {plan.feasible}")
```

## 📁 Repository Structure

```
AeroMorph/
├── aeromorph/                    # Main package
│   ├── __init__.py
│   ├── core/                     # Core data structures
│   │   ├── __init__.py
│   │   ├── decision_space.py     # Morphological Decision Space 𝒟_M
│   │   ├── config_space.py       # Morphing configuration manifold
│   │   ├── state.py              # Extended robot state
│   │   └── types.py              # Type definitions
│   │
│   ├── algorithms/               # Novel algorithms
│   │   ├── __init__.py
│   │   ├── p2ma.py               # Perception-to-Morphology-Action
│   │   ├── spatial_feasibility.py # Collision-free verification
│   │   └── unified_optimizer.py  # Joint optimization
│   │
│   ├── perception/               # Sensor fusion
│   │   ├── __init__.py
│   │   ├── sensor_fusion.py      # EKF-based fusion
│   │   ├── wind_estimator.py     # Wind field estimation
│   │   └── occupancy.py          # 3D occupancy grid
│   │
│   ├── planning/                 # Mission planning
│   │   ├── __init__.py
│   │   ├── energy_aware.py       # Energy-constrained planning
│   │   ├── predictive_morph.py   # Anticipatory morphing
│   │   └── mpc_controller.py     # Model Predictive Control
│   │
│   ├── swarm/                    # Multi-robot coordination
│   │   ├── __init__.py
│   │   ├── ea_smp.py             # Energy-Aware Swarm Morphing Protocol
│   │   ├── consensus.py          # Distributed consensus
│   │   └── leader_election.py    # Energy-balanced election
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── geometry.py           # Geometric computations
│       ├── visualization.py      # Plotting and visualization
│       └── logging.py            # Structured logging
│
├── examples/                     # Usage examples
│   ├── basic_p2ma.py             # P2MA algorithm demo
│   ├── spatial_check.py          # Feasibility verification
│   ├── swarm_coordination.py     # Multi-robot demo
│   ├── energy_planning.py        # Energy-aware planning
│   └── full_pipeline.py          # Complete integration
│
├── tests/                        # Unit and integration tests
│   ├── test_decision_space.py
│   ├── test_p2ma.py
│   ├── test_spatial.py
│   ├── test_swarm.py
│   └── test_integration.py
│
├── docs/                         # Documentation
│   ├── images/
│   ├── API.md
│   ├── THEORY.md
│   └── proposal.pdf
│
├── scripts/                      # Utility scripts
│   ├── setup_ros2.sh
│   └── run_simulation.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── setup.py
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## 🔬 Theoretical Foundation

### Extended Robot State

The robot state incorporating morphology:

```
x_ext = [x_pose ∈ SE(3), x_vel ∈ ℝ⁶, α ∈ ℳ, E_state ∈ ℝ⁺]ᵀ
```

### Morphing Utility Function

```
U_morph(α, p_t) = w₁·f_clearance(α, d) + w₂·f_aero(α, w) - w₃·E_morph(α)
```

### Swarm Consensus Dynamics

```
α̇ᵢ(t) = Σⱼ∈𝒩ᵢ aᵢⱼ(αⱼ(t) - αᵢ(t)) + bᵢ(α_leader(t) - αᵢ(t))
```

### Unified Optimization

```
min_{u,α} J = Σ||x_k - x_ref||²_Q + λ₁||Δα||² + λ₂·E_morph + λ₃·Sync_error

subject to:
    x_{k+1} = f(x_k, u_k, α_k)           # Coupled dynamics
    g(x_k, α_k, 𝒪) ≥ 0                   # Collision avoidance
    α_k ∈ ℳ_feasible                      # Reachable configs
    ||α̇_k|| ≤ α̇_max                       # Morphing rate limit
    Σ E_morph,k ≤ E_budget                # Energy constraint
```

## 📊 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Multi-terrain mission success | >30% improvement | Research |
| Energy efficiency gain | >20% vs fixed morphology | Research |
| Swarm synchronization error | <100ms across 10 robots | Research |
| Spatial safety | 0 collisions during morphing | Research |
| Perception-to-morph latency | <200ms | Research |

## 🔗 Related Publications

This framework builds upon recent advances in morphing robotics:

1. Polzin et al. "Robotic locomotion through active and passive morphological adaptation" - *Science Robotics* 2025
2. Guan et al. "Lattice structure musculoskeletal robots" - *Science Advances* 2025
3. Mandralis et al. "ATMO: Aerially transforming morphobot" - *Communications Engineering* 2025

## ⚠️ Status

**This is a research framework accompanying a PhD proposal.** The implementation provides:

- ✅ Complete architectural design
- ✅ Algorithm specifications with pseudocode
- ✅ Core data structures and interfaces
- ✅ Example usage patterns
- 🔄 Simulation validation (in progress)
- 📋 Hardware integration (planned)

The framework is designed for research exploration and extension. Contributions implementing specific components are welcome.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install dev dependencies
git clone https://github.com/hmshujaatzaheer/AeroMorph.git
cd AeroMorph
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black aeromorph/
```

## 📄 Citation

If you use AeroMorph in your research, please cite:

```bibtex
@misc{zaheer2026aeromorph,
  title={AeroMorph: Unified Perception-Driven Morphological Adaptation 
         Framework for Multi-Modal Aerial Robots},
  author={Zaheer, H M Shujaat},
  year={2026},
  note={PhD Research Proposal},
  url={https://github.com/hmshujaatzaheer/AeroMorph}
}
```

## 📧 Contact

- **Author:** H M Shujaat Zaheer
- **Email:** shujabis@gmail.com
- **GitHub:** [@hmshujaatzaheer](https://github.com/hmshujaatzaheer)

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>AeroMorph: Making morphing decisions as natural as flight control</b>
</p>
