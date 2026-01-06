"""
Basic P2MA Algorithm Example.

Demonstrates autonomous morphing decision-making using the
Perception-to-Morphology-Action algorithm.

Author: H M Shujaat Zaheer
"""

import numpy as np
from aeromorph import (
    MorphDecisionEngine,
    PerceptionState,
    MorphingConfigSpace
)


def main():
    """Run basic P2MA demonstration."""
    print("=" * 60)
    print("AeroMorph: P2MA Algorithm Demonstration")
    print("=" * 60)
    
    # Initialize decision engine
    engine = MorphDecisionEngine(
        config_space_dim=6,
        energy_budget=1000.0  # Joules
    )
    
    # Current morphing configuration (normalized 0-1)
    current_config = np.array([0.7, 0.3, 0.5, 0.5, 0.5, 0.4])
    print(f"\nCurrent configuration: {current_config}")
    
    # Simulate perception data
    # Scenario 1: Wide open space - no morphing needed
    print("\n--- Scenario 1: Open Environment ---")
    perception_open = PerceptionState(
        lidar_points=np.random.randn(1000, 3) * 10 + np.array([20, 0, 0]),
        imu_data=np.array([0.1, 0.1, 9.8, 0.01, 0.01, 0.01]),
        timestamp=0.0
    )
    
    decision = engine.decide(
        current_config=current_config,
        perception=perception_open
    )
    
    print(f"Decision: {decision.action.value}")
    print(f"Utility: {decision.utility:.3f}")
    print(f"Reason: {decision.trigger_reason}")
    
    # Scenario 2: Narrow passage detected
    print("\n--- Scenario 2: Narrow Passage ---")
    # Create LiDAR points representing a narrow passage
    left_wall = np.random.randn(200, 3) * 0.1 + np.array([5, -0.8, 0])
    right_wall = np.random.randn(200, 3) * 0.1 + np.array([5, 0.8, 0])
    narrow_points = np.vstack([left_wall, right_wall])
    
    perception_narrow = PerceptionState(
        lidar_points=narrow_points,
        imu_data=np.array([0.1, 0.1, 9.8, 0.01, 0.01, 0.01]),
        timestamp=1.0
    )
    
    decision = engine.decide(
        current_config=current_config,
        perception=perception_narrow
    )
    
    print(f"Decision: {decision.action.value}")
    print(f"Utility: {decision.utility:.3f}")
    print(f"Reason: {decision.trigger_reason}")
    if decision.target_config is not None:
        print(f"Target config: {decision.target_config}")
        print(f"Energy cost: {decision.energy_cost:.2f} J")
    
    # Scenario 3: High turbulence
    print("\n--- Scenario 3: Turbulent Conditions ---")
    perception_turbulent = PerceptionState(
        lidar_points=np.random.randn(1000, 3) * 10 + np.array([20, 0, 0]),
        imu_data=np.array([2.0, 1.5, 12.0, 0.5, 0.4, 0.3]),  # High accelerations
        timestamp=2.0
    )
    
    decision = engine.decide(
        current_config=current_config,
        perception=perception_turbulent
    )
    
    print(f"Decision: {decision.action.value}")
    print(f"Utility: {decision.utility:.3f}")
    print(f"Reason: {decision.trigger_reason}")
    if decision.target_config is not None:
        print(f"Target config: {decision.target_config}")
        print(f"Energy cost: {decision.energy_cost:.2f} J")
    
    # Print decision statistics
    print("\n--- Decision Statistics ---")
    stats = engine.get_decision_stats()
    print(f"Total decisions: {stats['total']}")
    print(f"Morph decisions: {stats['morphs']}")
    print(f"Hold decisions: {stats['holds']}")
    print(f"Morph rate: {stats['morph_rate']:.1%}")
    print(f"Total energy used: {stats['total_energy_used']:.2f} J")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
