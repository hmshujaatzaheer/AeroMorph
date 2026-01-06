"""
Energy-Aware Planning Example.

Demonstrates mission planning with morphing energy budget optimization.

Author: H M Shujaat Zaheer
"""

import numpy as np
from aeromorph import EnergyAwarePlanner, ExtendedRobotState
from aeromorph.core.state import Pose


def main():
    """Run energy-aware planning demonstration."""
    print("=" * 60)
    print("AeroMorph: Energy-Aware Mission Planning Demonstration")
    print("=" * 60)
    
    # Initialize planner
    planner = EnergyAwarePlanner(
        config_dim=6,
        morphing_budget_fraction=0.15,
        prediction_horizon=50
    )
    
    # Define mission
    start_state = ExtendedRobotState(
        pose=Pose(position=np.array([0.0, 0.0, 10.0])),
        morph_config=np.array([0.7, 0.3, 0.5, 0.5, 0.5, 0.4]),
        energy_state=5000.0
    )
    
    goal_state = ExtendedRobotState(
        pose=Pose(position=np.array([100.0, 50.0, 15.0])),
        morph_config=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    )
    
    print(f"\nMission Parameters:")
    print(f"  Start: {start_state.position}")
    print(f"  Goal: {goal_state.position}")
    print(f"  Battery: {start_state.energy_state:.0f} J")
    print(f"  Morphing budget: {start_state.energy_state * 0.15:.0f} J (15%)")
    
    # Create simple obstacle environment
    obstacles = np.random.randn(50, 3) * 10 + np.array([50, 25, 12])
    
    # Plan mission
    print("\n--- Planning Mission ---")
    plan = planner.plan(
        start_state=start_state,
        goal_state=goal_state,
        environment=obstacles,
        battery_capacity=5000.0
    )
    
    print(f"\nMission Plan:")
    print(f"  Feasible: {plan.feasible}")
    print(f"  Duration: {plan.estimated_duration:.1f} s")
    print(f"  Waypoints: {len(plan.trajectory)}")
    print(f"  Morphing events: {len(plan.morph_schedule)}")
    print(f"  Total morphing energy: {plan.total_morph_energy:.1f} J")
    
    if plan.morph_schedule:
        print("\n  Morphing Schedule:")
        for i, (time, config) in enumerate(plan.morph_schedule):
            print(f"    Event {i+1}: t={time:.1f}s, config={config[:3].round(2)}...")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
