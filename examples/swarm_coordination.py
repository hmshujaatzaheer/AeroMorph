"""
Swarm Coordination Example.

Demonstrates coordinated morphing across a multi-robot swarm
using the Energy-Aware Swarm Morphing Protocol (EA-SMP).

Author: H M Shujaat Zaheer
"""

import numpy as np
from aeromorph import SwarmMorphProtocol, ExtendedRobotState
from aeromorph.core.state import Pose, Velocity


def create_swarm_states(num_robots: int = 5) -> dict:
    """Create initial swarm states."""
    states = {}
    
    for i in range(num_robots):
        # Distribute robots in a formation
        angle = 2 * np.pi * i / num_robots
        radius = 5.0
        position = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            10.0  # Altitude
        ])
        
        states[i] = ExtendedRobotState(
            pose=Pose(position=position),
            velocity=Velocity(),
            morph_config=np.random.uniform(0.3, 0.7, 6),
            energy_state=4000.0 + np.random.uniform(-500, 500)
        )
    
    return states


def main():
    """Run swarm coordination demonstration."""
    print("=" * 60)
    print("AeroMorph: Swarm Morphing Coordination Demonstration")
    print("=" * 60)
    
    # Initialize swarm protocol
    num_robots = 5
    protocol = SwarmMorphProtocol(
        num_robots=num_robots,
        communication_range=15.0  # meters
    )
    
    # Create swarm states
    swarm_states = create_swarm_states(num_robots)
    
    print(f"\n--- Initial Swarm State ({num_robots} robots) ---")
    for robot_id, state in swarm_states.items():
        print(f"Robot {robot_id}: pos={state.position.round(1)}, "
              f"config={state.morph_config[:3].round(2)}..., "
              f"energy={state.energy_state:.0f}J")
    
    # Define target configuration for synchronized morphing
    target_config = np.array([0.3, 0.6, 0.5, 0.4, 0.5, 0.8])
    print(f"\nTarget configuration: {target_config}")
    
    # Coordinate swarm morphing
    print("\n--- Initiating Swarm Morphing Coordination ---")
    result = protocol.coordinate_morph(
        swarm_states=swarm_states,
        target_config=target_config,
        environment=None
    )
    
    print(f"\nCoordination Results:")
    print(f"  Leader elected: Robot {result.leader_id}")
    print(f"  Sync timestamp: {result.morph_timestamp:.3f}")
    print(f"  Participating robots: {result.participating}")
    print(f"  Excluded robots: {result.excluded}")
    print(f"  Synchronization error: {result.sync_error:.1f} ms")
    print(f"  Energy-balanced election: {result.energy_balanced}")
    
    # Check connectivity
    connectivity = protocol.get_connectivity()
    print(f"\n--- Communication Graph Analysis ---")
    print(f"Algebraic connectivity (λ₂): {connectivity:.3f}")
    print(f"Graph is {'connected' if connectivity > 0 else 'disconnected'}")
    
    # Simulate consensus steps
    print("\n--- Simulating Consensus Protocol ---")
    from aeromorph.core.state import SwarmState
    
    swarm_state = SwarmState(robot_states=swarm_states)
    
    for step in range(5):
        updates = protocol.step_consensus(swarm_state, target_config, dt=0.1)
        
        # Apply updates
        for robot_id, new_config in updates.items():
            swarm_states[robot_id].morph_config = new_config
        
        swarm_state = SwarmState(robot_states=swarm_states)
        variance = swarm_state.get_config_variance()
        
        print(f"Step {step + 1}: Config variance = {variance:.4f}")
    
    is_synced = protocol.is_synchronized(swarm_state, tolerance=0.1)
    print(f"\nSwarm synchronized: {is_synced}")
    
    print("\n--- Final Swarm State ---")
    for robot_id, state in swarm_states.items():
        error = np.linalg.norm(state.morph_config - target_config)
        print(f"Robot {robot_id}: config error = {error:.4f}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
