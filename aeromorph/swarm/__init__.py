"""
Swarm coordination module for AeroMorph framework.

Implements distributed protocols for coordinated morphing across
multi-robot teams with energy awareness and fault tolerance.
"""

from aeromorph.swarm.ea_smp import (
    SwarmMorphProtocol,
    SwarmConfig,
    SwarmRole,
    LeaderElection,
    ConsensusProtocol,
    TimeSynchronization
)

__all__ = [
    "SwarmMorphProtocol",
    "SwarmConfig",
    "SwarmRole",
    "LeaderElection",
    "ConsensusProtocol",
    "TimeSynchronization",
]
