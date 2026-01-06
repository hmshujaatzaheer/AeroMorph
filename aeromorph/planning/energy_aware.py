"""
Energy-Aware Mission Planning for Morphing Robots.

This module implements mission-level planning that optimizes morphing
frequency and extent based on energy budgets.

Reference: AeroMorph PhD Research Proposal, Section 3.4
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np

from aeromorph.core.types import MissionPlan
from aeromorph.core.state import ExtendedRobotState
from aeromorph.core.config_space import MorphingConfigSpace


@dataclass
class PlannerConfig:
    """Configuration for energy-aware planner."""
    morphing_budget_fraction: float = 0.15
    reserve_fraction: float = 0.10
    prediction_horizon: int = 50
    min_morph_benefit: float = 0.1
    max_morphs_per_mission: int = 20
    waypoint_spacing: float = 5.0


class EnergyModel:
    """Energy consumption model for morphing aerial robots."""
    
    def __init__(self, config_space: MorphingConfigSpace):
        self.config_space = config_space
        self.hover_power = 100.0
        self.cruise_power = 80.0
    
    def estimate_flight_energy(self, trajectory: np.ndarray, 
                               config: np.ndarray, speed: float = 5.0) -> float:
        if len(trajectory) < 2:
            return 0.0
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        total_distance = np.sum(distances)
        flight_time = total_distance / speed
        wingspan_factor = 0.5 + 1.0 * config[0]
        efficiency = 0.8 + 0.2 * wingspan_factor
        power = self.cruise_power / efficiency
        return power * flight_time
    
    def estimate_morph_energy(self, start: np.ndarray, end: np.ndarray) -> float:
        return self.config_space.compute_energy_cost(start, end)


class MorphingScheduler:
    """Schedules morphing events across a mission trajectory."""
    
    def __init__(self, config_space: MorphingConfigSpace,
                 energy_model: EnergyModel, config: PlannerConfig):
        self.config_space = config_space
        self.energy_model = energy_model
        self.planner_config = config
    
    def schedule(self, trajectory: np.ndarray, features: List[Dict],
                 energy_budget: float, initial_config: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        schedule = []
        current_config = initial_config.copy()
        remaining_budget = energy_budget
        
        for i, feat in enumerate(features):
            target, benefit = self._evaluate_need(current_config, feat)
            if benefit > self.planner_config.min_morph_benefit:
                cost = self.energy_model.estimate_morph_energy(current_config, target)
                if cost <= remaining_budget:
                    schedule.append((i, target))
                    remaining_budget -= cost
                    current_config = target.copy()
            if len(schedule) >= self.planner_config.max_morphs_per_mission:
                break
        return schedule
    
    def _evaluate_need(self, current: np.ndarray, feat: Dict) -> Tuple[np.ndarray, float]:
        passage = feat.get('passage_width', float('inf'))
        turbulence = feat.get('turbulence', 0.0)
        
        if passage < 2.0:
            return self.config_space.get_narrow_config(), 1.0
        elif turbulence > 0.5:
            return self.config_space.get_stable_config(), turbulence * 0.5
        elif passage > 5.0 and turbulence < 0.2:
            return self.config_space.get_efficient_config(), 0.3
        return current, 0.0


class EnergyAwarePlanner:
    """Energy-Aware Mission Planner for Morphing Robots."""
    
    def __init__(self, config_dim: int = 6, morphing_budget_fraction: float = 0.15,
                 prediction_horizon: int = 50, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig(
            morphing_budget_fraction=morphing_budget_fraction,
            prediction_horizon=prediction_horizon
        )
        self.config_space = MorphingConfigSpace(dim=config_dim)
        self.energy_model = EnergyModel(self.config_space)
        self.scheduler = MorphingScheduler(self.config_space, self.energy_model, self.config)
    
    def plan(self, start_state: ExtendedRobotState, goal_state: ExtendedRobotState,
             environment: Optional[np.ndarray] = None,
             battery_capacity: float = 5000.0) -> MissionPlan:
        morphing_budget = battery_capacity * self.config.morphing_budget_fraction
        trajectory = self._generate_trajectory(start_state.position, goal_state.position)
        features = self._estimate_features(trajectory, environment)
        morph_schedule = self.scheduler.schedule(trajectory, features, morphing_budget,
                                                  start_state.morph_config)
        
        total_morph_energy = sum(
            self.energy_model.estimate_morph_energy(
                start_state.morph_config if i == 0 else morph_schedule[i-1][1], config
            ) for i, (_, config) in enumerate(morph_schedule)
        )
        
        flight_energy = self.energy_model.estimate_flight_energy(trajectory, start_state.morph_config)
        feasible = (flight_energy + total_morph_energy) <= battery_capacity * (1 - self.config.reserve_fraction)
        distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        
        return MissionPlan(
            trajectory=[trajectory[i] for i in range(len(trajectory))],
            morph_schedule=[(i * 1.0, c) for i, c in morph_schedule],
            total_morph_energy=total_morph_energy,
            estimated_duration=distance / 5.0,
            feasible=feasible
        )
    
    def _generate_trajectory(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(goal - start)
        n = max(2, int(distance / self.config.waypoint_spacing))
        t = np.linspace(0, 1, n)
        return np.array([start + ti * (goal - start) for ti in t])
    
    def _estimate_features(self, trajectory: np.ndarray, env: Optional[np.ndarray]) -> List[Dict]:
        features = []
        for wp in trajectory:
            feat = {'passage_width': float('inf'), 'turbulence': 0.0}
            if env is not None and env.ndim == 2:
                distances = np.linalg.norm(env - wp, axis=1)
                feat['passage_width'] = np.min(distances) * 2 if len(distances) > 0 else float('inf')
            feat['turbulence'] = max(0, 1 - wp[2] / 20.0) * 0.5 if len(wp) > 2 else 0.0
            features.append(feat)
        return features
