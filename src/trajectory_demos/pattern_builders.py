"""Dispatch trajectory candidate builders by taxonomy pattern."""

from trajectory_demos.around_candidates import build_around_candidates
from trajectory_demos.linear_candidates import build_linear_candidates
from trajectory_demos.spherical_candidates import build_spherical_candidates
from trajectory_demos.sampler_candidates import (
    build_rotation_candidates,
)


def build_candidates(scene_name: str, controller, config, trajectory: str):
    if trajectory.startswith("around_"):
        return build_around_candidates(scene_name, controller, config, trajectory)
    if trajectory.startswith("spherical_"):
        return build_spherical_candidates(scene_name, controller, config, trajectory)
    if trajectory.startswith("rotation_"):
        return build_rotation_candidates(scene_name, controller, config, trajectory)
    return build_linear_candidates(scene_name, controller, config, trajectory)
