"""
Test suite for simulation module.
"""

import pytest
import numpy as np
from simulation.envs.gridworld import GridWorldEnv
from simulation.sensors.camera import Camera
from simulation.robot.kinematics import Kinematics


def test_gridworld_creation():
    """Test environment creation."""
    env = GridWorldEnv()
    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None


def test_reset():
    """Test environment reset."""
    env = GridWorldEnv()
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(info, dict)


def test_step():
    """Test environment step."""
    env = GridWorldEnv()
    obs, info = env.reset()
    
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)


def test_camera():
    """Test camera rendering."""
    camera = Camera(resolution=(64, 64), fov=90)
    world_state = np.zeros((32, 32))
    robot_pos = np.array([16, 16])
    robot_angle = 0.0
    
    image = camera.render(world_state, robot_pos, robot_angle)
    
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8


def test_kinematics():
    """Test kinematics model."""
    kin = Kinematics()
    
    # Test forward kinematics
    wheel_vels = np.array([1.0, 1.0])
    v_lin, v_ang = kin.forward_kinematics(wheel_vels, dt=0.1)
    
    assert isinstance(v_lin, float)
    assert isinstance(v_ang, float)
    
    # Test inverse kinematics
    wheel_vels_inv = kin.inverse_kinematics(v_lin=0.1, v_angular=0.1)
    
    assert wheel_vels_inv.shape == (2,)
    
    # Test pose update
    pose = np.array([0.0, 0.0, 0.0])
    pose_new = kin.update_pose(pose, v_linear=0.1, v_angular=0.0, dt=0.1)
    
    assert pose_new.shape == (3,)
    assert pose_new[0] != pose[0]  # x should change


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
