# navigation/train_rl.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import sys
sys.path.append('..')
from simulation.envs.gridworld import GridWorldEnv

def make_env():
    """Create environment instance"""
    def _init():
        return GridWorldEnv(size=10, grid_size=64)
    return _init

def train_navigation_policy(
    total_timesteps=100000,
    save_freq=10000,
    eval_freq=5000,
    log_dir='experiments/runs/rl',
    model_name='ppo_navigation'
):
    """
    Train PPO agent for robot navigation
    
    Args:
        total_timesteps: Total training steps
        save_freq: Frequency to save checkpoints
        eval_freq: Frequency to evaluate
        log_dir: Directory for logs and models
        model_name: Name for the model
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'{log_dir}/models', exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([make_env()])
    eval_env = DummyVecEnv([make_env()])
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f'{log_dir}/models',
        name_prefix=model_name
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{log_dir}/best_model',
        log_path=f'{log_dir}/eval',
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    # Create PPO model
    model = PPO(
        'CnnPolicy',
        env,
        verbose=1,
        tensorboard_log=f'{log_dir}/tensorboard',
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f'{log_dir}/models/{model_name}_final')
    print(f"Training complete! Model saved to {log_dir}/models/{model_name}_final")
    
    return model

def test_policy(model_path, num_episodes=10):
    """Test trained policy"""
    env = GridWorldEnv(size=10, grid_size=64, render_mode='human')
    model = PPO.load(model_path)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            env.render()
        
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, Steps={steps}, "
              f"Collision={info.get('collision', False)}, "
              f"Goal Reached={info.get('goal_reached', False)}")
    
    env.close()

if __name__ == '__main__':
    # Train the policy
    model = train_navigation_policy(
        total_timesteps=50000,  # Start small
        save_freq=5000,
        eval_freq=2500
    )
    
    # Test the trained policy
    # test_policy('experiments/runs/rl/models/ppo_navigation_final.zip')
