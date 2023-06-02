"""
Basic Training Example: Train a Hybrid PPO Agent on SupportBot Environment

This example demonstrates:
1. Environment setup with custom configuration
2. Agent initialization with LLM integration
3. Training loop with curriculum learning
4. Evaluation and model saving
"""

import os
import yaml
import logging
from pathlib import Path
import torch
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from src.agent_rllib.envs import SupportBotEnv
from src.agent_rllib.agents import HybridPPOAgent
from src.agent_rllib.training.callbacks import CustomCallbacks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_ray():
    """Initialize Ray cluster."""
    if not ray.is_initialized():
        ray.init(
            num_cpus=8,
            num_gpus=1 if torch.cuda.is_available() else 0,
            object_store_memory=2000000000,  # 2GB
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
        )
        logger.info("Ray initialized successfully")


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_ppo_config(config_dict: dict) -> PPOConfig:
    """Create PPO configuration from dictionary."""
    config = (
        PPOConfig()
        .environment(
            env=SupportBotEnv,
            env_config=config_dict.get("env_config", {})
        )
        .framework("torch")
        .training(
            lr=config_dict.get("lr", 3e-4),
            train_batch_size=config_dict.get("train_batch_size", 4096),
            sgd_minibatch_size=config_dict.get("sgd_minibatch_size", 128),
            num_sgd_iter=config_dict.get("num_sgd_iter", 10),
            gamma=config_dict.get("gamma", 0.99),
            lambda_=config_dict.get("lambda", 0.95),
            clip_param=config_dict.get("clip_param", 0.2),
            vf_clip_param=config_dict.get("vf_clip_param", 10.0),
            entropy_coeff=config_dict.get("entropy_coeff", 0.01),
            model=config_dict.get("model", {})
        )
        .rollouts(
            num_rollout_workers=config_dict.get("num_workers", 4),
            rollout_fragment_length=config_dict.get("rollout_fragment_length", 200)
        )
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0,
            num_cpus_per_worker=1
        )
        .evaluation(
            evaluation_interval=config_dict.get("evaluation_interval", 50),
            evaluation_duration=config_dict.get("evaluation_duration", 10),
            evaluation_config={
                "env_config": {
                    "difficulty": "hard",
                    "curriculum_level": 1.0,
                    "render_mode": None
                },
                "explore": False
            }
        )
        .callbacks(CustomCallbacks)
    )
    
    return config


def train_agent():
    """Main training function."""
    logger.info("Starting Agent-RLlib training...")
    
    # Setup
    setup_ray()
    
    # Load configuration
    config_path = "configs/ppo_default.yaml"
    config_dict = load_config(config_path)
    
    # Create PPO configuration
    ppo_config = create_ppo_config(config_dict)
    
    # Build algorithm
    algo = ppo_config.build()
    
    # Training loop
    best_reward = float('-inf')
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    try:
        for iteration in range(config_dict.get("stop", {}).get("training_iteration", 1000)):
            # Train one iteration
            result = algo.train()
            
            # Log metrics
            episode_reward_mean = result["episode_reward_mean"]
            episode_len_mean = result["episode_len_mean"]
            
            logger.info(
                f"Iteration {iteration}: "
                f"Reward={episode_reward_mean:.2f}, "
                f"Length={episode_len_mean:.1f}, "
                f"FPS={result.get('timers', {}).get('sample_throughput', 0):.0f}"
            )
            
            # Save best model
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                checkpoint_path = algo.save(checkpoint_dir / "best_model")
                logger.info(f"New best model saved: {checkpoint_path}")
            
            # Periodic checkpoint
            if iteration % 100 == 0:
                checkpoint_path = algo.save(checkpoint_dir / f"checkpoint_{iteration}")
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Early stopping
            if episode_reward_mean >= config_dict.get("stop", {}).get("episode_reward_mean", 50.0):
                logger.info(f"Target reward reached: {episode_reward_mean}")
                break
                
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    finally:
        # Final save
        final_checkpoint = algo.save(checkpoint_dir / "final_model")
        logger.info(f"Final model saved: {final_checkpoint}")
        
        # Cleanup
        algo.stop()
        ray.shutdown()
        logger.info("Training completed successfully")


def evaluate_model(checkpoint_path: str, num_episodes: int = 10):
    """Evaluate a trained model."""
    logger.info(f"Evaluating model: {checkpoint_path}")
    
    setup_ray()
    
    # Load configuration
    config_dict = load_config("configs/ppo_default.yaml")
    ppo_config = create_ppo_config(config_dict)
    
    # Load trained model
    algo = ppo_config.build()
    algo.restore(checkpoint_path)
    
    # Create evaluation environment
    env = SupportBotEnv(
        difficulty="hard",
        curriculum_level=1.0,
        render_mode="human"
    )
    
    total_rewards = []
    total_lengths = []
    success_rate = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        logger.info(f"\n=== Episode {episode + 1} ===")
        
        while not done:
            # Get action from trained agent
            action = algo.compute_single_action(obs, explore=False)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Render environment
            env.render()
            
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        
        # Check if issue was resolved successfully
        if info.get("issue_resolved", False):
            success_rate += 1
            
        logger.info(
            f"Episode {episode + 1} completed: "
            f"Reward={episode_reward:.2f}, "
            f"Length={episode_length}, "
            f"Success={info.get('issue_resolved', False)}"
        )
    
    # Calculate statistics
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_length = sum(total_lengths) / len(total_lengths)
    success_rate = success_rate / num_episodes
    
    logger.info(f"\n=== Evaluation Results ===")
    logger.info(f"Average Reward: {avg_reward:.2f}")
    logger.info(f"Average Length: {avg_length:.1f}")
    logger.info(f"Success Rate: {success_rate:.2%}")
    
    env.close()
    algo.stop()
    ray.shutdown()
    
    return {
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "success_rate": success_rate,
        "rewards": total_rewards,
        "lengths": total_lengths
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or evaluate Agent-RLlib model")
    parser.add_argument("--mode", choices=["train", "eval"], default="train",
                       help="Mode: train or evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint path for evaluation")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_agent()
    elif args.mode == "eval":
        if args.checkpoint is None:
            logger.error("Checkpoint path required for evaluation")
            exit(1)
        evaluate_model(args.checkpoint, args.episodes)
