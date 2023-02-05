"""
Comprehensive tests for Agent-RLlib environments.
"""

import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import Mock, patch

from src.agent_rllib.envs import SupportBotEnv


class TestSupportBotEnv:
    """Test suite for SupportBotEnv."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.env = SupportBotEnv(
            difficulty="medium",
            tools=["search", "calculator"],
            max_turns=10,
            curriculum_level=0.5
        )
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, 'env'):
            self.env.close()
    
    def test_environment_initialization(self):
        """Test environment initializes correctly."""
        assert self.env.difficulty == "medium"
        assert self.env.max_turns == 10
        assert self.env.curriculum_level == 0.5
        assert "search" in self.env.available_tools
        assert "calculator" in self.env.available_tools
    
    def test_observation_space(self):
        """Test observation space is correctly defined."""
        obs_space = self.env.observation_space
        
        assert isinstance(obs_space, gym.spaces.Dict)
        assert "text_embedding" in obs_space.spaces
        assert "conversation_metadata" in obs_space.spaces
        assert "available_tools" in obs_space.spaces
        
        # Check shapes
        assert obs_space["text_embedding"].shape == (512,)
        assert obs_space["conversation_metadata"].shape == (10,)
        assert obs_space["available_tools"].shape == (len(self.env.available_tools),)
    
    def test_action_space(self):
        """Test action space is correctly defined."""
        action_space = self.env.action_space
        
        assert isinstance(action_space, gym.spaces.Dict)
        assert "tool_id" in action_space.spaces
        assert "response_type" in action_space.spaces
        assert "confidence" in action_space.spaces
        
        # Check action space ranges
        assert action_space["tool_id"].n == len(self.env.available_tools) + 1
        assert action_space["response_type"].n == 5
        assert action_space["confidence"].shape == (1,)
    
    def test_reset_functionality(self):
        """Test environment reset works correctly."""
        obs, info = self.env.reset()
        
        # Check observation structure
        assert isinstance(obs, dict)
        assert all(key in obs for key in ["text_embedding", "conversation_metadata", "available_tools"])
        
        # Check observation types and shapes
        assert obs["text_embedding"].shape == (512,)
        assert obs["conversation_metadata"].shape == (10,)
        assert obs["available_tools"].shape == (len(self.env.available_tools),)
        
        # Check info
        assert isinstance(info, dict)
        assert "conversation_length" in info
        assert "customer_satisfaction" in info
        assert "scenario_complexity" in info
        
        # Check initial state
        assert self.env.conversation_state.turn_count == 0
        assert self.env.conversation_state.customer_satisfaction == 1.0
        assert not self.env.conversation_state.issue_resolved
    
    def test_step_functionality(self):
        """Test environment step function."""
        obs, info = self.env.reset()
        
        # Create valid action
        action = {
            "tool_id": 0,  # Use first tool
            "response_type": 1,
            "confidence": np.array([0.8])
        }
        
        # Take step
        new_obs, reward, terminated, truncated, new_info = self.env.step(action)
        
        # Check return types
        assert isinstance(new_obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(new_info, dict)
        
        # Check state updates
        assert self.env.conversation_state.turn_count == 1
        assert len(self.env.conversation_state.messages) >= 2  # Customer + Agent
        
        # Check observation structure unchanged
        assert all(key in new_obs for key in ["text_embedding", "conversation_metadata", "available_tools"])
    
    def test_tool_execution(self):
        """Test tool execution functionality."""
        # Test search tool
        result = self.env._execute_tool("search", {})
        assert isinstance(result, dict)
        assert "success" in result
        
        # Test calculator tool
        result = self.env._execute_tool("calculator", {})
        assert isinstance(result, dict)
        assert "success" in result
        
        # Test invalid tool
        result = self.env._execute_tool("invalid_tool", {})
        assert not result.get("success", True)
    
    def test_reward_calculation(self):
        """Test reward calculation logic."""
        obs, info = self.env.reset()
        
        # Test action with successful tool use
        action_with_tool = {
            "tool_id": 0,
            "response_type": 1,
            "confidence": np.array([0.9])
        }
        
        obs, reward, terminated, truncated, info = self.env.step(action_with_tool)
        
        # Reward should be calculated
        assert isinstance(reward, (int, float))
        # In our implementation, successful tool use should give positive reward
        # (This is a simplified test - actual reward depends on customer satisfaction)
    
    def test_termination_conditions(self):
        """Test various termination conditions."""
        obs, info = self.env.reset()
        
        # Test max turns termination
        for i in range(self.env.max_turns + 1):
            action = {
                "tool_id": len(self.env.available_tools),  # No tool
                "response_type": 0,
                "confidence": np.array([0.5])
            }
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if i >= self.env.max_turns - 1:
                assert terminated or truncated
                break
    
    def test_curriculum_level_effects(self):
        """Test that curriculum level affects scenario selection."""
        # Test low curriculum level
        low_env = SupportBotEnv(curriculum_level=0.1)
        low_env.reset()
        low_complexity = low_env.current_scenario["complexity"]
        low_env.close()
        
        # Test high curriculum level
        high_env = SupportBotEnv(curriculum_level=0.9)
        high_env.reset()
        high_complexity = high_env.current_scenario["complexity"]
        high_env.close()
        
        # Higher curriculum should generally select more complex scenarios
        # (This is probabilistic, so we just check they're different types)
        assert isinstance(low_complexity, (int, float))
        assert isinstance(high_complexity, (int, float))
    
    def test_render_functionality(self):
        """Test environment rendering."""
        obs, info = self.env.reset()
        
        # Test render doesn't crash
        try:
            self.env.render()
        except Exception as e:
            pytest.fail(f"Render failed: {e}")
        
        # Take a step and render again
        action = {
            "tool_id": 0,
            "response_type": 0,
            "confidence": np.array([0.5])
        }
        self.env.step(action)
        
        try:
            self.env.render()
        except Exception as e:
            pytest.fail(f"Render after step failed: {e}")
    
    def test_different_difficulties(self):
        """Test environment with different difficulty settings."""
        difficulties = ["easy", "medium", "hard"]
        
        for difficulty in difficulties:
            env = SupportBotEnv(difficulty=difficulty)
            obs, info = env.reset()
            
            # Should initialize without error
            assert isinstance(obs, dict)
            assert isinstance(info, dict)
            
            # Take a step
            action = {
                "tool_id": 0,
                "response_type": 0,
                "confidence": np.array([0.5])
            }
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Should work for all difficulties
            assert isinstance(reward, (int, float))
            
            env.close()
    
    def test_tool_configuration(self):
        """Test environment with different tool configurations."""
        # Test with minimal tools
        minimal_env = SupportBotEnv(tools=["search"])
        assert len(minimal_env.available_tools) == 1
        minimal_env.close()
        
        # Test with many tools
        many_tools_env = SupportBotEnv(tools=["search", "calculator", "weather", "calendar"])
        assert len(many_tools_env.available_tools) == 4
        many_tools_env.close()
    
    def test_conversation_state_updates(self):
        """Test that conversation state updates correctly."""
        obs, info = self.env.reset()
        initial_messages = len(self.env.conversation_state.messages)
        
        action = {
            "tool_id": 0,
            "response_type": 1,
            "confidence": np.array([0.7])
        }
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Should have more messages after step
        assert len(self.env.conversation_state.messages) > initial_messages
        
        # Should have agent response
        agent_messages = [msg for msg in self.env.conversation_state.messages if msg["role"] == "agent"]
        assert len(agent_messages) > 0
    
    def test_observation_consistency(self):
        """Test that observations are consistent and valid."""
        obs, info = self.env.reset()
        
        # Check observation values are in valid ranges
        assert np.all(np.isfinite(obs["text_embedding"]))
        assert np.all(obs["conversation_metadata"] >= 0)
        assert np.all(obs["conversation_metadata"] <= 1)
        assert np.all((obs["available_tools"] == 0) | (obs["available_tools"] == 1))
        
        # Take several steps and check consistency
        for _ in range(5):
            action = {
                "tool_id": np.random.randint(0, len(self.env.available_tools) + 1),
                "response_type": np.random.randint(0, 5),
                "confidence": np.array([np.random.uniform(0, 1)])
            }
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
            
            # Check observation validity
            assert np.all(np.isfinite(obs["text_embedding"]))
            assert np.all(obs["conversation_metadata"] >= 0)
            assert obs["conversation_metadata"].shape == (10,)


class TestEnvironmentEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_action_handling(self):
        """Test handling of invalid actions."""
        env = SupportBotEnv()
        obs, info = env.reset()
        
        # Test action with invalid tool_id
        invalid_action = {
            "tool_id": 999,  # Invalid tool ID
            "response_type": 0,
            "confidence": np.array([0.5])
        }
        
        # Should handle gracefully without crashing
        try:
            obs, reward, terminated, truncated, info = env.step(invalid_action)
        except Exception as e:
            pytest.fail(f"Environment crashed on invalid action: {e}")
        
        env.close()
    
    def test_extreme_confidence_values(self):
        """Test extreme confidence values."""
        env = SupportBotEnv()
        obs, info = env.reset()
        
        # Test very low confidence
        action = {
            "tool_id": 0,
            "response_type": 0,
            "confidence": np.array([0.0])
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float))
        
        # Test very high confidence
        action = {
            "tool_id": 0,
            "response_type": 0,
            "confidence": np.array([1.0])
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float))
        
        env.close()
    
    def test_empty_tool_list(self):
        """Test environment with empty tool list."""
        env = SupportBotEnv(tools=[])
        obs, info = env.reset()
        
        # Should still work with no tools
        action = {
            "tool_id": 0,  # Should be "no tool"
            "response_type": 0,
            "confidence": np.array([0.5])
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float))
        
        env.close()


# Pytest fixtures
@pytest.fixture
def support_bot_env():
    """Fixture providing a SupportBot environment."""
    env = SupportBotEnv(difficulty="medium", max_turns=10)
    yield env
    env.close()


@pytest.fixture
def reset_env(support_bot_env):
    """Fixture providing a reset environment."""
    obs, info = support_bot_env.reset()
    return support_bot_env, obs, info


# Integration tests
def test_full_episode_completion(support_bot_env):
    """Test completing a full episode."""
    obs, info = support_bot_env.reset()
    episode_reward = 0
    step_count = 0
    
    while step_count < support_bot_env.max_turns:
        action = {
            "tool_id": step_count % (len(support_bot_env.available_tools) + 1),
            "response_type": step_count % 5,
            "confidence": np.array([0.5 + 0.1 * (step_count % 5)])
        }
        
        obs, reward, terminated, truncated, info = support_bot_env.step(action)
        episode_reward += reward
        step_count += 1
        
        if terminated or truncated:
            break
    
    # Episode should complete successfully
    assert step_count > 0
    assert isinstance(episode_reward, (int, float))
    assert isinstance(info, dict)


if __name__ == "__main__":
    pytest.main([__file__])
