"""
Hybrid PPO Agent: Combines traditional PPO with LLM guidance for improved performance
in complex reasoning tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from dataclasses import dataclass

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.typing import ModelConfigDict, TensorType


@dataclass
class AgentConfig:
    """Configuration for the Hybrid PPO Agent."""
    hidden_dims: List[int] = None
    llm_model: str = "gpt-3.5-turbo"
    llm_guidance_weight: float = 0.3
    use_attention: bool = True
    dropout_rate: float = 0.1
    temperature: float = 0.8
    max_reasoning_steps: int = 5
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


class AttentionLayer(nn.Module):
    """Multi-head attention layer for processing conversation context."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection and residual connection
        output = self.w_o(context)
        output = self.layer_norm(output + x)
        
        return output


class HybridPPOModel(TorchModelV2, nn.Module):
    """
    Custom PyTorch model that combines traditional RL with LLM guidance.
    
    Architecture:
    1. Observation encoding (text + metadata)
    2. Attention layers for context processing  
    3. PPO policy and value heads
    4. LLM guidance integration
    """
    
    def __init__(
        self,
        obs_space,
        action_space, 
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.config = AgentConfig(**model_config.get("custom_model_config", {}))
        
        # Observation space components
        text_dim = obs_space["text_embedding"].shape[0]  # 512
        metadata_dim = obs_space["conversation_metadata"].shape[0]  # 10
        tools_dim = obs_space["available_tools"].shape[0]  # num_tools
        
        total_obs_dim = text_dim + metadata_dim + tools_dim
        
        # Encoder layers
        self.obs_encoder = nn.Sequential(
            nn.Linear(total_obs_dim, self.config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
        )
        
        # Attention layers for context processing
        if self.config.use_attention:
            self.attention_layers = nn.ModuleList([
                AttentionLayer(self.config.hidden_dims[1], n_heads=8, dropout=self.config.dropout_rate)
                for _ in range(2)
            ])
        
        # Policy head (for discrete actions)
        self.policy_head = nn.Sequential(
            nn.Linear(self.config.hidden_dims[1], self.config.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dims[2], num_outputs)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_dims[1], self.config.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dims[2], 1)
        )
        
        # LLM guidance components (simplified)
        self.llm_guidance_layer = nn.Sequential(
            nn.Linear(self.config.hidden_dims[1], self.config.hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dims[2], num_outputs),
            nn.Softmax(dim=-1)
        )
        
        # Store the last features for value function
        self._features = None
        
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Forward pass of the model."""
        obs = input_dict["obs"]
        
        # Concatenate observation components
        text_emb = obs["text_embedding"]
        metadata = obs["conversation_metadata"]
        tools = obs["available_tools"].float()
        
        # Flatten and concatenate
        combined_obs = torch.cat([
            text_emb,
            metadata, 
            tools
        ], dim=-1)
        
        # Encode observations
        features = self.obs_encoder(combined_obs)
        
        # Apply attention if enabled
        if self.config.use_attention:
            # Reshape for attention (add sequence dimension)
            features_seq = features.unsqueeze(1)  # [batch, 1, hidden_dim]
            
            for attention_layer in self.attention_layers:
                features_seq = attention_layer(features_seq)
            
            features = features_seq.squeeze(1)  # [batch, hidden_dim]
        
        # Store features for value function
        self._features = features
        
        # Generate policy logits
        policy_logits = self.policy_head(features)
        
        # LLM guidance (simplified - in practice would call actual LLM)
        llm_guidance = self.llm_guidance_layer(features)
        
        # Combine PPO policy with LLM guidance
        combined_logits = (
            (1 - self.config.llm_guidance_weight) * policy_logits +
            self.config.llm_guidance_weight * torch.log(llm_guidance + 1e-8)
        )
        
        return combined_logits, state
    
    def value_function(self) -> TensorType:
        """Compute value function estimate."""
        if self._features is None:
            return torch.zeros(1)
        return self.value_head(self._features).squeeze(-1)


class HybridPPOAgent:
    """
    High-level interface for the Hybrid PPO Agent.
    Provides easy-to-use methods for training and inference.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        config: Optional[AgentConfig] = None,
        device: str = "auto"
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config or AgentConfig()
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self._setup_model()
        
        # Training state
        self.training_step = 0
        self.episode_rewards = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def _setup_model(self):
        """Initialize the neural network model."""
        # Calculate number of outputs based on action space
        if hasattr(self.action_space, 'n'):
            # Discrete action space
            num_outputs = self.action_space.n
        else:
            # Dict action space - sum all discrete components
            num_outputs = sum([
                space.n if hasattr(space, 'n') else space.shape[0] 
                for space in self.action_space.spaces.values()
            ])
        
        self.model = HybridPPOModel(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=num_outputs,
            model_config={"custom_model_config": self.config.__dict__},
            name="hybrid_ppo_model"
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=3e-4,
            eps=1e-5
        )
    
    def get_action(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Dict[str, Any]:
        """Get action from the agent given an observation."""
        self.model.eval()
        
        with torch.no_grad():
            # Convert observation to tensors
            obs_tensor = {}
            for key, value in observation.items():
                obs_tensor[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
            
            # Forward pass
            logits, _ = self.model({"obs": obs_tensor}, [], None)
            
            # Sample action
            if deterministic:
                action_idx = torch.argmax(logits, dim=-1)
            else:
                dist = Categorical(logits=logits)
                action_idx = dist.sample()
            
            # Convert back to dict format for complex action spaces
            action = self._convert_action_index(action_idx.cpu().numpy()[0])
            
        return action
    
    def _convert_action_index(self, action_idx: int) -> Dict[str, Any]:
        """Convert flat action index back to structured action dict."""
        # Simplified conversion - in practice would need proper mapping
        tool_id = action_idx % 5  # Assuming 5 tools
        response_type = (action_idx // 5) % 5
        confidence = np.random.uniform(0.5, 1.0)  # Placeholder
        
        return {
            "tool_id": tool_id,
            "response_type": response_type,
            "confidence": np.array([confidence])
        }
    
    def update(
        self,
        observation: Dict[str, np.ndarray],
        reward: float,
        done: bool,
        info: Dict[str, Any]
    ):
        """Update the agent with new experience."""
        # Store experience for batch updates
        # In practice, would use a replay buffer
        self.training_step += 1
        
        if done:
            self.episode_rewards.append(reward)
            self.logger.info(f"Episode finished. Reward: {reward:.2f}")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "config": self.config.__dict__,
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint["training_step"]
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "training_step": self.training_step,
            "avg_episode_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            "total_episodes": len(self.episode_rewards),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
        }
