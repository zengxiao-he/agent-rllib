"""
Agent-RLlib: Multi-Agent Reinforcement Learning with LLM Integration

A sophisticated framework combining reinforcement learning with large language models
for building intelligent, tool-using agents.
"""

__version__ = "0.2.1"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .agents import HybridPPOAgent, LLMAgent
from .envs import SupportBotEnv, MultiAgentNegotiationEnv
from .tools import ToolRegistry

__all__ = [
    "HybridPPOAgent",
    "LLMAgent", 
    "SupportBotEnv",
    "MultiAgentNegotiationEnv",
    "ToolRegistry",
]
