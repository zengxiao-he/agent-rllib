"""
SupportBot Environment: A Gymnasium environment for training customer support agents
with tool calling capabilities and multi-turn conversation dynamics.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
import random
from dataclasses import dataclass

from ..tools import ToolRegistry


@dataclass
class ConversationState:
    """Represents the current state of a customer support conversation."""
    messages: List[Dict[str, str]]
    customer_satisfaction: float
    issue_resolved: bool
    tools_used: List[str]
    turn_count: int
    max_turns: int = 20


class SupportBotEnv(gym.Env):
    """
    A sophisticated environment for training customer support agents.
    
    The agent must:
    1. Understand customer queries through natural language
    2. Use appropriate tools (search, calculator, etc.)
    3. Provide helpful responses
    4. Maintain customer satisfaction
    5. Resolve issues efficiently
    
    Observation Space:
    - Text: Current conversation context
    - Vector: Conversation metadata (satisfaction, turn count, etc.)
    
    Action Space:
    - Discrete: Tool selection + response generation
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        difficulty: str = "medium",
        tools: List[str] = None,
        max_turns: int = 20,
        reward_shaping: bool = True,
        curriculum_level: float = 0.5,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.difficulty = difficulty
        self.max_turns = max_turns
        self.reward_shaping = reward_shaping
        self.curriculum_level = curriculum_level
        self.render_mode = render_mode
        
        # Initialize tool registry
        if tools is None:
            tools = ["search", "calculator", "weather", "calendar"]
        self.available_tools = tools
        self.tool_registry = ToolRegistry()
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Load scenario templates
        self._load_scenarios()
        
        # Initialize state
        self.conversation_state = None
        self.current_scenario = None
        
    def _setup_spaces(self):
        """Set up the observation and action spaces."""
        # Observation space: text embedding + metadata
        self.observation_space = spaces.Dict({
            "text_embedding": spaces.Box(
                low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32
            ),
            "conversation_metadata": spaces.Box(
                low=0, high=1, shape=(10,), dtype=np.float32
            ),
            "available_tools": spaces.MultiBinary(len(self.available_tools)),
        })
        
        # Action space: tool selection + response parameters
        self.action_space = spaces.Dict({
            "tool_id": spaces.Discrete(len(self.available_tools) + 1),  # +1 for "no tool"
            "response_type": spaces.Discrete(5),  # question, answer, clarification, etc.
            "confidence": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
    
    def _load_scenarios(self):
        """Load customer support scenarios based on difficulty."""
        scenarios = {
            "easy": [
                {
                    "customer_query": "What are your business hours?",
                    "expected_tools": [],
                    "complexity": 0.1,
                    "satisfaction_threshold": 0.8
                },
                {
                    "customer_query": "I need to reset my password",
                    "expected_tools": ["search"],
                    "complexity": 0.2,
                    "satisfaction_threshold": 0.7
                }
            ],
            "medium": [
                {
                    "customer_query": "My order hasn't arrived and I need to calculate a refund",
                    "expected_tools": ["search", "calculator"],
                    "complexity": 0.5,
                    "satisfaction_threshold": 0.6
                },
                {
                    "customer_query": "Can you help me find a product that fits my budget and needs?",
                    "expected_tools": ["search", "calculator"],
                    "complexity": 0.6,
                    "satisfaction_threshold": 0.7
                }
            ],
            "hard": [
                {
                    "customer_query": "I have a complex billing issue involving multiple accounts and need a detailed breakdown",
                    "expected_tools": ["search", "calculator", "calendar"],
                    "complexity": 0.8,
                    "satisfaction_threshold": 0.8
                }
            ]
        }
        
        self.scenarios = scenarios.get(self.difficulty, scenarios["medium"])
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        
        # Select a scenario based on curriculum level
        filtered_scenarios = [
            s for s in self.scenarios 
            if s["complexity"] <= self.curriculum_level + 0.1
        ]
        self.current_scenario = random.choice(filtered_scenarios or self.scenarios)
        
        # Initialize conversation state
        self.conversation_state = ConversationState(
            messages=[{
                "role": "customer",
                "content": self.current_scenario["customer_query"]
            }],
            customer_satisfaction=1.0,
            issue_resolved=False,
            tools_used=[],
            turn_count=0,
            max_turns=self.max_turns
        )
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.conversation_state.turn_count += 1
        
        # Process action
        tool_id = action["tool_id"]
        response_type = action["response_type"]
        confidence = action["confidence"][0]
        
        # Execute tool if selected
        tool_result = None
        if tool_id < len(self.available_tools):
            tool_name = self.available_tools[tool_id]
            tool_result = self._execute_tool(tool_name, action)
            self.conversation_state.tools_used.append(tool_name)
        
        # Generate agent response (simplified)
        agent_response = self._generate_response(response_type, tool_result, confidence)
        self.conversation_state.messages.append({
            "role": "agent",
            "content": agent_response
        })
        
        # Simulate customer response and update satisfaction
        customer_response, satisfaction_change = self._simulate_customer_response(
            agent_response, tool_result
        )
        
        if customer_response:
            self.conversation_state.messages.append({
                "role": "customer", 
                "content": customer_response
            })
        
        self.conversation_state.customer_satisfaction += satisfaction_change
        self.conversation_state.customer_satisfaction = np.clip(
            self.conversation_state.customer_satisfaction, 0.0, 1.0
        )
        
        # Check if issue is resolved
        if satisfaction_change > 0.3 or "thank you" in customer_response.lower():
            self.conversation_state.issue_resolved = True
        
        # Calculate reward
        reward = self._calculate_reward(action, tool_result, satisfaction_change)
        
        # Check termination conditions
        terminated = (
            self.conversation_state.issue_resolved or
            self.conversation_state.customer_satisfaction <= 0.1 or
            self.conversation_state.turn_count >= self.max_turns
        )
        
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_tool(self, tool_name: str, action: Dict) -> Dict:
        """Execute a tool and return results."""
        try:
            tool = self.tool_registry.get_tool(tool_name)
            # Simplified tool execution
            if tool_name == "search":
                return {"results": ["Relevant information found"], "success": True}
            elif tool_name == "calculator":
                return {"result": 42.0, "success": True}
            elif tool_name == "weather":
                return {"temperature": 22, "condition": "sunny", "success": True}
            else:
                return {"success": False, "error": "Tool not implemented"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_response(self, response_type: int, tool_result: Dict, confidence: float) -> str:
        """Generate agent response based on action parameters."""
        response_templates = [
            "I understand your concern. Let me help you with that.",
            "Based on my search, here's what I found:",
            "Could you please provide more details about your issue?",
            "I've calculated the information you requested.",
            "Thank you for your patience. Here's the solution:"
        ]
        
        base_response = response_templates[response_type % len(response_templates)]
        
        if tool_result and tool_result.get("success"):
            if "result" in tool_result:
                base_response += f" The result is {tool_result['result']}."
            elif "results" in tool_result:
                base_response += f" I found: {tool_result['results'][0]}."
        
        return base_response
    
    def _simulate_customer_response(self, agent_response: str, tool_result: Dict) -> Tuple[str, float]:
        """Simulate customer response and satisfaction change."""
        # Simplified customer simulation
        satisfaction_change = 0.0
        
        if tool_result and tool_result.get("success"):
            satisfaction_change += 0.2
            response = "That's helpful, thank you!"
        elif "more details" in agent_response.lower():
            satisfaction_change -= 0.05
            response = "I already explained my issue clearly."
        else:
            satisfaction_change += 0.1
            response = "Okay, please continue helping me."
        
        # Add some randomness
        satisfaction_change += random.uniform(-0.1, 0.1)
        
        return response, satisfaction_change
    
    def _calculate_reward(self, action: Dict, tool_result: Dict, satisfaction_change: float) -> float:
        """Calculate reward based on action effectiveness."""
        reward = 0.0
        
        # Base reward for customer satisfaction change
        reward += satisfaction_change * 10
        
        # Reward for successful tool usage
        if tool_result and tool_result.get("success"):
            reward += 2.0
        elif tool_result and not tool_result.get("success"):
            reward -= 1.0
        
        # Efficiency bonus (fewer turns is better)
        if self.conversation_state.issue_resolved:
            efficiency_bonus = max(0, (self.max_turns - self.conversation_state.turn_count) / self.max_turns)
            reward += efficiency_bonus * 5.0
        
        # Penalty for low confidence when customer is satisfied
        confidence = action["confidence"][0]
        if satisfaction_change > 0 and confidence < 0.5:
            reward -= 0.5
        
        return reward
    
    def _get_observation(self) -> Dict:
        """Get current observation."""
        # Simplified text embedding (in practice, use a real embedding model)
        conversation_text = " ".join([
            msg["content"] for msg in self.conversation_state.messages[-5:]  # Last 5 messages
        ])
        text_embedding = np.random.randn(512).astype(np.float32)  # Placeholder
        
        # Conversation metadata
        metadata = np.array([
            self.conversation_state.customer_satisfaction,
            self.conversation_state.turn_count / self.max_turns,
            float(self.conversation_state.issue_resolved),
            len(self.conversation_state.tools_used) / len(self.available_tools),
            len(self.conversation_state.messages) / (self.max_turns * 2),
            self.current_scenario["complexity"],
            self.current_scenario["satisfaction_threshold"],
            0.0,  # Reserved
            0.0,  # Reserved  
            0.0,  # Reserved
        ], dtype=np.float32)
        
        # Available tools
        available_tools = np.ones(len(self.available_tools), dtype=np.int8)
        
        return {
            "text_embedding": text_embedding,
            "conversation_metadata": metadata,
            "available_tools": available_tools,
        }
    
    def _get_info(self) -> Dict:
        """Get additional information about the current state."""
        return {
            "conversation_length": len(self.conversation_state.messages),
            "customer_satisfaction": self.conversation_state.customer_satisfaction,
            "issue_resolved": self.conversation_state.issue_resolved,
            "tools_used": self.conversation_state.tools_used.copy(),
            "scenario_complexity": self.current_scenario["complexity"],
            "turn_count": self.conversation_state.turn_count,
        }
    
    def render(self, mode: str = "human"):
        """Render the current state of the environment."""
        if mode == "human" or mode == "ansi":
            print("\n" + "="*50)
            print(f"Turn {self.conversation_state.turn_count}/{self.max_turns}")
            print(f"Customer Satisfaction: {self.conversation_state.customer_satisfaction:.2f}")
            print(f"Issue Resolved: {self.conversation_state.issue_resolved}")
            print(f"Tools Used: {', '.join(self.conversation_state.tools_used)}")
            print("-"*50)
            
            for msg in self.conversation_state.messages[-4:]:  # Show last 4 messages
                role = msg["role"].upper()
                content = msg["content"]
                print(f"{role}: {content}")
            
            print("="*50)
    
    def close(self):
        """Clean up resources."""
        pass
