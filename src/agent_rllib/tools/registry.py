"""
Tool Registry: Extensible system for registering and managing tools
that agents can use during task execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Callable
import json
import logging
from dataclasses import dataclass, asdict


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    return_type: str
    category: str
    version: str = "1.0.0"
    requires_auth: bool = False
    rate_limit: Optional[int] = None


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Tools are functions that agents can call to interact with external systems,
    perform calculations, or access information.
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.call_count = 0
        self.success_count = 0
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.
        
        Returns:
            Dict containing:
            - success: bool indicating if execution was successful
            - result: The actual result data
            - error: Error message if success is False
            - metadata: Additional information about the execution
        """
        pass
    
    def validate_params(self, **kwargs) -> bool:
        """Validate input parameters. Override in subclasses."""
        return True
    
    def get_metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        return ToolMetadata(
            name=self.name,
            description=self.description,
            parameters=self._get_parameter_schema(),
            return_type="Dict[str, Any]",
            category=self._get_category()
        )
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        """Override to provide parameter schema."""
        return {}
    
    def _get_category(self) -> str:
        """Override to specify tool category."""
        return "general"
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Make the tool callable."""
        self.call_count += 1
        
        try:
            if not self.validate_params(**kwargs):
                return {
                    "success": False,
                    "error": "Invalid parameters",
                    "result": None,
                    "metadata": {"call_count": self.call_count}
                }
            
            result = self.execute(**kwargs)
            
            if result.get("success", False):
                self.success_count += 1
            
            # Add metadata
            result["metadata"] = result.get("metadata", {})
            result["metadata"].update({
                "tool_name": self.name,
                "call_count": self.call_count,
                "success_rate": self.success_count / self.call_count
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "result": None,
                "metadata": {
                    "tool_name": self.name,
                    "call_count": self.call_count,
                    "success_rate": self.success_count / self.call_count
                }
            }


class ToolRegistry:
    """
    Central registry for managing tools available to agents.
    
    Provides methods to register, discover, and execute tools.
    Supports tool categories, versioning, and access control.
    """
    
    _instance = None
    _tools: Dict[str, BaseTool] = {}
    _categories: Dict[str, List[str]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self._access_control = {}
            self._rate_limits = {}
            self._initialized = True
    
    @classmethod
    def register(
        cls,
        name: str,
        tool_class: Type[BaseTool],
        *args,
        **kwargs
    ) -> None:
        """
        Register a new tool.
        
        Args:
            name: Unique name for the tool
            tool_class: Tool class inheriting from BaseTool
            *args, **kwargs: Arguments to pass to tool constructor
        """
        if name in cls._tools:
            logging.warning(f"Tool '{name}' already registered. Overwriting.")
        
        # Instantiate the tool
        tool_instance = tool_class(name, *args, **kwargs)
        cls._tools[name] = tool_instance
        
        # Update category index
        category = tool_instance._get_category()
        if category not in cls._categories:
            cls._categories[category] = []
        
        if name not in cls._categories[category]:
            cls._categories[category].append(name)
        
        logging.info(f"Registered tool: {name} (category: {category})")
    
    @classmethod
    def register_function(
        cls,
        name: str,
        func: Callable,
        description: str,
        category: str = "function",
        **metadata
    ) -> None:
        """
        Register a simple function as a tool.
        
        Args:
            name: Tool name
            func: Function to wrap
            description: Tool description
            category: Tool category
            **metadata: Additional metadata
        """
        
        class FunctionTool(BaseTool):
            def __init__(self):
                super().__init__(name, description)
                self.func = func
                self.category = category
            
            def execute(self, **kwargs) -> Dict[str, Any]:
                try:
                    result = self.func(**kwargs)
                    return {
                        "success": True,
                        "result": result,
                        "error": None
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "result": None,
                        "error": str(e)
                    }
            
            def _get_category(self) -> str:
                return self.category
        
        cls.register(name, FunctionTool)
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List available tools, optionally filtered by category."""
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())
    
    def list_categories(self) -> List[str]:
        """List all tool categories."""
        return list(self._categories.keys())
    
    def get_tool_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific tool."""
        tool = self.get_tool(name)
        return tool.get_metadata() if tool else None
    
    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{name}' not found",
                "result": None,
                "metadata": {}
            }
        
        return tool(**kwargs)
    
    def get_tools_by_category(self, category: str) -> List[BaseTool]:
        """Get all tools in a specific category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def search_tools(self, query: str) -> List[str]:
        """Search tools by name or description."""
        query = query.lower()
        matching_tools = []
        
        for name, tool in self._tools.items():
            if (query in name.lower() or 
                query in tool.description.lower()):
                matching_tools.append(name)
        
        return matching_tools
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the tool registry."""
        total_calls = sum(tool.call_count for tool in self._tools.values())
        total_successes = sum(tool.success_count for tool in self._tools.values())
        
        return {
            "total_tools": len(self._tools),
            "categories": len(self._categories),
            "total_calls": total_calls,
            "total_successes": total_successes,
            "overall_success_rate": total_successes / total_calls if total_calls > 0 else 0,
            "tools_by_category": {
                cat: len(tools) for cat, tools in self._categories.items()
            }
        }
    
    def export_schema(self) -> Dict[str, Any]:
        """Export tool schemas for documentation or API generation."""
        schema = {
            "version": "1.0.0",
            "tools": {},
            "categories": self._categories.copy()
        }
        
        for name, tool in self._tools.items():
            metadata = tool.get_metadata()
            schema["tools"][name] = asdict(metadata)
        
        return schema
    
    def clear_registry(self):
        """Clear all registered tools. Mainly for testing."""
        self._tools.clear()
        self._categories.clear()
        self.logger.info("Tool registry cleared")
    
    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def __iter__(self):
        """Iterate over tool names."""
        return iter(self._tools.keys())
