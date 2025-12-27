import json
import logging
import math
import re
from typing import Dict, List, Any, Optional, Tuple, Union

class ClarificationQuestion:
    def __init__(self):
        self.question_type: str = ""
        self.question_text: str = ""
        self.options: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "question_type": self.question_type,
            "question_text": self.question_text,
            "metadata": self.metadata
        }
        if self.options:
            result["options"] = self.options
        return result


class ClassificationPath:
    """
    Represents a single hypothesis/path in the multi-hypothesis exploration.
    Each path tracks its own classification journey and cumulative confidence.
    
    Now also maintains a chat trajectory - a list of messages that accumulates
    all LLM interactions for this path. The trajectory enables stateful
    conversations where the LLM can see all prior decisions and context.
    """
    def __init__(self, path_id: str, chapter: str, chapter_confidence: float, chapter_description: str):
        self.path_id = path_id
        self.chapter = chapter
        # Add log_score for numerical stability
        self.log_score = math.log(chapter_confidence + 1e-9)
        # Represents cumulative path-level confidence (product of path_confidence values)
        self.cumulative_confidence = chapter_confidence
        self.last_decision_confidence = chapter_confidence
        self.is_active = True
        self.is_complete = False
        self.current_node: Optional['HTSNode'] = None
        self.classification_path = [{
            "type": "chapter",
            "code": chapter,
            "description": chapter_description,
            # Backwards-compatible key
            "confidence": chapter_confidence,
            "decision_confidence": chapter_confidence,
            "path_confidence": chapter_confidence,
            "cumulative_confidence": chapter_confidence
        }]
        self.steps = []
        self.visited_nodes = []
        self.selection = {"chapter": chapter}
        self.reasoning_log = []
        self.failure_reason = None
        
        # Chat trajectory: list of messages in OpenAI format
        # [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        # The system message is added once at initialization; subsequent turns append user+assistant pairs
        self.trajectory: List[Dict[str, str]] = []
        self._trajectory_initialized = False
    
    def initialize_trajectory(self, system_prompt: str) -> None:
        """
        Initialize the trajectory with the unified system prompt.
        Should be called once when the path is first created.
        """
        if not self._trajectory_initialized:
            self.trajectory = [{"role": "system", "content": system_prompt}]
            self._trajectory_initialized = True
    
    def append_to_trajectory(self, user_message: str, assistant_response: str) -> None:
        """
        Append a user message and assistant response to the trajectory.
        This represents one complete turn in the conversation.
        
        Args:
            user_message: The user/task message (typically JSON prompt)
            assistant_response: The assistant's response
        """
        self.trajectory.append({"role": "user", "content": user_message})
        self.trajectory.append({"role": "assistant", "content": assistant_response})
    
    def get_trajectory_for_request(self, new_user_message: str) -> List[Dict[str, str]]:
        """
        Get the full trajectory including a new user message for the next LLM request.
        Does NOT modify the trajectory - use append_to_trajectory after receiving response.
        
        Args:
            new_user_message: The new user message to append
            
        Returns:
            List of messages including the new user message
        """
        return self.trajectory + [{"role": "user", "content": new_user_message}]
    
    def append_event_to_trajectory(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Append a system event to the trajectory for RL training.
        Events record decisions/state changes that result from LLM outputs.
        
        Args:
            event_type: Type of event (e.g., "classification_complete")
            event_data: Event details (code, confidence, path, etc.)
        """
        import json
        event = {"event": event_type, **event_data}
        self.trajectory.append({"role": "system", "content": json.dumps(event)})
        
    def add_step(self, stage: str, node: 'HTSNode', decision_confidence: float, path_confidence: float,
                 reasoning: str, options: List[Dict]):
        """Add a classification step to this path"""
        decision_confidence = max(0.0, min(1.0, decision_confidence))
        path_confidence = max(0.0, min(1.0, path_confidence))
        self.last_decision_confidence = decision_confidence

        # Use log probabilities for numerical stability
        self.log_score += math.log(path_confidence + 1e-9)
        self.cumulative_confidence *= path_confidence  # Human-readable global likelihood
        
        step = {
            "step": len(self.steps) + 1,
            "stage": stage,
            "current_code": self.current_node.htsno if self.current_node and self.current_node.htsno else "[GROUP]",
            "selected_code": node.htsno or "[GROUP]",
            "selected_description": node.description,
            "node_id": node.node_id,
            "is_group": node.is_group_node(),
            # Maintain legacy “confidence” key for UI consumers (decision confidence)
            "confidence": decision_confidence,
            "decision_confidence": decision_confidence,
            "path_confidence": path_confidence,
            "cumulative_confidence": self.cumulative_confidence,
            "cumulative_path_confidence": self.cumulative_confidence,
            "options": options,
            "reasoning": reasoning
        }
        self.steps.append(step)
        
        # Update current node and path
        self.current_node = node
        self.visited_nodes.append(node.node_id)
        self.selection[stage] = node.node_id
        
        # Add to classification path
        path_entry = {
            "type": stage,
            "code": node.htsno or "[GROUP]",
            "description": node.description,
            "is_group": node.is_group_node(),
            "node_id": node.node_id,
            # Preserve legacy key but store path confidence for the tree overview
            "confidence": path_confidence,
            "decision_confidence": decision_confidence,
            "path_confidence": path_confidence,
            "cumulative_confidence": self.cumulative_confidence
        }
        self.classification_path.append(path_entry)
        
        # Log reasoning
        self.reasoning_log.append(
            f"{stage}: Selected {node.htsno or '[GROUP]'} - {node.description} "
            f"(decision_conf: {decision_confidence:.3f}, path_conf: {path_confidence:.3f}, "
            f"cumulative_path: {self.cumulative_confidence:.3f})"
        )
        
    def mark_complete(self):
        """Mark this path as complete"""
        self.is_complete = True
        self.is_active = False
        
    def mark_pruned(self, reason: str):
        """Mark this path as pruned"""
        self.is_active = False
        self.failure_reason = reason
        
    def get_full_path_string(self) -> str:
        """Get the full classification path as a string"""
        path_parts = []
        for step in self.classification_path:
            if step.get("is_group"):
                path_element = f"[GROUP] {step.get('description')}"
            else:
                path_element = f"{step.get('code')} - {step.get('description')}"
            path_parts.append(path_element)
        return " > ".join(path_parts)
        
    def get_final_code(self) -> str:
        """Get the final classification code"""
        if self.current_node:
            if self.current_node.htsno:
                return self.current_node.htsno
            else:
                return f"GROUP:{self.current_node.node_id}"
        return None
    
    def clone(self) -> 'ClassificationPath':
        """Create a deep copy of this path, including the chat trajectory"""
        import uuid
        # Use UUID to ensure uniqueness instead of time.time() which can collide
        unique_id = str(uuid.uuid4())[:8]  # Short unique identifier
        new_path = ClassificationPath(
            f"{self.path_id}_clone_{unique_id}",
            self.chapter,
            1.0,  # Reset confidence, will be updated
            self.classification_path[0]["description"]
        )
        new_path.log_score = self.log_score  # Copy log score
        new_path.cumulative_confidence = self.cumulative_confidence
        new_path.last_decision_confidence = self.last_decision_confidence
        new_path.is_active = self.is_active
        new_path.is_complete = self.is_complete
        new_path.current_node = self.current_node
        new_path.classification_path = [p.copy() for p in self.classification_path]
        new_path.steps = [s.copy() for s in self.steps]
        new_path.visited_nodes = self.visited_nodes.copy()
        new_path.selection = self.selection.copy()
        new_path.reasoning_log = self.reasoning_log.copy()
        new_path.failure_reason = self.failure_reason
        
        # Deep copy the trajectory - each message dict is copied
        new_path.trajectory = [msg.copy() for msg in self.trajectory]
        new_path._trajectory_initialized = self._trajectory_initialized
        
        return new_path
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert path to dictionary for serialization using enhanced HTSSerializer"""
        try:
            from .serialization_utils import HTSSerializer
            HTSSerializer.log_serialization_operation(
                "SERIALIZE", "ClassificationPath", self.path_id, 
                before_data=f"log_score={self.log_score}, confidence={self.cumulative_confidence}"
            )
            
            # Use enhanced serialization
            result = HTSSerializer.serialize_classification_path(self)
            
            HTSSerializer.log_serialization_operation(
                "SERIALIZE", "ClassificationPath", self.path_id,
                after_data=f"checksum={result.get('_checksum', 'none')}, size={len(str(result))}",
                success=True
            )
            
            return result
            
        except Exception as e:
            HTSSerializer.log_serialization_operation(
                "SERIALIZE", "ClassificationPath", self.path_id,
                success=False, error=str(e)
            )
            raise
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], hts_tree: 'HTSTree') -> 'ClassificationPath':
        """Create ClassificationPath from dictionary using enhanced HTSSerializer"""
        try:
            from .serialization_utils import HTSSerializer
            # Extract path_id for logging
            path_id = data.get("path_id", "unknown")
            
            HTSSerializer.log_serialization_operation(
                "DESERIALIZE", "ClassificationPath", path_id, 
                before_data=f"checksum={data.get('_checksum', 'none')}, size={len(str(data))}"
            )
            
            # Use enhanced deserialization
            path = HTSSerializer.deserialize_classification_path(data, hts_tree)
            
            HTSSerializer.log_serialization_operation(
                "DESERIALIZE", "ClassificationPath", path_id,
                after_data=f"log_score={path.log_score}, confidence={path.cumulative_confidence}",
                success=True
            )
            
            return path
            
        except Exception as e:
            HTSSerializer.log_serialization_operation(
                "DESERIALIZE", "ClassificationPath", data.get("path_id", "unknown"),
                success=False, error=str(e)
            )
            raise

class HTSNode:
    """
    Represents a node in the HTS hierarchy.
    Each node can be:
      - A heading/subheading/tariff line with an HTS code
      - A group node (no htsno) with children
    """
    def __init__(self, data: Dict[str, Any], node_id: int):
        self.node_id = node_id  
        self.data = data
        self.htsno = data.get('htsno', '')
        self.indent = int(data.get('indent', '0'))
        self.description = data.get('description', '')
        self.is_superior = data.get('superior') == 'true'
        self.units = data.get('units', [])
        self.general = data.get('general', '')
        self.special = data.get('special', '')
        self.other = data.get('other', '')
        self.footnotes = data.get('footnotes', [])
        self.children: List['HTSNode'] = []
        self.parent: Optional['HTSNode'] = None  

    def add_child(self, child: 'HTSNode') -> None:
        child.parent = self
        self.children.append(child)

    def is_group_node(self) -> bool:
        return not self.htsno

    def get_chapter(self) -> Optional[str]:
        if self.htsno and len(self.htsno) >= 2:
            return self.htsno[:2]
        return None

    def get_node_type(self) -> str:
        if not self.htsno:
            return "group"
        clean_code = self.htsno.replace('.', '')
        if len(clean_code) <= 4:
            return "heading"
        elif len(clean_code) <= 6:
            return "subheading"
        elif len(clean_code) >= 10:
            return "tariff_line"
        else:
            return "intermediate"

    def get_clean_description(self) -> str:
        """Get description with HTML tags removed."""
        if not self.description:
            return ""
        return re.sub(r'</?i>', '', self.description)
    
    def get_inherited_general_rate(self) -> str:
        """Get general rate, inheriting from parent if not available for this node."""
        if self.general and self.general.strip():
            return self.general  # Use own rate if available
        
        # Otherwise, check parents recursively until we find a node with a rate
        current = self.parent
        while current:
            if current.general and current.general.strip():
                return current.general
            current = current.parent
            
        return ""  # No rate found in hierarchy
    
    def get_inherited_special_rate(self) -> str:
        """Get special rate, inheriting from parent if not available for this node."""
        if self.special and self.special.strip():
            return self.special  # Use own rate if available
        
        # Otherwise, check parents recursively until we find a node with a rate
        current = self.parent
        while current:
            if current.special and current.special.strip():
                return current.special
            current = current.parent
            
        return ""  # No rate found in hierarchy
    
    def get_inherited_other_rate(self) -> str:
        """Get other rate, inheriting from parent if not available for this node."""
        if self.other and self.other.strip():
            return self.other  # Use own rate if available
        
        # Otherwise, check parents recursively until we find a node with a rate
        current = self.parent
        while current:
            if current.other and current.other.strip():
                return current.other
            current = current.parent
            
        return ""  # No rate found in hierarchy
    
    def get_inherited_footnotes(self) -> List[str]:
        """Get footnotes, inheriting from parent if none available for this node."""
        if self.footnotes and len(self.footnotes) > 0:
            return self.footnotes  # Use own footnotes if available
        
        # Otherwise, check parents recursively until we find a node with footnotes
        current = self.parent
        while current:
            if current.footnotes and len(current.footnotes) > 0:
                return current.footnotes
            current = current.parent
            
        return []  # No footnotes found in hierarchy
            
    def to_dict(self, include_children: bool = True, visited_nodes: Optional[set] = None) -> Dict[str, Any]:
        """
        Convert node to dictionary for serialization.
        Uses visited_nodes to prevent infinite recursion with circular references.
        
        Args:
            include_children: Whether to include child nodes in the dictionary
            visited_nodes: Set of node IDs that have already been visited (to prevent loops)
            
        Returns:
            Dict representation of the node
        """
        # Initialize visited nodes set if not provided
        if visited_nodes is None:
            visited_nodes = set()
        
        # If we've visited this node before, return just the node ID to prevent recursion
        if self.node_id in visited_nodes:
            return {'node_id': self.node_id, 'reference_only': True}
        
        # Add this node to visited set
        visited_nodes.add(self.node_id)
        
        # Create the base dictionary
        result = {
            'node_id': self.node_id,
            'htsno': self.htsno,
            'description': self.get_clean_description(),
            'indent': self.indent,
            'is_superior': self.is_superior,
            'units': self.units,
            'general': self.general,
            'special': self.special,
            'other': self.other,
            'footnotes': self.footnotes,
            'node_type': self.get_node_type()
        }
        
        # Only include children if requested and we haven't visited them yet
        if include_children:
            result['children'] = [
                child.to_dict(include_children=True, visited_nodes=visited_nodes.copy()) 
                for child in self.children
            ]
        
        return result

    def __repr__(self) -> str:
        prefix = '  ' * self.indent
        clean_description = self.get_clean_description()
        if self.htsno:
            return f"{prefix}({self.node_id}) {self.htsno}: {clean_description} [{len(self.children)} children]"
        else:
            return f"{prefix}({self.node_id}) [GROUP] {clean_description} [{len(self.children)} children]"

class TreeUtils:
    """
    Utility class for tree operations, separated to make functionality more modular.
    """
    @staticmethod
    def format_node(node: HTSNode, index: Optional[int] = None, indent_level: int = 0) -> str:
        """Format a node for display with proper indentation and optional index."""
        indent = "  " * indent_level
        index_str = f"[{index}] " if index is not None else ""
        
        if node.htsno:
            # Clean description
            clean_description = node.get_clean_description()
            return f"{indent}{index_str}{node.htsno} - {clean_description}"
        else:
            # Group node
            clean_description = node.get_clean_description()
            return f"{indent}{index_str}[GROUP] {clean_description}"
    
    @staticmethod
    def format_nodes_list(nodes: List[HTSNode], with_indices: bool = True) -> str:
        """Format a list of nodes with optional indices."""
        result = []
        for i, node in enumerate(nodes, 1):
            result.append(TreeUtils.format_node(node, i if with_indices else None))
        return "\n".join(result)
    
    @staticmethod
    def format_subtree(node: HTSNode, max_depth: Optional[int] = None) -> str:
        """Format a subtree recursively starting from a node, with max depth limit."""
        lines = []
        
        def _format_subtree_helper(current_node: HTSNode, depth: int = 0):
            if max_depth is not None and depth > max_depth:
                return
            
            lines.append(TreeUtils.format_node(current_node, indent_level=depth))
            
            for child in current_node.children:
                _format_subtree_helper(child, depth + 1)
        
        _format_subtree_helper(node)
        return "\n".join(lines)
    
    @staticmethod
    def get_classification_path(node: HTSNode) -> str:
        """Get the classification path from root to the given node."""
        if not node:
            return "Unknown"
            
        parts = []
        current = node
        while current and (current.htsno or current.description):
            if current.htsno:
                parts.append(f"{current.htsno} - {current.description}")
            else:
                parts.append(f"[GROUP] {current.description}")
            current = current.parent
            
        parts.reverse()
        return " > ".join(parts)
    
    @staticmethod
    def determine_next_stage(node: HTSNode) -> str:
        """Determine the next classification stage based on node type."""
        node_type = node.get_node_type()
        if node_type == "heading":
            return "subheading"
        elif node_type == "subheading":
            return "tariff"
        else:
            return "tariff"
    
    @staticmethod
    def create_options_metadata(nodes: List[HTSNode]) -> List[Dict[str, Any]]:
        """Create metadata for node options in a standard format."""
        formatted_options = []
        for i, node in enumerate(nodes, 1):
            formatted_options.append({
                "index": i,
                "node_id": node.node_id,
                "code": node.htsno or "[GROUP]",
                "description": node.description,
                "general": node.general,
                "special": node.special,
                "other": node.other,
                "indent": node.indent,
                "superior": node.is_superior,
                "is_group": node.is_group_node()
            })
        return formatted_options
