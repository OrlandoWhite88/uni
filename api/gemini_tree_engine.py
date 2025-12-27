import json
import logging
import math
import re
import time
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from google.oauth2 import service_account

# Beam search configuration - adjust these values to experiment
CHAPTER_BEAM_SIZE = 3      # Beam size for initial chapter selection
CLASSIFICATION_BEAM_SIZE = 6  # Beam size for heading/subheading/tariff classification

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

class ConversationHistory:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        # Track asked questions by their text to avoid redundancy
        self.asked_questions: Dict[str, bool] = {}

    def add(self, question: str, answer: str, metadata: dict):
        self.entries.append({"question": question, "answer": answer, "metadata": metadata})
        # Mark this question as asked
        self.asked_questions[question] = True

    def get_by_stage(self, stage: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e.get("metadata", {}).get("stage") == stage]

    def format_for_prompt(self) -> str:
        s = ""
        for e in self.entries:
            s += f"Q: {e['question']}\nA: {e['answer']}\n"
        return s

    def to_list(self) -> List[Dict[str, Any]]:
        return self.entries
    
    def has_similar_question(self, question_text: str, similarity_threshold: float = 0.6) -> bool:
        """Check if a similar question has already been asked"""
        # This is a simple implementation - you could use more sophisticated similarity measures
        question_words = set(re.findall(r'\b\w+\b', question_text.lower()))
        if not question_words:
            return False
            
        for asked_question in self.asked_questions:
            asked_words = set(re.findall(r'\b\w+\b', asked_question.lower()))
            if not asked_words:
                continue
                
            # Calculate Jaccard similarity
            intersection = len(question_words.intersection(asked_words))
            union = len(question_words.union(asked_words))
            similarity = intersection / union if union > 0 else 0
            
            if similarity > similarity_threshold:
                return True
                
        return False

class ClassificationPath:
    """
    Represents a single hypothesis/path in the multi-hypothesis exploration.
    Each path tracks its own classification journey and cumulative confidence.
    """
    def __init__(self, path_id: str, chapter: str, chapter_confidence: float, chapter_description: str):
        self.path_id = path_id
        self.chapter = chapter
        # Add log_score for numerical stability
        self.log_score = math.log(chapter_confidence + 1e-9)
        self.cumulative_confidence = chapter_confidence  # Keep for human-readable output
        self.is_active = True
        self.is_complete = False
        self.current_node: Optional['HTSNode'] = None
        self.classification_path = [{
            "type": "chapter",
            "code": chapter,
            "description": chapter_description,
            "confidence": chapter_confidence
        }]
        self.steps = []
        self.visited_nodes = []
        self.selection = {"chapter": chapter}
        self.reasoning_log = []
        self.failure_reason = None
        
    def add_step(self, stage: str, node: 'HTSNode', confidence: float, reasoning: str, options: List[Dict]):
        """Add a classification step to this path"""
        # Use log probabilities for numerical stability
        self.log_score += math.log(confidence + 1e-9)
        self.cumulative_confidence *= confidence  # Keep for human-readable output
        
        step = {
            "step": len(self.steps) + 1,
            "stage": stage,
            "current_code": self.current_node.htsno if self.current_node and self.current_node.htsno else "[GROUP]",
            "selected_code": node.htsno or "[GROUP]",
            "is_group": node.is_group_node(),
            "node_id": node.node_id,
            "confidence": confidence,
            "cumulative_confidence": self.cumulative_confidence,
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
            "confidence": confidence,
            "cumulative_confidence": self.cumulative_confidence
        }
        self.classification_path.append(path_entry)
        
        # Log reasoning
        self.reasoning_log.append(f"{stage}: Selected {node.htsno or '[GROUP]'} - {node.description} (conf: {confidence:.3f}, cumulative: {self.cumulative_confidence:.3f})")
        
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
        """Create a deep copy of this path"""
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
        new_path.is_active = self.is_active
        new_path.is_complete = self.is_complete
        new_path.current_node = self.current_node
        new_path.classification_path = [p.copy() for p in self.classification_path]
        new_path.steps = [s.copy() for s in self.steps]
        new_path.visited_nodes = self.visited_nodes.copy()
        new_path.selection = self.selection.copy()
        new_path.reasoning_log = self.reasoning_log.copy()
        new_path.failure_reason = self.failure_reason
        return new_path
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert path to dictionary for serialization - FIXED VERSION"""
        result = {
            "path_id": self.path_id,
            "chapter": self.chapter,
            "log_score": self.log_score,  # CRITICAL: Include log_score for beam sorting
            "cumulative_confidence": self.cumulative_confidence,
            "is_active": self.is_active,
            "is_complete": self.is_complete,
            "classification_path": self.classification_path,
            "steps": self.steps,
            "visited_nodes": self.visited_nodes,
            "selection": self.selection,
            "reasoning_log": self.reasoning_log,
            "failure_reason": self.failure_reason,
            "full_path": self.get_full_path_string(),
            "final_code": self.get_final_code()
        }
        
        # CRITICAL: Serialize current_node properly
        if self.current_node:
            result["current_node"] = {
                "node_id": self.current_node.node_id,
                "htsno": self.current_node.htsno,
                "description": self.current_node.description,
                "indent": self.current_node.indent
            }
        else:
            result["current_node"] = None
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], hts_tree: 'HTSTree') -> 'ClassificationPath':
        """Create ClassificationPath from dictionary - NEW METHOD"""
        try:
            # Get initial chapter info
            chapter_description = "Unknown chapter"
            initial_confidence = 1.0
            
            # Try to get chapter description from classification_path
            classification_path = data.get("classification_path", [])
            if classification_path and len(classification_path) > 0:
                first_entry = classification_path[0]
                chapter_description = first_entry.get("description", chapter_description)
                initial_confidence = first_entry.get("confidence", initial_confidence)
            
            # Create the path object
            path = cls(
                path_id=data["path_id"],
                chapter=data["chapter"],
                chapter_confidence=initial_confidence,
                chapter_description=chapter_description
            )

            # Restore all the state
            path.log_score = data.get("log_score", math.log(initial_confidence + 1e-9))
            path.cumulative_confidence = data.get("cumulative_confidence", initial_confidence)
            path.is_active = data.get("is_active", True)
            path.is_complete = data.get("is_complete", False)
            path.classification_path = data.get("classification_path", [])
            path.steps = data.get("steps", [])
            path.visited_nodes = data.get("visited_nodes", [])
            path.selection = data.get("selection", {})
            path.reasoning_log = data.get("reasoning_log", [])
            path.failure_reason = data.get("failure_reason")

            # Restore current_node
            current_node_data = data.get("current_node")
            if current_node_data and isinstance(current_node_data, dict) and "node_id" in current_node_data:
                node_id = current_node_data["node_id"]
                path.current_node = hts_tree.get_node_by_id(node_id)
                if path.current_node is None:
                    logging.warning(f"Failed to restore current_node with ID {node_id} for path {path.path_id}")
                else:
                    logging.debug(f"Restored current_node {node_id} for path {path.path_id}")
            else:
                path.current_node = None
            
            return path
            
        except Exception as e:
            logging.error(f"Failed to create ClassificationPath from dict: {e}")
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

def get_vertex_ai_client():
    """Get Google Cloud credentials from environment variables"""
    
    # Check if running with a base64-encoded service account JSON
    if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON_B64'):
        # Decode the base64 string
        sa_json_b64 = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON_B64')
        sa_json = base64.b64decode(sa_json_b64).decode('utf-8')
        
        # Parse the JSON
        service_account_info = json.loads(sa_json)
        
        # Extract project ID from service account info
        project_id = service_account_info.get("project_id")
        if not project_id:
            logging.error("Project ID not found in service account JSON")
            project_id = "812624370602"  # Fallback
        
        # Create credentials from JSON dict
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]  # Add appropriate scope
        )
        
        # Initialize the client with the credentials - use project from service account
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location="us-central1",
            credentials=credentials
        )
        
        return client
    
    # Alternative method: check if a path to credentials file is provided
    elif os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        # Extract project ID from file
        try:
            with open(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'), 'r') as f:
                service_account_info = json.load(f)
                project_id = service_account_info.get("project_id", "812624370602")
        except:
            project_id = "812624370602"  # Fallback
            
        # Use application default credentials
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location="us-central1"
        )
        return client
    
    # Fall back to Application Default Credentials
    else:
        client = genai.Client(
            vertexai=True,
            project="812624370602",
            location="us-central1"
        )
        return client

class HTSTree:
    """Represents the full HTS hierarchy and the classification engine."""
    def __init__(self):
        self.root = HTSNode({'htsno': '', 'indent': '-1', 'description': 'ROOT', 'superior': None}, node_id=0)
        self.chapters: Dict[str, List[HTSNode]] = {}
        self.code_index: Dict[str, HTSNode] = {}
        self.node_index: Dict[int, HTSNode] = {}
        self.next_node_id = 1
        self.steps: List[Dict[str, Any]] = []
        self.max_questions_per_level = 3
        self.log_prompts = os.environ.get("LOG_PROMPTS", "false").lower() == "true"
        
        # Concurrency settings
        self.path_workers = int(os.environ.get("PATH_WORKERS", "4"))  # Number of concurrent path workers
        self.calibrate_workers = int(os.environ.get("CALIBRATE_WORKERS", "6"))  # Number of concurrent calibration workers

        self.chapters_map = self._init_chapters_map()

        # Initialize Vertex AI client
        self.client = get_vertex_ai_client()
        
        # Note: Removed instance variables for history, recent_questions, and product_attributes
        # These are now stored in the state dictionary for each request
        
    # ----------------------
    # Tree Access Methods
    # ----------------------
    
    def get_node_by_id(self, node_id: int) -> Optional[HTSNode]:
        """Get a node by its ID."""
        return self.node_index.get(node_id)
    
    def get_children(self, node: HTSNode) -> List[HTSNode]:
        """Get the immediate children of a node."""
        if not node:
            return []
        return node.children
        
    def get_chapter_nodes(self, chapter: str) -> List[HTSNode]:
        """Get all nodes for a given chapter."""
        # Don't rely on pre-built chapters index - directly scan the tree
        return [node for node in self.root.children 
                if node.htsno and node.htsno.startswith(chapter)]
    
    def get_chapter_headings(self, chapter: str) -> List[HTSNode]:
        """Get all heading nodes for a given chapter."""
        # Get all top-level nodes that start with this chapter
        chapter_nodes = self.get_chapter_nodes(chapter)
        # Sort them by HTS code to ensure correct order
        return sorted(chapter_nodes, key=lambda n: n.htsno)
    
    def create_chapter_parent(self, chapter: str) -> HTSNode:
        """Create a pseudo-node that has all chapter heading nodes as children."""
        chapter_parent = HTSNode({
            'htsno': chapter,
            'description': self.chapters_map.get(int(chapter), "Unknown chapter"),
            'indent': -1
        }, node_id=-1)
        
        for node in self.get_chapter_headings(chapter):
            chapter_parent.add_child(node)
            
        return chapter_parent
    
    def find_node_by_prefix(self, prefix: str) -> Optional[HTSNode]:
        """
        Find a node that matches the given prefix.
        
        Args:
            prefix: The prefix to search for (chapter or heading number)
            
        Returns:
            The matching node, or None if not found
        """
        # Check if it's an exact match first
        if prefix in self.code_index:
            return self.code_index[prefix]
        
        # If not exact match, try to find nodes with this prefix
        matching_nodes = []
        for code, node in self.code_index.items():
            if code.startswith(prefix):
                matching_nodes.append(node)
        
        if not matching_nodes:
            return None
        
        # Return the shortest matching node (most likely the parent)
        return min(matching_nodes, key=lambda n: len(n.htsno))
    
    # ----------------------
    # Tree Building Methods
    # ----------------------

    def _init_chapters_map(self) -> Dict[int, str]:
        return {
            1: "Live animals",
            2: "Meat and edible meat offal",
            3: "Fish and crustaceans, molluscs and other aquatic invertebrates",
            4: "Dairy produce; birds eggs; natural honey; edible products of animal origin, not elsewhere specified or included",
            5: "Products of animal origin, not elsewhere specified or included",
            6: "Live trees and other plants; bulbs, roots and the like; cut flowers and ornamental foliage",
            7: "Edible vegetables and certain roots and tubers",
            8: "Edible fruit and nuts; peel of citrus fruit or melons",
            9: "Coffee, tea, maté and spices",
            10: "Cereals",
            11: "Products of the milling industry; malt; starches; inulin; wheat gluten",
            12: "Oil seeds and oleaginous fruits; miscellaneous grains, seeds and fruits; industrial or medicinal plants; straw and fodder",
            13: "Lac; gums, resins and other vegetable saps and extracts",
            14: "Vegetable plaiting materials; vegetable products not elsewhere specified or included",
            15: "Animal or vegetable fats and oils and their cleavage products prepared edible fats; animal or vegetable waxes",
            16: "Preparations of meat, of fish or of crustaceans, molluscs or other aquatic invertebrates",
            17: "Sugars and sugar confectionery",
            18: "Cocoa and cocoa preparations",
            19: "Preparations of cereals, flour, starch or milk; bakers' wares",
            20: "Preparations of vegetables, fruit, nuts or other parts of plants",
            21: "Miscellaneous edible preparations",
            22: "Beverages, spirits and vinegar",
            23: "Residues and waste from the food industries; prepared animal feed",
            24: "Tobacco and manufactured tobacco substitutes",
            25: "Salt; sulfur; earths and stone; plastering materials, lime and cement",
            26: "Ores, slag and ash",
            27: "Mineral fuels, mineral oils and products of their distillation; bituminous substances; mineral waxes",
            28: "Inorganic chemicals; organic or inorganic compounds of precious metals, of rare-earth metals, of radioactive elements or of isotopes",
            29: "Organic chemicals",
            30: "Pharmaceutical products",
            31: "Fertilizers",
            32: "Tanning or dyeing extracts; dyes, pigments, paints, varnishes, putty and mastics",
            33: "Essential oils and resinoids; perfumery, cosmetic or toilet preparations",
            34: "Soap, organic surface-active agents, washing preparations, lubricating preparations, artificial waxes, prepared waxes, polishing or scouring preparations, candles and similar articles, modeling pastes, \"dental waxes\" and dental preparations with a basis of plaster",
            35: "Albuminoidal substances; modified starches; glues; enzymes",
            36: "Explosives; pyrotechnic products; matches; pyrophoric alloys; certain combustible preparations",
            37: "Photographic or cinematographic goods",
            38: "Miscellaneous chemical products",
            39: "Plastics and articles thereof",
            40: "Rubber and articles thereof",
            41: "Raw hides and skins (other than furskins) and leather",
            42: "Articles of leather; saddlery and harness; travel goods, handbags and similar containers; articles of animal gut (other than silkworm gut)",
            43: "Furskins and artificial fur; manufactures thereof",
            44: "Wood and articles of wood; wood charcoal",
            45: "Cork and articles of cork",
            46: "Manufactures of straw, of esparto or of other plaiting materials; basketware and wickerwork",
            47: "Pulp of wood or of other fibrous cellulosic material; waste and scrap of paper or paperboard",
            48: "Paper and paperboard; articles of paper pulp, of paper or of paperboard",
            49: "Printed books, newspapers, pictures and other products of the printing industry; manuscripts, typescripts and plans",
            50: "Silk",
            51: "Wool, fine or coarse animal hair; horsehair yarn and woven fabric",
            52: "Cotton",
            53: "Other vegetable textile fibers; paper yarn and woven fabric of paper yarn",
            54: "Man-made filaments",
            55: "Man-made staple fibers",
            56: "Wadding, felt and nonwovens; special yarns, twine, cordage, ropes and cables and articles thereof",
            57: "Carpets and other textile floor coverings",
            58: "Special woven fabrics; tufted textile fabrics; lace, tapestries; trimmings; embroidery",
            59: "Impregnated, coated, covered or laminated textile fabrics; textile articles of a kind suitable for industrial use",
            60: "Knitted or crocheted fabrics",
            61: "Articles of apparel and clothing accessories, knitted or crocheted",
            62: "Articles of apparel and clothing accessories, not knitted or crocheted",
            63: "Other made up textile articles; sets; worn clothing and worn textile articles; rags",
            64: "Footwear, gaiters and the like; parts of such articles",
            65: "Headgear and parts thereof",
            66: "Umbrellas, sun umbrellas, walking sticks, seatsticks, whips, riding-crops and parts thereof",
            67: "Prepared feathers and down and articles made of feathers or of down; artificial flowers; articles of human hair",
            68: "Articles of stone, plaster, cement, asbestos, mica or similar materials",
            69: "Ceramic products",
            70: "Glass and glassware",
            71: "Natural or cultured pearls, precious or semi-precious stones,precious metals, metals clad with precious metal and articles thereof; imitation jewelry; coin",
            72: "Iron and steel",
            73: "Articles of iron or steel",
            74: "Copper and articles thereof",
            75: "Nickel and articles thereof",
            76: "Aluminum and articles thereof",
            77: "(Reserved for possible future use)",
            78: "Lead and articles thereof",
            79: "Zinc and articles thereof",
            80: "Tin and articles thereof",
            81: "Other base metals; cermets; articles thereof",
            82: "Tools, implements, cutlery, spoons and forks, of base metal; parts thereof of base metal",
            83: "Miscellaneous articles of base metal",
            84: "Nuclear reactors, boilers, machinery and mechanical appliances; parts thereof",
            85: "Electrical machinery and equipment and parts thereof; sound recorders and reproducers, television image and sound recorders and reproducers, and parts and accessories of such articles",
            86: "Railway or tramway locomotives, rolling-stock and parts thereof; railway or tramway track fixtures and fittings and parts thereof; mechanical (including electro-mechanical) traffic signalling equipment of all kinds",
            87: "Vehicles other than railway or tramway rolling stock, and parts and accessories thereof",
            88: "Aircraft, spacecraft, and parts thereof",
            89: "Ships, boats and floating structures",
            90: "Optical, photographic, cinematographic, measuring, checking, precision, medical or surgical instruments and apparatus; parts and accessories thereof",
            91: "Clocks and watches and parts thereof",
            92: "Musical instruments; parts and accessories of such articles",
            93: "Arms and ammunition; parts and accessories thereof",
            94: "Furniture; bedding, mattresses, mattress supports, cushions and similar stuffed furnishings; lamps and lighting fittings, not elsewhere specified or included; illuminated sign illuminated nameplates and the like; prefabricated buildings",
            95: "Toys, games and sports requisites; parts and accessories thereof",
            96: "Miscellaneous manufactured articles",
            97: "Works of art, collectors' pieces and antiques",
            98: "Special classification provisions",
            99: "Temporary legislation; temporary modifications proclaimed pursuant to trade agreements legislation; additional import restrictions proclaimed pursuant to section 22 of the Agricultural Adjustment Act, as amended"
        }

    def build_from_json(self, json_data: Union[str, List[Dict[str, Any]]]) -> None:
        """Build the HTS hierarchy from JSON data. Also build a node index by node_id."""
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        logging.info(f"Building HTS tree with {len(data)} items")
        parents_by_indent = {-1: self.root}
        
        # Use enumeration to ensure deterministic node IDs based on position
        for index, item in enumerate(data):
            # Node ID is deterministic: position + 1 (since root is 0)
            node_id = index + 1
            node = HTSNode(item, node_id=node_id)
            self.node_index[node_id] = node
            if node.htsno:
                self.code_index[node.htsno] = node
            parent_indent = node.indent - 1
            while parent_indent >= -1:
                if parent_indent in parents_by_indent:
                    parent = parents_by_indent[parent_indent]
                    break
                parent_indent -= 1
            else:
                parent = self.root
            parent.add_child(node)
            parents_by_indent[node.indent] = node
            for indent in list(parents_by_indent.keys()):
                if indent > node.indent:
                    del parents_by_indent[indent]
            
            # Store in chapters mapping - we'll still build this for backward compatibility
            # but won't rely on it for chapter navigation
            if node.htsno:
                chapter = node.get_chapter()
                if chapter:
                    if chapter not in self.chapters:
                        self.chapters[chapter] = []
                    self.chapters[chapter].append(node)
        
        # Update next_node_id to match the number of nodes created
        self.next_node_id = len(data) + 1
        
        # Log verification to ensure consistency
        logging.info(f"Built tree with {len(self.node_index)} nodes, next_node_id={self.next_node_id}")

    # ----------------------
    # Helper Method for Vertex AI API Calls
    # ----------------------
    
    def send_vertex_ai_request(self, prompt, requires_json=False, temperature=0.7):
        """Send a request to the Vertex AI model and return the response."""
        
        # Extract project ID from client for consistency
        project_id = None
        try:
            # Get project from client if available
            if hasattr(self.client, '_project'):
                project_id = self.client._project
            elif hasattr(self.client, 'project'):
                project_id = self.client.project
            
            # Fall back to default if needed
            if not project_id:
                project_id = "812624370602"
                logging.debug(f"Using default project ID: {project_id}")
            else:
                logging.info(f"Using project ID from service account: {project_id}")
                
            # Create model path with the appropriate project ID
            model_name = f"projects/{project_id}/locations/us-central1/endpoints/8324240905682812928"
        except Exception as e:
            logging.error(f"Error getting project ID: {e}")
            # Fall back to default
            model_name = "projects/812624370602/locations/us-central1/endpoints/8324240905682812928"
        
        # Create the configuration exactly as in the Google playground example
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
            ],
        )
        
        # Set up for JSON responses if required
        if requires_json:
            generate_content_config.response_mime_type = "application/json"
        
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        
        try:
            # Match the exact API format from the Google playground example
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generate_content_config,
            )
            
            return response.text
        except Exception as e:
            logging.error(f"Error generating content: {e}")
            return f"Error: {str(e)}"

    # ----------------------
    # Multi-Hypothesis Path Exploration Methods
    # ----------------------

    def determine_top_chapters(self, product_description: str, k: int = CHAPTER_BEAM_SIZE, diagnosis: Optional[str] = None) -> List[Tuple[str, float, str]]:
        """
        Ask the LLM to pick the top K most likely chapters with confidence scores.

        Args:
            product_description: The product being classified
            k: Number of top chapters to return (default 3)
            diagnosis: Optional diagnosis from previous failed classification attempts

        Returns:
            List of tuples: (chapter_number, confidence_score, reasoning)
        """
        chapter_list = "\n".join([f"{num:02d}: {desc}" for num, desc in sorted(self.chapters_map.items())])

        # Include diagnosis information if available
        diagnosis_text = ""
        if diagnosis:
            diagnosis_text = f"""
    PREVIOUS CLASSIFICATION DIAGNOSIS:
    {diagnosis}

    IMPORTANT: Consider the above diagnosis when selecting chapters.
    """

        prompt = f"""Determine the top {k} most appropriate HS code chapters for this product:

    PRODUCT: {product_description}
    {diagnosis_text}
    CHAPTERS:
    {chapter_list}

    INSTRUCTIONS:
    Return the top {k} chapters that could potentially match this product, ranked by likelihood.
    For each chapter, provide a confidence score between 0.0 and 1.0.
    The confidence scores should reflect how well the chapter matches the product characteristics.

    CRITICAL: Your response must be a JSON array starting with [ and ending with ].
    Do NOT wrap the array in an object with keys like {{"top3Chapters": [...]}}.
    Do NOT include any explanatory text before or after the JSON.

    MANDATORY FORMAT - RESPOND WITH ONLY THIS JSON ARRAY:
    [
    {{
        "chapter": "02-digit chapter number",
        "confidence": decimal_between_0.0_and_1.0,
        "reasoning": "Brief explanation of why this chapter could apply"
    }},
    {{
        "chapter": "02-digit chapter number", 
        "confidence": decimal_between_0.0_and_1.0,
        "reasoning": "Brief explanation of why this chapter could apply"
    }},
    {{
        "chapter": "02-digit chapter number",
        "confidence": decimal_between_0.0_and_1.0, 
        "reasoning": "Brief explanation of why this chapter could apply"
    }}
    ]

    REQUIREMENTS:
    - Return exactly {k} chapters ordered from most likely to least likely
    - Confidence scores should sum to approximately 1.0 across all {k} chapters
    - Each chapter must be a 2-digit number (e.g., "03", "16", "01")
    - Each confidence must be between 0.0 and 1.0
    - Reasoning should be concise but specific

    EXAMPLE RESPONSE FORMAT:
    [
    {{"chapter": "03", "confidence": 0.70, "reasoning": "Fish products typically fall under chapter 03"}},
    {{"chapter": "16", "confidence": 0.20, "reasoning": "Could be fish preparations if processed"}},
    {{"chapter": "01", "confidence": 0.10, "reasoning": "Remote possibility if live fish"}}
    ]
    """
        
        if self.log_prompts:
            logging.info(f"==== TOP CHAPTERS DETERMINATION PROMPT ====\n{prompt}\n==== END PROMPT ====")

        logging.info(f"Requesting top {k} chapters from Vertex AI")
        
        # Try multiple times with different approaches
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Use the Vertex AI request function with JSON response
                chapters_response = self.send_vertex_ai_request(
                    prompt=prompt,
                    temperature=0.3,  # Lower temperature for more deterministic results
                    requires_json=True
                )

                if self.log_prompts:
                    logging.info(f"==== TOP CHAPTERS RESPONSE (Attempt {attempt + 1}) ====\n{chapters_response}\n==== END RESPONSE ====")

                # Clean the response - remove any text before/after JSON
                chapters_response = chapters_response.strip()
                
                # Find JSON array in response
                json_start = chapters_response.find('[')
                json_end = chapters_response.rfind(']') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_content = chapters_response[json_start:json_end]
                else:
                    # Fallback: try to parse entire response
                    json_content = chapters_response
                
                # Parse JSON
                chapters_data = json.loads(json_content)
                
                # Handle different response formats
                chapters_list = None
                
                if isinstance(chapters_data, list):
                    # Direct array format: [{"chapter": "03", ...}, ...]
                    chapters_list = chapters_data
                    logging.info(f"Parsed direct array format with {len(chapters_list)} items")
                    
                elif isinstance(chapters_data, dict):
                    # Nested object format: {"top3Chapters": [...], ...}
                    logging.warning(f"Received nested object format, extracting array...")
                    
                    # Try common key names
                    possible_keys = ["top3Chapters", f"top{k}Chapters", "chapters", "results", "data"]
                    for key in possible_keys:
                        if key in chapters_data and isinstance(chapters_data[key], list):
                            chapters_list = chapters_data[key]
                            logging.info(f"Extracted chapters from key '{key}'")
                            break
                    
                    # Fallback: find any array value in the dict
                    if chapters_list is None:
                        for key, value in chapters_data.items():
                            if isinstance(value, list) and len(value) > 0:
                                # Check if it looks like chapter data
                                if all(isinstance(item, dict) and "chapter" in item for item in value):
                                    chapters_list = value
                                    logging.info(f"Found chapters array in key '{key}'")
                                    break
                
                if chapters_list is None:
                    logging.error(f"Could not extract chapters list from response format: {type(chapters_data)}")
                    if attempt < max_attempts - 1:
                        logging.info(f"Retrying attempt {attempt + 2}...")
                        continue
                    else:
                        return []
                
                # Extract and validate results
                results = []
                for i, item in enumerate(chapters_list):
                    if not isinstance(item, dict):
                        logging.warning(f"Item {i} is not a dict: {item}")
                        continue
                        
                    chapter = item.get("chapter", "")
                    confidence_raw = item.get("confidence", 0.0)
                    reasoning = item.get("reasoning", "")
                    
                    # Validate and convert confidence
                    try:
                        confidence = float(confidence_raw)
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid confidence value: {confidence_raw}")
                        confidence = 0.0
                    
                    # Validate chapter format (must be 2-digit string)
                    if not isinstance(chapter, str):
                        chapter = str(chapter)
                    
                    # Handle both "03" and "3" formats
                    if chapter.isdigit():
                        chapter = f"{int(chapter):02d}"
                    
                    # Final validation
                    if re.match(r'^\d{2}$', chapter) and 0.0 <= confidence <= 1.0:
                        results.append((chapter, confidence, reasoning))
                        logging.debug(f"Valid chapter: {chapter} (confidence: {confidence:.3f})")
                    else:
                        logging.warning(f"Invalid chapter data - chapter: '{chapter}', confidence: {confidence}")
                
                # Check if we got the expected number of results
                if len(results) >= k:
                    # Take exactly k results
                    final_results = results[:k]
                    logging.info(f"Successfully extracted {len(final_results)} chapters: {[(r[0], r[1]) for r in final_results]}")
                    
                    # Validate confidence sum (should be approximately 1.0)
                    total_confidence = sum(r[1] for r in final_results)
                    if total_confidence < 0.5 or total_confidence > 1.5:
                        logging.debug(f"Total confidence {total_confidence:.3f} outside expected range")
                    
                    return final_results
                    
                elif len(results) > 0:
                    # We got some results but not enough
                    logging.warning(f"Expected {k} chapters, got {len(results)}. Using what we have.")
                    return results
                else:
                    # No valid results
                    logging.error(f"No valid chapters extracted from response")
                    if attempt < max_attempts - 1:
                        logging.info(f"Retrying attempt {attempt + 2}...")
                        continue
                    else:
                        return []
                        
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                logging.error(f"Response content: {chapters_response}")
                if attempt < max_attempts - 1:
                    logging.info(f"Retrying attempt {attempt + 2}...")
                    continue
                else:
                    return []
                    
            except Exception as e:
                logging.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    logging.info(f"Retrying attempt {attempt + 2}...")
                    continue
                else:
                    return []
        
        # If all attempts failed
        logging.error(f"All {max_attempts} attempts failed to determine top chapters")
        return []

    def initialize_classification_paths(self, product_description: str, k: int = CHAPTER_BEAM_SIZE, diagnosis: Optional[str] = None) -> List[ClassificationPath]:
        """
        Initialize multiple classification paths based on top K chapters.
        
        Args:
            product_description: The product being classified
            k: Number of paths to initialize
            diagnosis: Optional diagnosis from previous attempts
            
        Returns:
            List of ClassificationPath objects
        """
        # Get top chapters
        top_chapters = self.determine_top_chapters(product_description, k, diagnosis)
        
        if not top_chapters:
            logging.warning("No valid chapters returned, falling back to single chapter approach")
            # Fallback to original single chapter determination
            single_chapter = self.determine_chapter(product_description, diagnosis)
            if single_chapter:
                chapter_desc = self.chapters_map.get(int(single_chapter), "Unknown chapter")
                return [ClassificationPath("path_1", single_chapter, 1.0, chapter_desc)]
            else:
                return []
        
        # Create classification paths
        paths = []
        for i, (chapter, confidence, reasoning) in enumerate(top_chapters, 1):
            chapter_desc = self.chapters_map.get(int(chapter), "Unknown chapter")
            path = ClassificationPath(f"path_{i}", chapter, confidence, chapter_desc)
            path.reasoning_log.append(f"Initial chapter selection: {chapter} - {chapter_desc} (conf: {confidence:.3f}) - {reasoning}")
            paths.append(path)
            logging.info(f"Initialized path {i}: Chapter {chapter} with confidence {confidence:.3f}")
        
        return paths

    def generate_all_next_candidates(self, paths: List[ClassificationPath], product_description: str) -> List[Tuple[ClassificationPath, HTSNode, float, str]]:
        """
        Generate all possible next steps from all active paths.
        Now parallelized to process multiple paths concurrently.
        
        Args:
            paths: List of active classification paths
            product_description: The product being classified
            
        Returns:
            List of tuples: (parent_path, candidate_node, confidence, reasoning)
        """
        all_candidates = []
        active_paths = [p for p in paths if p.is_active and not p.is_complete]
        
        if not active_paths:
            return all_candidates
        
        # Prepare function for parallel execution
        def process_path(path):
            path_candidates = []
            
            # Determine current node
            current_node = path.current_node
            if current_node is None:
                # Need to select heading from chapter
                chapter_parent = self.create_chapter_parent(path.chapter)
                current_node = chapter_parent
            
            # Get children
            children = self.get_children(current_node)
            if not children:
                # No children, this path is complete
                path.mark_complete()
                return []
            
            # Score each child as a potential next step
            candidates = self.score_candidates(product_description, current_node, children)
            
            # Add each candidate with its parent path
            for child_node, confidence, reasoning in candidates:
                path_candidates.append((path, child_node, confidence, reasoning))
                
            return path_candidates
        
        # Process paths in parallel
        with ThreadPoolExecutor(max_workers=self.path_workers) as executor:
            futures = [executor.submit(process_path, path) for path in active_paths]
            
            for future in as_completed(futures):
                try:
                    path_results = future.result()
                    all_candidates.extend(path_results)
                except Exception as e:
                    logging.error(f"Error processing path in parallel: {e}")
        
        return all_candidates

    def score_candidates(self, product_description: str, parent_node: HTSNode, candidates: List[HTSNode]) -> List[Tuple[HTSNode, float, str]]:
        """
        Score candidate nodes using Select-Then-Calibrate approach to avoid probability dilution.
        
        This two-stage process:
        1. SELECT: Quickly identify top 3 most promising candidates
        2. CALIBRATE: Get high-quality confidence scores for just those 3
        """
        if not candidates:
            return []
        
        # If we have 3 or fewer candidates, skip selection and go straight to calibration
        if len(candidates) <= 3:
            return self._calibrate_candidates(product_description, parent_node, candidates, list(range(len(candidates))))
        
        # Stage 1: SELECT top 3 candidates
        selected_indices = self._select_top_candidates(product_description, parent_node, candidates, top_k=3)
        
        if not selected_indices:
            # Fallback if selection fails
            logging.error("Selection stage failed, falling back to scoring all candidates")
            return self._fallback_score_all(product_description, parent_node, candidates)
        
        # Stage 2: CALIBRATE the selected candidates
        return self._calibrate_candidates(product_description, parent_node, candidates, selected_indices)

    def _select_top_candidates(self, product_description: str, parent_node: HTSNode, candidates: List[HTSNode], top_k: int = 3) -> List[int]:
        """
        Stage 1: SELECT - Quickly identify the most promising candidates.
        Returns indices of selected candidates.
        """
        current_path = TreeUtils.get_classification_path(parent_node)
        
        # Format candidates with numbers for selection
        options_text = ""
        for i, node in enumerate(candidates, 1):
            if node.htsno:
                options_text += f"{i}. {node.htsno} - {node.description}\n"
            else:
                options_text += f"{i}. [GROUP] {node.description}\n"
        
        prompt = f"""QUICK SELECTION TASK: Identify the top {top_k} most relevant options for this product.

PRODUCT: {product_description}

CURRENT PATH: {current_path}

ALL OPTIONS:
{options_text}

INSTRUCTIONS:
From the numbered list above, identify the {top_k} options that are MOST LIKELY to match this product.
Consider the product's characteristics and the option descriptions.
Return ONLY a comma-separated list of numbers (e.g., "3, 7, 1").

OUTPUT FORMAT:
Just the numbers, nothing else. Example: 3, 7, 1
"""

        if self.log_prompts:
            logging.info(f"==== SELECTION PROMPT ====\n{prompt}\n==== END PROMPT ====")
        
        try:
            # Use lower temperature for more consistent selection
            response = self.send_vertex_ai_request(prompt=prompt, temperature=0.2)
            
            if self.log_prompts:
                logging.info(f"==== SELECTION RESPONSE ====\n{response}\n==== END RESPONSE ====")
            
            # Parse the response to extract numbers
            # Handle various formats like "1, 2, 3" or "1,2,3" or even with explanatory text
            numbers = re.findall(r'\d+', response)
            selected_indices = []
            
            for num_str in numbers[:top_k]:  # Take at most top_k numbers
                try:
                    idx = int(num_str) - 1  # Convert to 0-based index
                    if 0 <= idx < len(candidates):
                        selected_indices.append(idx)
                except ValueError:
                    continue
            
            if len(selected_indices) < top_k and len(selected_indices) < len(candidates):
                # If we didn't get enough valid selections, add some from the beginning
                for i in range(len(candidates)):
                    if i not in selected_indices:
                        selected_indices.append(i)
                        if len(selected_indices) >= top_k:
                            break
            
            logging.info(f"Selected indices: {[i+1 for i in selected_indices]} from {len(candidates)} candidates")
            return selected_indices[:top_k]
            
        except Exception as e:
            logging.error(f"Error in selection stage: {e}")
            return []

    def _calibrate_candidates(self, product_description: str, parent_node: HTSNode, 
                             all_candidates: List[HTSNode], selected_indices: List[int]) -> List[Tuple[HTSNode, float, str]]:
        """
        Stage 2: CALIBRATE - Get high-quality confidence scores for selected candidates.
        Now parallelized to process multiple candidates concurrently.
        """
        if not selected_indices:
            return []
        
        current_path = TreeUtils.get_classification_path(parent_node)
        subtree_representation = TreeUtils.format_subtree(parent_node, max_depth=2)
        
        # Generate prompts for each individual candidate - this enables parallelization
        def generate_prompt_for_candidate(idx):
            node = all_candidates[idx]
            if node.htsno:
                option_text = f"[Option {idx+1}] {node.htsno} - {node.description}"
            else:
                option_text = f"[Option {idx+1}] [GROUP] {node.description}"
                
            # Single-option prompt
            prompt = f"""CALIBRATION TASK: Score this specific classification option for the product.

PRODUCT DESCRIPTION: {product_description}

CURRENT CLASSIFICATION PATH: 
{current_path}

CONTEXT (showing hierarchy):
{subtree_representation}

OPTION TO SCORE:
{option_text}

INSTRUCTIONS:
Provide an INDEPENDENT confidence score from 0.0 to 1.0 for this option.

SCORING GUIDELINES:
- 0.9-1.0: The product description EXPLICITLY confirms ALL distinguishing features of this option
- 0.7-0.85: The option matches well but some specific details are not confirmed
- 0.4-0.65: The option is plausible but significant details are missing or uncertain
- 0.1-0.35: The option is unlikely but cannot be completely ruled out
- 0.0-0.1: The option clearly does not match the product

MANDATORY JSON FORMAT:
Return a JSON object as:
{{
  "option_number": {idx+1},
  "confidence": [decimal between 0.0-1.0],
  "reasoning": "[brief explanation of the score]"
}}
"""
            return prompt, idx
        
        # Function to score a single candidate
        def score_candidate(prompt_and_idx):
            prompt, idx = prompt_and_idx
            
            if self.log_prompts:
                logging.info(f"==== CALIBRATION PROMPT FOR OPTION {idx+1} ====\n{prompt}\n==== END PROMPT ====")
            
            # Retry logic for robust scoring
            for attempt in range(3):
                try:
                    content = self.send_vertex_ai_request(prompt=prompt, temperature=0.2, requires_json=True)
                    
                    if self.log_prompts:
                        logging.info(f"==== CALIBRATION RESPONSE (Attempt {attempt+1}) for option {idx+1} ====\n{content}\n==== END RESPONSE ====")
                    
                    # Parse JSON response
                    score_data = json.loads(content)
                    
                    option_num = score_data.get("option_number", 0)
                    confidence = float(score_data.get("confidence", 0.0))
                    reasoning = score_data.get("reasoning", "")
                    
                    # Validate the option number
                    if option_num != idx + 1:
                        logging.warning(f"Option number mismatch: expected {idx+1}, got {option_num}. Using expected index.")
                    
                    return all_candidates[idx], confidence, reasoning
                    
                except (json.JSONDecodeError, TypeError, ValueError, KeyError) as e:
                    logging.error(f"Error in calibration on attempt {attempt+1} for option {idx+1}: {e}")
                    if attempt < 2:
                        time.sleep(1)
                    continue
                    
                except Exception as e:
                    logging.error(f"Unexpected error in calibration for option {idx+1}: {e}")
                    break
            
            # If all attempts fail, return a fallback score
            logging.error(f"All calibration attempts failed for option {idx+1}. Using fallback score.")
            return all_candidates[idx], 0.1, f"Calibration error for option {idx+1}"
        
        # Generate prompts for all selected candidates
        prompts_and_indices = [generate_prompt_for_candidate(idx) for idx in selected_indices]
        
        # Score candidates in parallel
        scored_candidates = []
        with ThreadPoolExecutor(max_workers=self.calibrate_workers) as executor:
            futures = [executor.submit(score_candidate, prompt_idx) for prompt_idx in prompts_and_indices]
            
            for future in as_completed(futures):
                try:
                    node, confidence, reasoning = future.result()
                    scored_candidates.append((node, confidence, reasoning))
                    logging.debug(f"Calibrated {node.htsno or '[GROUP]'}: confidence={confidence:.3f}")
                except Exception as e:
                    logging.error(f"Error processing calibration future: {e}")
        
        # Sort results to maintain the original order
        if scored_candidates:
            scored_candidates.sort(key=lambda x: all_candidates.index(x[0]))
            
        return scored_candidates

    def _fallback_score_all(self, product_description: str, parent_node: HTSNode, candidates: List[HTSNode]) -> List[Tuple[HTSNode, float, str]]:
        """
        Fallback method if select-then-calibrate fails. Assigns uniform low scores.
        """
        logging.warning("Using fallback scoring method")
        return [(candidate, 0.1, "Fallback scoring due to error") for candidate in candidates]

    def advance_beam(self, current_beam: List[ClassificationPath], product_description: str, k: int) -> List[ClassificationPath]:
        """
        Corrected version: Advances the beam by finding the top K overall next steps.
        This version is simpler, safer, and adheres to beam search principles.
        """
        all_new_hypotheses = []

        # 1. Generate all possible next steps from all active paths in the current beam
        all_candidates_info = self.generate_all_next_candidates(current_beam, product_description)

        # 2. Create the full set of new potential hypotheses by combining parent paths with their scored children
        for parent_path, candidate_node, confidence, reasoning in all_candidates_info:
            # ALWAYS CLONE to prevent state corruption. This is a critical fix.
            new_path = parent_path.clone()
            
            # Determine stage and get options for logging
            stage = TreeUtils.determine_next_stage(parent_path.current_node) if parent_path.current_node else "heading"
            current_node_for_options = parent_path.current_node or self.create_chapter_parent(parent_path.chapter)
            children = self.get_children(current_node_for_options)
            options_metadata = TreeUtils.create_options_metadata(children)

            # Add the new step (this updates the log_score and cumulative_confidence)
            new_path.add_step(stage, candidate_node, confidence, reasoning, options_metadata)
            
            # Check if the new path has reached a terminal node
            if not self.get_children(candidate_node):
                new_path.mark_complete()
            
            all_new_hypotheses.append(new_path)
        
        # Also, carry over any already completed paths from the previous beam so they can continue to compete.
        for path in current_beam:
            if path.is_complete:
                all_new_hypotheses.append(path)

        # 3. PRUNING STEP: Sort the ENTIRE pool of generated hypotheses and simply take the top K.
        # This is the global leaderboard. We sort by log_score.
        all_new_hypotheses.sort(key=lambda p: p.log_score, reverse=True)
        

        new_beam = all_new_hypotheses[:k]
        
        # --- ENHANCED LOGGING ---
        logging.info(f"--- BEAM ADVANCEMENT COMPLETE ---")
        logging.info(f"Generated {len(all_new_hypotheses)} total candidate paths.")
        logging.info(f"Selected top {len(new_beam)} to form the new beam:")
        for i, path in enumerate(new_beam):
            status = "COMPLETE" if path.is_complete else "ACTIVE"
            logging.info(
                f"  [Beam Pos {i+1}] Path: {path.get_full_path_string()} "
                f"| Log Score: {path.log_score:.4f} | Status: {status}"
            )
        logging.info(f"---------------------------------")
        # --- END LOGGING ---

        return new_beam

    def prune_paths(self, paths: List[ClassificationPath], min_confidence: float = 0.05) -> List[ClassificationPath]:
        """
        Prune paths that fall below minimum confidence threshold.
        
        Args:
            paths: List of classification paths
            min_confidence: Minimum cumulative confidence to keep a path active
            
        Returns:
            List of active paths after pruning
        """
        active_paths = []
        
        for path in paths:
            if path.is_active and path.cumulative_confidence >= min_confidence:
                active_paths.append(path)
            elif path.is_active:
                path.mark_pruned(f"Confidence {path.cumulative_confidence:.3f} below threshold {min_confidence}")
                logging.info(f"Pruned {path.path_id}: {path.failure_reason}")
        
        # Ensure we always keep at least one path if any exist
        if not active_paths and paths:
            best_path = max(paths, key=lambda p: p.cumulative_confidence)
            if not best_path.is_complete:
                best_path.is_active = True
                best_path.failure_reason = None
                active_paths.append(best_path)
                logging.info(f"Rescued {best_path.path_id} as last remaining path")
        
        logging.info(f"After pruning: {len(active_paths)} active paths from {len(paths)} total")
        return active_paths

    def re_evaluate_beam_with_answer(self, beam: List[ClassificationPath], product_description: str, 
                                   question: ClarificationQuestion, answer: str, state: Dict) -> List[ClassificationPath]:
        """
        Re-evaluate all paths in the beam based on user's answer.
        
        Args:
            beam: List of classification paths in the beam
            product_description: The product being classified
            question: The clarification question that was asked
            answer: User's answer to the question
            state: Current state dictionary
            
        Returns:
            Updated beam with re-evaluated confidences
        """
        logging.info(f"Re-evaluating beam of {len(beam)} paths based on user answer")
        
        # Process the answer to get updated description
        options = question.metadata.get("options", [])
        updated_description, _ = self.process_answer(product_description, question, answer, options, state)
        
        # Re-evaluate each path
        for path in beam:
            if not path.is_active:
                continue
            
            # Re-evaluate path confidence based on the answer
            new_confidence = self._evaluate_path_with_answer(path, updated_description, answer, question)
            
            # Adjust the entire path's confidence based on the answer
            # This is a multiplicative adjustment factor
            adjustment_factor = new_confidence
            path.cumulative_confidence *= adjustment_factor
            
            path.reasoning_log.append(f"Re-evaluated after answer '{answer}': adjustment factor {adjustment_factor:.3f}, new cumulative {path.cumulative_confidence:.3f}")
            
            logging.info(f"Path {path.path_id}: confidence adjusted to {path.cumulative_confidence:.3f}")
        
        # Re-sort beam by confidence
        beam.sort(key=lambda p: p.cumulative_confidence, reverse=True)
        
        return beam

    def check_termination_conditions(self, beam: List[ClassificationPath]) -> Tuple[bool, Optional[ClassificationPath]]:
        """
        Check if termination conditions are met.
        
        Args:
            beam: Current beam of paths
            
        Returns:
            Tuple of (should_terminate, best_path)
        """
        if not beam:
            return True, None
        
        # Check if all paths are complete or pruned
        active_paths = [p for p in beam if p.is_active]
        if not active_paths:
            # Return the best completed path
            completed_paths = [p for p in beam if p.is_complete]
            if completed_paths:
                best_path = max(completed_paths, key=lambda p: p.log_score)
                return True, best_path
            else:
                return True, None
        
        # Check confident completion condition
        if beam[0].is_complete and len(beam) > 1:
            # Check if top path is significantly better than second path
            # A difference of 1.0 in log space represents ~2.7x probability difference
            if beam[0].log_score - beam[1].log_score > 1.0:
                logging.info(f"Confident completion: {beam[0].path_id} with log_score {beam[0].log_score:.3f} is significantly better than second place ({beam[1].log_score:.3f})")
                return True, beam[0]
        
        # Check if all paths in the beam are complete
        if all(p.is_complete for p in beam):
            # Return the path with the highest log_score
            best_path = max(beam, key=lambda p: p.log_score)
            logging.info(f"All paths complete, selecting best path: {best_path.path_id} with log_score {best_path.log_score:.3f}")
            return True, best_path
        
        return False, None

    def determine_chapter(self, product_description: str, diagnosis: Optional[str] = None) -> str:
        """
        Ask the LLM to pick the best 2-digit chapter.
        This is kept for backward compatibility with single-path mode.

        Args:
            product_description: The product being classified
            diagnosis: Optional diagnosis from previous failed classification attempts

        Returns:
            The 2-digit chapter number as a string
        """
        chapter_list = "\n".join([f"{num:02d}: {desc}" for num, desc in sorted(self.chapters_map.items())])

        # Include diagnosis information if available
        diagnosis_text = ""
        if diagnosis:
            diagnosis_text = f"""
PREVIOUS CLASSIFICATION DIAGNOSIS:
{diagnosis}

IMPORTANT: Consider the above diagnosis when selecting a more appropriate chapter.
"""

        prompt = f"""Determine the most appropriate HS code chapter for this product:

PRODUCT: {product_description}
{diagnosis_text}
CHAPTERS:
{chapter_list}

INSTRUCTIONS:
Return ONLY the 2-digit chapter number (e.g., "03") that best matches this product.
"""
        if self.log_prompts:
            logging.info(f"==== CHAPTER DETERMINATION PROMPT ====\n{prompt}\n==== END PROMPT ====")

        logging.info("Sending chapter determination prompt to Vertex AI")
        try:
            # Use the Vertex AI request function
            chapter_response = self.send_vertex_ai_request(
                prompt=prompt,
                temperature=0.3  # Lower temperature for more deterministic results
            )

            if self.log_prompts:
                logging.info(f"==== CHAPTER DETERMINATION RESPONSE ====\n{chapter_response}\n==== END RESPONSE ====")

            match = re.search(r'(\d{2})', chapter_response)
            if match:
                chapter = match.group(1)
                logging.info(f"Selected chapter: {chapter}")
                return chapter
            else:
                logging.warning(f"Could not parse chapter from response: {chapter_response}")
                return ""
        except Exception as e:
            logging.error(f"Error determining chapter: {e}")
            return ""

    def classify_next_level(self, product_description: str, node: HTSNode) -> tuple[Optional[HTSNode], float, str]:
        """
        Ask the LLM to choose among immediate children of the given node.
        
        Args:
            product_description: The product being classified
            node: The current node to classify within
        
        Returns:
            (selected_node, confidence, raw_llm_response)
        """
        if not node:
            logging.error("Cannot classify from a null node")
            return None, 0.0, "Error: Invalid node"
            
        # Get the immediate children to choose from
        options = self.get_children(node)
        
        if not options:
            logging.warning(f"No children found for node {node.node_id}: {node.description}")
            return None, 0.0, f"Error: No options available"
        
        # Get the full classification path for context
        current_path = TreeUtils.get_classification_path(node)
        
        # Get the full subtree for context
        subtree_representation = TreeUtils.format_subtree(node, max_depth=5)
        
        # Format the options with numbers for the LLM
        options_text = TreeUtils.format_nodes_list(options, with_indices=True)
        
        prompt = f"""Continue classifying this product at a more detailed level:

PRODUCT DESCRIPTION: {product_description}

CURRENT CLASSIFICATION PATH: 
{current_path}

FULL HIERARCHY:
{subtree_representation}

SELECT FROM THESE OPTIONS:
{options_text}

TASK:
Select the numbered option (1-{len(options)}) that best fits the product based on its specific characteristics.
Consider the full hierarchy when making your decision to understand where each option leads.

IMPORTANT NOTE ABOUT GROUP NODES:
Some options may be group nodes (marked with [GROUP]). These are valid selections that contain subclassifications.
For example, if option #2 is "[GROUP] Rifles", it would be a valid selection for a firearm that is a rifle.

MANDATORY CONFIDENCE SCORING REQUIREMENTS:

HIGH CONFIDENCE (0.9-1.0): 
- ONLY use when the product description EXPLICITLY mentions ALL distinguishing characteristics needed
- ALL technical requirements in the option description must be explicitly confirmed in the product description
- For specific measurements, weights, or packaging requirements, these MUST be explicitly stated
- When classifying firearms, HIGH CONFIDENCE requires explicit information about military vs. sporting/hunting use

MEDIUM CONFIDENCE (0.5-0.8): 
- Use when the option seems correct but specific details mentioned in the option are NOT confirmed in the description
- Use when you're making reasonable assumptions about the product that aren't explicitly stated
- Use when technical specifications (measurements, weight, processing methods) are absent
- For firearms without explicit end-use information (military vs. sporting)

LOW CONFIDENCE (0.1-0.4): 
- Use when critical information is missing and multiple options could apply
- Use when the description lacks fundamental details needed to distinguish between options
- When multiple valid classifications are possible based on the available information

FORMAT YOUR RESPONSE:
Return a JSON object with:
{{
  "selection": [selected option number, from 1 to {len(options)}],
  "confidence": [decimal between 0.0-1.0],
  "reasoning": "Your reasoning here"
}}

Please provide your answer in JSON format.
"""

        if self.log_prompts:
            logging.info(f"==== CLASSIFICATION PROMPT ====\n{prompt}\n==== END PROMPT ====")

        logging.info(f"Sending classification prompt with {len(options)} options")

        try:
            # Use Vertex AI with JSON response format
            content = self.send_vertex_ai_request(
                prompt=prompt,
                temperature=0.2,  # Lower temperature for classification tasks
                requires_json=True  # Request JSON response
            )

            if self.log_prompts:
                logging.info(f"==== CLASSIFICATION RESPONSE ====\n{content}\n==== END RESPONSE ====")
            
            result = json.loads(content)

            selection = result.get("selection", 0)
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "")

            if isinstance(selection, int) and 1 <= selection <= len(options):
                selected_node = options[selection - 1]
                return selected_node, confidence, content
            else:
                # Log the reason for null selection if available
                if selection is None and reasoning:
                    logging.warning(f"LLM returned null selection: {reasoning}")
                else:
                    logging.warning(f"Selected option index invalid: {selection}")

                return None, confidence, f"No valid selection: {reasoning}"
        except Exception as e:
            logging.error(f"Error during classification: {e}")
            return None, 0.0, f"Error: {str(e)}"

    def generate_clarification_question(self, product_description: str, node: HTSNode, stage: str, state: Dict) -> ClarificationQuestion:
        """
        Generate a user-friendly clarification question to help choose between immediate children.
        
        Args:
            product_description: The product being classified
            node: The current node we're classifying within
            stage: The classification stage (heading, subheading, etc.)
            state: The current state dictionary with history and product attributes
            
        Returns:
            A ClarificationQuestion object
        """
        if not node:
            logging.error("Cannot generate question from a null node")
            question = ClarificationQuestion()
            question.question_text = "Could you provide more details about your product?"
            question.metadata = {"stage": stage}
            return question
            
        # Get the immediate options to choose from
        options = self.get_children(node)
        
        if not options:
            logging.warning(f"No question options found for node {node.node_id}: {node.description}")
            question = ClarificationQuestion()
            question.question_text = "Could you provide more details about your product for classification?"
            question.metadata = {"stage": stage}
            return question
        
        # Format immediate children for display
        immediate_children_text = TreeUtils.format_nodes_list(options, with_indices=True)
        
        # Get conversation history from state
        history_text = self._format_history_for_prompt(state.get("history", []))
        
        # Get classification path for context
        path_context = f"Current classification path: {TreeUtils.get_classification_path(node)}"
        
        # Stage descriptions
        stage_prompts = {
            "chapter": "Determine which chapter (broad category) the product belongs to.",
            "heading": "Determine the specific 4-digit heading within the chapter.",
            "subheading": "Determine the 6-digit subheading that best matches the product.",
            "tariff": "Determine the most specific tariff line for the product."
        }
        stage_description = stage_prompts.get(stage, "Classify the product.")
        
        # Product attributes we already know - from state
        known_info = ""
        product_attributes = state.get("product_attributes", {})
        if product_attributes:
            known_info = "ALREADY KNOWN INFORMATION ABOUT THE PRODUCT:\n"
            for attr, value in product_attributes.items():
                known_info += f"- {attr}: {value}\n"
            known_info += "\n"

        prompt = f"""You are a customs classification expert creating a targeted question.

PRODUCT DESCRIPTION: {product_description}

CLASSIFICATION STAGE: {stage}
{stage_description}

{path_context}

{known_info}
IMMEDIATE CHILDREN TO CHOOSE FROM:
{immediate_children_text}

PREVIOUS CONVERSATION:
{history_text}

TASK:
Create a question that achieves a 60/40 balance:
- 60%: Direct reference to classification distinctions
- 40%: Product characteristics that determine those distinctions

QUESTION CREATION PROCESS:
1. Analyze the classification options to find the CORE DISTINCTION
   - What single characteristic separates these categories?
   - Examples: explosive vs mechanical, complete vs part, raw vs processed

2. Frame the question to TEST that distinction while referencing the product
   - Start with the product context: "Your [product description]..."
   - Ask about the distinguishing characteristic
   - Make it technically precise but understandable

3. Create options that:
   - Each map clearly to ONE classification category
   - Include both the characteristic AND what it means for classification
   - Cannot overlap (only one can be true)

BALANCE EXAMPLES:

❌ TOO INDIRECT (80% product, 20% classification):
"What material is your product made of?"
"How is your product manufactured?"

❌ TOO DIRECT (90% classification, 10% product):
"Which HS code applies: 9303, 9304, 9305, or 9306?"
"Is this classified as firearms, parts, ammunition, or other arms?"

✅ WELL BALANCED (60% classification, 40% product):
"Does your [product] fire projectiles using explosive charges (making it a firearm under 9303), 
or does it use mechanical/compressed air operation (placing it under 9304 as other arms)?"

"Is your [product] a complete unit that can fire independently (firearm classifications), 
or a component that must be attached to an existing weapon (parts/accessories under 9305)?"

STRUCTURE YOUR QUESTION:
1. Lead question: "Does/Is your [specific product reference] [key distinguishing characteristic]?"
2. Multiple choice options that each:
   - State the characteristic
   - Hint at the classification implication
   - Are mutually exclusive

FORMAT REQUIREMENTS:
- Question must reference the specific product from the description
- Each option must make the classification path clear without just stating codes
- Include enough technical detail to be accurate but remain accessible
- Options should help the user understand WHY their product belongs in that category

CRITICAL: Your question should teach the user about the classification logic while gathering 
the needed information. They should understand not just WHICH category, but WHY.

Note that the user cannot see the classification options you are creating the question for so do not say something like "From these options" and instead display the options through the question options. 

Respond in JSON format as:
{{
"question_type": "multiple_choice",
"question_text": "Your balanced question incorporating product context and classification distinction",
"options": [
    {{"id": "1", "text": "Option describing characteristic + classification implication"}},
    {{"id": "2", "text": "Option describing characteristic + classification implication"}}
]
}}
"""
        if self.log_prompts:
            logging.info(f"==== QUESTION GENERATION PROMPT ====\n{prompt}\n==== END PROMPT ====")
            
        try:
            logging.info(f"Generating clarification question for stage: {stage}")
            
            # Use Vertex AI with JSON response format
            content = self.send_vertex_ai_request(
                prompt=prompt,
                temperature=0.5,  # Medium temperature for creative question generation
                requires_json=True  # Request JSON response
            )

            if self.log_prompts:
                logging.info(f"==== QUESTION GENERATION RESPONSE ====\n{content}\n==== END RESPONSE ====")
            
            question_data = json.loads(content)
            question = ClarificationQuestion()
            question.question_type = question_data.get("question_type", "text")
            question.question_text = question_data.get("question_text", "")
            question.options = question_data.get("options", [])
            
            # Store options data in metadata for later use
            question.metadata = {
                "stage": stage,
                "options": TreeUtils.create_options_metadata(options),
                "node_id": node.node_id
            }
            
            # Check if this question is similar to one we've already asked
            if self._has_similar_question(question.question_text, state.get("history", [])):
                # Generate a fallback question that's more specific
                fallback_question = f"Can you provide more specific details about the {product_description} that would help distinguish between these options? For example, information about its processing, material, or intended use."
                question.question_text = fallback_question
                question.question_type = "text"
                question.options = []
            
            if not question.question_text:
                question.question_text = f"Can you provide more details about the product?"
                
            # Add to recent questions to avoid repetition (in state)
            recent_questions = state.get("recent_questions", [])
            recent_questions.append(question.question_text)
            if len(recent_questions) > 5:
                recent_questions.pop(0)
            state["recent_questions"] = recent_questions
                
            return question
        except Exception as e:
            logging.error(f"Error generating clarification question: {e}")
            question = ClarificationQuestion()
            question.question_text = "Could you provide more details about your product?"
            question.metadata = {"stage": stage}
            return question

    def process_answer(self, original_query: str, question: ClarificationQuestion, answer: str, options: List[Dict[str, Any]], state: Dict) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Process the user's answer to update the product description and select the best matching option.
        Returns a tuple of (updated_description, best_match).
        """
        # Get history from state instead of instance
        history_text = self._format_history_for_prompt(state.get("history", []))
        answer_text = answer
        
        # Handle multiple-choice answers
        if question.question_type == "multiple_choice" and question.options:
            try:
                if answer.strip().isdigit() and 1 <= int(answer.strip()) <= len(question.options):
                    option_index = int(answer.strip()) - 1
                    answer_text = question.options[option_index]["text"]
            except (ValueError, IndexError):
                pass
        
        # Format options text - with indices
        options_text = ""
        for i, opt in enumerate(options, 1):
            code = opt.get("code", "[GROUP]")
            description = opt.get("description", "")
            is_group = opt.get("is_group", False)
            group_marker = "[GROUP] " if is_group else ""
            options_text += f"{i}. {group_marker}{code} - {description}\n"
        
        # Include what we already know about the product - from state
        known_attributes = ""
        product_attributes = state.get("product_attributes", {})
        if product_attributes:
            known_attributes = "KNOWN PRODUCT ATTRIBUTES:\n"
            for attr, value in product_attributes.items():
                known_attributes += f"- {attr}: {value}\n"
            known_attributes += "\n"
        
        prompt = f"""You are a customs classification expert.
ORIGINAL PRODUCT DESCRIPTION: "{original_query}"
QUESTION ASKED: "{question.question_text}"
USER'S ANSWER: "{answer_text}"
{known_attributes}
AVAILABLE OPTIONS:
{options_text}
PREVIOUS CONVERSATION:
{history_text}

TASK:
1. Extract key attributes from the user's answer
2. Update the product description with the new information
3. Determine if one option clearly matches based on this information

NOTE ABOUT GROUP NODES:
Some options may be group nodes (indicated with [GROUP]). These are valid selections.
For example, if the user indicates their product is a 'rifle', and one option is '[GROUP] Rifles',
you should select this option.

Respond in JSON format as:
{{
  "updated_description": "Your updated description here",
  "extracted_attributes": {{"attribute_name": "attribute_value", ...}},
  "selected_option": [option number (1-{len(options)}), or null],
  "confidence": [decimal between 0.0-1.0],
  "reasoning": "Your reasoning here"
}}

The "extracted_attributes" should identify specific product characteristics mentioned in the answer, such as:
- material composition
- processing state (e.g., fresh, frozen, dried)
- physical form (e.g., whole, cut, ground)
- packaging details
- intended use
- dimensions or other measurable attributes
"""
        if self.log_prompts:
            logging.info(f"==== ANSWER PROCESSING PROMPT ====\n{prompt}\n==== END PROMPT ====")
        try:
            logging.info("Processing user's answer")
            
            # Use Vertex AI with JSON response format
            content = self.send_vertex_ai_request(
                prompt=prompt,
                temperature=0.3,  # Lower temperature for more consistent processing
                requires_json=True  # Request JSON response
            )

            if self.log_prompts:
                logging.info(f"==== ANSWER PROCESSING RESPONSE ====\n{content}\n==== END RESPONSE ====")

            result = json.loads(content)
            
            # Extract and store attributes in state
            extracted_attributes = result.get("extracted_attributes", {})
            if "product_attributes" not in state:
                state["product_attributes"] = {}
            state["product_attributes"].update(extracted_attributes)
            
            updated_description = result.get("updated_description", original_query)
            selected_option = result.get("selected_option")
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "")
            
            best_match = None
            if selected_option is not None and isinstance(selected_option, (int, float)) and 1 <= selected_option <= len(options):
                option_index = int(selected_option) - 1
                selected_opt = options[option_index]
                best_match = {
                    "index": selected_option,
                    "node_id": selected_opt.get("node_id"),
                    "code": selected_opt.get("code", "[GROUP]"),
                    "description": selected_opt.get("description", ""),
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "is_group": selected_opt.get("is_group", False)
                }

            # Store the reasoning in state for potential diagnosis
            if selected_option is None and state:
                state["no_match_reason"] = reasoning

            return updated_description, best_match
        except Exception as e:
            logging.error(f"Error processing answer: {e}")
            return original_query, None

    def explain_classification(self, original_query: str, enriched_query: str, full_path: str, conversation: List[Dict[str, Any]], state: Dict = None) -> str:
        """Generate an explanation of the classification."""
        conversation_text = ""
        for i, qa in enumerate(conversation):
            conversation_text += f"Q{i+1}: {qa['question']}\n"
            conversation_text += f"A{i+1}: {qa['answer']}\n\n"

        # Include extracted product attributes in the explanation from state
        product_attributes_text = ""
        product_attributes = state.get("product_attributes", {}) if state else {}
        if product_attributes:
            product_attributes_text = "KEY PRODUCT ATTRIBUTES:\n"
            for attr, value in product_attributes.items():
                product_attributes_text += f"- {attr}: {value}\n"
            product_attributes_text += "\n"

        # Include any classification diagnosis if available
        diagnosis_text = ""
        if state and "classification_diagnosis" in state:
            diagnosis_text = f"""
CLASSIFICATION CHALLENGES:
The system initially had difficulty classifying this product. The following diagnosis was provided:
{state['classification_diagnosis']}

"""

        # Include retry information if available
        retry_text = ""
        if state and state.get("global_retry_count", 0) > 1:
            retry_text = f"Note: This classification required {state.get('global_retry_count')} attempts to find the correct category.\n\n"

        # Include multi-hypothesis information if available
        multi_hypothesis_text = ""
        if state and state.get("used_multi_hypothesis"):
            paths_info = state.get("paths_considered", [])
            if paths_info:
                multi_hypothesis_text = f"""
MULTI-HYPOTHESIS ANALYSIS:
The system considered {len(paths_info)} possible classification paths:
"""
                for i, path_info in enumerate(paths_info, 1):
                    multi_hypothesis_text += f"Path {i}: {path_info.get('chapter', 'Unknown')} - {path_info.get('reasoning', 'No reasoning')}\n"
                multi_hypothesis_text += "\n"

        prompt = f"""As a customs classification expert, explain how this product was classified.

PRODUCT INFORMATION:
- ORIGINAL DESCRIPTION: {original_query}
- ENRICHED DESCRIPTION: {enriched_query}
{product_attributes_text}
{diagnosis_text}
{multi_hypothesis_text}
FINAL CLASSIFICATION:
{full_path}

CONVERSATION THAT LED TO THIS CLASSIFICATION:
{conversation_text}

{retry_text}
TASK:
Explain step-by-step why this classification is correct.
Focus on how each specific product characteristic led to classification decisions at each level.
Make your explanation clear, logical, and accessible to someone without customs expertise.
"""
        try:
            if self.log_prompts:
                logging.info(f"==== EXPLANATION PROMPT ====\n{prompt}\n==== END PROMPT ====")
            logging.info("Generating classification explanation")

            # Use Vertex AI for generating explanation
            explanation = self.send_vertex_ai_request(
                prompt=prompt,
                temperature=0.4  # Medium-low temperature for factual explanation
            )

            if self.log_prompts:
                logging.info(f"==== EXPLANATION RESPONSE ====\n{explanation}\n==== END RESPONSE ====")

            logging.info("Explanation generated successfully")
            return explanation
        except Exception as e:
            logging.error(f"Failed to generate explanation: {e}")
            return "Could not generate explanation due to an error."

    def _evaluate_path_with_answer(self, path: ClassificationPath, updated_description: str, 
                                 answer: str, question: ClarificationQuestion) -> float:
        """
        Evaluate how well a path aligns with the user's answer.
        
        Args:
            path: The classification path to evaluate
            updated_description: Updated product description
            answer: User's answer
            question: The clarification question
            
        Returns:
            New confidence score for this path
        """
        # Get the current path description
        path_description = path.get_full_path_string()
        
        prompt = f"""Evaluate how well this classification path aligns with the user's answer:

PRODUCT DESCRIPTION: {updated_description}
QUESTION ASKED: {question.question_text}
USER'S ANSWER: {answer}
CLASSIFICATION PATH: {path_description}

TASK:
Based on the user's answer, evaluate how likely it is that this classification path is correct.
Consider:
1. Does the answer support or contradict this classification path?
2. How well do the product characteristics revealed by the answer match this path?
3. Are there any obvious misalignments between the answer and this classification?

Return a confidence score between 0.0 and 1.0, where:
- 1.0 = The answer strongly supports this classification path
- 0.5 = The answer is neutral or somewhat supportive
- 0.0 = The answer contradicts or invalidates this path

FORMAT:
Return only a decimal number between 0.0 and 1.0.
"""
        
        try:
            response = self.send_vertex_ai_request(prompt, temperature=0.3)
            # Extract the confidence score
            confidence_match = re.search(r'(\d*\.?\d+)', response.strip())
            if confidence_match:
                confidence = float(confidence_match.group(1))
                return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            else:
                logging.warning(f"Could not parse confidence from response: {response}")
                return 0.5  # Default neutral confidence
        except Exception as e:
            logging.error(f"Error evaluating path with answer: {e}")
            return 0.5  # Default neutral confidence

    # ----------------------
    # Main Classification Workflow Methods (Enhanced)
    # ----------------------

    # Helper methods for stateless operation
    def _format_history_for_prompt(self, history_entries):
        """Convert history entries to formatted string for prompt."""
        s = ""
        for entry in history_entries:
            s += f"Q: {entry['question']}\nA: {entry['answer']}\n"
        return s

    def _has_similar_question(self, question_text, history_entries, similarity_threshold=0.6):
        """Check if a similar question exists in history."""
        question_words = set(re.findall(r'\b\w+\b', question_text.lower()))
        if not question_words:
            return False
            
        for entry in history_entries:
            asked_question = entry.get('question', '')
            asked_words = set(re.findall(r'\b\w+\b', asked_question.lower()))
            if not asked_words:
                continue
                
            # Calculate Jaccard similarity
            intersection = len(question_words.intersection(asked_words))
            union = len(question_words.union(asked_words))
            similarity = intersection / union if union > 0 else 0
            
            if similarity > similarity_threshold:
                return True
                
        return False

    def start_classification(self, product: str, interactive: bool = True, max_questions: int = 3, use_multi_hypothesis: bool = True, hypothesis_count: int = 3) -> Dict:
        """
        Called by /classify to begin classification. Returns either:
          - a "clarification_question" if we need user input
          - a final classification if we are done
          
        Args:
            product: Product description
            interactive: Whether to ask clarification questions
            max_questions: Maximum number of questions to ask
            use_multi_hypothesis: Whether to use multi-hypothesis path exploration
            hypothesis_count: Number of hypothesis paths to explore (if multi-hypothesis enabled)
        """
        # Create a fresh state dictionary with all necessary fields
        state = {
            "product": product,
            "original_query": product,
            "current_query": product,
            "questions_asked": 0,
            "selection": {},
            "current_node": None,
            "classification_path": [],
            "steps": [],
            "conversation": [],
            "pending_question": None,
            "pending_stage": None,
            "max_questions": max_questions,
            "visited_nodes": [],

            # NEW: Previously instance variables, now in state
            "history": [],  # Will store conversation history entries
            "product_attributes": {},  # Will store product attributes
            "recent_questions": [],  # Will store recently asked questions

            # Retry tracking
            "global_retry_count": 0,  # Track total retries across all chapters
            "classification_diagnosis": None,  # Will store diagnosis if classification fails
            
            # Multi-hypothesis specific state
            "use_multi_hypothesis": use_multi_hypothesis,
            "hypothesis_count": hypothesis_count,
            "paths": [],  # Will store ClassificationPath objects
            "beam": [],  # Current beam for beam search
            "paths_considered": [],  # For explanation purposes
            "active_question_path": None,  # Track which path generated the current question
            "used_multi_hypothesis": False  # Track if multi-hypothesis was actually used
        }

        return self.process_classification(state, interactive, max_questions)

    def continue_classification(self, state: Dict, answer: str, interactive: bool = True, max_questions: int = 3) -> Dict:
        """
        SIMPLIFIED VERSION: Just enriches the query and resumes the main loop.
        No manual path advancement - let beam search handle everything.
        """
        logging.info("=== STARTING continue_classification ===")
        
        # Restore beam from JSON if needed
        if state.get("beam"):
            logging.info(f"Restoring beam with {len(state['beam'])} paths")
            restored_beam = []
            
            for i, path_data in enumerate(state["beam"]):
                if isinstance(path_data, dict):
                    try:
                        path = ClassificationPath.from_dict(path_data, self)
                        restored_beam.append(path)
                        
                        # DEBUG: Check actual path state
                        logging.info(f"Restored path {i+1}: current_node={path.current_node.code if path.current_node else 'None'}")
                        logging.info(f"  Path depth: {len(path.path_history) if hasattr(path, 'path_history') else 'N/A'}")
                        logging.info(f"  Last choice: {path.path_history[-1] if hasattr(path, 'path_history') and path.path_history else 'None'}")
                        logging.info(f"  Visited nodes: {path.visited_nodes}")
                        logging.info(f"  Is active: {path.is_active}")
                        logging.info(f"  Is complete: {path.is_complete}")
                        
                        logging.info(f"[OK] Restored path {i+1}: {path.path_id}")
                    except Exception as e:
                        logging.error(f"Failed to restore path {i+1}: {e}")
                elif hasattr(path_data, 'path_id'):
                    restored_beam.append(path_data)
                    logging.info(f"[OK] Restored path {i+1}: {path_data.path_id}")
                
            if not restored_beam:
                logging.error("Beam restoration failed - restarting")
                state["beam"] = []
                state["global_retry_count"] = state.get("global_retry_count", 0) + 1
                return self.process_classification(state, interactive, max_questions)
            
            state["beam"] = restored_beam
            logging.info(f"[OK] Successfully restored beam with {len(restored_beam)} paths")
        
        # Validate pending question
        if not state.get("pending_question"):
            logging.warning("No pending question - resuming process")
            return self.process_classification(state, interactive, max_questions)
        
        # Process the answer to enrich the product description
        state["questions_asked"] += 1
        pending_question_dict = state.get("pending_question", {})
        question_text = pending_question_dict.get("question_text", "")
        
        # Log Q&A
        state["conversation"].append({"question": question_text, "answer": answer})
        if "history" not in state:
            state["history"] = []
        state["history"].append({"question": question_text, "answer": answer})
        
        # Create question object for processing
        question_obj = ClarificationQuestion()
        question_obj.question_text = question_text
        question_obj.question_type = pending_question_dict.get("question_type", "text")
        question_obj.options = pending_question_dict.get("options", [])
        question_obj.metadata = pending_question_dict.get("metadata", {})
        options_metadata = question_obj.metadata.get("options", [])
        
        # Process answer to update product description
        logging.info("Processing answer to enrich product description...")
        updated_query, _ = self.process_answer(state["current_query"], question_obj, answer, options_metadata, state)
        state["current_query"] = updated_query
        logging.info(f"Product description enriched: \"{state['current_query']}\"")
        
        # Clear pending question
        state["pending_question"] = None
        state["pending_stage"] = None
        
        # Resume main processing loop - beam search will naturally favor paths that match the enriched description
        logging.info("Resuming main classification with enriched description")
        logging.info("=== ENDING continue_classification ===")
        return self.process_classification(state, interactive, max_questions)

    def _diagnose_classification_issue(self, product_description: str, current_chapter: str, failed_stage: str, current_node: Optional[HTSNode] = None) -> str:
        """
        Ask the LLM to diagnose why classification failed and suggest a better chapter.

        Args:
            product_description: The product being classified
            current_chapter: The current chapter we tried to classify in
            failed_stage: The stage where classification failed
            current_node: The current node where classification failed (if available)

        Returns:
            A diagnostic explanation and suggested chapter
        """
        chapter_list = "\n".join([f"{num:02d}: {desc}" for num, desc in sorted(self.chapters_map.items())])

        # Include information about the current classification path and available options
        classification_path = ""
        available_options = ""

        if current_node:
            # Get the classification path
            path = []
            node = current_node
            while node and node.node_id != 0:  # Stop at root
                if node.htsno:
                    path.append(f"{node.htsno} - {node.description}")
                else:
                    path.append(f"[GROUP] {node.description}")
                node = node.parent

            path.reverse()
            classification_path = "\n".join(path)

            # Get the available options that failed
            children = self.get_children(current_node)
            if children:
                options_list = []
                for i, child in enumerate(children, 1):
                    if child.htsno:
                        options_list.append(f"{i}. {child.htsno} - {child.description}")
                    else:
                        options_list.append(f"{i}. [GROUP] {child.description}")
                available_options = "\n".join(options_list)

        # Build the prompt with all available information
        path_section = f"""
CURRENT CLASSIFICATION PATH:
{classification_path}
""" if classification_path else ""

        options_section = f"""
AVAILABLE OPTIONS THAT FAILED:
{available_options}
""" if available_options else ""

        prompt = f"""Analyze why this product classification failed and suggest a better chapter:

PRODUCT DESCRIPTION: {product_description}

CURRENT CHAPTER: {current_chapter} - {self.chapters_map.get(int(current_chapter) if current_chapter.isdigit() else 0, "Unknown")}{path_section}{options_section}

CLASSIFICATION FAILED AT: {failed_stage}

AVAILABLE CHAPTERS:
{chapter_list}

TASK:
1. Analyze why the product doesn't fit in the current chapter or any of the available options
2. Identify key product characteristics that were missed or misinterpreted
3. Suggest a more appropriate chapter with explanation
4. Return your analysis in plain text

FORMAT:
Provide a brief analysis explaining why classification failed and suggest a better chapter number.
"""
        try:
            logging.info(f"Diagnosing classification issue for product in chapter {current_chapter}")

            # Use Vertex AI for diagnosis
            diagnosis = self.send_vertex_ai_request(
                prompt=prompt,
                temperature=0.3  # Lower temperature for more factual analysis
            )

            logging.info(f"Classification diagnosis generated")
            return diagnosis
        except Exception as e:
            logging.error(f"Error generating classification diagnosis: {e}")
            return f"Classification failed at {failed_stage}. The product may not belong in chapter {current_chapter}."

    def process_classification(self, state: Dict, interactive: bool, max_questions: int) -> Dict:
        """
        Core classification logic with multi-hypothesis support.
        ENHANCED VERSION with better beam state management.
        """
        # --- Initialization Step (only runs once at the very beginning) ---
        if not state.get("beam"):
            logging.info("--- Initializing New Classification ---")
            state["global_retry_count"] = state.get("global_retry_count", 0) + 1
            if state["global_retry_count"] >= 5:
                return {"error": "Classification failed after multiple attempts.", "final": False, "state": state}

            hypothesis_count = state.get("hypothesis_count", 3)
            diagnosis = state.get("classification_diagnosis")
            
            paths = self.initialize_classification_paths(state["current_query"], hypothesis_count, diagnosis)
            if not paths:
                return {"error": "Could not initialize classification paths.", "final": False, "state": state}
            
            state["beam"] = paths
            state["used_multi_hypothesis"] = True
            state["paths_considered"] = [{"chapter": p.chapter, "reasoning": p.reasoning_log[0] if p.reasoning_log else ""} for p in paths]
            logging.info(f"Initialized beam with {len(paths)} paths")
        # --- End of Initialization ---

        beam = state.get("beam", [])
        k = CLASSIFICATION_BEAM_SIZE
        
        # Validate beam
        if not beam:
            logging.error("Empty beam in process_classification")
            return {"error": "Empty classification beam.", "final": False, "state": state}
        
        max_iterations = 10
        iteration = state.get("iteration_count", 0)

        while iteration < max_iterations:
            iteration += 1
            state["iteration_count"] = iteration
            logging.info(f"--- Beam Search Iteration {iteration} (Beam Size: {len(beam)}) ---")
            
            # Debug beam state
            for i, path in enumerate(beam):
                current_node_info = f"node_id={path.current_node.node_id}" if path.current_node else "None"
                logging.debug(f"  Beam[{i}]: {path.path_id} | log_score={path.log_score:.4f} | current_node={current_node_info} | active={path.is_active}")
            
            # 1. Check for termination
            should_terminate, best_path = self.check_termination_conditions(beam)
            if should_terminate:
                if best_path:
                    logging.info(f"Termination condition met. Returning best path: {best_path.path_id}")
                    return self._convert_path_to_result(best_path, state)
                else:
                    logging.warning("Termination met, but no best path found. All paths pruned. Triggering diagnosis.")
                    diagnosis = self._diagnose_classification_issue(state['current_query'], "multiple", "completion failure")
                    state["classification_diagnosis"] = diagnosis
                    state["beam"] = []
                    return self.process_classification(state, interactive, max_questions)

            # 2. Check if a question is needed for the top path
            top_path = beam[0]
            if top_path.is_active and interactive and state["questions_asked"] < max_questions:
                node_for_next_step = top_path.current_node or self.create_chapter_parent(top_path.chapter)
                options = self.get_children(node_for_next_step)
                
                if options:
                    scored_candidates = self.score_candidates(state["current_query"], node_for_next_step, options)
                    if scored_candidates:
                        _, best_confidence, _ = max(scored_candidates, key=lambda item: item[1])
                        
                        if best_confidence < 0.85:
                            logging.info(f"Top path ({top_path.path_id}) has low confidence ({best_confidence:.2f}) for next step. Generating question.")
                            stage = TreeUtils.determine_next_stage(node_for_next_step)
                            question = self.generate_clarification_question(state["current_query"], node_for_next_step, stage, state)
                            
                            # Save state and return the question to the user
                            state["pending_question"] = question.to_dict()
                            state["pending_stage"] = stage
                            # CRITICAL: Serialize the beam properly
                            state["beam"] = [p.to_dict() for p in beam]
                            
                            return {
                                "final": False, 
                                "clarification_question": question, 
                                "state": state,
                                "debug_info": {"next_step_confidence_trigger": best_confidence}
                            }

            # 3. If no question was asked, advance the beam
            logging.info(f"Confidence sufficient or questioning disabled. Advancing the beam.")
            new_beam = self.advance_beam(beam, state["current_query"], k)
            
            beam = new_beam
            state["beam"] = beam
            
            if not beam:
                logging.error("Beam became empty after advancement. Triggering diagnosis.")
                diagnosis = self._diagnose_classification_issue(state['current_query'], "multiple", "advancement failure")
                state["classification_diagnosis"] = diagnosis
                state["beam"] = []
                return self.process_classification(state, interactive, max_questions)

        # If loop finishes due to max iterations, return the best path
        if not beam:
            return {"error": "Classification failed after maximum iterations.", "final": False, "state": state}
             
        best_path = max(beam, key=lambda p: p.log_score)
        logging.warning(f"Max iterations reached. Returning best available path: {best_path.path_id}")
        return self._convert_path_to_result(best_path, state)

    # ----------------------
    # Server-Sent Events (SSE) Streaming Methods
    # ----------------------

    async def run_classification_with_events(self, product: str, interactive: bool = True, max_questions: int = 3, 
                                           use_multi_hypothesis: bool = True, hypothesis_count: int = 3):
        """
        Run classification while yielding Server-Sent Events for real-time updates.
        Implements full streaming beam search with event yielding.
        
        Yields events for:
        - Chapter selection
        - Path initialization
        - Beam advancement
        - Candidate scoring
        - Question generation
        - Final classification
        """
        import asyncio
        from datetime import datetime, timezone
        
        # Yield start event
        yield {
            "type": "classification_start",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "product": product,
                "interactive": interactive,
                "max_questions": max_questions,
                "use_multi_hypothesis": use_multi_hypothesis,
                "hypothesis_count": hypothesis_count
            }
        }
        
        # Create enhanced state that tracks streaming
        state = {
            "product": product,
            "original_query": product,
            "current_query": product,
            "questions_asked": 0,
            "selection": {},
            "current_node": None,
            "classification_path": [],
            "steps": [],
            "conversation": [],
            "pending_question": None,
            "pending_stage": None,
            "max_questions": max_questions,
            "visited_nodes": [],
            "history": [],
            "product_attributes": {},
            "recent_questions": [],
            "global_retry_count": 0,
            "classification_diagnosis": None,
            "use_multi_hypothesis": use_multi_hypothesis,
            "hypothesis_count": hypothesis_count,
            "paths": [],
            "beam": [],
            "paths_considered": [],
            "active_question_path": None,
            "used_multi_hypothesis": False,
            "streaming": True,  # Flag to indicate streaming mode
            "iteration_count": 0  # Initialize iteration count
        }
        
        try:
            # Initialize classification paths with streaming events
            yield {
                "type": "initialization_start",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "status": "Determining top chapters",
                    "hypothesis_count": hypothesis_count
                }
            }
            
            # Get top chapters with events
            diagnosis = state.get("classification_diagnosis")
            top_chapters = self.determine_top_chapters(state["current_query"], hypothesis_count, diagnosis)
            
            if not top_chapters:
                yield {
                    "type": "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "error": "No chapters found",
                        "message": "Could not determine appropriate chapters for this product"
                    }
                }
                return
            
            yield {
                "type": "chapter_selection",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "chapters": [
                        {
                            "chapter": chapter,
                            "confidence": confidence,
                            "reasoning": reasoning,
                            "description": self.chapters_map.get(int(chapter), "Unknown chapter")
                        }
                        for chapter, confidence, reasoning in top_chapters
                    ]
                }
            }
            
            # Initialize paths from chapters
            paths = []
            for i, (chapter, confidence, reasoning) in enumerate(top_chapters, 1):
                chapter_desc = self.chapters_map.get(int(chapter), "Unknown chapter")
                path = ClassificationPath(f"path_{i}", chapter, confidence, chapter_desc)
                path.reasoning_log.append(f"Initial chapter selection: {chapter} - {chapter_desc} (conf: {confidence:.3f}) - {reasoning}")
                paths.append(path)
            
            state["beam"] = paths
            state["used_multi_hypothesis"] = True
            state["paths_considered"] = [{"chapter": p.chapter, "reasoning": p.reasoning_log[0] if p.reasoning_log else ""} for p in paths]
            
            # Now run the beam search with event streaming
            beam = state["beam"]
            k = CLASSIFICATION_BEAM_SIZE
            max_iterations = 10
            iteration = state.get("iteration_count", 0)
            
            while iteration < max_iterations:
                iteration += 1
                state["iteration_count"] = iteration
                
                yield {
                    "type": "iteration_start",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "iteration": iteration,
                        "beam_size": len(beam)
                    }
                }
                
                # Check for termination
                should_terminate, best_path = self.check_termination_conditions(beam)
                if should_terminate:
                    if best_path:
                        # Generate explanation
                        explanation = ""
                        if state.get("conversation"):
                            explanation = self.explain_classification(
                                state.get("original_query", ""),
                                state.get("current_query", ""),
                                best_path.get_full_path_string(),
                                state.get("conversation", []),
                                state
                            )
                        
                        yield {
                            "type": "classification_complete",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "data": {
                                "final_code": best_path.get_final_code(),
                                "full_path": best_path.get_full_path_string(),
                                "confidence": best_path.cumulative_confidence,
                                "log_score": best_path.log_score,
                                "reasoning": best_path.reasoning_log,
                                "explanation": explanation
                            }
                        }
                        return
                    else:
                        yield {
                            "type": "error",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "data": {
                                "error": "No valid classification path found",
                                "message": "All paths were pruned"
                            }
                        }
                        return
                
                # Check if question needed for the top path
                top_path = beam[0]
                if top_path.is_active and interactive and state["questions_asked"] < max_questions:
                    node_for_next_step = top_path.current_node or self.create_chapter_parent(top_path.chapter)
                    options = self.get_children(node_for_next_step)
                    
                    if options:
                        yield {
                            "type": "candidate_scoring_start",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "data": {
                                "path_id": top_path.path_id,
                                "current_node": node_for_next_step.htsno or "[GROUP]",
                                "options_count": len(options)
                            }
                        }
                        
                        scored_candidates = self.score_candidates(state["current_query"], node_for_next_step, options)
                        
                        if scored_candidates:
                            _, best_confidence, _ = max(scored_candidates, key=lambda item: item[1])
                            
                            yield {
                                "type": "candidate_scoring_complete",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "data": {
                                    "candidates": [
                                        {
                                            "code": node.htsno or "[GROUP]",
                                            "description": node.description,
                                            "confidence": confidence,
                                            "reasoning": reasoning
                                        }
                                        for node, confidence, reasoning in scored_candidates
                                    ],
                                    "best_confidence": best_confidence
                                }
                            }
                            
                            if best_confidence < 0.85:
                                stage = TreeUtils.determine_next_stage(node_for_next_step)
                                question = self.generate_clarification_question(state["current_query"], node_for_next_step, stage, state)
                                
                                # Save state before returning question
                                state["pending_question"] = question.to_dict()
                                state["pending_stage"] = stage
                                # Serialize the beam for state preservation
                                state["beam"] = [p.to_dict() for p in beam]
                                
                                yield {
                                    "type": "question_generated",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "data": {
                                        "question": question.to_dict(),
                                        "confidence_trigger": best_confidence,
                                        "stage": stage,
                                        "state": state  # Include state for continuation
                                    }
                                }
                                return  # Stop here and wait for user response
                
                # Advance beam
                yield {
                    "type": "beam_advancement_start",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "current_beam_size": len(beam)
                    }
                }
                
                new_beam = self.advance_beam(beam, state["current_query"], k)
                
                yield {
                    "type": "beam_leaderboard",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "beam": [
                            {
                                "position": i + 1,
                                "path_id": path.path_id,
                                "chapter": path.chapter,
                                "current_path": path.get_full_path_string(),
                                "log_score": path.log_score,
                                "cumulative_confidence": path.cumulative_confidence,
                                "is_active": path.is_active,
                                "is_complete": path.is_complete
                            }
                            for i, path in enumerate(new_beam)
                        ]
                    }
                }
                
                beam = new_beam
                state["beam"] = beam
                
                if not beam:
                    yield {
                        "type": "error",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "error": "Beam became empty",
                            "message": "All paths were eliminated during advancement"
                        }
                    }
                    return
            
            # Max iterations reached
            if beam:
                best_path = max(beam, key=lambda p: p.log_score)
                yield {
                    "type": "classification_complete",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "final_code": best_path.get_final_code(),
                        "full_path": best_path.get_full_path_string(),
                        "confidence": best_path.cumulative_confidence,
                        "log_score": best_path.log_score,
                        "reasoning": best_path.reasoning_log,
                        "note": "Max iterations reached"
                    }
                }
            else:
                yield {
                    "type": "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "error": "Classification failed",
                        "message": "Maximum iterations reached with no valid paths"
                    }
                }
                
        except Exception as e:
            logging.error(f"Error in run_classification_with_events: {e}", exc_info=True)
            yield {
                "type": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "error": str(e),
                    "message": "Classification stream error"
                }
            }

    async def continue_classification_with_events(self, state: Dict, answer: str, interactive: bool = True, max_questions: int = 3):
        """
        Continue streaming classification after receiving an answer to a clarification question.
        This version processes the answer and continues beam search with full event streaming.
        
        Args:
            state: The classification state (including serialized beam)
            answer: User's answer to the clarification question
            interactive: Whether to ask clarification questions
            max_questions: Maximum number of questions to ask
            
        Yields:
            SSE events for the continuation process
        """
        import asyncio
        from datetime import datetime, timezone
        import hashlib
        
        # DEBUG: Log initial state
        logging.info("=== STREAMING CONTINUE: Starting continue_classification_with_events ===")
        logging.info(f"Answer received: '{answer}'")
        logging.info(f"Questions asked so far: {state.get('questions_asked', 0)}")
        logging.info(f"Beam size in state: {len(state.get('beam', []))}")
        
        # DEBUG: Create a state checksum for validation
        state_str = json.dumps(state.get('beam', []), sort_keys=True)
        state_checksum = hashlib.md5(state_str.encode()).hexdigest()[:8]
        logging.info(f"State checksum (beam): {state_checksum}")
        
        # Yield continuation start event
        yield {
            "type": "continuation_start",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "answer": answer,
                "questions_asked": state.get("questions_asked", 0),
                "max_questions": max_questions,
                "beam_size": len(state.get("beam", [])),
                "state_checksum": state_checksum  # For debugging
            }
        }
        
        try:
            # Validate state has a pending question
            if not state.get("pending_question"):
                logging.error("No pending question in state!")
                yield {
                    "type": "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "error": "No pending question",
                        "message": "Cannot continue without a pending question"
                    }
                }
                return
            
            # Yield answer processing event
            yield {
                "type": "answer_processing",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "question": state.get("pending_question", {}).get("question_text", ""),
                    "answer": answer,
                    "stage": state.get("pending_stage", "unknown")
                }
            }
            
            # Process the answer to enrich the product description (like continue_classification does)
            logging.info("Processing answer to enrich product description...")
            
            # Restore beam from JSON if needed
            if state.get("beam"):
                logging.info(f"Restoring beam with {len(state['beam'])} paths")
                restored_beam = []
                
                for i, path_data in enumerate(state["beam"]):
                    if isinstance(path_data, dict):
                        try:
                            path = ClassificationPath.from_dict(path_data, self)
                            restored_beam.append(path)
                            logging.info(f"[OK] Restored path {i+1}: {path.path_id}")
                        except Exception as e:
                            logging.error(f"Failed to restore path {i+1}: {e}")
                    elif hasattr(path_data, 'path_id'):
                        restored_beam.append(path_data)
                        logging.info(f"[OK] Path {i+1} already restored: {path_data.path_id}")
                
                if not restored_beam:
                    logging.error("Beam restoration failed")
                    yield {
                        "type": "error",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "error": "Beam restoration failed",
                            "message": "Could not restore classification paths from state"
                        }
                    }
                    return
                
                state["beam"] = restored_beam
                logging.info(f"[OK] Successfully restored beam with {len(restored_beam)} paths")
            
            # Process the answer
            state["questions_asked"] += 1
            pending_question_dict = state.get("pending_question", {})
            question_text = pending_question_dict.get("question_text", "")
            
            # Log Q&A
            state["conversation"].append({"question": question_text, "answer": answer})
            if "history" not in state:
                state["history"] = []
            state["history"].append({"question": question_text, "answer": answer})
            
            # Create question object for processing
            question_obj = ClarificationQuestion()
            question_obj.question_text = question_text
            question_obj.question_type = pending_question_dict.get("question_type", "text")
            question_obj.options = pending_question_dict.get("options", [])
            question_obj.metadata = pending_question_dict.get("metadata", {})
            options_metadata = question_obj.metadata.get("options", [])
            
            # Process answer to update product description
            updated_query, _ = self.process_answer(state["current_query"], question_obj, answer, options_metadata, state)
            state["current_query"] = updated_query
            logging.info(f"Product description enriched: \"{state['current_query']}\"")
            
            # Yield the enriched description event
            yield {
                "type": "description_enriched",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "original_description": state.get("original_query", ""),
                    "previous_description": question_obj.metadata.get("original_query", state.get("original_query", "")),
                    "updated_description": updated_query,
                    "extracted_attributes": state.get("product_attributes", {}),
                    "question_asked": question_text,
                    "answer_provided": answer
                }
            }
            
            # Clear pending question
            state["pending_question"] = None
            state["pending_stage"] = None
            
            # Continue with beam search iterations WITH EVENT STREAMING
            beam = state.get("beam", [])
            k = CLASSIFICATION_BEAM_SIZE
            max_iterations = 10
            iteration = state.get("iteration_count", 0)
            
            while iteration < max_iterations:
                iteration += 1
                state["iteration_count"] = iteration
                
                yield {
                    "type": "iteration_start",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "iteration": iteration,
                        "beam_size": len(beam)
                    }
                }
                
                # Check for termination
                should_terminate, best_path = self.check_termination_conditions(beam)
                if should_terminate:
                    if best_path:
                        # Generate explanation
                        explanation = ""
                        if state.get("conversation"):
                            explanation = self.explain_classification(
                                state.get("original_query", ""),
                                state.get("current_query", ""),
                                best_path.get_full_path_string(),
                                state.get("conversation", []),
                                state
                            )
                        
                        yield {
                            "type": "classification_complete",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "data": {
                                "final_code": best_path.get_final_code(),
                                "full_path": best_path.get_full_path_string(),
                                "confidence": best_path.cumulative_confidence,
                                "log_score": best_path.log_score,
                                "reasoning": best_path.reasoning_log,
                                "explanation": explanation
                            }
                        }
                        return
                    else:
                        yield {
                            "type": "error",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "data": {
                                "error": "No valid classification path found",
                                "message": "All paths were pruned"
                            }
                        }
                        return
                
                # Check if question needed
                top_path = beam[0]
                if top_path.is_active and interactive and state["questions_asked"] < max_questions:
                    node_for_next_step = top_path.current_node or self.create_chapter_parent(top_path.chapter)
                    options = self.get_children(node_for_next_step)
                    
                    if options:
                        yield {
                            "type": "candidate_scoring_start",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "data": {
                                "path_id": top_path.path_id,
                                "current_node": node_for_next_step.htsno or "[GROUP]",
                                "options_count": len(options)
                            }
                        }
                        
                        scored_candidates = self.score_candidates(state["current_query"], node_for_next_step, options)
                        
                        if scored_candidates:
                            _, best_confidence, _ = max(scored_candidates, key=lambda item: item[1])
                            
                            yield {
                                "type": "candidate_scoring_complete",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "data": {
                                    "candidates": [
                                        {
                                            "code": node.htsno or "[GROUP]",
                                            "description": node.description,
                                            "confidence": confidence,
                                            "reasoning": reasoning
                                        }
                                        for node, confidence, reasoning in scored_candidates
                                    ],
                                    "best_confidence": best_confidence
                                }
                            }
                            
                            if best_confidence < 0.85:
                                stage = TreeUtils.determine_next_stage(node_for_next_step)
                                question = self.generate_clarification_question(state["current_query"], node_for_next_step, stage, state)
                                
                                # Save state before returning question
                                state["pending_question"] = question.to_dict()
                                state["pending_stage"] = stage
                                # Serialize the beam for state preservation
                                state["beam"] = [p.to_dict() for p in beam]
                                
                                yield {
                                    "type": "question_generated",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "data": {
                                        "question": question.to_dict(),
                                        "confidence_trigger": best_confidence,
                                        "stage": stage,
                                        "state": state  # Include state for continuation
                                    }
                                }
                                return
                
                # Advance beam
                yield {
                    "type": "beam_advancement_start",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "current_beam_size": len(beam)
                    }
                }
                
                new_beam = self.advance_beam(beam, state["current_query"], k)
                
                yield {
                    "type": "beam_leaderboard",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "beam": [
                            {
                                "position": i + 1,
                                "path_id": path.path_id,
                                "chapter": path.chapter,
                                "current_path": path.get_full_path_string(),
                                "log_score": path.log_score,
                                "cumulative_confidence": path.cumulative_confidence,
                                "is_active": path.is_active,
                                "is_complete": path.is_complete
                            }
                            for i, path in enumerate(new_beam)
                        ]
                    }
                }
                
                beam = new_beam
                state["beam"] = beam
                
                if not beam:
                    yield {
                        "type": "error",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": {
                            "error": "Beam became empty",
                            "message": "All paths were eliminated during advancement"
                        }
                    }
                    return
            
            # Max iterations reached
            if beam:
                best_path = max(beam, key=lambda p: p.log_score)
                yield {
                    "type": "classification_complete",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "final_code": best_path.get_final_code(),
                        "full_path": best_path.get_full_path_string(),
                        "confidence": best_path.cumulative_confidence,
                        "log_score": best_path.log_score,
                        "reasoning": best_path.reasoning_log,
                        "note": "Max iterations reached"
                    }
                }
            else:
                yield {
                    "type": "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {
                        "error": "Classification failed",
                        "message": "Maximum iterations reached with no valid paths"
                    }
                }
                
        except Exception as e:
            logging.error(f"Error in continue_classification_with_events: {e}", exc_info=True)
            yield {
                "type": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "error": str(e),
                    "message": "Classification continuation stream error"
                }
            }
        finally:
            logging.info("=== STREAMING CONTINUE: Ending continue_classification_with_events ===")

    def _convert_path_to_result(self, path: ClassificationPath, state: Dict) -> Dict:
        """
        Convert a ClassificationPath to a result dictionary.
        
        Args:
            path: The path to convert
            state: The current state dictionary
            
        Returns:
            Dictionary with classification result
        """
        # Get the final code and full path
        final_code = path.get_final_code() or ""
        full_path = path.get_full_path_string()
        
        # Create the classification result
        result = {
            "final": True,
            "classification": {
                "code": final_code,
                "path": full_path,
                "confidence": path.cumulative_confidence,
                "log_score": path.log_score,
                "chapter": path.chapter,
                "steps": path.steps,
                "reasoning": path.reasoning_log
            },
            "state": state
        }
        
        # Add explanation if we have conversation history
        if state.get("conversation"):
            explanation = self.explain_classification(
                state.get("original_query", ""),
                state.get("current_query", ""),
                full_path,
                state.get("conversation", []),
                state
            )
            result["explanation"] = explanation
        
        return result
    
    def _process_beam_search_classification(self, state: Dict, interactive: bool, max_questions: int) -> Dict:
        """
        Process classification using beam search with robust state management and question logic.
        """
        # A confidence score below this for the BEST next option will trigger a question.
        CONFIDENCE_THRESHOLD = 0.85
        
        beam = state.get("beam", [])
        k = state.get("hypothesis_count", 3)
        
        if not beam:
            logging.warning("Beam is empty. Restarting classification process.")
            state["beam"] = [] # Ensure beam is cleared before restart
            return self.process_classification(state, interactive, max_questions)
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logging.info(f"--- Beam Search Iteration {iteration} (Beam Size: {len(beam)}) ---")
            
            # 1. Check for termination
            should_terminate, best_path = self.check_termination_conditions(beam)
            if should_terminate:
                if best_path:
                    logging.info(f"Termination condition met. Returning best path: {best_path.path_id}")
                    return self._convert_path_to_result(best_path, state)
                else:
                    logging.warning("Termination met, but no best path. All paths pruned. Triggering diagnosis.")
                    diagnosis = self._diagnose_classification_issue(state['current_query'], "multiple", "completion failure")
                    state["classification_diagnosis"] = diagnosis
                    state["beam"] = []
                    return self.process_classification(state, interactive, max_questions)

            # 2. Check if a question is needed for the top path before advancing
            top_path = beam[0]

            # Only ask a question if the top path is active and we are in interactive mode.
            if top_path.is_active and interactive and state["questions_asked"] < max_questions:
                node_for_next_step = top_path.current_node or self.create_chapter_parent(top_path.chapter)
                options = self.get_children(node_for_next_step)
                
                if options:
                    scored_candidates = self.score_candidates(state["current_query"], node_for_next_step, options)
                    if scored_candidates:
                        # Find the confidence of the single best next step for the top path
                        _, best_confidence, _ = max(scored_candidates, key=lambda item: item[1])
                        
                        # ASK A QUESTION IF: the confidence for the best choice is below our threshold.
                        if best_confidence < CONFIDENCE_THRESHOLD:
                            logging.info(f"Top path ({top_path.path_id}) has low confidence ({best_confidence:.2f}) for its next step. Generating question.")
                            stage = TreeUtils.determine_next_stage(node_for_next_step)
                            question = self.generate_clarification_question(state["current_query"], node_for_next_step, stage, state)
                            
                            # Save state and return the question to the user
                            state["pending_question"] = question.to_dict()
                            state["pending_stage"] = stage
                            # Serialize the beam objects to dictionaries before saving state
                            state["beam"] = [p.to_dict() for p in beam]
                            
                            return {
                                "final": False,
                                "clarification_question": question,
                                "state": state,
                                "debug_info": {
                                    "beam_size": len(beam),
                                    "top_path_log_score": top_path.log_score,
                                    "next_step_confidence_trigger": best_confidence
                                }
                            }

            # 3. If no question was asked, advance the entire beam
            logging.info(f"Confidence sufficient or questioning disabled. Advancing the beam.")
            new_beam = self.advance_beam(beam, state["current_query"], k)
            
            # This assignment is the critical fix for the state corruption.
            beam = new_beam
            state["beam"] = beam
            
            if not beam:
                logging.error("Beam became empty after advancement. Triggering diagnosis.")
                diagnosis = self._diagnose_classification_issue(state['current_query'], "multiple", "advancement failure")
                state["classification_diagnosis"] = diagnosis
                state["beam"] = []
                return self.process_classification(state, interactive, max_questions)

        # If loop finishes due to max iterations, return the best path we have.
        if not beam:
             return {"error": "Classification failed after maximum iterations.", "final": False, "state": state}
             
        best_path = max(beam, key=lambda p: p.log_score)
        logging.warning(f"Max iterations reached. Returning best available path: {best_path.path_id}")
        return self._convert_path_to_result(best_path, state)
