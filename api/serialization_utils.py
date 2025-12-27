import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Union

class SerializationError(Exception):
    """Custom exception for serialization errors"""
    pass

class HTSSerializer:
    """Robust serializer for HTS classification objects"""
    
    @staticmethod
    def serialize_node_reference(node: Optional['HTSNode']) -> Optional[Dict[str, Any]]:
        """Serialize a node as a reference (not full object)"""
        if node is None:
            return None
        return {
            "_type": "node_ref",
            "node_id": node.node_id,
            "htsno": node.htsno,
            "description": node.description[:50] + "..." if len(node.description) > 50 else node.description
        }
    
    @staticmethod
    def serialize_classification_path(path: 'ClassificationPath') -> Dict[str, Any]:
        """Serialize a ClassificationPath with full integrity checks"""
        try:
            logging.debug(f"Serializing ClassificationPath: {path.path_id}")
            
            # Create checksum of critical data
            critical_data = f"{path.path_id}:{path.chapter}:{path.log_score}:{path.cumulative_confidence}"
            checksum = hashlib.md5(critical_data.encode()).hexdigest()[:8]
            
            serialized = {
                "_type": "classification_path",
                "_checksum": checksum,
                "path_id": path.path_id,
                "chapter": path.chapter,
                "log_score": float(path.log_score),  # Ensure float
                "cumulative_confidence": float(path.cumulative_confidence),
                "is_active": bool(path.is_active),
                "is_complete": bool(path.is_complete),
                "current_node": HTSSerializer.serialize_node_reference(path.current_node),
                "classification_path": path.classification_path[:],  # Copy list
                "steps": path.steps[:],  # Copy list
                "visited_nodes": path.visited_nodes[:],  # Copy list
                "selection": dict(path.selection),  # Copy dict
                "reasoning_log": path.reasoning_log[:],  # Copy list
                "failure_reason": path.failure_reason,
                # Trajectory support - serialize the chat trajectory for stateful classification
                "trajectory": [msg.copy() for msg in getattr(path, 'trajectory', [])],
                "_trajectory_initialized": getattr(path, '_trajectory_initialized', False)
            }
            
            # Validate serialization
            json.dumps(serialized)  # This will raise if not serializable
            
            logging.debug(f"Successfully serialized ClassificationPath {path.path_id} with checksum {checksum}")
            return serialized
            
        except Exception as e:
            logging.error(f"Failed to serialize ClassificationPath {path.path_id}: {e}")
            raise SerializationError(f"Failed to serialize ClassificationPath {path.path_id}: {e}")
    
    @staticmethod
    def deserialize_classification_path(data: Dict[str, Any], hts_tree: 'HTSTree') -> 'ClassificationPath':
        """Deserialize a ClassificationPath with validation"""
        try:
            path_id = data.get('path_id', 'unknown')
            logging.debug(f"Deserializing ClassificationPath: {path_id}")
            
            # Validate type
            if data.get("_type") != "classification_path":
                raise ValueError(f"Invalid type: {data.get('_type')}")
            
            # Validate checksum if present
            if "_checksum" in data:
                critical_data = f"{data['path_id']}:{data['chapter']}:{data['log_score']}:{data['cumulative_confidence']}"
                expected_checksum = hashlib.md5(critical_data.encode()).hexdigest()[:8]
                if data["_checksum"] != expected_checksum:
                    logging.warning(f"Checksum mismatch for path {data['path_id']}: expected {expected_checksum}, got {data['_checksum']}")
            
            # Get chapter description safely
            chapter_desc = ""
            if hasattr(hts_tree, 'chapters_map'):
                chapter_desc = hts_tree.chapters_map.get(int(data['chapter']), "Unknown chapter")
            
            # Create path with minimal data
            from .groq_tree_engine import ClassificationPath  # Import here to avoid circular imports
            path = ClassificationPath(
                path_id=data["path_id"],
                chapter=data["chapter"],
                chapter_confidence=1.0,  # Reset confidence, will be updated
                chapter_description=chapter_desc
            )
            
            # Restore all fields with type validation
            path.log_score = float(data.get("log_score", 0.0))
            path.cumulative_confidence = float(data.get("cumulative_confidence", 1.0))
            path.is_active = bool(data.get("is_active", True))
            path.is_complete = bool(data.get("is_complete", False))
            
            # Restore lists (ensure they are lists)
            path.classification_path = list(data.get("classification_path", []))
            path.steps = list(data.get("steps", []))
            path.visited_nodes = list(data.get("visited_nodes", []))
            path.reasoning_log = list(data.get("reasoning_log", []))
            
            # Restore dict (ensure it's a dict)
            path.selection = dict(data.get("selection", {}))
            
            # Restore optional fields
            path.failure_reason = data.get("failure_reason")
            
            # Restore current_node reference
            node_ref = data.get("current_node")
            if node_ref and isinstance(node_ref, dict) and node_ref.get("node_id") is not None:
                path.current_node = hts_tree.get_node_by_id(node_ref["node_id"])
                if path.current_node is None:
                    logging.warning(f"Could not restore node {node_ref['node_id']} for path {path.path_id}")
            else:
                path.current_node = None
            
            # Restore trajectory for stateful classification
            path.trajectory = [msg.copy() for msg in data.get("trajectory", [])]
            path._trajectory_initialized = bool(data.get("_trajectory_initialized", False))
            
            logging.debug(f"Successfully deserialized ClassificationPath {path_id}")
            return path
            
        except Exception as e:
            logging.error(f"Failed to deserialize ClassificationPath: {e}")
            raise SerializationError(f"Failed to deserialize ClassificationPath: {e}")
    
    @staticmethod
    def serialize_beam(beam: List['ClassificationPath']) -> List[Dict[str, Any]]:
        """Serialize entire beam with validation"""
        logging.info(f"Serializing beam with {len(beam)} paths")
        
        serialized_beam = []
        for i, path in enumerate(beam):
            try:
                serialized_path = HTSSerializer.serialize_classification_path(path)
                serialized_path["_beam_position"] = i
                serialized_beam.append(serialized_path)
                logging.debug(f"Serialized beam path {i}: {path.path_id}")
            except Exception as e:
                logging.error(f"Failed to serialize beam path {i}: {e}")
                raise SerializationError(f"Beam serialization failed at position {i}: {e}")
        
        logging.info(f"Successfully serialized beam with {len(serialized_beam)} paths")
        return serialized_beam
    
    @staticmethod
    def deserialize_beam(data: List[Dict[str, Any]], hts_tree: 'HTSTree') -> List['ClassificationPath']:
        """Deserialize entire beam with validation"""
        logging.info(f"Deserializing beam with {len(data)} paths")
        
        beam = []
        for i, path_data in enumerate(data):
            try:
                # Validate beam position if present
                if "_beam_position" in path_data and path_data["_beam_position"] != i:
                    logging.warning(f"Beam position mismatch: expected {i}, got {path_data['_beam_position']}")
                
                path = HTSSerializer.deserialize_classification_path(path_data, hts_tree)
                beam.append(path)
                logging.debug(f"Deserialized beam path {i}: {path.path_id}")
            except Exception as e:
                logging.error(f"Failed to deserialize beam path {i}: {e}")
                raise SerializationError(f"Beam deserialization failed at position {i}: {e}")
        
        # Validate beam is properly sorted by log_score
        log_scores = [p.log_score for p in beam]
        if log_scores != sorted(log_scores, reverse=True):
            logging.warning("Beam not properly sorted after deserialization, re-sorting")
            beam.sort(key=lambda p: p.log_score, reverse=True)
        
        logging.info(f"Successfully deserialized beam with {len(beam)} paths")
        return beam
    
    @staticmethod
    def serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize entire state dictionary"""
        try:
            logging.info("Starting state serialization")
            
            # Create a copy to avoid modifying original
            serialized_state = {}
            
            # Handle simple fields
            simple_fields = [
                "product", "original_query", "current_query", "questions_asked",
                "max_questions", "global_retry_count", "classification_diagnosis",
                "use_multi_hypothesis", "hypothesis_count", "used_multi_hypothesis",
                "iteration_count", "pending_stage"
            ]
            
            for field in simple_fields:
                if field in state:
                    serialized_state[field] = state[field]
                    logging.debug(f"Serialized simple field: {field}")
            
            # Handle lists that should be copied
            list_fields = [
                "visited_nodes", "recent_questions", "paths_considered",
                "_treerl_pruned_leaves"  # TreeRL pruned paths for reward computation
            ]
            
            for field in list_fields:
                if field in state and isinstance(state[field], list):
                    serialized_state[field] = state[field][:]
                    logging.debug(f"Serialized list field: {field} with {len(state[field])} items")
            
            # Handle dicts that should be copied
            dict_fields = [
                "selection", "product_attributes"
            ]
            
            for field in dict_fields:
                if field in state and isinstance(state[field], dict):
                    serialized_state[field] = dict(state[field])
                    logging.debug(f"Serialized dict field: {field} with {len(state[field])} items")
            
            # Handle complex fields
            
            # Conversation and history
            if "conversation" in state:
                serialized_state["conversation"] = [dict(item) for item in state["conversation"]]
                logging.debug(f"Serialized conversation with {len(state['conversation'])} entries")
            
            if "history" in state:
                serialized_state["history"] = [dict(item) for item in state["history"]]
                logging.debug(f"Serialized history with {len(state['history'])} entries")
            
            # Handle beam (list of ClassificationPath objects)
            if "beam" in state and state["beam"]:
                if isinstance(state["beam"][0], dict):
                    # Already serialized
                    serialized_state["beam"] = state["beam"]
                    logging.debug(f"Beam already serialized with {len(state['beam'])} paths")
                else:
                    # Need to serialize
                    serialized_state["beam"] = HTSSerializer.serialize_beam(state["beam"])
                    logging.debug(f"Serialized beam with {len(state['beam'])} paths")
            else:
                serialized_state["beam"] = []
                logging.debug("No beam to serialize")
            
            # Handle paths (older field, might still exist)
            if "paths" in state and state["paths"]:
                if isinstance(state["paths"][0], dict):
                    serialized_state["paths"] = state["paths"]
                    logging.debug(f"Paths already serialized with {len(state['paths'])} paths")
                else:
                    serialized_state["paths"] = [
                        HTSSerializer.serialize_classification_path(p) 
                        for p in state["paths"]
                    ]
                    logging.debug(f"Serialized paths with {len(state['paths'])} paths")
            
            # Handle current_node
            if "current_node" in state and state["current_node"] is not None:
                if hasattr(state["current_node"], 'node_id'):
                    serialized_state["current_node"] = HTSSerializer.serialize_node_reference(state["current_node"])
                    logging.debug(f"Serialized current_node: {state['current_node'].node_id}")
                else:
                    serialized_state["current_node"] = state["current_node"]
                    logging.debug("Current_node already serialized")
            
            # Handle classification_path list
            if "classification_path" in state:
                serialized_state["classification_path"] = state["classification_path"][:]
                logging.debug(f"Serialized classification_path with {len(state['classification_path'])} steps")
            
            # Handle steps list
            if "steps" in state:
                serialized_state["steps"] = state["steps"][:]
                logging.debug(f"Serialized steps with {len(state['steps'])} steps")
            
            # Handle pending_question
            if "pending_question" in state and state["pending_question"] is not None:
                if hasattr(state["pending_question"], 'to_dict'):
                    serialized_state["pending_question"] = state["pending_question"].to_dict()
                    logging.debug("Serialized pending_question object")
                else:
                    serialized_state["pending_question"] = state["pending_question"]
                    logging.debug("Pending_question already serialized")
            
            # Add metadata
            serialized_state["_serialization_version"] = "1.0"
            serialized_state["_state_checksum"] = HTSSerializer._calculate_state_checksum(serialized_state)
            
            # Validate the entire state is serializable
            json.dumps(serialized_state)
            
            logging.info(f"Successfully serialized state with checksum {serialized_state['_state_checksum']}")
            return serialized_state
            
        except Exception as e:
            logging.error(f"Failed to serialize state: {e}")
            raise SerializationError(f"Failed to serialize state: {e}")
    
    @staticmethod
    def deserialize_state(data: Dict[str, Any], hts_tree: 'HTSTree') -> Dict[str, Any]:
        """Deserialize state dictionary"""
        try:
            logging.info("Starting state deserialization")
            
            # Validate serialization version
            version = data.get("_serialization_version", "0.0")
            if version != "1.0":
                logging.warning(f"Deserializing state from version {version}, current version is 1.0")
            
            # Create new state dict
            state = {}
            
            # Copy simple fields
            simple_fields = [
                "product", "original_query", "current_query", "questions_asked",
                "max_questions", "global_retry_count", "classification_diagnosis",
                "use_multi_hypothesis", "hypothesis_count", "used_multi_hypothesis",
                "iteration_count", "pending_stage", "pending_question"
            ]
            
            for field in simple_fields:
                if field in data:
                    state[field] = data[field]
                    logging.debug(f"Deserialized simple field: {field}")
            
            # Copy lists
            list_fields = [
                "visited_nodes", "recent_questions", "paths_considered",
                "conversation", "history", "classification_path", "steps"
            ]
            
            for field in list_fields:
                if field in data:
                    state[field] = data[field][:]
                    logging.debug(f"Deserialized list field: {field} with {len(data[field])} items")
            
            # Copy dicts
            dict_fields = ["selection", "product_attributes"]
            
            for field in dict_fields:
                if field in data:
                    state[field] = dict(data[field])
                    logging.debug(f"Deserialized dict field: {field} with {len(data[field])} items")
            
            # Deserialize beam
            if "beam" in data and data["beam"]:
                state["beam"] = HTSSerializer.deserialize_beam(data["beam"], hts_tree)
                logging.debug(f"Deserialized beam with {len(state['beam'])} paths")
            else:
                state["beam"] = []
                logging.debug("No beam to deserialize")
            
            # Deserialize paths (if present)
            if "paths" in data and data["paths"]:
                state["paths"] = [
                    HTSSerializer.deserialize_classification_path(p, hts_tree)
                    for p in data["paths"]
                ]
                logging.debug(f"Deserialized paths with {len(state['paths'])} paths")
            
            # Deserialize current_node
            if "current_node" in data and data["current_node"] is not None:
                if isinstance(data["current_node"], dict) and data["current_node"].get("_type") == "node_ref":
                    node_id = data["current_node"].get("node_id")
                    if node_id is not None:
                        state["current_node"] = hts_tree.get_node_by_id(node_id)
                        if state["current_node"] is None:
                            logging.warning(f"Could not restore current_node with ID {node_id}")
                        else:
                            logging.debug(f"Deserialized current_node: {node_id}")
                else:
                    state["current_node"] = None
                    logging.debug("Current_node data invalid")
            
            # Initialize missing fields
            state.setdefault("history", [])
            state.setdefault("product_attributes", {})
            state.setdefault("recent_questions", [])
            
            logging.info("Successfully deserialized state")
            return state
            
        except Exception as e:
            logging.error(f"Failed to deserialize state: {e}")
            raise SerializationError(f"Failed to deserialize state: {e}")
    
    @staticmethod
    def _calculate_state_checksum(state: Dict[str, Any]) -> str:
        """Calculate checksum for state validation"""
        # Use key fields for checksum
        key_data = f"{state.get('product', '')}:{state.get('questions_asked', 0)}:{len(state.get('beam', []))}"
        return hashlib.md5(key_data.encode()).hexdigest()[:8]
    
    @staticmethod
    def log_serialization_operation(operation: str, object_type: str, object_id: str = None, 
                                   before_data: Any = None, after_data: Any = None, 
                                   success: bool = True, error: str = None):
        """Comprehensive logging for serialization operations"""
        log_prefix = f"SERIALIZATION[{operation}][{object_type}]"
        
        if object_id:
            log_prefix += f"[{object_id}]"
        
        if success:
            logging.info(f"{log_prefix}: SUCCESS")
            if before_data is not None:
                logging.debug(f"{log_prefix}: BEFORE - Type: {type(before_data)}, Size: {len(str(before_data))}")
            if after_data is not None:
                logging.debug(f"{log_prefix}: AFTER - Type: {type(after_data)}, Size: {len(str(after_data))}")
        else:
            logging.error(f"{log_prefix}: FAILED - {error}")
            if before_data is not None:
                logging.error(f"{log_prefix}: FAILED_DATA - {str(before_data)[:200]}...")
    
    @staticmethod
    def validate_json_serializable(obj: Any, context: str = "") -> bool:
        """Validate that an object can be serialized to JSON"""
        try:
            json.dumps(obj)
            logging.debug(f"JSON validation SUCCESS for {context}")
            return True
        except (TypeError, OverflowError) as e:
            logging.error(f"JSON validation FAILED for {context}: {e}")
            return False
