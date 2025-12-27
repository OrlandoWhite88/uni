import json
import os
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FileSessionStore:
    """
    File-based session storage with automatic cleanup and TTL support.
    Each session is stored as a JSON file with metadata.
    """
    
    def __init__(self, sessions_dir: str = "./sessions", default_ttl_minutes: int = 30):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.default_ttl_minutes = default_ttl_minutes
        logger.info(f"FileSessionStore initialized: {self.sessions_dir} (TTL: {default_ttl_minutes}m)")
    
    def create_session(self, initial_state: Dict[str, Any], ttl_minutes: Optional[int] = None) -> str:
        """Create a new session and return session_id"""
        session_id = str(uuid.uuid4())
        ttl = ttl_minutes or self.default_ttl_minutes
        expires_at = datetime.now() + timedelta(minutes=ttl)
        
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "state": initial_state
        }
        
        session_file = self.sessions_dir / f"{session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Created session {session_id} (expires: {expires_at.strftime('%H:%M:%S')})")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session state by ID, return None if not found or expired"""
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            logger.warning(f"Session {session_id} not found")
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Check if expired
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if datetime.now() > expires_at:
                logger.info(f"Session {session_id} expired, removing")
                session_file.unlink(missing_ok=True)
                return None
            
            # Update last accessed time
            session_data["last_accessed"] = datetime.now().isoformat()
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
            
            return session_data["state"]
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error reading session {session_id}: {e}")
            # Remove corrupted session file
            session_file.unlink(missing_ok=True)
            return None
    
    def update_session(self, session_id: str, state: Dict[str, Any]) -> bool:
        """Update session state, return True if successful"""
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            logger.warning(f"Cannot update non-existent session {session_id}")
            return False
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Check if expired
            expires_at = datetime.fromisoformat(session_data["expires_at"])
            if datetime.now() > expires_at:
                logger.info(f"Session {session_id} expired during update, removing")
                session_file.unlink(missing_ok=True)
                return False
            
            # Update state and timestamp
            session_data["state"] = state
            session_data["last_accessed"] = datetime.now().isoformat()
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
            
            return True
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error updating session {session_id}: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        session_file = self.sessions_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            logger.info(f"Deleted session {session_id}")
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions, return count of cleaned sessions"""
        cleaned_count = 0
        now = datetime.now()
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                expires_at = datetime.fromisoformat(session_data["expires_at"])
                if now > expires_at:
                    session_file.unlink()
                    cleaned_count += 1
                    logger.debug(f"Cleaned expired session {session_file.stem}")
                    
            except Exception as e:
                logger.error(f"Error checking session {session_file}: {e}")
                # Remove corrupted files
                session_file.unlink(missing_ok=True)
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned {cleaned_count} expired sessions")
        
        return cleaned_count
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about current sessions"""
        total_sessions = 0
        expired_sessions = 0
        now = datetime.now()
        
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                total_sessions += 1
                expires_at = datetime.fromisoformat(session_data["expires_at"])
                if now > expires_at:
                    expired_sessions += 1
                    
            except Exception:
                total_sessions += 1  # Count corrupted files too
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": total_sessions - expired_sessions,
            "expired_sessions": expired_sessions,
            "sessions_directory": str(self.sessions_dir)
        }

# Global session store instance
session_store = None

def get_session_store() -> FileSessionStore:
    """Get or create the global session store instance"""
    global session_store
    if session_store is None:
        # Create sessions directory in project root or use /tmp for serverless
        sessions_dir = os.environ.get("SESSIONS_DIR", "./sessions")
        ttl_minutes = int(os.environ.get("SESSION_TTL_MINUTES", "30"))
        session_store = FileSessionStore(sessions_dir, ttl_minutes)
    return session_store
