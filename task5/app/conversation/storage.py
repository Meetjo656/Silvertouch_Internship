from typing import Dict, Optional
import time
from app.conversation.session import ConversationSession

class Storage:
    """In-memory or persistent storage interface for conversation sessions."""
    
    _sessions: Dict[str, dict] = {}

    @classmethod
    def save(cls, session: ConversationSession) -> None:
        """Save a session to storage."""
        cls._sessions[session.conversation_id] = session.to_dict()

    @classmethod
    def load(cls, session_id: str) -> Optional[ConversationSession]:
        """Load a session from storage."""
        data = cls._sessions.get(session_id)
        if data:
            return ConversationSession.from_dict(data)
        return None

    @classmethod
    def delete(cls, session_id: str) -> bool:
        """Delete a session from storage."""
        if session_id in cls._sessions:
            del cls._sessions[session_id]
            return True
        return False

    @classmethod
    def exists(cls, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in cls._sessions

    @classmethod
    def cleanup_expired(cls, max_age_seconds: int = 3600) -> int:
        """Remove sessions older than the max age."""
        now = time.time()
        expired = []
        for sid, data in cls._sessions.items():
            last_activity = data.get("last_activity", 0.0)
            if now - last_activity > max_age_seconds:
                expired.append(sid)
        
        for sid in expired:
            del cls._sessions[sid]
            
        return len(expired)
