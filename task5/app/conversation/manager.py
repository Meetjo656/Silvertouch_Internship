from typing import Optional, Any
from app.conversation.session import ConversationSession
from app.conversation.storage import Storage

class ConversationManager:
    """Manages conversational state transitions and orchestrates session data."""

    def __init__(self):
        self.storage = Storage()

    def start_session(self, conversation_id: str, workflow_id: str = "") -> ConversationSession:
        """Initialize a new session and save it."""
        session = ConversationSession(conversation_id=conversation_id, workflow_id=workflow_id)
        self.storage.save(session)
        return session

    def get_session(self, conversation_id: str) -> Optional[ConversationSession]:
        """Retrieve an existing session."""
        return self.storage.load(conversation_id)

    def update_session(self, session: ConversationSession) -> None:
        """Persist session updates."""
        self.storage.save(session)

    def close_session(self, conversation_id: str) -> None:
        """Mark a session as closed."""
        session = self.get_session(conversation_id)
        if session:
            session.conversation_state = "closed"
            self.update_session(session)

    def pause_session(self, conversation_id: str) -> None:
        """Pause an active session."""
        session = self.get_session(conversation_id)
        if session:
            session.conversation_state = "paused"
            self.update_session(session)

    def resume_session(self, conversation_id: str) -> None:
        """Resume a paused session."""
        session = self.get_session(conversation_id)
        if session:
            session.conversation_state = "active"
            self.update_session(session)

    def store_entity(self, conversation_id: str, entity_name: str, entity_value: Any) -> bool:
        """Store an entity in the session."""
        session = self.get_session(conversation_id)
        if session:
            session.add_entity(entity_name, entity_value)
            self.update_session(session)
            return True
        return False

    def get_missing_entity(self, conversation_id: str) -> Optional[str]:
        """Get the name of the next required entity missing from the session."""
        session = self.get_session(conversation_id)
        if session and not session.is_complete():
            for req_ent in session.required_entities:
                if req_ent not in session.collected_entities:
                    return req_ent
        return None

    def mark_complete(self, conversation_id: str) -> None:
        """Mark a session as fully completed."""
        session = self.get_session(conversation_id)
        if session:
            session.conversation_state = "completed"
            self.update_session(session)