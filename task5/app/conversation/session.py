import time
from typing import Dict, Any, Optional

class ConversationSession:
    """Represents a single conversation session and its state."""

    def __init__(
        self, 
        conversation_id: str, 
        workflow_id: str = "", 
        current_intent: str = "", 
        current_subintent: str = "", 
        collected_entities: Optional[Dict[str, Any]] = None,
        required_entities: Optional[Dict[str, Any]] = None,
        waiting_for: str = "",
        conversation_state: str = "active",
        last_activity: Optional[float] = None
    ):
        self.conversation_id = conversation_id
        self.workflow_id = workflow_id
        self.current_intent = current_intent
        self.current_subintent = current_subintent
        self.collected_entities = collected_entities or {}
        self.required_entities = required_entities or {}
        self.waiting_for = waiting_for
        self.conversation_state = conversation_state
        self.last_activity = last_activity or time.time()

    def is_waiting(self) -> bool:
        """Check if the session is currently waiting for a specific entity."""
        return bool(self.waiting_for)

    def add_entity(self, entity_name: str, entity_value: Any) -> None:
        """Add a collected entity and update the waiting status."""
        self.collected_entities[entity_name] = entity_value
        if self.waiting_for == entity_name:
            self.waiting_for = ""
        self.last_activity = time.time()

    def is_complete(self) -> bool:
        """Check if all required entities have been collected."""
        if not self.required_entities:
            return True
        return all(entity in self.collected_entities for entity in self.required_entities)

    def clear(self) -> None:
        """Clear the current conversational context (keeps conversation_id)."""
        self.workflow_id = ""
        self.current_intent = ""
        self.current_subintent = ""
        self.collected_entities = {}
        self.required_entities = {}
        self.waiting_for = ""
        self.conversation_state = "active"
        self.last_activity = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the session to a dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "workflow_id": self.workflow_id,
            "current_intent": self.current_intent,
            "current_subintent": self.current_subintent,
            "collected_entities": self.collected_entities,
            "required_entities": self.required_entities,
            "waiting_for": self.waiting_for,
            "conversation_state": self.conversation_state,
            "last_activity": self.last_activity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Create a ConversationSession from a dictionary."""
        return cls(
            conversation_id=data.get("conversation_id", ""),
            workflow_id=data.get("workflow_id", ""),
            current_intent=data.get("current_intent", ""),
            current_subintent=data.get("current_subintent", ""),
            collected_entities=data.get("collected_entities", {}),
            required_entities=data.get("required_entities", {}),
            waiting_for=data.get("waiting_for", ""),
            conversation_state=data.get("conversation_state", "active"),
            last_activity=data.get("last_activity", time.time())
        )



    
