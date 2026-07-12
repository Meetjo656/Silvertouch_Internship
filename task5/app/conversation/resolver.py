from typing import Dict, Any
from app.conversation.session import ConversationSession

class ResolutionEngine:
    """Decides the next action a chatbot should take in a conversation."""

    @classmethod
    def resolve(cls, session: ConversationSession, user_input: str) -> Dict[str, Any]:
        """Determine the next step in the dialogue."""
        if cls.should_extract_entity(session):
            return {
                "action": "extract_entity", 
                "input": user_input, 
                "entity": session.waiting_for
            }
            
        if cls.should_classify(session):
            return {
                "action": "classify", 
                "input": user_input
            }
            
        if cls.should_ask_entity(session):
            missing = next((k for k in session.required_entities if k not in session.collected_entities), None)
            if missing:
                session.waiting_for = missing
                return {
                    "action": "ask_entity", 
                    "entity": missing
                }
                
        if cls.should_continue(session):
            return {
                "action": "execute_intent"
            }
            
        if cls.should_fallback(session):
            return {
                "action": "fallback"
            }
            
        return {"action": "unknown"}

    @classmethod
    def should_classify(cls, session: ConversationSession) -> bool:
        """Should classify if we don't have an active intent and aren't waiting for an entity."""
        return not session.current_intent and not session.is_waiting()

    @classmethod
    def should_extract_entity(cls, session: ConversationSession) -> bool:
        """Should extract if we are actively waiting for an entity to fill a slot."""
        return session.is_waiting()

    @classmethod
    def should_ask_entity(cls, session: ConversationSession) -> bool:
        """Should ask for an entity if we have an intent but are missing required entities."""
        return bool(session.current_intent) and not session.is_complete()

    @classmethod
    def should_continue(cls, session: ConversationSession) -> bool:
        """Should execute the intent if intent is identified and all entities are collected."""
        return bool(session.current_intent) and session.is_complete()

    @classmethod
    def should_fallback(cls, session: ConversationSession) -> bool:
        """Should fallback if the dialogue state is stuck or undefined."""
        return False
