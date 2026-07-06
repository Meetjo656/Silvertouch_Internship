from typing import Dict, Any, Optional
from app.conversation.session import ConversationSession
from app.conversation.pending_action import PendingActionRecovery

class ContextResolver:
    """
    Resolves the context of an incoming message based on session state.
    """
    
    @staticmethod
    def resolve_context(session: Optional[ConversationSession], user_input: str) -> Dict[str, Any]:
        """
        Evaluates the incoming message against the current session.
        
        Flow:
        waiting_for_choice? -> waiting_for_entity? -> waiting_for_confirmation? -> normal classification
        """
        if session is not None and session.conversation_state == "active":
            
            # 1. Waiting for a choice? (e.g., "Second one")
            if session.waiting_for_choice:
                return {
                    "action": "handle_choice",
                    "input": user_input
                }
                
            # 2. Waiting for an entity? (e.g., "IFP-12345")
            if session.is_waiting():
                # Skip Intent Detection & execute pending action
                return PendingActionRecovery.recover(session, user_input)
                
            # 3. Waiting for confirmation? (e.g., "Yes", "No", "Correction")
            if session.waiting_for_confirmation:
                return {
                    "action": "handle_confirmation",
                    "input": user_input
                }
                
        # 4. Fallback to normal intent classification
        return {
            "action": "classify_intent",
            "input": user_input
        }
