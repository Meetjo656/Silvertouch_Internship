from typing import Dict, Any
from app.conversation.session import ConversationSession

class PendingActionRecovery:
    """
    Handles the recovery workflow when a session is waiting for a specific entity.
    """
    
    @staticmethod
    def recover(session: ConversationSession, user_input: str) -> Dict[str, Any]:
        """
        Skips intent detection and instructs the system to extract the 
        missing entity directly from the user's input, then continue the workflow.
        """
        entity_name = session.waiting_for
        
        return {
            "action": "extract_entity",
            "entity": entity_name,
            "input": user_input,
            "next_step": "continue_workflow"
        }
