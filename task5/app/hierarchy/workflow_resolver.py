import logging
from typing import Optional

logger = logging.getLogger("mybot-nlu")

class WorkflowResolver:
    """
    Resolves the workflow context for the current request.
    This could use the explicitly provided workflow from the request, 
    or the prediction from the global Workflow Classifier.
    """
    def __init__(self):
        pass
        
    def resolve(self, request_workflow_id: Optional[str], predicted_workflow: Optional[str] = None) -> str:
        """
        Finalizes the workflow ID.
        Favors an explicitly passed workflow_id from the request.
        Falls back to the predicted workflow.
        """
        # If the request explicitly provides a workflow (and it's not the default), use it.
        if request_workflow_id and request_workflow_id != "welcome-intent":
            return request_workflow_id
            
        # Otherwise use the predicted workflow
        if predicted_workflow:
            return predicted_workflow
            
        # Fallback if everything is missing
        logger.warning("No workflow provided or predicted. Falling back to default.")
        return "welcome-intent"

workflow_resolver = WorkflowResolver()
