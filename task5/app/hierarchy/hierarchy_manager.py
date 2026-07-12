import logging
from typing import Dict, Any, List, Optional

from app.hierarchy.workflow_resolver import workflow_resolver
from app.hierarchy.candidate_filter import candidate_filter
from app.hierarchy.intent_ranker import intent_ranker
from app.hierarchy.metadata_provider import metadata_provider

logger = logging.getLogger("mybot-nlu")

class HierarchyManager:
    """
    Orchestrates the decision process for a user utterance.
    Consumes raw prediction results from the inference layer and processes them
    through filtering, ranking, and validation.
    """
    def __init__(self):
        pass

    def resolve(
        self, 
        request_workflow_id: Optional[str], 
        predicted_workflow: Optional[str],
        global_intent_predictions: List[Dict[str, Any]],
        threshold: float = 0.30
    ) -> Dict[str, Any]:
        """
        Main orchestration method.
        
        Args:
            request_workflow_id: Workflow provided in the API request (if any)
            predicted_workflow: Top workflow prediction from WorkflowClassifier (if any)
            global_intent_predictions: Top K predictions from the Global Intent Classifier
            threshold: Confidence threshold for intents
            
        Returns:
            A dictionary containing the resolved workflow, intent/subintent, and required entities.
        """
        # Step 1: Resolve Workflow
        final_workflow = self.resolve_workflow(request_workflow_id, predicted_workflow)
        
        # Step 2: Filter Intent Candidates
        filtered_predictions = candidate_filter.filter_predictions(final_workflow, global_intent_predictions)
        
        # Step 3: Rank and Resolve Intent/SubIntent
        resolved_intent = self.resolve_intent(filtered_predictions, threshold)
        
        # Step 4: Determine Required Entities (subintent resolver step)
        required_entities = []
        if resolved_intent:
            required_entities = metadata_provider.get_required_entities(final_workflow, resolved_intent["name"])
            
        return self.build_response(final_workflow, resolved_intent, required_entities)

    def resolve_workflow(self, request_workflow_id: Optional[str], predicted_workflow: Optional[str]) -> str:
        return workflow_resolver.resolve(request_workflow_id, predicted_workflow)

    def resolve_intent(self, filtered_predictions: List[Dict[str, Any]], threshold: float) -> Optional[Dict[str, Any]]:
        return intent_ranker.rank(filtered_predictions, threshold)

    def build_response(self, workflow: str, resolved_intent: Optional[Dict[str, Any]], required_entities: List[dict]) -> Dict[str, Any]:
        response = {
            "workflow": workflow,
            "intent": resolved_intent,
            "requiredEntities": required_entities,
            "fallback": resolved_intent is None
        }
        return response

hierarchy_manager = HierarchyManager()