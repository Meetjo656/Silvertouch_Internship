from typing import List, Dict, Any, Optional

class KnowledgeGraph:
    """
    In-memory knowledge graph for conversational workflows, intents, subintents, and entities.
    """
    
    def __init__(
        self, 
        training_json: List[Dict[str, Any]], 
        build_graph: bool = True, 
        build_workflow_map: bool = True, 
        build_intent_map: bool = True, 
        build_subintent_map: bool = True, 
        build_entity_map: bool = True
    ):
        # Store configuration
        self.training_json = training_json
        
        # Internal dictionaries for O(1) lookups
        self.workflow_map: Dict[str, Any] = {}
        self.intent_map: Dict[str, Any] = {}
        self.subintent_map: Dict[str, Any] = {}
        self.entity_map: Dict[str, List[str]] = {}
        
        # Optionally build the graph immediately
        if build_graph:
            self.build_graph(
                build_workflow_map, 
                build_intent_map, 
                build_subintent_map, 
                build_entity_map
            )

    def build_graph(
        self, 
        build_workflow_map: bool = True, 
        build_intent_map: bool = True, 
        build_subintent_map: bool = True, 
        build_entity_map: bool = True
    ):
        """Iterates through the training JSON to construct fast lookup maps."""
        self.workflow_map.clear()
        self.intent_map.clear()
        self.subintent_map.clear()
        self.entity_map.clear()

        # The JSON could be a single object or a list of objects.
        # Normalize to list to iterate safely.
        workflows = self.training_json if isinstance(self.training_json, list) else [self.training_json]

        for workflow in workflows:
            wf_id = workflow.get('workflowId')
            if not wf_id:
                continue

            # 1. Build Workflow Map
            if build_workflow_map:
                self.workflow_map[wf_id] = workflow
                
            # Note: In our schema, workflows and intents are often synonymous or
            # workflows contain subIntents directly. 
            if build_intent_map:
                self.intent_map[wf_id] = workflow

            # Iterate over SubIntents
            for subintent in workflow.get('subIntents', []):
                sub_name = subintent.get('name')
                if not sub_name:
                    continue
                
                # 2. Build SubIntent Map
                if build_subintent_map:
                    self.subintent_map[sub_name] = subintent
                    # Attach parent workflow reference
                    self.subintent_map[sub_name]['_parentWorkflowId'] = wf_id
                
                # 3. Build Entity Map
                if build_entity_map:
                    for entity in subintent.get('requiredEntities', []):
                        if entity not in self.entity_map:
                            self.entity_map[entity] = []
                        self.entity_map[entity].append(sub_name)

    def get_workflows(self) -> List[str]:
        """Returns a list of all workflow IDs."""
        return list(self.workflow_map.keys())

    def get_intents(self) -> List[str]:
        """Returns a list of all main intent IDs (which correspond to workflows)."""
        return list(self.intent_map.keys())

    def get_subintents(self, workflow_id: Optional[str] = None) -> List[str]:
        """Returns all subintents, optionally filtered by a specific workflow."""
        if workflow_id:
            workflow = self.workflow_map.get(workflow_id)
            if workflow:
                return [sub.get('name') for sub in workflow.get('subIntents', []) if sub.get('name')]
            return []
        
        return list(self.subintent_map.keys())

    def get_required_entities(self, subintent_name: str) -> List[str]:
        """Gets the list of required entities for a given subintent."""
        subintent = self.subintent_map.get(subintent_name)
        if subintent:
            return subintent.get('requiredEntities', [])
        return []

    def workflow_exists(self, workflow_id: str) -> bool:
        """Checks if a workflow ID exists in the graph."""
        return workflow_id in self.workflow_map

    def intent_exists(self, intent_id: str) -> bool:
        """Checks if an intent ID exists in the graph."""
        return intent_id in self.intent_map

    def subintent_exists(self, subintent_name: str) -> bool:
        """Checks if a subintent name exists in the graph."""
        return subintent_name in self.subintent_map

    def export_graph(self) -> Dict[str, Any]:
        """Exports the internal graph maps as a dictionary."""
        return {
            "workflows": self.workflow_map,
            "intents": self.intent_map,
            "subintents": self.subintent_map,
            "entities": self.entity_map
        }

    def reload(self, training_json: List[Dict[str, Any]]):
        """Reloads the graph with new training data."""
        self.training_json = training_json
        self.build_graph()
