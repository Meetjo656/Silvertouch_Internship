from typing import Dict, List, Any, Optional
# pyrefly: ignore [missing-import]
from .panel import Panel

class PanelGraph:
    def __init__(self):
        self.nodes: Dict[str, Panel] = {}
        self.edges: List[Dict[str, Any]] = []

    def add_node(self, panel: Panel):
        self.nodes[panel.id] = panel
        
    def add_edge(self, source_id: str, target_id: str, relation: str, properties: Optional[Dict] = None):
        self.edges.append({
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "properties": properties or {}
        })
        if target_id not in self.nodes[source_id].neighbors:
            self.nodes[source_id].neighbors.append(target_id)
        if source_id not in self.nodes[target_id].neighbors:
            self.nodes[target_id].neighbors.append(source_id)

    def get_panels_by_class(self, class_name: str) -> List[Panel]:
        return [p for p in self.nodes.values() if p.class_name == class_name]

    def remove_node(self, panel_id: str):
        if panel_id in self.nodes:
            del self.nodes[panel_id]
        
        self.edges = [e for e in self.edges if e["source"] != panel_id and e["target"] != panel_id]
        
        for p in self.nodes.values():
            if panel_id in p.neighbors:
                p.neighbors.remove(panel_id)

    def get_neighbors(self, panel_id: str) -> List[Panel]:
        if panel_id not in self.nodes:
            return []
        return [self.nodes[nid] for nid in self.nodes[panel_id].neighbors if nid in self.nodes]
