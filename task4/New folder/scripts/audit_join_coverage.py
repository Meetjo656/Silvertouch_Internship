import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.schema.schema_registry import SchemaRegistry
from src.semantic.join_path_resolver import JoinPathResolver

def audit_join_coverage(data_dir: str):
    registry = SchemaRegistry(data_dir)
    resolver = JoinPathResolver(data_dir, registry)
    
    tables = list(registry.tables.keys())
    
    with_paths = set()
    no_paths = set()
    total_edges = 0
    
    # Try to resolve every pair of tables
    # For reporting purposes, we can just do a quadratic sweep
    for i in range(len(tables)):
        has_edge = False
        for j in range(len(tables)):
            if i != j:
                path = resolver.resolve_join_path([tables[i], tables[j]])
                if path:
                    has_edge = True
                    total_edges += 1
        if has_edge:
            with_paths.add(tables[i])
        else:
            no_paths.add(tables[i])
            
    # Each edge is counted twice in the loops above, divide by 2
    actual_edges = total_edges / 2
    
    report = {
        "Total Tables": len(tables),
        "Tables With Join Paths": len(with_paths),
        "Tables Without Join Paths": len(no_paths),
        "Total Unique Edges Resolvable": actual_edges,
        "Average Join Degree": actual_edges / len(tables) if len(tables) > 0 else 0
    }
    
    out_path = os.path.join(data_dir, "..", "outputs", "join_coverage_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    print("Join Coverage Audit Complete:")
    print(json.dumps(report, indent=2))
    print(f"Saved to {out_path}")
    
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    audit_join_coverage(os.path.join(base_dir, "data"))
