import os
import json
import re
from typing import Dict, Any, List

class SchemaRegistry:
    """
    Authoritative source of truth for Schema structures.
    Supports Authoritative DDL (priority 1) and Inferred Schema (priority 2).
    """
    def __init__(self, data_dir: str):
        self.tables = {}
        self.metadata = {}
        self._load_authoritative(data_dir)
        self._load_inferred(data_dir)

    def _load_authoritative(self, data_dir: str):
        private_data_dir = os.path.join(data_dir, "raw", "Private Data")
        if not os.path.exists(private_data_dir):
            return
            
        for root, dirs, files in os.walk(private_data_dir):
            for file in files:
                if "ddl" in file.lower() and file.endswith(".json"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            for item in data:
                                payload = item.get("payload", {})
                                ddl = payload.get("ddl", "")
                                
                                match = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s\(]+)", ddl, re.IGNORECASE)
                                if match:
                                    t_name = match.group(1).replace('"', '').replace('`', '').lower()
                                    
                                    ddl_no_comments = re.sub(r'/\*.*?\*/', '', ddl, flags=re.DOTALL)
                                    inner_match = re.search(r'\((.*)\)', ddl_no_comments, flags=re.DOTALL)
                                    cols = []
                                    if inner_match:
                                        for p in inner_match.group(1).split(','):
                                            p_clean = p.strip()
                                            if p_clean and not p_clean.lower().startswith(('primary key', 'foreign key', 'constraint', 'unique')):
                                                cols.append(p_clean.split()[0].replace('"', '').replace('`', '').lower())
                                    
                                    self.tables[t_name] = {
                                        "ddl": ddl,
                                        "columns": cols,
                                        "source": "AUTHORITATIVE"
                                    }
                                    self.metadata[t_name] = payload
                    except Exception as e:
                        print(f"Error loading {path}: {e}")

    def _load_inferred(self, data_dir: str):
        inferred_path = os.path.join(data_dir, "..", "outputs", "inferred_schema_catalog.json")
        if os.path.exists(inferred_path):
            try:
                with open(inferred_path, 'r', encoding='utf-8') as f:
                    inferred_data = json.load(f)
                    for item in inferred_data:
                        if item.get("schema_status") == "INFERRED":
                            t_name = item["table"].lower()
                            # Priority: Authoritative > Inferred
                            if t_name not in self.tables:
                                cols = item.get("inferred_columns", [])
                                # STRICT: No synthetic DDL. Only metadata string.
                                metadata_str = f"Table: {t_name}\nColumns: {', '.join(cols)}\nSource: INFERRED"
                                self.tables[t_name] = {
                                    "ddl": metadata_str,
                                    "columns": cols,
                                    "source": "INFERRED"
                                }
            except Exception as e:
                print(f"Error loading inferred schema: {e}")

    def table_exists(self, table_name: str) -> bool:
        return table_name.lower() in self.tables

    def column_exists(self, table_name: str, column_name: str) -> bool:
        t_name = table_name.lower()
        c_name = column_name.lower()
        if t_name in self.tables:
            return c_name in self.tables[t_name]["columns"]
        return False

    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        return self.metadata.get(table_name.lower(), {})
        
    def get_missing_tables(self, required_tables: List[str]) -> List[str]:
        return [t for t in required_tables if not self.table_exists(t)]
        
    def get_coverage_stats(self) -> Dict[str, int]:
        authoritative = sum(1 for t in self.tables.values() if t["source"] == "AUTHORITATIVE")
        inferred = sum(1 for t in self.tables.values() if t["source"] == "INFERRED")
        return {
            "total": len(self.tables),
            "authoritative": authoritative,
            "inferred": inferred
        }
