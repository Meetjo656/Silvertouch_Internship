import os
import json

class DDLRetriever:
    def __init__(self, raw_data_dir):
        # Load all ddl
        legacy_dir = os.path.join(raw_data_dir, 'Private Data', 'qdrant_export_20260615_062842', 'LEGACY')
        ddl_path = os.path.join(legacy_dir, 'ddl.json')
        with open(ddl_path, 'r', encoding='utf-8') as f:
            self.ddls = json.load(f)
            
    def retrieve(self, predicted_modules, extracted_raw_entities):
        """
        Retrieves DDL schemas. Always includes the geographic master if location entities are present.
        """
        target_modules = [m['module'] for m in predicted_modules]
        
        # Determine if we need the geographic master
        needs_geo = False
        for e in extracted_raw_entities:
            if e.get('concept_type') in ['state', 'district', 'geography', 'administrative_unit']:
                needs_geo = True
                
        relevant_tables = set()
        
        for item in self.ddls:
            payload = item['payload']
            mod = payload.get('module')
            # The table definitions in LEGACY might be missing 'table_name', so we fall back to tables array
            tables = payload.get('tables', [])
            
            if mod in target_modules:
                relevant_tables.update(tables)
                
        if needs_geo:
            relevant_tables.add("chatbot_districts_blocks_gp_villages")
            
        return list(relevant_tables)
