import os
import re
from src.utils.file_ops import load_json

class EntityExtractor:
    def __init__(self, catalog_path):
        self.catalog = load_json(catalog_path)
        
    def extract(self, query):
        """
        Extracts entities from the query using the semantic catalog.
        Returns a dictionary of matched entities and their catalog entries.
        """
        query_lower = query.lower()
        extracted_entities = {
            "locations": [],
            "sectors": [],
            "institution_types": [],
            "business_concepts": []
        }
        
        extracted_raw = []

        for entry in self.catalog:
            # Check concept name
            name = entry['concept']
            aliases = entry.get('aliases', []) + entry.get('abbreviations', [])
            
            # Build matching patterns
            patterns = [r'\b' + re.escape(name.lower()) + r'\b']
            for alias in aliases:
                patterns.append(r'\b' + re.escape(alias.lower()) + r'\b')
                
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    extracted_raw.append(entry)
                    
                    # Group by type
                    ctype = entry.get('concept_type', '')
                    if ctype in ['state', 'district', 'geography']:
                        extracted_entities['locations'].append(name)
                    elif ctype == 'sector':
                        extracted_entities['sectors'].append(name)
                    elif ctype == 'institution_type':
                        extracted_entities['institution_types'].append(name)
                    else:
                        extracted_entities['business_concepts'].append(name)
                    break # Stop checking patterns if one matches
                    
        return extracted_entities, extracted_raw
