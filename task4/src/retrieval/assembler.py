import os
import sys

# Add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.routing.extractor import EntityExtractor
from src.routing.semantic_router import SemanticModuleRouter
from src.retrieval.doc_retriever import DocRetriever
from src.retrieval.ddl_retriever import DDLRetriever
from src.retrieval.sql_retriever import SQLRetriever

class ContextAssembler:
    def __init__(self, raw_data_dir, catalog_path):
        self.extractor = EntityExtractor(catalog_path)
        embeddings_dir = os.path.join(os.path.dirname(catalog_path), '../embeddings')
        self.router = SemanticModuleRouter(embeddings_dir)
        self.doc_retriever = DocRetriever(raw_data_dir)
        self.ddl_retriever = DDLRetriever(raw_data_dir)
        self.sql_retriever = SQLRetriever(raw_data_dir)
        
    def assemble(self, query):
        """
        Orchestrates the entire retrieval pipeline to output the Context Package.
        """
        # 1. Entity Extraction
        entities, raw_entities = self.extractor.extract(query)
        
        # 2. Module Prediction
        predicted_modules = self.router.predict(query)
        
        # 3. Document Retrieval
        relevant_docs = self.doc_retriever.retrieve(predicted_modules, raw_entities)
        
        # 4. DDL Retrieval
        relevant_tables = self.ddl_retriever.retrieve(predicted_modules, raw_entities)
        
        # 5. SQL Example Retrieval
        similar_sql = self.sql_retriever.retrieve(query, predicted_modules)
        
        # Final Package
        context_package = {
            "question": query,
            "entities": entities,
            "predicted_modules": predicted_modules,
            "relevant_docs": relevant_docs,
            "relevant_tables": relevant_tables,
            "similar_sql_examples": similar_sql
        }
        
        return context_package

if __name__ == "__main__":
    # Test execution for Phase 1 Deliverable
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    catalog_path = os.path.join(project_root, 'data', 'semantic_catalog', 'semantic_catalog.json')
    
    assembler = ContextAssembler(raw_data_dir, catalog_path)
    pkg = assembler.assemble("Current rankings for Dairy societies in Anand")
    
    import json
    print(json.dumps(pkg, indent=2))
