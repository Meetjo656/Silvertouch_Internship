import re
from typing import Dict, Any

class EnumExtractor:
    """
    Extracts business semantics (enums, status mappings) directly from DDL comments and table documentation.
    """
    def extract_from_ddl(self, ddl: str) -> Dict[str, Dict[str, str]]:
        """
        Parses DDL string to find columns with inline semantic mappings.
        Returns mapping: { column_name: { "value": "meaning" } }
        """
        semantics = {}
        # DDL usually has: column_name datatype /* comment */
        inner_match = re.search(r'\((.*)\)', ddl, flags=re.DOTALL)
        if not inner_match:
            return semantics
            
        inner = inner_match.group(1)
        
        # Split by comma but respect comments
        # Find all `column_name type /* comment */` patterns
        pattern = r'([a-zA-Z0-9_]+)\s+[A-Z0-9_]+(?:.*?)?/\*\s*(.*?)\s*\*/'
        matches = re.findall(pattern, inner, flags=re.IGNORECASE)
        
        for col, comment in matches:
            col = col.lower()
            mapping = self._parse_mapping_from_comment(comment)
            if mapping:
                semantics[col] = mapping
                
        return semantics
        
    def extract_from_doc(self, doc: str) -> Dict[str, Dict[str, str]]:
        """
        Extracts mappings from markdown documentation.
        Assumes format like `column_name: 1 = Active, 0 = Inactive`
        """
        semantics = {}
        lines = doc.split('\n')
        for line in lines:
            line = line.strip()
            # Look for lines starting with a potential column name and a colon
            m = re.match(r'^([a-zA-Z0-9_]+)\s*[:\-]\s*(.*)$', line)
            if m:
                col = m.group(1).lower()
                rest = m.group(2)
                mapping = self._parse_mapping_from_comment(rest)
                if mapping:
                    semantics[col] = mapping
        return semantics
        
    def _parse_mapping_from_comment(self, comment: str) -> Dict[str, str]:
        mapping = {}
        # Match "1=Active", "0: Inactive", "1 - Approved" but NOT "dis-trict"
        pairs = re.findall(r'\b([0-9A-Za-z_]+)\s*(?:=|:|\s-\s)\s*([A-Za-z_]+)\b', comment)
        if pairs:
            for k, v in pairs:
                mapping[str(k).lower()] = v.lower()
                
        if not mapping:
            # Look for explicit "X means Y"
            m = re.search(r'\b([0-9A-Za-z_]+)\s+(?:means|is)\s+([a-zA-Z_]+)\b', comment, re.IGNORECASE)
            if m:
                mapping[str(m.group(1)).lower()] = m.group(2).lower()
                
        return mapping
