import os
import json
import sqlglot
import sqlglot.expressions as exp
from collections import defaultdict

def mine_filters(data_dir: str, out_path: str):
    sql_dir = os.path.join(data_dir, "raw", "Private Data")
    filter_patterns = defaultdict(lambda: defaultdict(int))
    
    if not os.path.exists(sql_dir):
        print(f"SQL Data dir not found: {sql_dir}")
        return
        
    print(f"Mining filters from {sql_dir}...")
    
    for root, dirs, files in os.walk(sql_dir):
        for file in files:
            if file.endswith('.json'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            payload = item.get("payload", {})
                            sql = payload.get("sql", "")
                            if not sql: continue
                            
                            try:
                                asts = sqlglot.parse(sql, read="postgres")
                                if not asts: continue
                                ast = asts[0]
                                
                                for eq in ast.find_all(exp.EQ):
                                    left = eq.left
                                    right = eq.right
                                    
                                    if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                                        col_name = left.name.lower()
                                        val = right.this
                                        # Focus on state/enum columns
                                        if any(k in col_name for k in ['status', 'is_', 'has_', 'state', 'type', 'approval']):
                                            filter_patterns[col_name][val] += 1
                            except Exception:
                                pass 
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    
    output = {}
    for col, vals in filter_patterns.items():
        if len(vals) > 0:
            sorted_vals = sorted(vals.items(), key=lambda x: x[1], reverse=True)
            output[col] = [f"{col} = '{v[0]}' ({v[1]} usages)" for v in sorted_vals]
            
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
        
    print(f"Mined {len(output)} enum columns. Saved to {out_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    mine_filters(os.path.join(base_dir, "data"), os.path.join(base_dir, "data", "processed", "filter_pattern_library.json"))
