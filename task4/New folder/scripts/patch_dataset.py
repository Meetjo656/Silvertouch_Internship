import os
import json
import glob
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation.query_complexity_classifier import QueryComplexityClassifier

def fix_dataset(filepath):
    print(f"Fixing {filepath}...")
    fixed_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            
            # 1. Fix Markdown SQL Target
            assistant_msg = data['messages'][2]['content']
            if assistant_msg.startswith('```sql\n'):
                assistant_msg = assistant_msg.replace('```sql\n', '', 1)
            if assistant_msg.endswith('\n```'):
                assistant_msg = assistant_msg[:-4]
            data['messages'][2]['content'] = assistant_msg
            
            # 1.5 Fix System Prompt
            sys_msg = data['messages'][0]['content']
            sys_msg = sys_msg.replace(' Respond with ONLY the raw SQL wrapped in ```sql\n...\n```.', ' Respond with ONLY the raw SQL.')
            data['messages'][0]['content'] = sys_msg
            
            # 2. Fix Missing Relevant Columns
            user_msg = data['messages'][1]['content']
            if 'No tables provided.' in user_msg and 'Relevant Columns: None' not in user_msg:
                user_msg = user_msg.replace('No tables provided.', 'No tables provided.\nRelevant Columns: None')
                data['messages'][1]['content'] = user_msg
                
            # 3. Fix UNKNOWN Complexity & SQL
            if data['metadata'].get('complexity') == 'UNKNOWN':
                sql = assistant_msg
                
                # Fix specific bad SQLs
                if sql.startswith('Gujarat and Kerala for National Seed Federation.,"'):
                    sql = sql.replace('Gujarat and Kerala for National Seed Federation.,"', '')
                    if sql.endswith('"'):
                        sql = sql[:-1]
                elif sql.startswith('Jharkhand, and West Bengal,'):
                    sql = sql.replace('Jharkhand, and West Bengal,', '')
                elif sql.startswith('SELECT * FROM ( /* 1 UCB'):
                    sql = sql.replace('/* 1 UCB ', '')
                    if sql.endswith('*/'):
                        sql = sql[:-2].strip()
                elif sql.startswith('SELECT * FROM ( /* 1 Rural'):
                    sql = sql.replace('/* 1 Rural (SCB + DCB + SCARDB + PCARDB) ', '')
                    if sql.endswith('*/'):
                        sql = sql[:-2].strip()
                        
                data['messages'][2]['content'] = sql
                new_complexity = QueryComplexityClassifier.classify(sql)
                
                if new_complexity == 'UNKNOWN':
                    print(f"Still UNKNOWN at {filepath}:{idx}: {sql}")
                    new_complexity = "SIMPLE_SELECT"
                    
                data['metadata']['complexity'] = new_complexity
            
            fixed_lines.append(json.dumps(data))
            
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + '\n')

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'training'))
    for file in glob.glob(os.path.join(data_dir, '*.jsonl')):
        fix_dataset(file)

if __name__ == "__main__":
    main()
