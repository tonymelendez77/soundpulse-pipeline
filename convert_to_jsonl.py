import json
from pathlib import Path

def convert_json_to_jsonl(json_file):
    """Convert JSON array to newline-delimited JSON"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Write as JSONL (one object per line)
    jsonl_file = str(json_file).replace('.json', '.jsonl')
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"✓ Converted {json_file} → {jsonl_file}")
    return jsonl_file

if __name__ == "__main__":
    data_dir = Path('data/raw')
    
    # Find all JSON files in all subdirectories
    json_files = list(data_dir.glob('*/*.json'))
    
    if not json_files:
        print("⚠ No JSON files found in data/raw/ subdirectories")
    else:
        for filepath in json_files:
            convert_json_to_jsonl(filepath)
        print(f"\n✓ Converted {len(json_files)} files")