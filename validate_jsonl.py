import json

with open('test_latest.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"ERROR on line {i}: {e}")
            print(f"Line content: {line[:200]}...")
            break
    else:
        print(f"✅ All {i} lines are valid JSON")