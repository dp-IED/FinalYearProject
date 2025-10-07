import csv
import json
import re

input_file = 'OBD Reading.csv'
output_file = 'OBD Reading Clean.csv'

def fix_json_string(s):
    # Remove leading/trailing whitespace and fix double double-quotes
    s = s.strip()
    s = s.replace('""', '"')
    # Remove any leading/trailing quotes
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s

def parse_row(row):
    s = fix_json_string(row)
    # Try to parse as JSON
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        # Try to fix common issues
        s = re.sub(r'\"', '"', s)
        obj = json.loads(s)
    # Flatten the structure
    flat = {}
    for k, v in obj.items():
        if isinstance(v, dict) and 'S' in v:
            flat[k] = v['S']
        else:
            flat[k] = v
    return flat

def main():
    rows = []
    all_keys = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            if not line:
                continue
            flat = parse_row(line[0])
            rows.append(flat)
            all_keys.update(flat.keys())
    all_keys = sorted(all_keys)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in all_keys})

if __name__ == '__main__':
    main()
