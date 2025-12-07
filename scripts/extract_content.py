#!/usr/bin/env python3
"""
Automatic extraction script: Extract the assistant's content from v1_synthesized_test_output.jsonl
and save it as results/v1_test_235B.json in JSON array format.
"""

import json
import os

def extract_content(input_file, output_file):
    contents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                content = ""
                if 'messages' in data:
                    for msg in data['messages']:
                        if msg.get('role') == 'assistant':
                            content = msg.get('content', '')
                            break
                else:
                    content = data.get('content', '')
                contents.append(content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(contents, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Define file paths
    input_file = "dataset/data/processed/sft/v1_synthesized_test_output.jsonl"
    output_file = "results/v1_test_235B.json"
    
    # Ensure input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        exit(1)
    
    # Extract content
    extract_content(input_file, output_file)
    print(f"Successfully extracted content to {output_file}.")