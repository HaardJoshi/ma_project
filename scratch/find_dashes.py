import os
import re

directory = 'report/chapters'
dash_char = '—'

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.typ'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if dash_char in content:
                print(f"File: {path}")
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if dash_char in line:
                        print(f"  {i+1}: {line.strip()}")
