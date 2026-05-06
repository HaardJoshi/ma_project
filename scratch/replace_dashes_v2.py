import os
import re

directory = 'report/chapters'
files_to_process = [os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files if file.endswith('.typ')]
main_path = 'report/main.typ'
if os.path.exists(main_path):
    files_to_process.append(main_path)

for path in files_to_process:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Replace literal em-dash
    content = content.replace('—', ' - ')
    
    # 2. Replace Typst em-dash (---) with spaces or at start/end of breaks
    # But avoid replacing --- in other contexts if any
    content = content.replace(' --- ', ' - ')
    content = re.sub(r'(\w)---(\w)', r'\1 - \2', content)
    content = re.sub(r'(\w)--- ', r'\1 - ', content)
    content = re.sub(r' ---(\w)', r' - \1', content)

    # 3. Handle en-dashes used as parentheticals ( -- with spaces)
    # But keep them for ranges (no spaces, e.g. 2000--2023)
    content = content.replace(' -- ', ' - ')
    
    # Clean up double spaces if any were introduced
    content = content.replace('  -  ', ' - ')
    content = content.replace(' -  ', ' - ')
    content = content.replace('  - ', ' - ')

    if content != original_content:
        print(f"Updating {path}...")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
