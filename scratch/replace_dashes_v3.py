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
    
    # Replace literal em-dash
    content = content.replace('—', ' - ')
    
    # Replace Typst em-dash (---)
    content = content.replace(' --- ', ' - ')
    content = content.replace('---', ' - ')
    
    # Replace Typst en-dash (--) only if it has spaces around it (parenthetical use)
    content = content.replace(' -- ', ' - ')
    
    # Clean up multiple spaces that might have been introduced
    content = re.sub(r' +', ' ', content)
    # Fix ' - ' with multiple spaces
    content = content.replace(' - ', ' - ') # already handled by re.sub
    
    # Special case: if we have ' - - ', fix it
    content = content.replace(' - - ', ' - ')

    if content != original_content:
        print(f"Updating {path}...")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
