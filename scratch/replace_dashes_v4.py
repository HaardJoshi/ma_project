import os

directory = 'report/chapters'
files_to_process = [os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files if file.endswith('.typ')]
main_path = 'report/main.typ'
if os.path.exists(main_path):
    files_to_process.append(main_path)

for path in files_to_process:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if ' --- ' in content:
        print(f"Found ' --- ' in {path}")
        content = content.replace(' --- ', ' - ')
    
    if '---' in content:
        print(f"Found '---' in {path}")
        content = content.replace('---', ' - ')

    if ' — ' in content:
        print(f"Found ' — ' in {path}")
        content = content.replace(' — ', ' - ')
        
    if '—' in content:
        print(f"Found '—' in {path}")
        content = content.replace('—', ' - ')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
