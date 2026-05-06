import os

directory = 'report/chapters'
old_dash = '—'
new_dash = '-'

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.typ'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if old_dash in content:
                print(f"Processing {path}...")
                new_content = content.replace(old_dash, new_dash)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

# Also check report/main.typ
main_path = 'report/main.typ'
if os.path.exists(main_path):
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if old_dash in content:
        print(f"Processing {main_path}...")
        new_content = content.replace(old_dash, new_dash)
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
