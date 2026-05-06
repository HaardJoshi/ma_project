import os
import re

chapters_dir = 'report/chapters'
main_file = 'report/main.typ'
bib_file = 'report/works.bib'

# 1. Extract all citation keys from the .typ files
cited_keys = set()
typ_files = [os.path.join(chapters_dir, f) for f in os.listdir(chapters_dir) if f.endswith('.typ')]
if os.path.exists(main_file):
    typ_files.append(main_file)

for fpath in typ_files:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
        # Find all occurrences of @key (ignoring characters that aren't part of a bib key)
        keys = re.findall(r'@([\w:-]+)', content)
        for k in keys:
            # Exclude Typst internal labels like <ch-intro> which are sometimes misidentified
            # In Typst, labels are <label>, citations are @cite.
            # But let's check if there are any obvious labels being caught.
            # Typst labels can be @label if they are references.
            cited_keys.add(k)

# 2. Extract all defined keys from the bib file
defined_keys = set()
if os.path.exists(bib_file):
    with open(bib_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # Find @type{key,
        keys = re.findall(r'@\w+\{([\w:-]+),', content)
        for k in keys:
            defined_keys.add(k)

# 3. Cross-reference
missing_in_bib = cited_keys - defined_keys
unused_in_chapters = defined_keys - cited_keys

# Filter out internal Typst labels from missing_in_bib
# Typst labels are used for chapters/sections, e.g., @ch-introduction.
# I should identify which ones start with ch-, sec-, fig-, tbl-
internal_prefixes = ['ch-', 'sec-', 'fig-', 'tbl-', 'eq-']
missing_refs = [k for k in missing_in_bib if not any(k.startswith(p) for p in internal_prefixes)]

print("--- Reference Check ---")
if not missing_refs:
    print("SUCCESS: All academic citations have matching bibliography entries.")
else:
    print(f"WARNING: Found {len(missing_refs)} citations missing from works.bib:")
    for k in missing_refs:
        print(f"  - @{k}")

print("\n--- Typst Internal Links ---")
internal_links = [k for k in cited_keys if any(k.startswith(p) for p in internal_prefixes)]
print(f"Found {len(internal_links)} internal cross-references (chapters, figures, tables).")

print("\n--- Bibliography Coverage ---")
print(f"Total references defined in bib: {len(defined_keys)}")
print(f"References used in text: {len(defined_keys - unused_in_chapters)}")
if unused_in_chapters:
    print(f"References defined but not used: {len(unused_in_chapters)}")
    # Optional: print them? Maybe too many.
