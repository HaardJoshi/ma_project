import os
import re

directory = 'report/chapters'
main_path = 'report/main.typ'
bib_path = 'report/works.bib'

total_words = 0
citation_tags = set()
total_citations_count = 0

def count_words(text):
    # Remove Typst comments
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove Typst commands like #figure, #set, etc.
    text = re.sub(r'#\w+\(.*?\)', '', text, flags=re.DOTALL)
    text = re.sub(r'#\w+', '', text)
    # Remove markers like <ch-intro>
    text = re.sub(r'<\w+.*?>', '', text)
    # Count words
    words = re.findall(r'\w+', text)
    return len(words)

# Process Chapters
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.typ')]
if os.path.exists(main_path):
    files.append(main_path)

stats = {}

for path in sorted(files):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    word_count = count_words(content)
    total_words += word_count
    
    # Find citations
    cites = re.findall(r'@([\w:-]+)', content)
    unique_cites = set(cites)
    citation_tags.update(unique_cites)
    total_citations_count += len(cites)
    
    stats[os.path.basename(path)] = {
        'words': word_count,
        'citations': len(cites),
        'unique_citations': len(unique_cites)
    }

# Process Bibliography
ref_count = 0
if os.path.exists(bib_path):
    with open(bib_path, 'r', encoding='utf-8') as f:
        bib_content = f.read()
    # Count @type{key, entries
    refs = re.findall(r'@\w+\{([\w:-]+),', bib_content)
    ref_count = len(refs)

print(f"Total Words: {total_words}")
print(f"Total Reference Entries in Bib: {ref_count}")
print(f"Unique Citations Used: {len(citation_tags)}")
print(f"Total Citation Occurrences: {total_citations_count}")
print("\nBreakdown by file:")
for f, s in stats.items():
    print(f"  {f}: {s['words']} words, {s['citations']} citations")
