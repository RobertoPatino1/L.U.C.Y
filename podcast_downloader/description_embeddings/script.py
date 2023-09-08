import json
import re

with open('./Podcast-Downloader/description_embeddings/psi-mammoliti.json', 'r') as f:
    dicty = json.load(f)

values = dicty.values()

def simplify_title(title):
    file_name = re.sub(r'[%/&!@#\*\$\?\+\^\\.\\\\]', '', title)[:100].replace(' ', '-')
    return file_name

new_dicty = {'description_embeddings':[{'title':d['title'], 'description':'', 'embedding': d['embedding']} for d in values]}

with open(f'./Podcast-Downloader/description_embeddings/{simplify_title("Psicología al desnudo")}.json', 'w') as f:
    json.dump(new_dicty, f)