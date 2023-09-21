import re
import unicodedata
import os
import json
import sys
sys.path.append('./')


# Parse methods
def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

# OS methods
def get_root_dir():
    return './podcast_downloader'

def get_embeddings_dir():
    return get_dir('Embedding_store', get_root_dir())

def get_desc_emb_dir():
    return get_dir('description_embeddings', get_embeddings_dir())

def get_par_emb_dir():
    return get_dir('paragraph_embeddings', get_embeddings_dir())

def get_desc_emb_meta_path():
    return get_dir('metadata.json', get_desc_emb_dir(), 'episode_records')

def get_dir(file_name, dir, key=None):   
    if not file_name in os.listdir(dir):
        if file_name.endswith('.json'):
            with open(f'{get_desc_emb_dir()}/metadata.json', 'w') as f:
                json.dump({key:[]}, f)
        else:
            os.mkdir(f"{dir}/{file_name}")     
    return f"{dir}/{file_name}"

# Array methods
def flatten(l):
    return [item for sublist in l for item in sublist]


