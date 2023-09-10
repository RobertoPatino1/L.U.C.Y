import re
import unicodedata
import numpy as np
import openai
import streamlit as st
import pickle
import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_base = st.secrets["API_BASE"]

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
    return get_dir('metadata.json', get_desc_emb_dir(), 'episodes_record')

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

# Embeddings methods
def get_embeddings_transformer():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    return embeddings

def store_embeddings(texts, store_name, path, embeddings = get_embeddings_transformer(), document=False):
    if not document:
        vectorStore = FAISS.from_texts(texts, embeddings)
    else:
        vectorStore = FAISS.from_documents(texts, embeddings)

    with open(f"{path}/faiss_{store_name}.pkl", "wb") as f:
        pickle.dump(vectorStore, f)

def load_embeddings(store_name, path):
    with open(f"{path}/faiss_{store_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
    return VectorStore

