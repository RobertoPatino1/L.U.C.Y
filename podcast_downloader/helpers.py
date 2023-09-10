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
import sys
sys.path.append('./')


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

# Embeddings methods
def get_embeddings_transformer(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    return embeddings


def update_embeddings(vectorStore, texts, store_name, path, embeddings = get_embeddings_transformer(), host_documents=False):
    # Create a dictionary containing the metadata
    metadata = {
        'store_name': store_name,
        'host_documents': host_documents,
        'embeddings_model_name': embeddings.model_name,
        'texts': texts,
        'faiss_index': vectorStore.serialize_to_bytes()  # Serialize the FAISS index
    }

    with open(f"{path}/faiss_{store_name}.pkl", "wb") as f:
        pickle.dump(metadata, f)

def load_embeddings(store_name:str, path:str, host_documents:bool):
    embeddings_path = f"{path}/faiss_{store_name}.pkl"
    if not os.exists(embeddings_path):
        embeddings_model = get_embeddings_transformer()
        embeddings_model_name = embeddings_model.name
        texts = ['']
        if not host_documents:
            faiss_index = FAISS.from_texts(texts, embeddings_model)
        else:
            faiss_index = FAISS.from_documents(texts, embeddings_model)
            
        update_embeddings(faiss_index, texts, store_name, path, embeddings_model, host_documents)
    else:
        with open(embeddings_path, "rb") as f:
            metadata = pickle.load(f)
        
        texts = metadata['texts']
        embeddings_model_name = metadata['embeddings_model_name']

        # Deserialize the FAISS index
        faiss_index = FAISS.deserialize_from_bytes(metadata['faiss_index'], 
                                                get_embeddings_transformer(embeddings_model_name))
    return {
        'store_name': store_name,
        'path': path,
        'host_documents': host_documents,
        'embeddings_model_name': embeddings_model_name,
        'texts': texts,
        'faiss_index': faiss_index
    }

# Test methods
def test():
    create_embeddings(['hola mundo'], 'test', './')
    db = load_embeddings('test', './')
    db.add_texts(['adios mundo', 'saludos mundo'])
    retriever = db.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents("hola mundo")
    print([doc.page_content for doc in docs])

def test2():
    # store_embeddings(['hola mundo'], 'test', './')
    # db_metadata = load_embeddings('test', './')
    # db = db_metadata['faiss_index']
    # added_texts = ['adios mundo', 'saludos mundo']
    # db.add_texts(added_texts)
    # texts = db_metadata['texts'] + added_texts
    # store_embeddings(texts, db_metadata['store_name'], './')

    db_metadata = load_embeddings('test', './')
    db = db_metadata['faiss_index']
    retriever = db.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents("hola mundo")
    print([doc.page_content for doc in docs])

def test3():
    # create_embeddings(['hola mundo'], 'test', './')
    # db_metadata = load_embeddings('test', './')
    # db = db_metadata['faiss_index']
    # added_texts = ['adios mundo', 'saludos mundo']
    # db.add_texts(added_texts)
    # texts = db_metadata['texts'] + added_texts
    # update_embeddings(db, texts, db_metadata['store_name'], db_metadata['path'])
    

    db_metadata = load_embeddings('test', './')
    db = db_metadata['faiss_index']
    retriever = db.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents("hola mundo")
    print([doc.page_content for doc in docs])

if __name__ == '__main__':
    # test2()
    test3()