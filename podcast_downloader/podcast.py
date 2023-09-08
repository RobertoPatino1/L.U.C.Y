import requests
import os
import re
from bs4 import BeautifulSoup
import openai
import numpy as np
from dotenv import load_dotenv, find_dotenv
import time
import json
import unicodedata

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("API_BASE")

class Podcast:
    def __init__(self, name, rss_feed_url):
        self.name = name
        self.rss_feed_url = rss_feed_url
        
        
        self.download_directory = f'./Podcast-Downloader/downloads/{self.slugify(name)}'
        if not os.path.exists(self.download_directory):
            os.makedirs(self.download_directory)

        self.transcription_directory = f'./Podcast-Downloader/transcripts/{self.slugify(name)}'
        if not os.path.exists(self.transcription_directory):
            os.makedirs(self.transcription_directory)   

        self.description_embeddings_path = f'./Podcast-Downloader/description_embeddings/{self.slugify(name)}.json'
           

    def get_items(self):
        page = requests.get(self.rss_feed_url)
        soup = BeautifulSoup(page.text, 'xml')
        return soup.find_all('item')

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input = text, model=model)['data'][0]['embedding']
    
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def simplify_title(self):
        file_name = re.sub(r'[%/&!@#\*\$\?\+\^\\.\\\\]', '', self.name)[:100].replace(' ', '_')
        return file_name
    
    def slugify(self, value, allow_unicode=False):
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

    def save_description_embeddings(self, description_embeddings):
        description_embeddings_json = {'description_embeddings':description_embeddings}
        with open(self.description_embeddings_path, 'w') as f:
                json.dump(description_embeddings_json, f)

    def add_description_embeddings(self, items_limit):
        # Obtener el arreglo de description_embeddings
        description_embeddings = self.get_description_embeddings()['description_embeddings']
        # Obtener los episodios del podcast
        items = self.get_items()
        i = 0
        for item in items:
            title = item.find('title').text
            # Obtener la descripción del episodio
            description = item.find('description').text
            soup = BeautifulSoup(description, 'html.parser')
            description = "\n".join([p.get_text(strip=True) for p in soup.find_all('p')])

            if (len(description_embeddings) == 0 or title not in [d['title'] for d in description_embeddings]) and i < items_limit:
                # Dormir el lazo 8 segundos por cada 10 embeddings
                if (i % 9 == 0):
                    time.sleep(8)
                # Obtener el embedding de la descripción del episodio
                description_embedding = self.get_embedding(description)
                # Agregar la descripción del episodio con su embedding
                description_embeddings += [{'title': title, 
                                            'description': description,
                                            'embedding': description_embedding}]
                i += 1
        # Actualizar description_embeddings
        self.save_description_embeddings(description_embeddings)

    def get_description_embeddings(self):
        description_embeddings = None

        # Declarar el archivo de embeddings de las descripciones de los episodios
        description_embeddings_dir = f'./Podcast-Downloader/description_embeddings'
        if not os.path.exists(description_embeddings_dir):
            os.mkdir(description_embeddings_dir)  
        
        if not os.path.exists(self.description_embeddings_path): 
            # Crear el json de description_embeddings 
            description_embeddings = []
            self.save_description_embeddings(description_embeddings)
        else:
            # Cargar el archivo json de description_embeddings
            with open(self.description_embeddings_path, 'r') as f:
                description_embeddings = json.load(f)['description_embeddings']

        return description_embeddings
    
    def search_items(self, search_embedding, top_limit=2, items_limit=10):
        # Obtener los episodios del podcast
        items = self.get_items()

        # Agregar description_embeddings 
        self.add_description_embeddings(items_limit)
        description_embeddings = self.get_description_embeddings()

        # Sorting de description_embeddings
        sorted_description_embeddings = sorted(description_embeddings, 
                                               key=lambda x: self.cosine_similarity(x['embedding'], search_embedding), 
                                               reverse=True)
        
        # Obtener los títulos de todos los podcasts
        items_titles = [podcast.find('title').text for podcast in items]
        # Obtener los episodios indexados por título
        matched_podcasts = []
        for description_embedding in sorted_description_embeddings[:top_limit]:
            title_index = items_titles.index(description_embedding['title'])
            matched_podcasts += [items[title_index]]

        return matched_podcasts

            
        