import requests
import os
from bs4 import BeautifulSoup
import time
import json
import re
import podcast_downloader.helpers as helpers
from podcast_downloader.helpers import slugify, cosine_similarity, get_embedding
import subprocess

class Podcast:
    def __init__(self, name, rss_feed_url):
        # Definir atributos de clase
        self.name = name
        self.rss_feed_url = rss_feed_url
        
        # Definir directorios de clase
        base_path = helpers.get_base_dir()
        self.download_directory = f'{base_path}/downloads/{slugify(name)}'
        self.transcription_directory = f'{base_path}/transcripts/{slugify(name)}'
        self.paragraph_embeddings_directory = f'{base_path}/paragraph_embeddings/{slugify(name)}'
        self.description_embeddings_path = f'{base_path}/description_embeddings/{slugify(name)}.json'
    
        # Crear directorios de clase
        for dir in [self.download_directory, self.transcription_directory, self.paragraph_embeddings_directory]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def search_items(self, search_embedding, TOP_LIMIT=2):
        # Obtener los episodios del podcast
        items = self.get_items()

        # Obtener description_embeddings
        description_embeddings = self.get_description_embeddings()

        # Sorting de description_embeddings
        sorted_description_embeddings = sorted(description_embeddings, 
                                               key=lambda x: cosine_similarity(x['embedding'], search_embedding), 
                                               reverse=True)
        
        # Obtener los títulos de todos los podcasts
        items_titles = [podcast.find('title').text for podcast in items]
        # Obtener los episodios indexados por título
        matched_podcasts = []
        for description_embedding in sorted_description_embeddings[:TOP_LIMIT]:
            title_index = items_titles.index(description_embedding['title'])
            matched_podcasts += [items[title_index]]

        return matched_podcasts

    def get_description_embeddings(self):
        description_embeddings = None

        # Declarar el archivo de embeddings de las descripciones de los episodios
        base_dir = helpers.get_base_dir()
        description_embeddings_dir = f'{base_dir}/description_embeddings'
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

    def get_items(self):
        page = requests.get(self.rss_feed_url)
        soup = BeautifulSoup(page.text, 'xml')
        return soup.find_all('item')

    def save_description_embeddings(self, description_embeddings):
        description_embeddings_json = {'description_embeddings': description_embeddings}
        with open(self.description_embeddings_path, 'w') as f:
                json.dump(description_embeddings_json, f)

    def update_description_embeddings(self, items_limit=5):
        # Obtener el arreglo de description_embeddings
        description_embeddings = self.get_description_embeddings()
        # Obtener los episodios del podcast
        items = self.get_items()
        i = 0
        j = 0
        while i < items_limit:
            item = items[j]
            title = item.find('title').text

            if ((len(description_embeddings) == 0) or ((title not in [d['title'] for d in description_embeddings]))):
                # Dormir el lazo 8 segundos por cada 10 embeddings
                if ((i+1) % 10 == 0):
                    time.sleep(8)
                # Obtener la descripción del episodio
                description = item.find('description').text
                soup = BeautifulSoup(description, 'html.parser')
                description = "\n".join([p.get_text(strip=True) for p in soup.find_all('p')])
                # Obtener el embedding de la descripción del episodio
                description_embedding = get_embedding(description)
                # Agregar la descripción del episodio con su embedding
                description_embeddings += [{'title': title, 
                                            'description': description,
                                            'embedding': description_embedding}]
                i += 1
            else:
                i = items_limit
            j += 1            
        # Actualizar description_embeddings
        self.save_description_embeddings(description_embeddings)
    
    def get_paragraph_embeddings(self, episode_path):
        paragraph_embeddings = None
        paragraph_embeddings_path = f'{self.paragraph_embeddings_directory}/{episode_path}'

        if not os.path.exists(paragraph_embeddings_path):
            paragraph_embeddings = []
            self.save_paragraph_embeddings(paragraph_embeddings, episode_path)
        else:
            with open(paragraph_embeddings_path, 'r') as f:
                paragraph_embeddings = json.load(f)['paragraph_embeddings']
        
        return paragraph_embeddings

    def save_paragraph_embeddings(self, paragraph_embeddings, episode_path):
        paragraph_embeddings_json = {'paragraph_embeddings': paragraph_embeddings}
        with open(f'{self.paragraph_embeddings_directory}/{episode_path}', 'w') as f:
            json.dump(paragraph_embeddings_json, f)

    def update_paragraph_embeddings(self, episode_path, url, paragraphs_limit = 5):
        transcripts_paths = os.listdir(self.transcription_directory)
        paragraph_embeddings = self.get_paragraph_embeddings(episode_path)
        
        if episode_path not in transcripts_paths:
            base_dir = helpers.get_base_dir()
            download_episode_path = f'{self.download_directory}/{re.sub(r"[.]json$", ".mp3",episode_path)}'

            episode_metadata_json = {'url': url, 'download_episode_path': download_episode_path}
            with open(f'{base_dir}/podcast_metadata.json', 'w') as f:
                json.dump(episode_metadata_json, f)
            
            # subprocess.run([f'{base_dir}/run_all.sh'])
            subprocess.call(['python', f'{base_dir}/download_podcasts.py'])
            subprocess.call(['python', f'{base_dir}/transcriptions.py'])
            
        with open(f'{self.transcription_directory}/{episode_path}', 'r') as f:
            paragraphs = [x['text'] for x in json.load(f)['paragraphs']]

        i = 0 
        j = 0
        while i < paragraphs_limit:
            if ((len(paragraph_embeddings) == 0) or ((paragraphs[j] not in [x['paragraph'] for x in paragraph_embeddings]))):
                if ((i+1) % 10 == 0):
                    time.sleep(8)
                paragraph_embeddings += [{'paragraph': paragraphs[j] , 'embedding': get_embedding(paragraphs[j])}]
                i += 1
            else:
                i = paragraphs_limit

            j += 1
        self.save_paragraph_embeddings(paragraph_embeddings, episode_path)
        
            





        

    
    
    

            
        