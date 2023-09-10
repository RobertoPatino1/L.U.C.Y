import requests
import os
from bs4 import BeautifulSoup
import time
import json
import re
import podcast_downloader.helpers as helpers
from podcast_downloader.helpers import slugify, get_embedding, store_embeddings, load_embeddings
import subprocess

class Podcast:
    def __init__(self, name, rss_feed_url):
        # Definir atributos de clase
        self.name = name
        self.rss_feed_url = rss_feed_url
        
        # Definir directorios de clase
        base_path = helpers.get_root_dir()
        self.download_directory = f'{base_path}/downloads/{slugify(name)}'
        self.transcription_directory = f'{base_path}/transcripts/{slugify(name)}'
        self.paragraph_embeddings_directory = f'{base_path}/paragraph_embeddings/{slugify(name)}'
    
        # Crear directorios de clase
        for dir in [self.download_directory, self.transcription_directory, self.paragraph_embeddings_directory]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def search_items(self, message, **kwargs):
        matched_podcasts = []
        # Obtener los items del podcast
        items = self.get_items()
        # Obtener los embeddings del podcast respecto a sus descripciones
        db_description_embeddings = load_embeddings(slugify(self.name))
        # Instanciar retriever
        retriever = db_description_embeddings.as_retriever(search_kwargs=kwargs)
        # Obtener descripciones que se asimilen al mensaje
        docs = retriever.get_relevant_documents(message)
        # Obtener los episodios indexados por título
        doc_descriptions = [x.page_content for x in docs]
        items_descriptions = [self.get_cleaned_description(x.find('description').text) for x in items]

        for doc_description in doc_descriptions:
            ind_description = items_descriptions.index(doc_description)
            matched_podcasts += [items[ind_description]]

        return matched_podcasts
    
    def update_description_embeddings(self, items_limit=10):
        '''
        Actualizar description_embeddings del podcast con un máximo de items_limit 
        '''
        # Obtener los embeddings del podcast respecto a sus descripciones
        db_description_embeddings = load_embeddings(slugify(self.name))

        # Obtener episode records del podcast
        records = self.get_episode_records()
        episode_records = [x for x in records if x['podcast'] == self.name] 
        # Obtener episodios del podcast
        items = self.get_items()

        if slugify(self.name) not in os.listdir(helpers.get_desc_emb_dir()):
            # Empezar db con el primer episodio más reciente
            item = items[0]
            episode_title = item.find('title').text
            store_embeddings([self.get_cleaned_description(item)], f'{slugify(self.name)}')
            self.add_episode_records(episode_title)
        
        # Obtener los títulos de episode_records
        titles = [x['title'] for x in episode_records]

        i = 0 
        j = 0 
        while i < items_limit: 
            item = items[j]
            title = item.find('title').text
            if title not in titles:
                # Agregar description embedding 
                description = self.get_cleaned_description(item)
                db_description_embeddings.add_texts([description])
                i += 1
            elif len(titles) == len(items):
                i = items_limit
            j += 1
    
    # Episode records methods
    def get_episode_records(self):
        with open(f'{helpers.get_desc_emb_meta_path()}', 'r') as f:
            records = json.load(f)['episode_records']
        return records
    
    def save_episode_records(self, records):
        with open(f'{helpers.get_desc_emb_meta_path()}', 'w') as f:
            json.dump(records, f)
        
    def add_episode_records(self, episode_title):
        records = self.get_episode_records()
        dicty = {'podcast':self.name, 'title': episode_title}
        records += [dicty]
        self.save_episode_records(records)

    # Paragraph embeddings methods
    def get_paragraph_embeddings(self, episode_path):
        paragraph_embeddings = None

        episodes_path = helpers.get_dir(slugify(self.name), helpers.get_par_emb_dir())
        if episode_path not in os.listdir(episodes_path):
            # Empezar db con el el primer párrafo
            item = items[0]
            episode_title = item.find('title').text
            store_embeddings([self.get_cleaned_description(item)], f'{slugify(self.name)}')
            self.add_episode_records(episode_title)


















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
            base_dir = helpers.get_root_dir()
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
        


    # Helpers methods
    def get_items(self):
        page = requests.get(self.rss_feed_url)
        soup = BeautifulSoup(page.text, 'xml')
        return soup.find_all('item')
    
    def get_cleaned_description(self, item):
        raw_description = item.find('description').text
        bs_description = BeautifulSoup(raw_description, 'html.parser')
        description = "\n".join([p.get_text(strip=True) for p in bs_description.find_all('p')])
        return description
    
    
            





        

    
    
    

            
        