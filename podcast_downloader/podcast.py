import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import podcast_downloader.helpers as helpers
from podcast_downloader.helpers import slugify, store_embeddings, load_embeddings
import os
import json
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

    
        # Crear directorios de clase
        for dir in [self.download_directory, self.transcription_directory]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def search_items(self, message, **kwargs):
        matched_podcasts = []
        # Obtener los items del podcast
        items = self.get_items()
        # Obtener los embeddings del podcast respecto a sus descripciones
        db_description_embeddings = load_embeddings(slugify(self.name), path = helpers.get_desc_emb_dir())
        # Instanciar retriever
        retriever = db_description_embeddings.as_retriever(search_kwargs=kwargs)
        # Obtener descripciones que se asimilen al mensaje
        docs = retriever.get_relevant_documents(message)
        # Obtener los episodios indexados por título
        doc_descriptions = [x.page_content for x in docs]
        items_descriptions = [self.get_cleaned_description(x) for x in items]

        for doc_description in doc_descriptions:
            ind_description = items_descriptions.index(doc_description)
            matched_podcasts += [items[ind_description]]

        return matched_podcasts
    
    def update_description_embeddings(self, items_limit=10):
        '''
        Actualizar description_embeddings del podcast con un máximo de items_limit 
        '''
        # Obtener episodios del podcast
        items = self.get_items()
        if f'{slugify(self.name)}.pkl' not in os.listdir(helpers.get_desc_emb_dir()):
            # Empezar db con el primer episodio más reciente
            item = items[0]
            episode_title = item.find('title').text
            store_embeddings([self.get_cleaned_description(item)], f'{slugify(self.name)}', path = helpers.get_desc_emb_dir())
            self.add_episode_records(episode_title)
        
        # Obtener los embeddings del podcast respecto a sus descripciones
        db_description_embeddings = load_embeddings(slugify(self.name), path=helpers.get_desc_emb_dir())

        # Obtener episode records del podcast
        records = self.get_episode_records()
        episode_records = [x for x in records if x['podcast'] == self.name] 
    
        # Obtener los títulos de episode_records
        titles = [x['title'] for x in episode_records]

        i = 0 
        j = 0 
        while i < items_limit: 
            item = items[j]
            title = item.find('title').text
            if (len(titles) == 0) or (title not in titles):
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
            json.dump({'episode_records': records}, f)
        
    def add_episode_records(self, episode_title):
        records = self.get_episode_records()
        dicty = {'podcast':self.name, 'title': episode_title}
        records += [dicty]
        self.save_episode_records(records)

    # Paragraph embeddings methods    
    def update_paragraph_embeddings(self, slugified_episode, url):
        transcripts_paths = os.listdir(self.transcription_directory)
        if slugified_episode not in transcripts_paths:
            self.generate_transcript(slugified_episode, url)

        episodes_embeddings_path = helpers.get_dir(slugify(self.name), helpers.get_par_emb_dir())
        if f'{slugified_episode}.pkl' not in os.listdir(episodes_embeddings_path):
            loader = TextLoader(f'{self.transcription_directory}/{slugified_episode}.txt')
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            store_embeddings(docs, slugified_episode, episodes_embeddings_path, document=True)

    def generate_transcript(self, episode_path, url):
        base_dir = helpers.get_root_dir()
        download_episode_path = f'{self.download_directory}/{episode_path}.mp3)'

        episode_metadata_json = {'url': url, 'download_episode_path': download_episode_path}
        with open(f'{base_dir}/podcast_metadata.json', 'w') as f:
            json.dump(episode_metadata_json, f)
        
        # subprocess.run([f'{base_dir}/run_all.sh'])
        subprocess.call(['python', f'{base_dir}/download_podcasts.py'])
        subprocess.call(['python', f'{base_dir}/transcriptions.py'])
        
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
    
    
            





        

    
    
    

            
        