import json
import time
import requests
import os

import chainlit as cl
from chainlit.input_widget import Select, Switch, TextInput
from chainlit import make_async

from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from translate import Translator

from podcast_downloader.podcast import load_embeddings, convert_language_variable
from podcast_downloader.podcast import Podcast

# Default Podcast variable
default_podcast = Podcast('Psicologia Al Desnudo | @psi.mammoliti','https://anchor.fm/s/28fef6f0/podcast/rss')

# Faiss Index Paths
DB_FAISS_PATH = 'vectorstore/db_faiss'
RAW_PODCAST_LIST_PATH = './podcast_downloader/podcast_list.json'

# Audio variables
generated_file_path = './generated_files'
audio_path = f'{generated_file_path}/audio_files'
speaker_id = '21m00Tcm4TlvDq8ikWAM'

# Assembly AI supported languages
language_codes = {
    'US English': 'en_us',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Dutch': 'nl',
    'Hindi': 'hi',
    'Chinese': 'zh',
    'Finnish': 'fi',
    'Russian': 'ru',
    'Ukrainian': 'uk',
    'Vietnamese': 'vi'
}


def get_audio(text:str, chat_id:int, speaker_id:str, api_key:str):
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)

    output_path = f'{audio_path}/audio_{chat_id}.mp3'

    CHUNK_SIZE = 1024
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{speaker_id}"

    headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": api_key,
    }

    data = {
    "text": text,
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
    }
    }

    response = requests.post(url, json=data, headers=headers)
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    return output_path

aget_audio = make_async(get_audio)

def get_episodes_metadata(podcast_items):
    episode_urls = [podcast.find('enclosure')['url'] for podcast in podcast_items]
    episode_titles = [podcast.find('title').text for podcast in podcast_items]
    return list(zip(episode_urls, episode_titles))


async def update_data(message, **kwargs):
    # Traducir mensaje
    with open('podcast.json', 'r') as f:
        podcast_data = json.load(f)
    podcast = Podcast(podcast_data['name'], podcast_data['rss_feed_url'])

    # Actualizar description_embeddings del podcast
    podcast.update_description_embeddings()
    
    # Obtengo la metadata de top_limit = 2 episodios con mayor similitud
    podcast_items = podcast.search_items(message, **kwargs)
    episodes_metadata = get_episodes_metadata(podcast_items)
    for episode in episodes_metadata:
        url, title = episode
        # Actualizar paragraph_embbeddings
        podcast.update_paragraph_embeddings(title, url)
    

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    with open('podcast.json', 'r') as f:
        podcast_data = json.load(f)
    with open('ts_prompts.json', 'r') as f:
        ts_prompts = json.load(f)
    
    language = "en_us" if podcast_data['language'] == "en" else podcast_data['language']
    ts_prompt = ts_prompts[language]


    prompt = PromptTemplate(template=ts_prompt,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
@cl.cache
def load_llm(open_ai_api_key:str=None):
    if cl.user_session.get('settings')['Model'] == 'Llama-2-7B-Chat-GGML':
        # Load the locally downloaded model here
        llm = CTransformers(
            model = "TheBloke/Llama-2-7B-Chat-GGML",
            model_type="llama",
            max_new_tokens = 512,
            temperature = 0.5
        )
    elif cl.user_session.get('settings')['Model'] == 'gpt-3.5-turbo':
        llm = OpenAI(model_name='gpt-3.5-turbo', openai_api_key=open_ai_api_key)

    return llm

#QA Model Function
def qa_bot():
    # openai_api_key = cl.user_session.get('settings')['gpt_api_key']
    openai_api_key = None
    podcast = cl.user_session.get('podcast')
    embeddings = load_embeddings()
    db = FAISS.load_local(podcast.db_faiss_path, embeddings)
    llm = load_llm(openai_api_key)
    qa_prompt = set_custom_prompt()
    
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

aqa_bot = make_async(qa_bot)

def get_ts_message(message:str, src:str, dst:str):
    '''
    Translate a message
    '''
    if src != dst:
        translator= Translator(from_lang=src, to_lang=dst)
        return translator.translate(message)
    else:
        return message
    
aget_ts_message = make_async(get_ts_message)

# chainlit code
@cl.on_chat_start
async def start():
    # Set starting variables
    # await cl.Message(content=f'Starting with, \n Podcast: {default_podcast.name}\n Your spoken language: {list(language_codes.keys())[0]}').send()
    await cl.Message(content='Complete the required fields on Chat Settings to start').send()

    cl.user_session.set('podcast', default_podcast)
    cl.user_session.set('audios', [])

    # Set the settings tab
    settings_widgets = [TextInput(id="assembly_ai_api_key", label="(*) Assembly AI API Key for Transcriptions"),
                        Select(id="src", 
                               label="Your language", 
                               tooltip="We are in charge of identifying the language of your Google Podcast",
                               items=language_codes),
                        TextInput(id="PodcastName", 
                                    label="Podcast name",
                                    placeholder='Psicologia Al Desnudo | @psi.mammoliti',),
                        TextInput(id="RSS", 
                                label="Podcast RSS Feed URL",
                                placeholder='https://anchor.fm/s/28fef6f0/podcast/rss'),

                        Switch(id="text_to_speech", label="Text to speech", initial=False),
                        TextInput(id="eleven_labs_api_key", label="Eleven Labs API Key for Text to Speech"),
                        
                        Select(
                            id="Model",
                            label="Model",
                            values=["Llama-2-7B-Chat-GGML", "gpt-3.5-turbo"],
                            initial_index=0,
                        ),
                        TextInput(id="gpt_api_key", label="OpenAI API Key for GPT Model"),  
                        ]
    settings = await cl.ChatSettings(settings_widgets).send() 

    with open('podcast.json', 'w') as f:
                    json.dump({'name':default_podcast.name, 
                            'rss_feed_url':default_podcast.rss_feed_url,
                                'language': default_podcast.get_ts_language(),
                                'assembly_key': None,
                                'src': "en"}, f)

@cl.on_settings_update
async def setup_agent(settings):
    reverse_dict = {f'{x[1]}': f'{x[0]}' for x in language_codes.items()}
    if settings['assembly_ai_api_key'] != None:
        src = settings["src"] if settings["src"] != None else 'en_us'
        cl.user_session.set('able_to_chat', True)
        if settings['PodcastName'] != None and settings['RSS'] != None and requests.get(settings['RSS']).status_code == 200:
            if settings['Model'] == 'gpt-3.5-turbo' and settings['gpt_api_key'] == None: 
                await cl.Message(content='Enter a Open API Key').send()
            elif settings['text_to_speech'] and settings['eleven_labs_api_key'] == None:
                await cl.Message(content='Enter a Eleven Labs API Key').send()
            else:
                podcast = Podcast(settings['PodcastName'], settings['RSS'])
                cl.user_session.set('podcast', podcast)
        else:
            podcast = cl.user_session.get('podcast')
     
        await cl.Message(f'Successful podcast load,\n Podcast: {podcast.name}\n Your spoken language: {reverse_dict[src]}').send()
        with open('podcast.json', 'w') as f:
                    json.dump({'name':podcast.name, 
                            'rss_feed_url':podcast.rss_feed_url,
                                'language': podcast.get_ts_language(),
                                'assembly_key': settings['assembly_ai_api_key'],
                                'src': convert_language_variable(src)}, f)
        cl.user_session.set('able_to_chat', True)
    else:
        await cl.Message(content='Enter an Assembly AI API Key').send()
        cl.user_session.set('able_to_chat', False)
        
            
    cl.user_session.set('settings', settings)

@cl.on_message
async def main(message, message_id):
    # Empezar temporizador
    start_time = time.time()

    if cl.user_session.get('able_to_chat'):
        # Traducir mensaje
        with open('podcast.json', 'r') as f:
            podcast_data = json.load(f)

        src = "en" if podcast_data['src'] == "en_us" else podcast_data['src']
        dst = "en" if podcast_data['language'] == "en_us" else podcast_data['language']

        ts_message = await aget_ts_message(message, src, dst)

        await update_data(ts_message, k=2)
        chain = await aqa_bot()

        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True
        res = await chain.acall(ts_message, callbacks=[cb])
        # Obtener respuesta del LLM
        answer = res["result"] 
        sources = res["source_documents"]

        ts_answer = await aget_ts_message(answer, dst, src)
        
        await cl.Message(content=ts_answer).send()
        src_message = "Sources:\n\n\n" 
        for document in sources:
            src_message += f"Podcast: {document.metadata['podcast']}\n Episode: {document.metadata['episode']}\n"
            src_message += f"Content: \n{document.page_content}\n\n"

        if cl.user_session.get('settings')['text_to_speech']:
            output_path = await aget_audio(ts_answer, message_id, speaker_id, cl.user_session.get('settings')['eleven_labs_api_key'])
            cl.user_session.get('audios').append(cl.Audio(path=output_path, display='inline'))
            audios = cl.user_session.get('audios')
            await cl.Message(
                content='Generated audio',
                elements=[audios[-1]],
            ).send()
            
        await cl.Message(content=src_message).send()

    else:
        await cl.Message(content='Complete the required fields on Chat Settings').send()
    print("--- %s seconds ---" % (time.time() - start_time))
