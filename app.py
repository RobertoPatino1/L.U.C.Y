import json
import time
import requests
import os

import chainlit as cl
from chainlit.input_widget import Select, Switch, TextInput

from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from podcast_downloader.podcast import load_embeddings
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
    'Portuguese': 'pt',
    'Dutch': 'nl',
    'Hindi': 'hi',
    'Japanese': 'ja',
    'Chinese': 'zh',
    'Finnish': 'fi',
    'Korean': 'ko',
    'Polish': 'pl',
    'Russian': 'ru',
    'Turkish': 'tr',
    'Ukrainian': 'uk',
    'Vietnamese': 'vi'
}

# Custom prompt template variable
# custom_prompt_template = """Utilice la siguiente información para responder la pregunta del usuario.
# Si no sabe la respuesta, simplemente diga que no la sabe, no intente inventar una respuesta.

# Contexto: {context}
# Pregunta: {question}

# Solo devuelva la útil respuesta a continuación y nada más.
# Respuesta útil:
# """

# custom_prompt_template = """Use the following information to answer the user's question.
# If you don't know the answer, just say you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Just return the helpful answer below and nothing more.
# Helpful answer:"""

custom_prompt_template = """
Utilisez les informations suivantes pour répondre à la question de l'utilisateur.
Si vous ne connaissez pas la réponse, dites simplement que vous ne la savez pas, n'essayez pas d'inventer une réponse.

Contexte: {context}
Demande: {question}

Renvoyez simplement la réponse utile ci-dessous et rien de plus.
Réponse utile:"""


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

def get_episodes_metadata(podcast_items):
    episode_urls = [podcast.find('enclosure')['url'] for podcast in podcast_items]
    episode_titles = [podcast.find('title').text for podcast in podcast_items]
    return list(zip(episode_urls, episode_titles))


async def update_data(message, **kwargs):
    podcast = cl.user_session.get('podcast')
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
    prompt = PromptTemplate(template=custom_prompt_template,
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
def load_llm(open_ai_api_key:str):
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
async def qa_bot():
    openai_api_key = cl.user_session.get('settings')['gpt_api_key']
    podcast = cl.user_session.get('podcast')
    embeddings = load_embeddings()
    db = FAISS.load_local(podcast.db_faiss_path, embeddings)
    llm = load_llm(openai_api_key)
    qa_prompt = set_custom_prompt()
    
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def load_settings_widgets(current_podcast):
    settings_widgets = [TextInput(id="assembly_ai_api_key", label="Assembly AI API Key for Transcriptions"),
                        TextInput(id="PodcastName", 
                      label="Podcast name",
                      placeholder='Psicologia Al Desnudo | @psi.mammoliti',
                      description=f'Current Podcast: {current_podcast.name}'),
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
    return settings_widgets

def set_podcast(podcast: Podcast):
    cl.user_session.set('podcast', podcast)
    assembly_ai_api_key = cl.user_session.get('settings')['assembly_ai_api_key']
    with open('podcast.json', 'w') as f:
        json.dump({'name':podcast.name, 
                   'rss_feed_url':podcast.rss_feed_url,
                     'language': podcast.get_ts_language(),
                     'assembly_key': assembly_ai_api_key}, f)
    

# chainlit code
@cl.on_chat_start
async def start():
    cl.user_session.set('podcast', default_podcast)
    current_podcast = cl.user_session.get('podcast')
    settings_widgets = load_settings_widgets(current_podcast) 
    settings = await cl.ChatSettings(settings_widgets).send()
    cl.user_session.set('settings', settings)
    set_podcast(current_podcast)
    # Settings widgets list
    cl.user_session.set('audios', [])

@cl.on_settings_update
async def setup_agent(settings):
    podcast_name = settings['PodcastName']
    rss = settings['RSS']
    if podcast_name != None and rss != None and requests.get(rss).status_code == 200: 
        set_podcast(Podcast(podcast_name, rss))

    cl.user_session.set('assembly_ai_api_key', settings['assembly_ai_api_key'])

    current_podcast = cl.user_session.get('podcast')
    settings_widgets = load_settings_widgets(current_podcast) 
    settings = await cl.ChatSettings(settings_widgets).send()
    cl.user_session.set('settings', settings)
    
@cl.on_message
async def main(message, message_id):
    # Empezar temporizador
    start_time = time.time()

    eleven_labs_api_key = cl.user_session.get('settings')['eleven_labs_api_key']
    open_ai_api_key = cl.user_session.get('settings')['gpt_api_key']

    if cl.user_session.get('assembly_ai_api_key') == None:
        await cl.Message(content='Enter an Assembly AI API Key').send()
    elif cl.user_session.get('settings')['Model'] == 'gpt-3.5-turbo' and open_ai_api_key == None: 
        await cl.Message(content='Enter a Open API Key').send()
    elif cl.user_session.get('settings')['text_to_speech'] and eleven_labs_api_key == None:
        await cl.Message(content='Enter a Eleven Labs API Key').send()
    else:
        await update_data(message, k=2)
        chain = await qa_bot()
        msg = cl.Message(content="Getting data...")
        await msg.send()

        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True
        res = await chain.acall(message, callbacks=[cb])
        # Obtener respuesta del LLM
        answer = res["result"] 
        # await cl.Message(content=answer).send()
        sources = res["source_documents"]

        if cl.user_session.get('settings')['text_to_speech']:
            output_path = get_audio(answer, message_id, speaker_id, eleven_labs_api_key)
            cl.user_session.get('audios').append(cl.Audio(path=output_path, display='inline'))
            audios = cl.user_session.get('audios')
            await cl.Message(
                content='Generated audio',
                elements=[audios[-1]],
            ).send()

        src_message = "Sources:\n\n\n" 
        for document in sources:
            src_message += f"Podcast: {document.metadata['podcast']}\n Episode: {document.metadata['episode']}\n"
            src_message += f"Content: \n{document.page_content}\n\n"
            
        await cl.Message(content=src_message).send()

    print("--- test with translated podcast: %s seconds ---" % (time.time() - start_time))
    
# Testeos
podcast = Podcast('Confidentiel','https://www.rtl.fr/podcast/confidentiel.xml')
# output function
def update_data(message, **kwargs):
    
    # Actualizar description_embeddings del podcast
    podcast.update_description_embeddings()
    # Obtengo la metadata de top_limit = 2 episodios con mayor similitud
    podcast_items = podcast.search_items(message, **kwargs)
    episodes_metadata = get_episodes_metadata(podcast_items)
    for episode in episodes_metadata:
        url, title = episode
        # Actualizar paragraph_embbeddings
        podcast.update_paragraph_embeddings(title, url)

def qa_bot():
    embeddings = load_embeddings()
    db = FAISS.load_local(podcast.db_faiss_path, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# def get_result(message):
#     start_time = time.time()
#     update_data(message, k=2)
#     if not openai_use:
#         response = final_result(message)
#     else:
#         with get_openai_callback() as cb:
#             response = final_result(message)
#         print(cb)
#     print(response)

def get_result(message):
    start_time = time.time()
    update_data(message, k=2)
    response = final_result(message)
    print(response)

    print(f"{podcast.name} test" + "--- %s seconds ---" % (time.time() - start_time))


    
    
    