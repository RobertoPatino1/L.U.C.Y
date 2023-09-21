import time
import json
import chainlit as cl

from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from podcast_downloader.podcast import load_embeddings
from podcast_downloader.podcast import Podcast



DB_FAISS_PATH = 'vectorstore/db_faiss'
RAW_PODCAST_LIST_PATH = './podcast_downloader/podcast_list.json'

custom_prompt_template = """Utilice la siguiente información para responder la pregunta del usuario.
Si no sabe la respuesta, simplemente diga que no la sabe, no intente inventar una respuesta.

Contexto: {context}
Pregunta: {question}

Solo devuelva la útil respuesta a continuación y nada más.
Respuesta útil:
"""

def save_podcast_data(podcast_list):
    l_podcast_json = {'podcast_list':podcast_list}
    json_path = './podcast_downloader/podcast_list.json'
    with open(json_path, 'w') as f:
        json.dump(l_podcast_json, f)

def get_podcast_list(raw_podcast_list):
    podcast_list = []

    for raw_podcast in raw_podcast_list:
        podcast_list += [Podcast(raw_podcast['name'], raw_podcast['rss_feed_url'])]
    
    return podcast_list

def get_episodes_metadata(podcast_items):
    episode_urls = [podcast.find('enclosure')['url'] for podcast in podcast_items]
    episode_titles = [podcast.find('title').text for podcast in podcast_items]
    return list(zip(episode_urls, episode_titles))


async def update_data(message, **kwargs):
    with open(RAW_PODCAST_LIST_PATH, 'r') as f:
        raw_podcast_list = json.load(f)['podcast_list']
    podcast_list = [Podcast(raw_podcast['name'], raw_podcast['rss_feed_url']) for raw_podcast in raw_podcast_list]


    for podcast in podcast_list:
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
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 1054,
        temperature = 0.5
    )
    return llm

#QA Model Function
async def qa_bot():
    embeddings = load_embeddings()
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# chainlit code
@cl.on_chat_start
async def start():
    # msg = cl.Message(content="Starting the bot...")
    # await msg.send()
    # msg.content = "Hola, soy Lucy. Qué quieres hablar conmigo hoy?"
    # await msg.update()
    pass

@cl.on_message
async def main(message):
    start_time = time.time()
    await update_data(message, k=2)
    # await ingest()
    chain = await qa_bot()
    msg = cl.Message(content="Getting data...")
    await msg.send()

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    # answer = res["result"] 
    # await cl.Message(content=answer).send()
    sources = res["source_documents"]

    src_message = "Fuentes:\n\n\n" 
    for document in sources:
        src_message += f"Podcast: {document.metadata['podcast']}\n Episodio: {document.metadata['episode']}\n"
        src_message += f"Contenido: \n{document.page_content}\n\n"
        
    await cl.Message(content=src_message).send()

    print("--- entire_process: %s seconds ---" % (time.time() - start_time))
    
# Testeos

# output function
# def final_result(query):
#     qa_result = qa_bot()
#     response = qa_result({'query': query})
#     return response

# def get_result(message):
#     start_time = time.time()
#     update_data(message, k=2)
#     ingest()
#     response = final_result(message)
#     print(response)
#     print("--- %s seconds ---" % (time.time() - start_time))


# # get_result('¿Cómo funcionan las emociones prohibidas en los niños?')
# get_result('¿Qué es la inteligencia emocional?')

    
    
    