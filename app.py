import streamlit as st
import openai
import numpy as np
import json
import pandas as pd
import os
from PIL import Image
from podcast_downloader.podcast import Podcast
import dateutil.parser

def search(message_embedding, paragraph_emb_df, q_results = 5):
    paragraph_emb_df['similarity'] = paragraph_emb_df['embedding'].apply(lambda x: cosine_similarity(x, message_embedding))
    paragraph_emb_df.sort_values('similarity', ascending=False)
    return paragraph_emb_df.iloc[:q_results]['paragraph'].values

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = text, model=model)['data'][0]['embedding']

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_json(EPISODE_PATH):
    with open(EPISODE_PATH, 'r') as f:
        dictionary = json.load(f) 
    return dictionary

def save_podcast_data(podcast_list):
    l_podcast_json = {'podcast_list':podcast_list}
    json_path = './Podcast-Downloader/podcast_list.json'
    with open(json_path, 'w') as f:
        json.dump(l_podcast_json, f)

        
def select_settings(podcast_list):
    # Session State initializing
    if 'store' not in st.session_state:
        st.session_state.store = False 

    if st.sidebar.button('Agrega tu Google Podcast favorito') or st.session_state.store:
        st.session_state.store = True
        with st.empty():
            st.sidebar.write('El RSS Feed URL puedes obtenerlo en: https://getrssfeed.com/')
            # image = Image.open('./step_by_step_rss_feed.gif')
            # st.sidebar.image(image, caption='¿Cómo obtener el rss feed url?')
            # q = st.sidebar.number_input('Cantidad', min_value=1, max_value=2)
            # if st.sidebar.button(f'Ingresar podcast'):
                # for i in range(q):
            name = st.sidebar.text_input(f'Nombre del podcast')
            rss_feed_url = st.sidebar.text_input('RSS Feed URL del podcast')

            if st.sidebar.button('Guardar podcast'):        
                if rss_feed_url not in [podcast['rss_feed_url'] for podcast in podcast_list]:
                    podcast_list += [{'name':name, 'rss_feed_url':rss_feed_url}]
                    save_podcast_data(podcast_list)
                elif name != '' and rss_feed_url != '':
                    st.sidebar.write('Escribir los datos del podcast')
                else:
                    st.sidebar.write('Podcast actualmente en existencia')
            
            st.empty()

    podcast_options = st.sidebar.multiselect('Escoge algunos de los podcast que conozco', [podcast['name'] for podcast in podcast_list])
    
    
    # model_name = st.sidebar.radio("Choose LLM:",
    #                               ("gpt-3.5-turbo-0613", "gpt-4"))
    # temperature = st.sidebar.slider("Temperature:", min_value=0.0,
    #                                 max_value=1.0, value=0.0, step=0.01)
    # return ChatOpenAI(temperature=temperature, model_name=model_name)

def get_podcast_list(raw_podcast_list):
    podcast_list = []

    for raw_podcast in raw_podcast_list:
        podcast_list += [Podcast(raw_podcast['name'], raw_podcast['rss_feed_url'])]
    
    return podcast_list

def get_episodes_metadata(podcast_items):
    episode_urls = [podcast.find('enclosure')['url'] for podcast in podcast_items]
    episode_titles = [podcast.find('title').text for podcast in podcast_items]
    episode_release_dates = [parse_date(podcast.find('pubDate').text) for podcast in podcast_items]
    episode_descriptions = [podcast.find('description').text for podcast in podcast_items]
    return list(zip(episode_urls, episode_titles, episode_release_dates, episode_descriptions))


def main():
    # Iniciar chat y obtener el arreglo de podcast disponibles en formato json
    # Set page config
    st.set_page_config(
        page_title="Chatty", page_icon="🎯")
    
    # Empezar el chat inicial informativo del bot 
    podcast_downloader_dir = './podcast_downloader'
    podcast_list_path = f'{podcast_downloader_dir}/podcast_list.json'

    # Obtener los podcast disponibles
    raw_podcast_list = load_json(podcast_list_path)['podcast_list']

    # Inicializar podcast_list en streamlit session
    if not 'podcast_list' in st.session_state:
        st.session_state['podcast_list'] = raw_podcast_list
        
    # Mostrar mensaje inicial informativo en el chat 
    initial_message = """
    Hola! Soy Lucy, tu coach personal, mis respuestas se basan en tus podcasts favoritos,
    actualmente conozco de los siguientes podcast:\n
    {podcasts}\n
    Vamos cuéntame, de qué quieres hablar conmigo hoy 😊
    """
    coach = st.chat_message("assistant", avatar='👩')
    coach.write(initial_message.format(podcasts="\n".join([d['name'] for d in raw_podcast_list])))
    
    
    # Set OpenAI API key from Streamlit secrets
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_base = st.secrets["API_BASE"]

    # Definir la orden predeterminada para el LLM
    template = """
    Eres la mejor coach de mejora personal, te llamas Lucy, para personas de entre 18 y 27 años.
    Te voy a compartir una solicitud de una de estas personas que pertenecen al público objetivo.
    Tu te encargarás de ofrecer la mejor respuesta acorde a qué dicen las mejores coach o incluso psicólogas respecto a la temática relacionada.
    Además tendrás que seguir cuidadosamente las siguientes TODAS las reglas de a continuación:

    1/ La respuesta debe ser demasiado similar o incluso idéntica a qué dicen las expertas en el tema, 
    esto en función de su forma de hablar, sus argumentos lógicos y cualquier otro detalle que identifiques.

    Debajo se encuentra la solicitud:
    {message}

    Ahora te muestro a continuación qué dicen las expertas acerca del tema, con lo cual puedas basar tu respuesta:
    {experts}

    Por favor, escribe cómo le responderías a esta persona que ha acudido a ti como coach:
    """

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4"
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Accept user input
    if prompt := st.chat_input("¿Qué tal?, cuéntame, estoy para escucharte!"):
        if len(st.session_state.messages) > 0:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
        else:
            # Asimilar spotify search al empezar el chat
            message_embedding = get_embedding(prompt)
            





            # Obtener arreglo con objetos tipo podcast
            podcast_list = [Podcast(raw_podcast['name'], raw_podcast['rss_feed_url']) for raw_podcast in raw_podcast_list]

            for podcast in podcast_list:
                # Obtengo la metadata de top_limit = 2 episodios con mayor similitud
                podcast_items = podcast.search_items(message_embedding, top_limit = 2)
                episodes_metadata = get_episodes_metadata(podcast_items)
                
                for episode in episodes_metadata:
                    url, title, release_date, description = episode

                    










            paragraph_emb_df = get_embeddings(message_embedding)
            similarities = search(message_embedding, paragraph_emb_df)
            
            custom_prompt = template.format(message=prompt, experts=similarities)
            st.session_state.messages.append({"role": "user", "content": custom_prompt})
        # Display user message in chat message container
        with st.chat_message("user", '🗿'):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant", '👩'):
            message_placeholder = st.empty()
            full_response = ""

        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
            allow_fallback=True
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main()
