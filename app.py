import streamlit as st
import openai
import json
import os
import podcast_downloader.helpers as hp
from podcast_downloader.podcast import Podcast
from podcast_downloader.helpers import slugify, cosine_similarity, get_embedding

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
    return list(zip(episode_urls, episode_titles))

def get_matched_paragraphs(raw_podcast_list, message_embedding, TOP_LIMIT = 2):
    
    matched_paragraphs = []
    # Obtener arreglo con objetos tipo podcast
    podcast_list = [Podcast(raw_podcast['name'], raw_podcast['rss_feed_url']) for raw_podcast in raw_podcast_list]

    for podcast in podcast_list:
        # Actualizar description_embeddings del podcast
        podcast.update_description_embeddings()
        # Obtengo la metadata de top_limit = 2 episodios con mayor similitud
        podcast_items = podcast.search_items(message_embedding, TOP_LIMIT = 2)
        episodes_metadata = get_episodes_metadata(podcast_items)
        
        for episode in episodes_metadata:
            url, title = episode
            episode_path = f'{slugify(title)}.json'

            # Actualizar paragraph_embbeddings
            podcast.update_paragraph_embeddings(episode_path, url)
            # Obtener el paragraph_embeddings del episodio
            paragraph_embeddings = podcast.get_paragraph_embeddings(episode_path)
            paragraph_embeddings_sorted = sorted(paragraph_embeddings, 
                                                key = lambda x: cosine_similarity(x['embedding'], message_embedding),
                                                reverse=True)

            matched_paragraphs += [x['paragraph'] for x in paragraph_embeddings_sorted[:TOP_LIMIT]]
        
    return matched_paragraphs


def main():
    # Iniciar chat y obtener el arreglo de podcast disponibles en formato json
    # Set page config
    st.set_page_config(
        page_title="Chatty", page_icon="🎯")
    
    # Empezar el chat inicial informativo del bot 
    podcast_downloader_dir = hp.get_base_dir()
    podcast_list_path = f'{podcast_downloader_dir}/podcast_list.json'

    # Obtener los podcast disponibles
    with open(podcast_list_path, 'r') as f:
        raw_podcast_list = json.load(f)['podcast_list']

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
            matched_paragraphs = get_matched_paragraphs(raw_podcast_list, message_embedding)
            custom_prompt = template.format(message=prompt, experts="\n".join(matched_paragraphs))
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

def test():
    # message_embedding = get_embedding('Hola, últimamente me he sentido muy bien, crees que me pueda mantener así?')
    # with open('message_embedding.json', 'w') as f:
    #     json.dump({'message_embedding':message_embedding}, f)

    # Empezar el chat inicial informativo del bot 
    podcast_downloader_dir = hp.get_base_dir()
    podcast_list_path = f'{podcast_downloader_dir}/podcast_list.json'

    # Obtener los podcast disponibles
    with open(podcast_list_path, 'r') as f:
        raw_podcast_list = json.load(f)['podcast_list']


    with open('message_embedding.json') as f:
        message_embedding = json.load(f)['message_embedding']
    matched_paragraphs = get_matched_paragraphs(raw_podcast_list, message_embedding)
    print(matched_paragraphs)

if __name__ == '__main__':
    # main()
    test()
