import streamlit as st
import openai
import json
import podcast_downloader.helpers as hp
from podcast_downloader.podcast import Podcast
from podcast_downloader.helpers import slugify

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

def get_matched_paragraphs(message, raw_podcast_list, **kwargs):
    matched = []
    # Obtener arreglo con objetos tipo podcast
    podcast_list = [Podcast(raw_podcast['name'], raw_podcast['rss_feed_url']) for raw_podcast in raw_podcast_list]

    for podcast in podcast_list:
        # Actualizar description_embeddings del podcast
        podcast.update_description_embeddings()
        # Obtengo la metadata de top_limit = 2 episodios con mayor similitud
        podcast_items = podcast.search_items(message, **kwargs)
        episodes_metadata = get_episodes_metadata(podcast_items)
        
        for episode in episodes_metadata:
            url, title = episode
            slugified_episode = f'{slugify(title)}'
            # Actualizar paragraph_embbeddings
            podcast.update_paragraph_embeddings(slugified_episode, url)
            # Obtener los top_limit = 2 párrafos del episodio con mayor similitud
            db_transcription_embeddings = hp.load_embeddings(slugified_episode, hp.get_par_emb_dir())
            retriever = db_transcription_embeddings.as_retriever(search_kwargs=kwargs)
            docs = retriever.get_relevant_documents(message)
            matched_paragraphs = [x.page_content for x in docs]
            matched += [{'podcast':podcast.name, 'title': title, 'matched_paragraphs':matched_paragraphs}]
            
    return matched


def main():
    # Iniciar chat y obtener el arreglo de podcast disponibles en formato json
    # Set page config
    st.set_page_config(
        page_title="Chatty", page_icon="🎯")
    
    # Empezar el chat inicial informativo del bot 
    podcast_downloader_dir = hp.get_root_dir()
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

    with st.chat_message("assistant", avatar='👩'):
        st.markdown(initial_message.format(podcasts="\n".join([d['name'] for d in raw_podcast_list])))
    
    
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

    Además menciona que te basaste en los siguientes episodios:
    {episodes}

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
            matched = get_matched_paragraphs(message, raw_podcast_list, k=2)
            matched_paragraphs = hp.flatten([x['matched_paragraphs'] for x in matched])
            titles = [x['title'] for x in matched]
            custom_prompt = template.format(message=prompt, experts="\n".join(matched_paragraphs), episodes="\n".join(titles))
            st.session_state.messages.append({"role": "user", "content": custom_prompt})
        # Display user message in chat message container
        with st.chat_message("user", avatar='🗿'):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar='👩'):
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
    podcast_downloader_dir = hp.get_root_dir()
    podcast_list_path = f'{podcast_downloader_dir}/podcast_list.json'

    # Obtener los podcast disponibles
    with open(podcast_list_path, 'r') as f:
        raw_podcast_list = json.load(f)['podcast_list']


    with open('message_embedding.json') as f:
        message_embedding = json.load(f)['message_embedding']
    matched_paragraphs = get_matched_paragraphs(raw_podcast_list, message_embedding)
    print(matched_paragraphs)

def test1():
    podcast = Podcast('Psicologia Al Desnudo | @psi.mammoliti', 'https://anchor.fm/s/28fef6f0/podcast/rss')
    db_instructEmbedd = hp.load_embeddings(f'{slugify(podcast.name)}')
    retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents("Hola, últimamente me he sentido muy bien, crees que me pueda mantener así?")
    print(docs[0].page_content)

def test2():
    message = 'Hola, últimamente me he sentido muy bien, crees que me pueda mantener así?'
    # Obtener los podcast disponibles
    podcast_downloader_dir = hp.get_root_dir()
    podcast_list_path = f'{podcast_downloader_dir}/podcast_list.json'
    with open(podcast_list_path, 'r') as f:
        raw_podcast_list = json.load(f)['podcast_list']
    matched = get_matched_paragraphs(message, raw_podcast_list, k=2)
    matched_paragraphs = hp.flatten([x['matched_paragraphs'] for x in matched])
    print(matched_paragraphs)


if __name__ == '__main__':
    # main()
    test2()