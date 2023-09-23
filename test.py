import time

start_time = time.time()

from app import get_result

get_result()

# from transformers import pipeline
# import json

# language_codes = {
#     'US English': 'en_us',
#     'Spanish': 'es',
#     'French': 'fr',
#     'German': 'de',
#     'Italian': 'it',
#     'Portuguese': 'pt',
#     'Dutch': 'nl',
#     'Hindi': 'hi',
#     'Japanese': 'ja',
#     'Chinese': 'zh',
#     'Finnish': 'fi',
#     'Korean': 'ko',
#     'Polish': 'pl',
#     'Russian': 'ru',
#     'Turkish': 'tr',
#     'Ukrainian': 'uk',
#     'Vietnamese': 'vi'
# }

# custom_prompt_template = """Use the following information to answer the user's question.
# If you don't know the answer, just say you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Just return the helpful answer below and nothing more.
# Helpful answer:"""
# # source & destination languages
# src = "en"

# ts_prompts = {"en": custom_prompt_template}

# for dst in (list(language_codes.values())[1:]):
#     task_name = f"translation_{src}_to_{dst}"
#     model_name = f"Helsinki-NLP/opus-mt-{src}-{dst}"
#     translator  = pipeline(task_name, model=model_name, tokenizer=model_name)

#     chunks = ["Use the following information to answer the user's question.", 
#             "If you don't know the answer, just say you don't know, don't try to make up an answer.",
#             "Context:","Question:", "Just return the helpful answer below and nothing more.","Helpful answer:"]

#     ts_chunks = [translator(x)[0]["translation_text"] for x in chunks]
#     ts_prompt_template_1 = f"{ts_chunks[0]}\n{ts_chunks[1]}\n\n{ts_chunks[2]}"
#     ts_prompt_template_2 = " {context}\n" + f"{ts_chunks[3]}" + " {question}\n\n" + f"{ts_chunks[4]}\n{ts_chunks[5]}"

#     custom_prompt_template = ts_prompt_template_1 + ts_prompt_template_2
#     ts_prompts[dst] = custom_prompt_template

# with open('ts_prompts.json', 'w') as f:
#     json.dumps(ts_prompts, f)
# print("---  %s seconds ---" % (time.time() - start_time))

# import requests
# from bs4 import BeautifulSoup

# page = requests.get('https://anchor.fm/s/28fef6f0/podcast/rss')
# soup = BeautifulSoup(page.text, 'xml')
# original_variable = soup.find('language').text

# import re

# def convert_language_variable(language_variable):
#     # Define el patrón de búsqueda utilizando expresiones regulares
#     pattern = r'^(en)$|([a-z]{2})[-_]?([a-z]{2})?$'

#     # Intenta hacer el reemplazo
#     match = re.match(pattern, language_variable)

#     if match:
#         # Si hay coincidencia con el patrón, toma la parte correspondiente del idioma
#         if match.group(1):
#             return 'en_us'
#         elif match.group(2):
#             return match.group(2)
#     else:
#         return language_variable

# print(original_variable, convert_language_variable(original_variable))

# # Variables de ejemplo
# variables_ejemplo = ['en', 'es-ar', 'fr-fr', 'pt-br']

# # Convertir las variables utilizando la función
# for variable in variables_ejemplo:
#     converted_variable = convert_language_variable(variable)
#     print(f'Variable original: {variable}, Variable convertida: {converted_variable}')



# import re

# def convert_language_variable(language_variable):
#     # Define el patrón de búsqueda utilizando expresiones regulares
#     pattern = r'^([a-z]{2})[-_]?([a-z]{2})?$'

#     # Intenta hacer el reemplazo
#     match = re.match(pattern, language_variable)

#     if match:
#         # Si hay coincidencia con el patrón, toma la primera parte del idioma (es en es-ar)
#         return match.group(1)
#     else:
#         return language_variable

# # Variables de ejemplo
# variables_ejemplo = ['en', 'es-ar', 'fr-fr', 'pt-br']

# # Convertir las variables utilizando la función
# for variable in variables_ejemplo:
#     converted_variable = convert_language_variable(variable)
#     print(f'Variable original: {variable}, Variable convertida: {converted_variable}')

# import re

# def convert_language_variable(language_variable):
#     # Define el patrón de búsqueda utilizando expresiones regulares
#     patterns = [r'^en$', r'^es$']
#     replace = ['en_us','es']

#     for i, pattern in enumerate(patterns):
#         # Si la variable tiene el valor 'en', la reemplaza por 'en_us'
#         if re.match(pattern, language_variable):
#             replace = replace[i]
#             return re.sub(pattern, replace, language_variable)
#         else:
#             return language_variable

# # Variable con valor 'en'
# # original_variable = 'en'

# # Convertir la variable utilizando la función
# converted_variable = convert_language_variable(original_variable)

# print('Variable original:', original_variable)
# print('Variable convertida:', converted_variable)




















# if 'es' in '<![CDATA[es-ar]]>':
#     print('felicida')

# import time

# start_time = time.time()
# import whisper


# model = whisper.load_model("small")
# result = model.transcribe("347678490-44100-2-f6b021f57fae4.mp3")

# with open('result2.txt', 'w') as f:
#     f.write(result['text'])

# print("--- %s seconds ---" % (time.time() - start_time))
# print(result["text"])

# import requests

# r = requests.get('https://feeds.megaphone.fm/huberanlab')
# if r.status_code == 200:
#     print('Felicida')
# else:
#     print(':c')


# import chainlit as cl
# from chainlit.input_widget import Select


# @cl.on_chat_start
# async def start():
#     settings = await cl.ChatSettings(
#         [
#             Select(
#                 id="Model",
#                 label="OpenAI - Model",
#                 values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
#                 initial_index=0,
#                 tooltip = 'Select the model'
#             )
#         ]
#     ).send()
#     value = settings["Model"]


# from app import get_result

# get_result('How can I be more present?')































# import chainlit as cl
# from chainlit.input_widget import Select, Switch, TextInput

# import requests
# import os
# # from podcast_downloader.podcast import Podcast

# generated_file_path = './generated_files'
# audio_path = f'{generated_file_path}/audio_files'
# speaker_id = '21m00Tcm4TlvDq8ikWAM'

# def llm_output(message):
#     return message   

# def get_audio(text:str, chat_id:int, speaker_id:str, api_key:str):
#     if not os.path.exists(audio_path):
#         os.makedirs(audio_path)

#     output_path = f'{audio_path}/audio_{chat_id}.mp3'

#     CHUNK_SIZE = 1024
#     url = f"https://api.elevenlabs.io/v1/text-to-speech/{speaker_id}"

#     headers = {
#     "Accept": "audio/mpeg",
#     "Content-Type": "application/json",
#     "xi-api-key": api_key,
#     }

#     data = {
#     "text": text,
#     "model_id": "eleven_multilingual_v2",
#     "voice_settings": {
#         "stability": 0.5,
#         "similarity_boost": 0.5
#     }
#     }

#     response = requests.post(url, json=data, headers=headers)
#     with open(output_path, 'wb') as f:
#         for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
#             if chunk:
#                 f.write(chunk)

#     return output_path

# @cl.on_chat_start
# async def start():

#     languages_with_emojis = [
#     ("English (USA)", "🇺🇸"),
#     ("English (UK)", "🇬🇧"),
#     ("English (Australia)", "🇦🇺"),
#     ("English (Canada)", "🇨🇦"),
#     ("Japanese", "🇯🇵"),
#     ("Chinese", "🇨🇳"),
#     ("German", "🇩🇪"),
#     ("Hindi", "🇮🇳"),
#     ("French (France)", "🇫🇷"),
#     ("French (Canada)", "🇨🇦"),
#     ("Korean", "🇰🇷"),
#     ("Portuguese (Brazil)", "🇧🇷"),
#     ("Portuguese (Portugal)", "🇵🇹"),
#     ("Italian", "🇮🇹"),
#     ("Spanish (Spain)", "🇪🇸"),
#     ("Spanish (Mexico)", "🇲🇽"),
#     ("Indonesian", "🇮🇩"),
#     ("Dutch", "🇳🇱"),
#     ("Turkish", "🇹🇷"),
#     ("Filipino", "🇵🇭"),
#     ("Polish", "🇵🇱"),
#     ("Swedish", "🇸🇪"),
#     ("Bulgarian", "🇧🇬"),
#     ("Romanian", "🇷🇴"),
#     ("Arabic (Saudi Arabia)", "🇸🇦"),
#     ("Arabic (UAE)", "🇦🇪"),
#     ("Czech", "🇨🇿"),
#     ("Greek", "🇬🇷"),
#     ("Finnish", "🇫🇮"),
#     ("Croatian", "🇭🇷"),
#     ("Malay", "🇲🇾"),
#     ("Slovak", "🇸🇰"),
#     ("Danish", "🇩🇰"),
#     ("Tamil", "🇮🇳"),
#     ("Ukrainian", "🇺🇦")
# ]

#     # Create a formatted string with emojis
#     languages = [f"{emoji} {language}" for language, emoji in languages_with_emojis]
    # settings = await cl.ChatSettings(
    #     [
    #         TextInput(id="PodcastName", label="Podcast name"),
    #         TextInput(id="RSS", label="Podcast RSS Feed URL"),

    #         Switch(id="text_to_speech", label="Text to speech", initial=False),
    #         TextInput(id="eleven_labs_api_key", label="Eleven Labs API Key for Text to Speech"),
            
    #         Select(
    #             id="Model",
    #             label="Model",
    #             values=["Llama-2-7B-Chat-GGML", "gpt-3.5-turbo"],
    #             initial_index=0,
    #         ),
    #         TextInput(id="gpt_api_key", label="OpenAI API Key for GPT Model"),  
    #     ]
    # ).send()

    # cl.user_session.set('settings', settings)
    # cl.user_session.set('audios', [])

# @cl.on_settings_update
# async def setup_agent(settings):
#     podcast_name = cl.user_session.get('PodcastName')
#     rss = cl.user_session.get('RSS')
#     # podcast = Podcast(podcast_name, rss)


# @cl.on_message
# async def main(message, message_id):    
#     eleven_labs_api_key = cl.user_session.get('settings')['eleven_labs_api_key']
#     open_ai_api_key = cl.user_session.get('settings')['gpt_api_key']
    
#     output = llm_output(message)

#     if cl.user_session.get('settings')['text_to_speech']:
#         if eleven_labs_api_key != None:
#             output_path = get_audio(output, message_id, speaker_id, eleven_labs_api_key)
#             cl.user_session.get('audios').append(cl.Audio(path=output_path, display='inline'))
#             audios = cl.user_session.get('audios')
#             await cl.Message(
#                 content='Generated audio',
#                 elements=[audios[-1]],
#             ).send()
#         else:
#             await cl.Message(content='Enter a valid Eleven Labs API Key').send()
#     else:
#         if cl.user_session.get('settings')['Model'] == 'Llama-2-7B-Chat-GGML':
#             await cl.Message(content=output).send()
#         elif cl.user_session.get('settings')['Model'] == 'gpt-3.5-turbo': 
#             if open_ai_api_key != None:
#                 await cl.Message(content=output + " GPT").send()
#             else:
#                 await cl.Message(content='Enter a valid Open API Key').send()




