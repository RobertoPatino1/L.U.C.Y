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

# custom_prompt_template = """
# Utilisez les informations suivantes pour répondre à la question de l'utilisateur.
# Si vous ne connaissez pas la réponse, dites simplement que vous ne la savez pas, n'essayez pas d'inventer une réponse.

# Contexte: {context}
# Demande: {question}

# Renvoyez simplement la réponse utile ci-dessous et rien de plus.
# Réponse utile:"""


    
# # Testeos
# from functools import cache
# @cache
# def load_pipeline(src:str, dst:str):
#     task_name = f"translation_{src}_to_{dst}"
#     model_name = f"Helsinki-NLP/opus-mt-{src}-{dst}"
#     translator  = pipeline(task_name, model=model_name, tokenizer=model_name)
#     return translator

# @cache
# def load_llm():
#     llm = CTransformers(
#         model = "TheBloke/Llama-2-7B-Chat-GGML",
#         model_type="llama",
#         max_new_tokens = 512,
#         temperature = 0.5
#     )
    
#     return llm

# def get_ts_message(message:str, src:str, dst:str):
#     if src != dst:
#         translator = load_pipeline(src, dst)
#         return translator(message)[0]["translation_text"]
#     else:
#         return message

# podcast = Podcast('Confidentiel','https://www.rtl.fr/podcast/confidentiel.xml')
# # output function
# def update_data(message, **kwargs):
#     # Traducir mensaje
#     with open('podcast.json', 'r') as f:
#         podcast_data = json.load(f)

#     src = podcast_data['src']
#     dst = podcast_data['language']

#     ts_message = get_ts_message(message, src, dst)
#     # Actualizar description_embeddings del podcast
#     podcast.update_description_embeddings() 
#     # Obtengo la metadata de top_limit = 2 episodios con mayor similitud
#     podcast_items = podcast.search_items(ts_message, **kwargs)
#     episodes_metadata = get_episodes_metadata(podcast_items)
#     for episode in episodes_metadata:
#         url, title = episode
#         # Actualizar paragraph_embbeddings
#         podcast.update_paragraph_embeddings(title, url)

# def qa_bot():
#     embeddings = load_embeddings()
#     db = FAISS.load_local(podcast.db_faiss_path, embeddings)
#     llm = load_llm()
#     qa_prompt = set_custom_prompt()
    
#     qa = retrieval_qa_chain(llm, qa_prompt, db)

#     return qa

# def final_result(query):
#     # Traducir mensaje
#     with open('podcast.json', 'r') as f:
#         podcast_data = json.load(f)

#     src = podcast_data['src']
#     dst = podcast_data['language']

#     ts_message = get_ts_message(query, src, dst)
    
#     qa_result = qa_bot()
#     response = qa_result({'query': ts_message})
#     response['result'] = get_ts_message(response['result'], dst, src)

#     return response

# # def get_result(message):
# #     start_time = time.time()
# #     update_data(message, k=2)
# #     if not openai_use:
# #         response = final_result(message)
# #     else:
# #         with get_openai_callback() as cb:
# #             response = final_result(message)
# #         print(cb)
# #     print(response)

# def get_result(message):
#     # update_data(message, k=2)
#     response = final_result(message)
#     return response


    
    
    