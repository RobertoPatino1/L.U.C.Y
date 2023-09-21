import time

start_time = time.time()

from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from podcast_downloader.podcast import load_embeddings

from functools import cache

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Utilice la siguiente información para responder la pregunta del usuario.
Si no sabe la respuesta, simplemente diga que no la sabe, no intente inventar una respuesta.

Contexto: {context}
Pregunta: {question}

Solo devuelva la útil respuesta a continuación y nada más.
Respuesta útil:
"""

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
@cache
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = load_embeddings()
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

chain = qa_bot()

print("--- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
qa_bot = qa_bot()
print("--- %s seconds ---" % (time.time() - start_time))


















# start_time = time.time()
# from app import retrieval_qa_chain
# qa_chain = retrieval_qa_chain()
# print("--- %s seconds ---" % (time.time() - start_time))



# # start_time = time.time()
# # from podcast_downloader import transcriptions
# # print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# from podcast_downloader.podcast import Podcast
# print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# from podcast_downloader.podcast import load_embeddings
# print("--- %s seconds ---" % (time.time() - start_time))




# start_time = time.time()
# load_embeddings()
# print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# load_embeddings()
# print("--- %s seconds ---" % (time.time() - start_time))























# import time
# start_time = time.time()
# # from langchain.llms import CTransformers
# from functools import cache

# @cache
# def testing(model):
#     llm = CTransformers(
#         model = model,
#         model_type="llama",
#         max_new_tokens = 512,
#         temperature = 0.5
#     )
#     return llm

# llm = testing("TheBloke/Llama-2-7B-Chat-GGML")

# @cache
# def to_cache():
#     from podcast_downloader.podcast import Podcast

# start_time = time.time()
# to_cache()
# print("--- %s seconds ---" % (time.time() - start_time))
# start_time = time.time()
# to_cache()
# print("--- %s seconds ---" % (time.time() - start_time))


# from podcast_downloader.podcast import Podcast
# from langchain.embeddings import HuggingFaceEmbeddings


# # podcast = Podcast('Psicologia Al Desnudo | @psi.mammoliti', 'https://anchor.fm/s/28fef6f0/podcast/rss')





# # podcast.update_description_embeddings()
# # embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",
# #                                        model_kwargs={'device': 'cpu'})


# from podcast_downloader.podcast import Podcast
# import time
# start_time = time.time()
# # import podcast_downloader.podcast as podcast
# # from podcast_downloader.podcast import Podcast
# podcast = Podcast('Psicologia Al Desnudo | @psi.mammoliti', 'https://anchor.fm/s/28fef6f0/podcast/rss')
# print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# import podcast_downloader.podcast as podcast
# # from podcast_downloader.podcast import Podcast
# podcast = Podcast('Psicologia Al Desnudo | @psi.mammoliti', 'https://anchor.fm/s/28fef6f0/podcast/rss')
# print("--- %s seconds ---" % (time.time() - start_time))