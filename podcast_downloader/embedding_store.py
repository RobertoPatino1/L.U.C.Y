from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os

import sys
sys.path.append('./')

class EmbeddingStore:

    def __init__(self, init_text:str, store_name:str, path:str, host_documents:bool):
        self.store_name = store_name
        self.path = path
        self.host_documents = host_documents
        self.init_text = init_text
        
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                                 model_kwargs={"device": "cpu"})
        
        self.vectorStore = self.load_embeddings['faiss_index']

    def add_texts(self, texts_to_add:list):
        # Create a dictionary containing the metadata
        previous_metadata = self.load_embeddings()
        self.vectorStore.add_texts(texts_to_add)
        texts = previous_metadata['texts'] + texts_to_add
        metadata = {
            'store_name': self.store_name,
            'is_document': self.host_documents,
            'embeddings_model_name': self.embeddings.model_name,
            'texts': texts,
            'faiss_index': self.vectorStore.serialize_to_bytes()  # Serialize the FAISS index
        }

        with open(f"{self.path}/faiss_{self.store_name}.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load_embeddings(self):
        if not os.exists(f"{self.path}/faiss_{self.store_name}.pkl"):
            if not self.host_documents:
                faiss_index = FAISS.from_texts(self.init_text, self.embeddings)
            else:
                faiss_index = FAISS.from_documents(self.init_text, self.embeddings)
        else:
            with open(f"{self.path}/faiss_{self.store_name}.pkl", "rb") as f:
                metadata = pickle.load(f)
        
            # Deserialize the FAISS index
            faiss_index = FAISS.deserialize_from_bytes(metadata['faiss_index'], self.embeddings)

        return {
            'store_name': self.store_name,
            'path': self.path,
            'is_document': self.host_documents,
            'embeddings_model_name': self.embeddings.model_name,
            'texts': metadata['texts'],
            'faiss_index': faiss_index
        }
        
        
    