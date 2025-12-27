import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """Embed a list of documents"""
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        """Embed a single query"""
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

#Download the Embeddings
def download_hugging_face_embeddings():
    return SentenceTransformerEmbeddings()