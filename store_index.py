from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as PineconeClient
from pinecone import ServerlessSpec
from langchain_pinecone import Pinecone
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Process the data
extracted_data=load_pdf_file(data='Data/')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = PineconeClient(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# Check if index already exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index '{index_name}' created successfully!")
else:
    print(f"Index '{index_name}' already exists, skipping creation.") 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = Pinecone.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
) 