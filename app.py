from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import Pinecone
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from src.prompt import *
import os
import random

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Load existing index
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = OpenAI(temperature=0.4, max_tokens=500)

# Create RAG chain using LCEL (LangChain Expression Language)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_function(question):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    formatted_prompt = system_prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    # Handle different response formats
    if hasattr(response, 'content'):
        full_response = response.content
    else:
        full_response = str(response)

    # Parser la réponse structurée
    try:
        # Extraire la réponse médicale - essayer plusieurs formats
        reponse_start = -1
        for pattern in ["RÉPONSE :", "RÉPONSE:", "RESPONSE :", "RESPONSE:"]:
            reponse_start = full_response.find(pattern)
            if reponse_start != -1:
                break

        emotion_start = -1
        for pattern in ["EMOTION_AVATAR :", "EMOTION_AVATAR:", "EMOTION_AVATAR :", "EMOTION_AVATAR:"]:
            emotion_start = full_response.find(pattern)
            if emotion_start != -1:
                break

        if reponse_start != -1 and emotion_start != -1:
            answer = full_response[reponse_start + len("RÉPONSE :"):emotion_start].strip()
            emotion_text = full_response[emotion_start + len("EMOTION_AVATAR :"):].strip()
            # Nettoyer l'émotion
            emotion = emotion_text.upper().strip()
            # Supprimer les caractères spéciaux
            emotion = emotion.replace(":", "").replace(".", "").strip()
        else:
            # Fallback si le format n'est pas respecté
            answer = full_response
            emotion = "NEUTRE"
    except:
        answer = full_response
        emotion = "NEUTRE"

    return {"answer": answer, "emotion": emotion}

rag_chain = RunnableLambda(rag_function)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke(msg)
    print("Response : ", response["answer"])
    print("Emotion : ", response["emotion"])

    # Retourner la réponse en JSON avec l'émotion
    return jsonify({
        "reponse": response["answer"],
        "emotion_avatar": response["emotion"]
    })




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8088, debug= True)