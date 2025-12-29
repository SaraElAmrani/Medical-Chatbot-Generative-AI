from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_pinecone import Pinecone
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.predict import DermaAIPredictor
from app.model_loader import ModelLoader
from src.prompt import image_analysis_prompt
from src.helper import download_hugging_face_embeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize FastAPI app
app = FastAPI(
    title="DermaAI - Intelligent Skin Disease Detection API",
    description="Multi-model system for skin disease classification with intelligent routing",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables
predictor: Optional[DermaAIPredictor] = None
embeddings = None
retriever = None
llm = None
docsearch = None

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[DermaAIPredictor] = None

# Pydantic models for request/response
class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    risk_level: str
    medical_category: str
    recommendation: str
    explanation: str
    disclaimer: str
    emotion: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    model1_available: bool
    model2_available: bool
    rag_available: bool

class TextQuery(BaseModel):
    question: str

class TextResponse(BaseModel):
    question: str
    answer: str
    emotion: str
    

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    message: str
    response: str
    
    timestamp: str

class AddDocumentRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, str]] = None

class AddDocumentResponse(BaseModel):
    status: str
    message: str
    document_id: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models and RAG on startup"""
    global predictor, embeddings, retriever, llm, docsearch
    
    print("🚀 Starting DermaAI API...")
    print("📦 Loading models...")
    
    try:
        # Initialize model loader
        model_loader = ModelLoader()
        
        # Load both models
        model1 = model_loader.load_model1()
        model2 = model_loader.load_model2()
        
        # Initialize predictor
        predictor = DermaAIPredictor(model1, model2)
        
        print("✅ Models loaded successfully!")
        
        # Load RAG components
        print("📚 Loading RAG system...")
        try:
            embeddings = download_hugging_face_embeddings()
            
            index_name = "medicalbot"
            docsearch = Pinecone.from_existing_index(
                index_name=index_name,
                embedding=embeddings
            )
            
            retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
            llm = OpenAI(temperature=0.4, max_tokens=1000)
            
            print("✅ RAG system loaded successfully!")
        except Exception as e:
            print(f"⚠️ RAG system not available: {str(e)}")
            print("🔄 Continuing without RAG - using fallback responses")
            retriever = None
            llm = None
        print("🎯 DermaAI API is ready!")
        
    except Exception as e:
        print(f"❌ Error loading systems: {str(e)}")
        raise

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "DermaAI - Intelligent Skin Disease Detection API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if predictor is not None and retriever is not None else "unhealthy",
        models_loaded=predictor is not None,
        model1_available=predictor is not None and predictor.model1 is not None,
        model2_available=predictor is not None and predictor.model2 is not None,
        rag_available=retriever is not None
    )



def generate_medical_explanation(cnn_result: Dict, user_query: str = "") -> Dict:
    """
    Generate medical explanation using CNN results and RAG
    
    Args:
        cnn_result: Results from CNN prediction
        user_query: User's question/context
        
    Returns:
        Dictionary with explanation, recommendation, disclaimer, emotion
    """
    if retriever is None or llm is None:
        # Fallback without RAG
        explanation = f"The analysis suggests {cnn_result['disease']} with {cnn_result['confidence']:.1%} confidence. This is a probabilistic assessment and not a medical diagnosis."
        recommendation = cnn_result['recommendation']
        disclaimer = "This information is for educational purposes only and is not medical advice. Please consult a healthcare professional for proper evaluation."
        emotion = "ATTENTIF"
        return {
            "explanation": explanation,
            "recommendation": recommendation,
            "disclaimer": disclaimer,
            "emotion": emotion
        }
    
    try:
        # Get relevant medical context
        query = f"{cnn_result['disease']} {cnn_result['medical_category']} skin condition"
        print(f"🔍 RAG Query: {query}")
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        print(f"📄 Retrieved RAG Context (First 500 chars):\n{context[:500]}...")
        
        # Create prompt
        prompt = image_analysis_prompt.format(
            disease=cnn_result['disease'],
            confidence=cnn_result['confidence'],
            context=context
        )
        print("🤖 Sending Prompt to LLM...")
        
        # Get LLM response
        response = llm.invoke(prompt)
        full_response = response.content if hasattr(response, 'content') else str(response)

        # Since the format has changed, we return the full response as explanation
        # The prompt itself handles the structure
        explanation = full_response
        recommendation = "Please consult a healthcare professional."
        disclaimer = "This information is educational and based on retrieved medical sources. It does not represent a medical diagnosis."
        emotion = "ATTENTIF"

        return {
            "explanation": explanation,
            "recommendation": recommendation,
            "disclaimer": disclaimer,
            "emotion": emotion
        }

    except Exception as e:
        # Fallback on error
        if 'context' in locals() and context:
             explanation = f"1. Image Analysis Summary:\n   'The image analysis suggests {cnn_result['disease']} (confidence: {cnn_result['confidence']:.2%}%).'\n\n2. Medical Definition (from knowledge base):\n   {context[:500]}...\n\n3. Disclaimer:\n   'This information is educational and based on retrieved medical sources. It does not represent a medical diagnosis.'"
        else:
             explanation = f"The analysis suggests {cnn_result['disease']} with {cnn_result['confidence']:.1%} confidence. This is a probabilistic assessment and not a medical diagnosis."
        
        recommendation = cnn_result['recommendation']
        disclaimer = "This information is for educational purposes only and is not medical advice."
        emotion = "ATTENTIF"
    
    return {
        "explanation": explanation,
        "recommendation": recommendation,
        "disclaimer": disclaimer,
        "emotion": emotion
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG)"),
    user_query: Optional[str] = Form(None, description="Optional user context/question"),
    force_cancer_model: Optional[bool] = Form(False, description="Force cancer model usage")
):
    """
    Predict skin disease from an uploaded image
    
    - **file**: Image file to analyze
    - **user_query**: Optional context (e.g., "Is this skin cancer?")
    - **force_cancer_model**: Force using the cancer detection model
    """
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only JPEG and PNG are supported."
        )
    
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Make prediction
        cnn_result = predictor.predict_from_bytes(
            image_bytes=image_bytes,
            user_query=user_query,
            force_cancer_model=force_cancer_model
        )
        
        # Generate medical explanation
        explanation_data = generate_medical_explanation(cnn_result, user_query or "")
        
        # Format response
        return PredictionResponse(
            disease=cnn_result['disease'],
            confidence=cnn_result['confidence'],
            risk_level=cnn_result['risk_level'],
            medical_category=cnn_result['medical_category'],
            recommendation=cnn_result['recommendation'],
            explanation=explanation_data['explanation'],
            disclaimer=explanation_data['disclaimer'],
            emotion=explanation_data['emotion']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/ask", response_model=TextResponse, tags=["RAG"])
async def ask_question(query: TextQuery):
    """
    Ask a medical question and get an answer from the RAG system
    
    - **question**: Your medical question (e.g., "What is melanoma?")
    
    Returns an answer based on the medical knowledge base with sources.
    """
    
    if retriever is None or llm is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG system not available. Please ensure Pinecone and OpenAI are configured."
        )
    
    try:
        # Retrieve relevant documents
        print(f"🔍 Question: {query.question}")
        docs = retriever.invoke(query.question)
        
        # Extract context and sources
        context = "\n\n".join(doc.page_content for doc in docs)
        
        
        print(f"📄 Retrieved {len(docs)} documents")
        
        # Create prompt for LLM
        prompt = f"""Based on the following medical information, answer the question clearly and concisely.

Medical Context:
{context}

Question: {query.question}

Instructions:
- Provide a clear, accurate answer based on the medical context
- Use simple language that patients can understand
- If the context doesn't contain enough information, say so
- Do not make up information

Answer:"""
        
        # Get LLM response
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Determine emotion based on content
        emotion = "ATTENTIF"  # Default emotion for medical queries
        
        print(f"✅ Answer generated successfully")
        
        return TextResponse(
            question=query.question,
            answer=answer.strip(),
            emotion=emotion,
        )
        
    except Exception as e:
        print(f"❌ Error in RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG query error: {str(e)}")

