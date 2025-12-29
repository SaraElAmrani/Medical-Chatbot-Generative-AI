# DermaAI - Intelligent Skin Disease Detection API

DermaAI is a multi-model system for skin disease classification with intelligent routing, powered by FastAPI, CNN models, and a RAG (Retrieval-Augmented Generation) system for medical explanations.

## 🚀 Key Features
- **Multi-Model Analysis**: Uses dual CNN models for accurate skin disease detection (General vs. Cancer specific).
- **Intelligent Routing**: Automatically routes images to the most appropriate model based on initial analysis.
- **RAG System**: Provides detailed medical explanations using Pinecone vector database and OpenAI.
- **FastAPI Backend**: High-performance, asynchronous API.

---

## 🛠️ Prerequisites
- **Python**: 3.10+
- **Anaconda** (Recommended for environment management)

---

## 📥 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Medical-Chatbot-Generative-AI
```

### 2. Create a Conda Environment
```bash
conda create -n medicalChatbot python=3.10 -y
conda activate medicalChatbot
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables
Create a `.env` file in the root directory and add your API keys:

```ini
PINECONE_API_KEY="your_pinecone_api_key"
OPENAI_API_KEY="your_openai_api_key"
```

### 5. Initialize Knowledge Base (First Run Only)
If you need to populate the vector database with medical data:
```bash
python store_index.py
```

---

## 🏃‍♂️ How to Run

### Option 1: Quick Start (Windows)
Use the provided batch script to start the server and run a connectivity test:
```bash
run_test.bat
```
*(This will start the server on port 8001 and wait for it to be ready)*

### Option 2: Manual Start
Run the FastAPI server using Uvicorn:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```
The API will be available at `http://localhost:8001`.

---

## 🧪 Testing

### Integration Tests
To verify the full pipeline (Image -> CNN -> RAG -> Response):
```bash
python test_integration.py
```

### Quick Server Check
```bash
python test_server.py
```

---

## 📚 API Documentation

Once the server is running, explore the interactive API docs:
- **Swagger UI**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **ReDoc**: [http://localhost:8001/redoc](http://localhost:8001/redoc)

---

## 🚢 AWS CI/CD Deployment (GitHub Actions)

### 1. IAM User Permissions
Ensure the IAM user has the following policies:
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

### 2. ECR Repository
Create an ECR repository to store the Docker images:
- URI Example: `970547337635.dkr.ecr.ap-south-1.amazonaws.com/medicalchatbot`

### 3. EC2 Setup (Ubuntu)
1. Launch an Ubuntu EC2 instance.
2. Install Docker:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```
3. Configure as a self-hosted runner in GitHub Settings > Actions > Runners.

### 4. GitHub Secrets
Add the following secrets to your GitHub repository:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `ECR_REPO`
- `PINECONE_API_KEY`
- `OPENAI_API_KEY`