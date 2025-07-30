from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import uvicorn

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ERP content from file
with open("erp_data.txt", "r", encoding="utf-8") as f:
    documents = f.read().split("\n")

# Initialize SentenceTransformer model and FAISS index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedding_model.encode(documents)
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings))

# Chat schema
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.message
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    context = documents[I[0][0]]

    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response_text = f"(Simulated reply based on context)\n{prompt}"
    return {"reply": response_text}    
    """
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })
    result = response.json()
    return {"reply": result.get("response", "").strip()}
    """
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
