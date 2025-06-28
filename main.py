import os
import uuid
from datetime import datetime

import torch
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import uvicorn

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
llm_client = OpenAI(api_key=OPENAI_API_KEY)

# Setup Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "customer-service-kb"

# Recreate index if exists
existing_indexes = [index.name for index in pinecone.list_indexes()]
if INDEX_NAME in existing_indexes:
    pinecone.delete_index(INDEX_NAME)

pinecone.create_index(
    name=INDEX_NAME,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)

kb_index = pinecone.Index(INDEX_NAME)

# Load knowledge base CSV
csv_path = r'C:\Users\Acer\Desktop\Company\Cyfuture\data\data\complaint_knowledge.csv'
df = pd.read_csv(csv_path)
texts = df['text'].tolist() if 'text' in df.columns else df.iloc[:, 0].tolist()
embeddings = embedding_model.encode(texts).tolist()

# ✅ Batch upsert to avoid 2MB limit
def batch_upsert(index, texts, embeddings, batch_size=100):
    for i in range(0, len(texts), batch_size):
        batch = [
            (str(j), embeddings[j], {"text": texts[j]})
            for j in range(i, min(i + batch_size, len(texts)))
        ]
        index.upsert(vectors=batch)

batch_upsert(kb_index, texts, embeddings)
print("Knowledge base uploaded to Pinecone.")

# In-memory complaint store
complaint_db = {}

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "RAG Chatbot is running. Visit /docs to test the API."}

# Pydantic models
class ComplaintRequest(BaseModel):
    name: str
    phone_number: str = Field(..., pattern=r"^\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$")
    email: EmailStr
    complaint_details: str

class ComplaintResponse(BaseModel):
    complaint_id: str
    name: str
    phone_number: str
    email: str
    complaint_details: str
    created_at: str

@app.post("/complaints")
def create_complaint(complaint: ComplaintRequest):
    complaint_id = str(uuid.uuid4())[:8].upper()
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    complaint_db[complaint_id] = {
        "complaint_id": complaint_id,
        "name": complaint.name,
        "phone_number": complaint.phone_number,
        "email": complaint.email,
        "complaint_details": complaint.complaint_details,
        "created_at": created_at,
    }
    return {
        "complaint_id": complaint_id,
        "message": "Complaint created successfully"
    }

@app.get("/complaints/{complaint_id}", response_model=ComplaintResponse)
def get_complaint(complaint_id: str):
    if complaint_id not in complaint_db:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return complaint_db[complaint_id]

@app.post("/chat")
def chatbot(query: str = Query(..., description="User query")):
    # Step 1: Try to find a matching complaint
    matched_complaint = None
    for cid, complaint in complaint_db.items():
        if cid in query or complaint["phone_number"] in query or complaint["email"] in query:
            matched_complaint = complaint
            break

    # Step 2: If complaint found, generate a follow-up response
    if matched_complaint:
        context = f"""
Complaint ID: {matched_complaint['complaint_id']}
Name: {matched_complaint['name']}
Phone Number: {matched_complaint['phone_number']}
Email: {matched_complaint['email']}
Details: {matched_complaint['complaint_details']}
Created At: {matched_complaint['created_at']}
"""
        prompt = f"""The following is a customer's previous complaint and their latest query. 
Respond as a helpful support assistant, referencing their earlier complaint if relevant.
If they are following up, be empathetic and offer the next steps or status.

Context:
{context}

User: {query}
Response:"""
        try:
            completion = llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            return {"response": completion.choices[0].message.content}
        except Exception as e:
            return {"error": f"LLM failure: {str(e)}"}

    # Step 3: If no complaint matched, fallback to knowledge base RAG
    try:
        query_embedding = embedding_model.encode([query]).tolist()[0]
        results = kb_index.query(vector=query_embedding, top_k=3, include_metadata=True)
        context = "\n".join([match['metadata']['text'] for match in results['matches']])

        prompt = f"""Use the following knowledge base to answer the user's question. 
If it sounds like a new complaint, politely ask for their name, phone number, email, and complaint details.

Context:
{context}

User: {query}
Response:"""

        completion = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        return {"error": f"RAG fallback failed: {str(e)}"}

# Run server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


# Run using:
# uvicorn main:app --reload
# uvicorn rag_bot:app --reload
