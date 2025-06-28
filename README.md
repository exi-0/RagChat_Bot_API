🛠️ RAG-Based Complaint Management Chatbot
A FastAPI-powered chatbot that supports complaint management and retrieval using Retrieval-Augmented Generation (RAG), powered by Pinecone, OpenAI, and SentenceTransformers.
Video Link:
https://drive.google.com/file/d/1DDuY9vSYW-N4PxL0vgu7ppwmsHa5bEkA/view?usp=sharing
🚀 Features
📌 File and retrieve complaints via REST API

🔎 Semantic search using Pinecone vector index

💬 Chat interface with context-aware responses

🧠 LLM fallback using OpenAI GPT (RAG style)

🧾 CSV-based knowledge base ingestion

🔐 Phone, email validation with Pydantic

🌐 CORS-enabled for frontend integration

🧱 Tech Stack
Layer	Tools Used
Backend API	FastAPI
LLM	OpenAI (gpt-3.5-turbo)
Embeddings	SentenceTransformers
Vector DB	Pinecone
Data Parsing	pandas
Deployment	Uvicorn

📂 Project Structure
bash
Copy
Edit
📁 project-root/
├── main.py                 # FastAPI server + logic
├── .env                   # Secrets (API keys)
├── complaint_knowledge.csv # Knowledge base (CSV)
└── README.md              # This file
🧪 API Endpoints
Method	Endpoint	Description
GET	/	Health check
POST	/complaints	Create a complaint
GET	/complaints/{id}	Retrieve complaint by ID
POST	/chat?query=...	Get chatbot response (context-aware)

📥 Setup Instructions
1. Clone and Install Dependencies
bash
Copy
Edit
git clone https://github.com/your-username/rag-complaint-chatbot.git
cd rag-complaint-chatbot
pip install -r requirements.txt
2. Prepare .env File
Create a .env file with the following content:

env
Copy
Edit
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
3. Place Your Knowledge Base
Ensure complaint_knowledge.csv is placed in the correct path:

kotlin
Copy
Edit
C:\Users\Acer\Desktop\Company\Cyfuture\data\data\complaint_knowledge.csv
4. Run the App
bash
Copy
Edit
uvicorn main:app --reload
Visit Swagger UI: http://127.0.0.1:8000/docs

📊 Example Chat Flow
User: Internet not working since morning
Bot: I'm sorry to hear that. Could you please provide your name, phone number, email, and complaint details so we can assist further?

📌 Notes
The app recreates the Pinecone index on startup (for dev use).

All complaint records are stored in-memory; use a database for production.

Handles both retrieval (RAG) and complaint matching via ID/phone/email.

🙌 Acknowledgements
Pinecone

OpenAI

SentenceTransformers

FastAPI
