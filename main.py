from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils import search_query_multi
from gpt_service import generate_answer_with_gpt

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def handle_query(req: QueryRequest):
    try:
        query = req.query
        print("🔍 Received query:", query)

        # File paths
        PPC_EMB_PATH = "data/ppc_section_embeddings2.json"
        CRPC_EMB_PATH = "data/crpc_embeddings.json"
        PPC_DATA_PATH = "data/ppc_data_final.json"
        CRPC_DATA_PATH = "data/CrPC.json"

        top_sections = search_query_multi(
            query=query,
            embedding_paths=[PPC_EMB_PATH, CRPC_EMB_PATH],
            section_data_paths=[PPC_DATA_PATH, CRPC_DATA_PATH],
            top_k=10
        )

        if not top_sections:
            return {"answer": "No relevant sections found.", "sections": []}

        answer = generate_answer_with_gpt(query, top_sections)
        return {"answer": answer, "sections": top_sections}

    except Exception as e:
        print("❌ ERROR:", e)
        return {"error": str(e)}
