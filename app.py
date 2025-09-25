from fastapi import FastAPI, Query, Request
import time
import os
from rag_pipeline import load_data, create_vectorstore, get_qa

app = FastAPI()
qa = None  # global QA chain

# Logging request time
@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"⏱️ {request.url.path} took {duration:.2f}s")
    return response

# Load RAG pipeline on startup
@app.on_event("startup")
def startup():
    global qa
    print("🔄 Starting up...")
    if os.path.exists("vectorstore"):
        vectordb = create_vectorstore(docs=None)
    else:
        docs = load_data("my_data.csv")
        vectordb = create_vectorstore(docs)
    qa = get_qa(vectordb)
    print("✅ QA system ready.")

# Query endpoint
@app.get("/ask")
def ask(query: str = Query(..., description="Your question")):
    try:
        answer = qa.run(query)
        return {"query": query, "answer": answer}
    except Exception as e:
        return {"error": str(e)}
