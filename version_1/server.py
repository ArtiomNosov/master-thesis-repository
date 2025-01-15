from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import sqlite3
import asyncio
from transformers import pipeline

# FastAPI instance
app = FastAPI()

# Database initialization
def init_db():
    conn = sqlite3.connect("candidates.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            resume TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES jobs (id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Models for API
class JobAndResumes(BaseModel):
    job_description: str
    resumes: List[str]

class RetrainRequest(BaseModel):
    model_name: str

# Dummy BERT pipeline
bert_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Async function for BERT model processing
async def process_with_model(job_description: str, resumes: List[str]):
    results = []
    for resume in resumes:
        await asyncio.sleep(0.1)  # Simulate processing time
        result = bert_model(resume, candidate_labels=[job_description])
        results.append(result)
    return results

# API Endpoints
@app.post("/submit")
async def submit_job_and_resumes(data: JobAndResumes):
    conn = sqlite3.connect("candidates.db")
    cursor = conn.cursor()

    # Insert job description
    cursor.execute("INSERT INTO jobs (description) VALUES (?)", (data.job_description,))
    job_id = cursor.lastrowid

    # Insert resumes
    for resume in data.resumes:
        cursor.execute("INSERT INTO resumes (job_id, resume) VALUES (?, ?)", (job_id, resume))

    conn.commit()
    conn.close()

    # Process data with the BERT model asynchronously
    results = await process_with_model(data.job_description, data.resumes)

    return {"job_id": job_id, "results": results}

@app.post("/retrain")
async def retrain_model(request: RetrainRequest):
    # Simulate retraining process
    await asyncio.sleep(2)  # Dummy retrain time
    return {"message": f"Model {request.model_name} retrained successfully"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
