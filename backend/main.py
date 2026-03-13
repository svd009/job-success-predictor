from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from backend.scorer import get_top_recommendations, compute_score, extract_skills, JOB_DESCRIPTIONS

app = FastAPI(title="Job Application Success Predictor", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ResumeRequest(BaseModel):
    resume_text: str
    job_title: Optional[str] = None

class SingleJobRequest(BaseModel):
    resume_text: str
    job_id: str

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/score")
def score_resume(req: ResumeRequest):
    if len(req.resume_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Resume text too short")
    result = get_top_recommendations(req.resume_text)
    return {"success": True, **result}

@app.post("/score/single")
def score_single(req: SingleJobRequest):
    jd = next((j for j in JOB_DESCRIPTIONS if j["id"] == req.job_id), None)
    if not jd:
        raise HTTPException(status_code=404, detail="Job not found")
    result = compute_score(req.resume_text, jd["description"])
    return {"success": True, "job": jd["title"], "company": jd["company"], **result}

@app.get("/jobs")
def list_jobs():
    return {"jobs": [{"id":j["id"],"title":j["title"],"company":j["company"]} for j in JOB_DESCRIPTIONS]}
