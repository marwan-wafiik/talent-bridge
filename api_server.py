# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from hybrid_matcher import HybridMatcher
from explanation_agent import ExplanationAgent
from hallucination_agent import audit_experience_timeline
from bias_auditor import audit_algorithmic_bias
from logger import log_event, log_error

app = FastAPI(title="TalentBridge AI Engine API", version="1.0.0")

class MatchRequest(BaseModel):
    job_description: str
    candidates: list[str]

class AuditRequest(BaseModel):
    job_description: str
    candidate_resume: str

@app.get("/")
def health_check():
    log_event("API", "Health check endpoint pinged.")
    return {"status": "online", "system": "TalentBridge AI Recruitment Engine"}

@app.post("/api/match")
def match_candidates(data: MatchRequest):
    try:
        log_event("HybridMatcher", f"Processing match query across {len(data.candidates)} candidates.")
        matcher = HybridMatcher(data.candidates)
        results = matcher.hybrid_match(data.job_description, top_k=5)
        
        return {
            "status": "success",
            "matches": [{"candidate_index": idx, "rrf_score": round(score, 4)} for idx, score in results]
        }
    except Exception as e:
        log_error("HybridMatcher", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audit-candidate")
def audit_candidate(data: AuditRequest):
    try:
        timeline_audit = audit_experience_timeline(data.candidate_resume)
        bias_audit = audit_algorithmic_bias(data.job_description, data.candidate_resume)
        
        return {
            "status": "success",
            "timeline_audit": timeline_audit,
            "bias_audit": bias_audit
        }
    except Exception as e:
        log_error("AuditAgent", str(e))
        raise HTTPException(status_code=500, detail=str(e))