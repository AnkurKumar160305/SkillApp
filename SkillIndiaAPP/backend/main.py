from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import sys
from .ml_models import recommender
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Skill Development App API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SkillRequest(BaseModel):
    skills: str

@app.on_event("startup")
async def startup_event():
    print("Startup: Initializing Recommender System...")
    try:
        recommender.load_data()
        print("Startup: Recommender System initialized successfully.")
    except Exception as e:
        print(f"Startup ERROR: {str(e)}")

@app.get("/api/health")
def health_check():
    import os
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jobs_exists = os.path.exists(os.path.join(base_path, "Naukri_Jobs_Data.csv"))
    courses_exists = os.path.exists(os.path.join(base_path, "udemy_courses.csv"))
    artifacts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "artifacts.pkl")
    artifacts_exists = os.path.exists(artifacts_path)
    
    return {
        "status": "online",
        "cwd": os.getcwd(),
        "base_path": base_path,
        "files": {
            "jobs_csv": jobs_exists,
            "courses_csv": courses_exists,
            "artifacts_pkl": artifacts_exists
        },
        "recommender": {
            "jobs_loaded": len(recommender.jobs_df) > 0,
            "courses_loaded": len(recommender.courses_df) > 0,
            "best_model": recommender.best_model_name
        }
    }

@app.get("/api")
def read_root():
    return {"message": "Skill Development App API is running"}

@app.post("/api/recommend_jobs")
def get_jobs(request: SkillRequest):
    try:
        if not recommender.jobs_df and not recommender.load_artifacts():
             recommender.load_data()
        
        recommendations = recommender.recommend_jobs(request.skills)
        return {"recommendations": recommendations}
    except Exception as e:
        print(f"Error in /api/recommend_jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend_courses")
def get_courses(request: SkillRequest):
    try:
        if not recommender.courses_df and not recommender.load_artifacts():
             recommender.load_data()

        recommendations = recommender.recommend_courses(request.skills)
        return {"recommendations": recommendations}
    except Exception as e:
        print(f"Error in /api/recommend_courses: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model_info")
def get_model_info():
    try:
        info = recommender.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))