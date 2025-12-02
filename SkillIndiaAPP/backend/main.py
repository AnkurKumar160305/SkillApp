from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from ml_models import recommender
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Skill Development App API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SkillRequest(BaseModel):
    skills: str

@app.on_event("startup")
async def startup_event():
    recommender.load_data()

@app.get("/")
def read_root():
    return {"message": "Skill Development App API is running"}

@app.post("/recommend_jobs")
def get_jobs(request: SkillRequest):
    try:
        recommendations = recommender.recommend_jobs(request.skills)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend_courses")
def get_courses(request: SkillRequest):
    try:
        recommendations = recommender.recommend_courses(request.skills)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
def get_model_info():
    try:
        info = recommender.get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



# To run write ==> python main.py