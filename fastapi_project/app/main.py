from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import shutil
from services.search import add_documents_from_files, semantic_search, answer_question

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploaded_files"


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload files, extract text, and index them into FAISS.
    """
    saved_paths = []
    
    try:
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)
        
        response = add_documents_from_files(saved_paths)
        return {"message": "Files uploaded successfully!", "indexing_response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.post("/search/")
async def search_documents(query: str = Form(...)):

    try:
        results = semantic_search(query)
        return {"query": query, "results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/ask/")
async def ask_question(query: str = Form(...)):

    try:
        response = answer_question(query)
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")
