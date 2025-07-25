# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tempfile
import os
import shutil
from pathlib import Path
import sqlite3

# Import your iris system class from the new file
from iris_system import FixedSimpleIrisSystem

app = FastAPI()
iris_system = FixedSimpleIrisSystem()

# Serve your HTML file - FIXED ENCODING
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("iris_ui.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/enroll")
async def enroll_user(
    subject_id: str = Form(...),
    eye: str = Form(...),
    image: UploadFile = File(...)
):
    file_path = None
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Save uploaded file temporarily
        file_path = uploads_dir / f"temp_{image.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Call iris system enrollment
        success = iris_system.enroll_user(str(file_path), subject_id, eye)
        
        if success:
            return {
                "success": True,
                "message": f"User {subject_id}_{eye} enrolled successfully",
                "subject_id": subject_id,
                "eye": eye
            }
        else:
            raise HTTPException(status_code=400, detail="Enrollment failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if file_path and file_path.exists():
            file_path.unlink()

@app.post("/authenticate")
async def authenticate_user(image: UploadFile = File(...)):
    file_path = None
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Save uploaded file temporarily
        file_path = uploads_dir / f"temp_auth_{image.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Call iris system authentication
        success, matched_id, confidence = iris_system.authenticate_user(str(file_path))
        
        return {
            "success": success,
            "matched_id": matched_id,
            "confidence": confidence,
            "hamming_distance": 1 - confidence,
            "threshold": iris_system.threshold
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if file_path and file_path.exists():
            file_path.unlink()

@app.get("/users")
async def list_users():
    try:
        # Get enrolled users from database
        conn = sqlite3.connect(iris_system.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT subject_id, eye, image_path, feature_hash, created_at FROM iris_codes ORDER BY created_at')
        users = cursor.fetchall()
        conn.close()
        
        user_list = []
        for subject_id, eye, image_path, feature_hash, created_at in users:
            user_list.append({
                "subject_id": subject_id,
                "eye": eye,
                "image_path": Path(image_path).name,
                "feature_hash": feature_hash,
                "created_at": created_at
            })
        
        return {"users": user_list}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_user/{subject_id}/{eye}")
async def delete_user(subject_id: str, eye: str):
    try:
        conn = sqlite3.connect(iris_system.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM iris_codes WHERE subject_id = ? AND eye = ?', (subject_id, eye))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        conn.commit()
        conn.close()
        
        return {"message": f"User {subject_id}_{eye} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear_all_users")
async def clear_all_users():
    try:
        conn = sqlite3.connect(iris_system.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM iris_codes')
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return {"message": f"All {deleted_count} users cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))