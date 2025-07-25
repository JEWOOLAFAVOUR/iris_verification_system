# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tempfile
import os
import shutil
from pathlib import Path
import sqlite3
import uuid
from datetime import datetime

# Import your iris system class from the new file
from iris_system import FixedSimpleIrisSystem

app = FastAPI()
iris_system = FixedSimpleIrisSystem()

# Create directories for permanent storage
UPLOADS_DIR = Path("uploads")
ENROLLED_IMAGES_DIR = Path("enrolled_images")
UPLOADS_DIR.mkdir(exist_ok=True)
ENROLLED_IMAGES_DIR.mkdir(exist_ok=True)

# Serve static files (images)
app.mount("/enrolled_images", StaticFiles(directory="enrolled_images"), name="enrolled_images")

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
    temp_file_path = None
    permanent_file_path = None
    
    try:
        # Generate unique filename for permanent storage
        file_extension = Path(image.filename).suffix if image.filename else '.jpg'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        permanent_filename = f"{subject_id}_{eye}_{timestamp}{file_extension}"
        permanent_file_path = ENROLLED_IMAGES_DIR / permanent_filename
        
        # Save uploaded file temporarily for processing
        temp_file_path = UPLOADS_DIR / f"temp_{uuid.uuid4()}{file_extension}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Call iris system enrollment with temp file
        success = iris_system.enroll_user(str(temp_file_path), subject_id, eye)
        
        if success:
            # Move temp file to permanent storage only if enrollment succeeds
            shutil.move(str(temp_file_path), str(permanent_file_path))
            temp_file_path = None  # Prevent deletion in finally block
            
            # Update database with permanent image path
            conn = sqlite3.connect(iris_system.db_path)
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE iris_codes SET image_path = ? WHERE subject_id = ? AND eye = ?',
                (str(permanent_file_path), subject_id, eye)
            )
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "message": f"User {subject_id}_{eye} enrolled successfully",
                "subject_id": subject_id,
                "eye": eye,
                "image_path": permanent_filename
            }
        else:
            raise HTTPException(status_code=400, detail="Enrollment failed")
            
    except Exception as e:
        # Clean up permanent file if enrollment failed
        if permanent_file_path and permanent_file_path.exists():
            permanent_file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()

@app.post("/authenticate")
async def authenticate_user(image: UploadFile = File(...)):
    file_path = None
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Save uploaded file temporarily
        file_extension = Path(image.filename).suffix if image.filename else '.jpg'
        file_path = uploads_dir / f"temp_auth_{uuid.uuid4()}{file_extension}"
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
            # Convert absolute path to relative URL for frontend
            image_filename = Path(image_path).name if image_path else None
            image_url = f"/enrolled_images/{image_filename}" if image_filename else None
            
            user_list.append({
                "subject_id": subject_id,
                "eye": eye,
                "image_path": Path(image_path).name if image_path else "No image",
                "image_url": image_url,
                "feature_hash": feature_hash[:16] + "..." if feature_hash else "N/A",  # Truncate hash for display
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
        
        # Get image path before deletion
        cursor.execute('SELECT image_path FROM iris_codes WHERE subject_id = ? AND eye = ?', (subject_id, eye))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="User not found")
        
        image_path = result[0]
        
        # Delete from database
        cursor.execute('DELETE FROM iris_codes WHERE subject_id = ? AND eye = ?', (subject_id, eye))
        conn.commit()
        conn.close()
        
        # Delete image file if it exists
        if image_path and Path(image_path).exists():
            Path(image_path).unlink()
        
        return {"message": f"User {subject_id}_{eye} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear_all_users")
async def clear_all_users():
    try:
        conn = sqlite3.connect(iris_system.db_path)
        cursor = conn.cursor()
        
        # Get all image paths before deletion
        cursor.execute('SELECT image_path FROM iris_codes')
        image_paths = cursor.fetchall()
        
        # Delete all records
        cursor.execute('DELETE FROM iris_codes')
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        # Delete all image files
        for (image_path,) in image_paths:
            if image_path and Path(image_path).exists():
                Path(image_path).unlink()
        
        return {"message": f"All {deleted_count} users cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))