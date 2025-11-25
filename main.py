from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
from datetime import datetime, timedelta
from typing import Optional
import os
from dotenv import load_dotenv

from database import SessionLocal, engine, Base
from models import User, NoiseReading
from schemas import UserCreate, UserLogin, UserResponse, Token, UserUpdate, NoiseReadingCreate, NoiseReadingResponse
from auth import verify_password, get_password_hash, create_access_token, verify_token
from tempfile import NamedTemporaryFile
import shutil
from pathlib import Path
from typing import List

# Lazy import model to keep startup light; it will load on first request
ana_model = None

# Load environment variables
load_dotenv()

# Create database tables
Base.metadata.create_all(bind=engine)

# Create uploads directory if it doesn't exist
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ANA Backend API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get current user
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    username = verify_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

@app.get("/")
async def root():
    return {"message": "ANA Backend API is running!"}

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    place_name: Optional[str] = Form(None),
    address: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    global ana_model
    tmp_path = None

    try:
        # 1) Save upload to a temporary file
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # 2) Lazy import model module
        if ana_model is None:
            from saved_models import model as ana_model_module
            ana_model = ana_model_module

        # 3) Run model
        result = ana_model.predict_audio(tmp_path)
        if not result:
            raise HTTPException(status_code=500, detail="Model failed to analyze audio")

        # 4) Clean up location fields (safe even if None)
        place_name_value = place_name if (place_name and place_name.strip()) else None
        address_value = address if (address and address.strip()) else None

        # 5) Save ONLY derived data (no raw audio path)
        noise_reading = NoiseReading(
            user_id=current_user.id,
            db_level=result["db_level"],
            a_weighted_db=result.get("a_weighted_db"),
            weighted_db=result.get("weighted_db"),
            npi_score=result.get("npi_score"),
            noise_type=result["noise_type"],
            confidence=result["confidence"],
            pollution_category=result["pollution_category"],
            health_impact=result.get("health_impact"),
            recommendation=result.get("recommendation"),
            latitude=latitude or 0.0,
            longitude=longitude or 0.0,
            place_name=place_name_value,
            address=address_value,
            audio_file_path=None,  # IMPORTANT: do not keep file reference
            created_at=datetime.now(),
        )

        db.add(noise_reading)
        db.commit()
        db.refresh(noise_reading)

        # Add reading id to response
        result["reading_id"] = noise_reading.id
        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /analyze: {e}")
        raise HTTPException(status_code=500, detail="Error analyzing audio")
    finally:
        # 6) Always clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.post("/auth/signup", response_model=UserResponse)
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Enforce bcrypt password length constraint (<=72 bytes)
    if len(user.password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=400,
            detail="Password too long for bcrypt (max 72 bytes). Please use a shorter password."
        )
    # Check if user already exists
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    # Check if email already exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        created_at=datetime.utcnow()
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse(
        id=db_user.id,
        username=db_user.username,
        email=db_user.email,
        created_at=db_user.created_at
    )

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    # Authenticate user
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    
    return Token(access_token=access_token, token_type="bearer")

@app.get("/auth/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at
    )

@app.put("/auth/update", response_model=UserResponse)
async def update_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile information"""
    
    # If updating password, verify current password
    if user_update.new_password:
        if not user_update.current_password:
            raise HTTPException(
                status_code=400,
                detail="Current password is required to set new password"
            )
        if not verify_password(user_update.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=400,
                detail="Current password is incorrect"
            )
        # Update password
        if len(user_update.new_password.encode("utf-8")) > 72:
            raise HTTPException(
                status_code=400,
                detail="Password too long for bcrypt (max 72 bytes)"
            )
        current_user.hashed_password = get_password_hash(user_update.new_password)
    
    # Update username if provided
    if user_update.username and user_update.username != current_user.username:
        # Check if username already exists
        existing_user = db.query(User).filter(
            User.username == user_update.username,
            User.id != current_user.id
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Username already taken"
            )
        current_user.username = user_update.username
    
    # Update email if provided
    if user_update.email and user_update.email != current_user.email:
        # Check if email already exists
        existing_user = db.query(User).filter(
            User.email == user_update.email,
            User.id != current_user.id
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Email already taken"
            )
        current_user.email = user_update.email
    
    db.commit()
    db.refresh(current_user)
    
    print(f"User updated: {current_user.username} - Email: {current_user.email}")
    
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at
    )

@app.get("/history", response_model=List[NoiseReadingResponse])
async def get_history(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get recent noise readings for the current user"""
    readings = db.query(NoiseReading)\
        .filter(NoiseReading.user_id == current_user.id)\
        .order_by(NoiseReading.created_at.desc())\
        .limit(limit)\
        .all()
    return readings

@app.get("/history/recent", response_model=List[NoiseReadingResponse])
async def get_recent_history(
    limit: int = 3,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the most recent noise readings for the current user (default: 3)"""
    readings = db.query(NoiseReading)\
        .filter(NoiseReading.user_id == current_user.id)\
        .order_by(NoiseReading.created_at.desc())\
        .limit(limit)\
        .all()
    return readings

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
