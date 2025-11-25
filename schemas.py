from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class NoiseReadingCreate(BaseModel):
    db_level: float
    a_weighted_db: Optional[float] = None
    weighted_db: Optional[float] = None
    npi_score: Optional[float] = None
    noise_type: str
    confidence: float
    pollution_category: str
    health_impact: Optional[str] = None
    recommendation: Optional[str] = None
    latitude: float
    longitude: float
    place_name: Optional[str] = None
    address: Optional[str] = None

class NoiseReadingResponse(BaseModel):
    id: int
    user_id: int
    db_level: float
    a_weighted_db: Optional[float] = None
    weighted_db: Optional[float] = None
    npi_score: Optional[float] = None
    noise_type: str
    confidence: float
    pollution_category: str
    health_impact: Optional[str] = None
    recommendation: Optional[str] = None
    latitude: float
    longitude: float
    place_name: Optional[str] = None
    address: Optional[str] = None
    audio_file_path: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True