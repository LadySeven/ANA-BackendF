from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    
    # Relationship to noise readings
    noise_readings = relationship("NoiseReading", back_populates="user")

class NoiseReading(Base):
    __tablename__ = "noise_readings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Analysis results
    db_level = Column(Float, nullable=False)
    a_weighted_db = Column(Float, nullable=True)
    weighted_db = Column(Float, nullable=True)
    npi_score = Column(Float, nullable=True)
    noise_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    pollution_category = Column(String, nullable=False)
    health_impact = Column(Text, nullable=True)
    recommendation = Column(Text, nullable=True)
    
    # Location data
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    place_name = Column(String, nullable=True)
    address = Column(Text, nullable=True)
    
    # Audio file reference
    audio_file_path = Column(String, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationship to user
    user = relationship("User", back_populates="noise_readings")
