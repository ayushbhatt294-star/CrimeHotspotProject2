from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..db import Base

class CrimeORM(Base):
    __tablename__ = 'crimes'

    id = Column(String(64), primary_key=True, index=True)
    crime_type = Column(String(50), nullable=False)
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    description = Column(Text, nullable=True)
    severity = Column(Integer, default=1)
    area_name = Column(String(128), nullable=True)

class RouteORM(Base):
    __tablename__ = 'routes'

    route_id = Column(String(64), primary_key=True, index=True)
    geojson = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class UserORM(Base):
    __tablename__ = 'users'

    id = Column(String(64), primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
