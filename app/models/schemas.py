from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from enum import Enum

class CrimeType(str, Enum):
    ASSAULT = "assault"
    BURGLARY = "burglary"
    THEFT = "theft"
    VANDALISM = "vandalism"
    ROBBERY = "robbery"
    VEHICLE_THEFT = "vehicle_theft"
    FRAUD = "fraud"

class Location(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)

class CrimeIncident(BaseModel):
    id: Optional[str] = None
    # allow either a known CrimeType enum or a free-form string (defensive)
    crime_type: Union[CrimeType, str]
    location: Location
    timestamp: datetime
    description: Optional[str] = None
    severity: int = Field(default=1, ge=1, le=5)
    area_name: Optional[str] = None

class PatrolPoint(BaseModel):
    location: Location
    priority: int = Field(default=1, ge=1, le=10)
    estimated_time: int  # minutes

class HotspotZone(BaseModel):
    center: Location
    radius: float  # in kilometers
    crime_count: int
    crime_types: Dict[str, int]
    risk_score: float
    polygon: List[Location]

class GeoJSONFeature(BaseModel):
    type: str = Field(default="Feature")
    geometry: dict
    properties: dict

class GeoJSONRoute(GeoJSONFeature):
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v != "Feature":
            raise ValueError("Type must be 'Feature'")
        return v

    @classmethod
    def validate_geometry(cls, v: dict) -> dict:
        if v.get('type') != 'LineString':
            raise ValueError("Geometry type must be 'LineString'")
        if not isinstance(v.get('coordinates', []), list):
            raise ValueError("coordinates must be a list")
        return v

class PatrolRoute(BaseModel):
    route_id: str
    waypoints: List[Location]
    total_distance: float  # in kilometers
    estimated_duration: int  # in minutes
    priority_zones: List[str]
    route_type: str  # 'optimized', 'hotspot_focused', 'balanced'


class UserCreate(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: str
    username: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str
    exp: Optional[int] = None