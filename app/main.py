from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, cast
from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import json
from uuid import uuid4

from .core.config import settings
from .core.analysis import calculate_hotspots, nearest_neighbor_route, calculate_distance
from .core.risk_analysis import risk_analyzer
from .core.logging_config import configure_logging
from .models.schemas import (
    CrimeIncident, 
    CrimeType, 
    Location, 
    HotspotZone, 
    PatrolPoint,
    PatrolRoute,
    GeoJSONRoute
)
from .models.orm import CrimeORM, RouteORM
from .db import SessionLocal, engine, Base
from sqlalchemy.orm import Session
from collections import Counter
from typing import cast as _cast

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# hotspot cache (derived, kept in-memory for now)
hotspot_cache: List[HotspotZone] = []

# Create required directories
os.makedirs(settings.ROUTES_DIR, exist_ok=True)
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)


@app.on_event("startup")
def on_startup():
    """Create DB tables on startup (development only)"""
    configure_logging()
    Base.metadata.create_all(bind=engine)


# Some runtimes (e.g. tests using TestClient) may not run the startup events.
# Create tables at import time as a safe fallback for test environments.
try:
    Base.metadata.create_all(bind=engine)
except Exception:
    # ignore DB creation errors at import time (e.g., permission issues)
    pass


@app.middleware("http")
async def log_requests(request: Request, call_next):
    import logging
    logger = logging.getLogger("app.requests")
    logger.info(f"Incoming request {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Completed {request.method} {request.url} -> {response.status_code}")
    return response

def _to_aware_utc(dt: datetime) -> Optional[datetime]:
    """Convert datetime to UTC timezone-aware"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def orm_to_pydantic(r) -> CrimeIncident:
    """Convert a SQLAlchemy ORM CrimeORM instance into a CrimeIncident Pydantic model."""
    try:
        ct = CrimeType(r.crime_type)
    except Exception:
        ct = str(r.crime_type)

    lat = float(cast(Any, r.lat)) if r.lat is not None else 0.0
    lng = float(cast(Any, r.lng)) if r.lng is not None else 0.0
    timestamp = cast(Any, r.timestamp) if r.timestamp else datetime.now(timezone.utc)
    description = str(r.description) if r.description is not None else None
    severity = int(cast(Any, r.severity)) if r.severity is not None else 1
    area_name = str(r.area_name) if r.area_name is not None else None

    return CrimeIncident(
        id=str(r.id),
        crime_type=ct,
        location=Location(lat=lat, lng=lng),
        timestamp=timestamp,
        description=description,
        severity=severity,
        area_name=area_name,
    )

# === ROUTE ENDPOINTS ===

@app.post("/api/routes")
async def save_route_geojson(route: GeoJSONRoute):
    """Save a patrol route as GeoJSON"""
    route_id = str(uuid4())
    filename = f"{route_id}.geojson"
    filepath = os.path.join(settings.ROUTES_DIR, filename)
    
    route.properties['route_id'] = route_id
    if 'created_at' not in route.properties:
        route.properties['created_at'] = datetime.now(timezone.utc).isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(route.dict(), f, indent=2)
    return {"route_id": route_id, "message": "Route saved successfully"}

@app.get("/api/routes/{route_id}")
async def get_route(route_id: str):
    """Get a specific route by ID"""
    filepath = os.path.join(settings.ROUTES_DIR, f"{route_id}.geojson")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Route not found")
    
    with open(filepath, 'r') as f:
        return json.load(f)

@app.get("/api/routes")
async def list_routes():
    """List all saved routes"""
    if not os.path.exists(settings.ROUTES_DIR):
        return []
        
    routes = []
    for filename in os.listdir(settings.ROUTES_DIR):
        if not filename.endswith('.geojson'):
            continue
        filepath = os.path.join(settings.ROUTES_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                route = json.load(f)
                routes.append(route)
        except Exception as e:
            print(f"Error reading route {filename}: {e}")
    
    return routes

@app.delete("/api/routes/{route_id}")
async def delete_route(route_id: str):
    """Delete a route by ID"""
    filepath = os.path.join(settings.ROUTES_DIR, f"{route_id}.geojson")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Route not found")
    
    try:
        os.remove(filepath)
        return {"message": "Route deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting route: {str(e)}")

# === CRIME INCIDENT ENDPOINTS (DB-backed) ===

@app.post("/api/crimes", response_model=CrimeIncident)
def add_crime_incident(crime: CrimeIncident, db: Session = Depends(get_db)):
    """Add a new crime incident to the database (stored via SQLAlchemy)"""
    # generate id if not provided
    cid = crime.id or f"CRIME_{uuid4().hex[:8]}"

    crime_orm = CrimeORM(
        id=cid,
        crime_type=crime.crime_type.value if isinstance(crime.crime_type, CrimeType) else str(crime.crime_type),
        lat=crime.location.lat,
        lng=crime.location.lng,
        timestamp=crime.timestamp,
        description=crime.description,
        severity=crime.severity,
        area_name=crime.area_name,
    )

    db.add(crime_orm)
    db.commit()
    db.refresh(crime_orm)
    hotspot_cache.clear()

    return orm_to_pydantic(crime_orm)


@app.get("/api/crimes", response_model=List[CrimeIncident])
def get_crimes(
    crime_type: Optional[CrimeType] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(default=1000, le=10000),
    db: Session = Depends(get_db),
):
    """Retrieve crime incidents with filters from the DB"""
    query = db.query(CrimeORM)

    if crime_type:
        query = query.filter(CrimeORM.crime_type == crime_type.value)

    if start_date:
        sd = _to_aware_utc(start_date)
        if sd is not None:
            query = query.filter(CrimeORM.timestamp >= sd)

    if end_date:
        ed = _to_aware_utc(end_date)
        if ed is not None:
            query = query.filter(CrimeORM.timestamp <= ed)

    rows = query.order_by(CrimeORM.timestamp.desc()).limit(limit).all()

    result: List[CrimeIncident] = []
    for r in rows:
        try:
            result.append(orm_to_pydantic(r))
        except Exception as e:
            print(f"Error converting crime {r.id}: {e}")
            continue

    return result


@app.delete("/api/crimes/{crime_id}")
def delete_crime(crime_id: str, db: Session = Depends(get_db)):
    """Delete a crime incident from DB"""
    row = db.query(CrimeORM).filter(CrimeORM.id == crime_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Crime not found")
    db.delete(row)
    db.commit()
    hotspot_cache.clear()
    return {"message": "Crime deleted successfully"}

# === HOTSPOT AND RISK ANALYSIS ENDPOINTS ===

@app.get("/api/hotspots", response_model=List[HotspotZone])
async def get_hotspots(
    recalculate: bool = False,
    eps_km: float = Query(default=settings.EPS_KM_CLUSTERING, ge=0.1, le=5.0),
    min_samples: int = Query(default=settings.MIN_SAMPLES_CLUSTERING, ge=2, le=20),
    days_back: int = Query(default=settings.DEFAULT_DAYS_BACK, ge=1, le=365)
):
    """Get crime hotspots"""
    if hotspot_cache and not recalculate:
        return hotspot_cache

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)

    from sqlalchemy.orm import Session as _Session
    db = SessionLocal()
    try:
        rows = db.query(CrimeORM).filter(CrimeORM.timestamp >= cutoff_date).all()
        recent_crimes = [orm_to_pydantic(r) for r in rows]
    finally:
        db.close()

    hotspots = calculate_hotspots(recent_crimes, eps_km, min_samples)
    hotspot_cache.clear()
    hotspot_cache.extend(hotspots)

    return hotspots

@app.get("/api/risk-analysis")
def get_risk_analysis(area_name: str, time_window: Optional[int] = 24, db: Session = Depends(get_db)):
    """Get risk analysis for a specific area"""
    try:
        rows = db.query(CrimeORM).filter(CrimeORM.area_name == area_name).all()
        if not rows:
            return {
                "area_name": area_name,
                "risk_score": 0,
                "risk_category": "Low",
                "predicted_crimes": 0,
                "crime_trend": "stable"
            }
        
        area_crimes = [orm_to_pydantic(r) for r in rows]
        return risk_analyzer.calculate_area_risk(area_crimes, area_name, time_window)
    except Exception as e:
        print(f"Error in risk-analysis: {e}")
        return {
            "area_name": area_name,
            "risk_score": 0,
            "risk_category": "Low",
            "predicted_crimes": 0,
            "crime_trend": "stable"
        }

@app.get("/api/risk-predictions")
def get_risk_predictions(db: Session = Depends(get_db)):
    """Get risk predictions for all areas"""
    try:
        areas_rows = db.query(CrimeORM.area_name).filter(CrimeORM.area_name != None).distinct().all()
        areas = [a[0] for a in areas_rows if a[0]]
        
        if not areas:
            return {
                "total_areas": 0,
                "realtime_updated_areas": 0,
                "risk_predictions": []
            }
        
        predictions = []
        for area in areas:
            try:
                rows = db.query(CrimeORM).filter(CrimeORM.area_name == area).all()
                if not rows:
                    continue
                    
                area_crimes = [orm_to_pydantic(r) for r in rows]
                risk_data = risk_analyzer.calculate_area_risk(area_crimes, area, time_window=24)
                
                # Ensure all values are JSON serializable
                safe_risk_data = {
                    "area_name": str(risk_data.get("area_name", area)),
                    "risk_score": float(risk_data.get("risk_score", 0)),
                    "risk_category": str(risk_data.get("risk_category", "Low")),
                    "predicted_crimes": int(risk_data.get("predicted_crimes", 0)),
                    "crime_rate_per_1000": float(risk_data.get("crime_rate_per_1000", 0)),
                    "population": int(risk_data.get("population", 100000)),
                    "police_stations": int(risk_data.get("police_stations", 2)),
                    "has_realtime_update": bool(risk_data.get("has_realtime_update", False)),
                    "crime_trend": str(risk_data.get("crime_trend", "stable"))
                }
                predictions.append(safe_risk_data)
            except Exception as e:
                print(f"Error processing area {area}: {e}")
                continue

        return {
            "total_areas": len(areas),
            "realtime_updated_areas": sum(1 for p in predictions if p.get('crime_trend') != 'stable'),
            "risk_predictions": sorted(predictions, key=lambda x: x['risk_score'], reverse=True)
        }
    except Exception as e:
        print(f"Error in risk-predictions: {e}")
        import traceback
        traceback.print_exc()
        return {
            "total_areas": 0,
            "realtime_updated_areas": 0,
            "risk_predictions": []
        }


# === STATUS AND HEALTH ENDPOINTS ===

@app.get("/api/status")
def get_status(db: Session = Depends(get_db)):
    """Return system status for frontend"""
    try:
        # count total crimes
        total_rows = db.query(CrimeORM).count()
        areas_rows = db.query(CrimeORM.area_name).filter(CrimeORM.area_name != None).distinct().all()
        total_areas = len([a[0] for a in areas_rows if a[0]])
        
        # count new crimes in last 24 hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_rows = db.query(CrimeORM).filter(CrimeORM.timestamp >= cutoff).all()
        new_count = len(recent_rows)
        recent_areas = len({_cast(Any, r.area_name) for r in recent_rows if _cast(Any, r.area_name) is not None})
        
        return {
            'api': 'running',
            'historical_data': {
                'total_crimes': total_rows,
                'total_areas': total_areas
            },
            'realtime_status': {
                'new_crimes_reported': new_count,
                'areas_affected': recent_areas,
                'last_update': datetime.now(timezone.utc).isoformat()
            }
        }
    except Exception as e:
        print(f"Error in status endpoint: {e}")
        return {
            'api': 'running',
            'historical_data': {'total_crimes': 0, 'total_areas': 0},
            'realtime_status': {
                'new_crimes_reported': 0,
                'areas_affected': 0,
                'last_update': datetime.now(timezone.utc).isoformat()
            }
        }


@app.get("/api/health")
def health_check(db: Session = Depends(get_db)):
    """Basic health endpoint for monitoring"""
    try:
        from sqlalchemy import text
        _ = db.execute(text("SELECT 1")).first()
        db_ok = True
    except Exception:
        db_ok = False

    return {"status": "ok", "db_ok": db_ok}


@app.get("/api/realtime-crimes")
def get_realtime_crimes(db: Session = Depends(get_db)):
    """Get crimes from last 24 hours"""
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        rows = db.query(CrimeORM).filter(CrimeORM.timestamp >= cutoff).order_by(CrimeORM.timestamp.desc()).all()
        crimes = []
        for r in rows:
            try:
                crime_dict = orm_to_pydantic(r).dict()
                # Ensure timestamp is serializable
                if isinstance(crime_dict.get('timestamp'), datetime):
                    crime_dict['timestamp'] = crime_dict['timestamp'].isoformat()
                crimes.append(crime_dict)
            except Exception as e:
                print(f"Error converting crime {r.id}: {e}")
                continue
                
        areas = len({r.get('area_name') for r in crimes if r.get('area_name')})
        return {
            'total_new_crimes': len(crimes),
            'crimes': crimes,
            'areas_affected': areas,
            'last_update': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        print(f"Error in realtime-crimes: {e}")
        return {
            'total_new_crimes': 0,
            'crimes': [],
            'areas_affected': 0,
            'last_update': datetime.now(timezone.utc).isoformat()
        }


@app.get("/api/realtime-stats")
def get_realtime_stats(db: Session = Depends(get_db)):
    """Get real-time statistics for the last 24 hours"""
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        rows = db.query(CrimeORM).filter(CrimeORM.timestamp >= cutoff).all()
        
        crime_types = Counter([str(r.crime_type) for r in rows if r.crime_type])
        areas = [str(r.area_name) for r in rows if r.area_name]
        area_counts = Counter(areas)
        
        return {
            'total_new_crimes': len(rows),
            'areas_affected': len(set(areas)),
            'crime_types': dict(crime_types),
            'top_affected_areas': dict(area_counts.most_common(5)),
            'last_update': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        print(f"Error in realtime-stats: {e}")
        return {
            'total_new_crimes': 0,
            'areas_affected': 0,
            'crime_types': {},
            'top_affected_areas': {},
            'last_update': datetime.now(timezone.utc).isoformat()
        }


@app.get("/api/crime-stats")
def get_crime_stats(db: Session = Depends(get_db)):
    """Return aggregated historical and realtime stats"""
    try:
        all_rows = db.query(CrimeORM).order_by(CrimeORM.timestamp.desc()).all()
        
        if not all_rows:
            return {
                'historical': {
                    'total_crimes': 0,
                    'total_areas': 0,
                    'crime_types': {},
                    'top_crime_areas': {}
                },
                'realtime': {
                    'new_crimes_count': 0,
                    'total_new_crimes': 0,
                    'areas_affected': 0,
                    'crime_types': {}
                },
                'combined': {'trend': 'stable'}
            }
        
        total_crimes = len(all_rows)
        areas = [str(r.area_name) for r in all_rows if r.area_name is not None]
        total_areas = len(set(areas))

        # historical crime types
        hist_counter = Counter([str(r.crime_type) for r in all_rows if r.crime_type])
        historical = {
            'total_crimes': total_crimes,
            'total_areas': total_areas,
            'crime_types': dict(hist_counter),
            'top_crime_areas': {},
        }

        # top crime areas
        area_counter = Counter([str(r.area_name) if r.area_name else 'Unknown' for r in all_rows])
        historical['top_crime_areas'] = dict(area_counter.most_common(10))

        # realtime window (24h)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent = [r for r in all_rows if r.timestamp and r.timestamp >= cutoff]
        recent_counter = Counter([str(r.crime_type) for r in recent if r.crime_type])
        recent_area_counter = Counter([str(r.area_name) for r in recent if r.area_name is not None])

        realtime = {
            'new_crimes_count': len(recent),
            'total_new_crimes': len(recent),
            'areas_affected': len(recent_area_counter),
            'crime_types': dict(recent_counter),
        }

        # simple trend heuristic
        if all_rows and all_rows[-1].timestamp:
            days_diff = (datetime.now(timezone.utc) - all_rows[-1].timestamp).days
        else:
            days_diff = 1
            
        avg_per_day = total_crimes / max(1, days_diff or 1)
        combined = {
            'trend': 'increasing' if len(recent) > avg_per_day else 'stable'
        }

        return {
            'historical': historical,
            'realtime': realtime,
            'combined': combined
        }
    except Exception as e:
        print(f"Error in crime-stats: {e}")
        import traceback
        traceback.print_exc()
        return {
            'historical': {
                'total_crimes': 0,
                'total_areas': 0,
                'crime_types': {},
                'top_crime_areas': {}
            },
            'realtime': {
                'new_crimes_count': 0,
                'total_new_crimes': 0,
                'areas_affected': 0,
                'crime_types': {}
            },
            'combined': {'trend': 'stable'}
        }


@app.post("/api/report-crime")
def report_crime(payload: Dict[str, Any], db: Session = Depends(get_db)):
    """Accept a crime report and return prediction"""
    try:
        cid = payload.get('id') or f"CRIME_{uuid4().hex[:8]}"
        crime_type = payload.get('crime_type') or payload.get('crimeType') or 'unknown'
        lat = float(payload.get('lat') or payload.get('location', {}).get('lat') or payload.get('coordinates', {}).get('lat', 0.0))
        lng = float(payload.get('lng') or payload.get('location', {}).get('lng') or payload.get('coordinates', {}).get('lon', 0.0))
        area_name = payload.get('area_name') or payload.get('area')
        desc = payload.get('description') or payload.get('notes')
        severity = int(payload.get('severity', 1))

        crime_orm = CrimeORM(
            id=cid,
            crime_type=str(crime_type),
            lat=lat,
            lng=lng,
            timestamp=datetime.now(timezone.utc),
            description=desc,
            severity=severity,
            area_name=area_name,
        )

        db.add(crime_orm)
        db.commit()
        db.refresh(crime_orm)
        hotspot_cache.clear()

        # compute area risk
        rows = db.query(CrimeORM).filter(CrimeORM.area_name == area_name).all() if area_name else [crime_orm]
        area_crimes = [orm_to_pydantic(r) for r in rows]
        prediction = risk_analyzer.calculate_area_risk(area_crimes, area_name or 'Unknown', time_window=24)

        return {
            'crime_id': cid,
            'timestamp': crime_orm.timestamp.isoformat(),
            'prediction': prediction
        }
    except Exception as e:
        print(f"Error in report-crime: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/patrol-recommendations")
def get_patrol_recommendations(db: Session = Depends(get_db)):
    """Return patrol recommendations"""
    try:
        # ensure hotspots exist
        if not hotspot_cache:
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            rows = db.query(CrimeORM).filter(CrimeORM.timestamp >= cutoff).all()
            recent = [orm_to_pydantic(r) for r in rows]
            hs = calculate_hotspots(recent, settings.EPS_KM_CLUSTERING, settings.MIN_SAMPLES_CLUSTERING)
            hotspot_cache.clear()
            hotspot_cache.extend(hs)

        priority_hotspots = []
        patrol_suggestions = []
        for idx, h in enumerate(hotspot_cache[:10], start=1):
            area_name = getattr(h, 'area_name', f'hotspot_{idx}')
            priority_hotspots.append({
                'center_lat': h.center.lat,
                'center_lon': h.center.lng,
                'crime_count': h.crime_count,
                'risk_score': h.risk_score,
                'area_name': area_name
            })

            suggestion = {
                'priority': idx,
                'type': 'hotspot',
                'area': area_name,
                'reason': f'{h.crime_count} recent incidents; risk {h.risk_score:.1f}',
                'coordinates': {'lat': h.center.lat, 'lon': h.center.lng},
                'recommended_time': 'IMMEDIATE' if h.risk_score >= 7.5 else 'SCHEDULE'
            }
            patrol_suggestions.append(suggestion)

        # current time period
        hour = datetime.now().hour
        if 6 <= hour < 12:
            period = 'morning'
        elif 12 <= hour < 17:
            period = 'afternoon'
        elif 17 <= hour < 21:
            period = 'evening'
        else:
            period = 'night'

        return {
            'current_time_period': period,
            'patrol_suggestions': patrol_suggestions,
            'priority_hotspots': priority_hotspots,
        }
    except Exception as e:
        print(f"Error in patrol-recommendations: {e}")
        import traceback
        traceback.print_exc()
        return {
            'current_time_period': 'unknown',
            'patrol_suggestions': [],
            'priority_hotspots': [],
        }


@app.get("/api/area/{area_name}")
def get_area_details(area_name: str, db: Session = Depends(get_db)):
    """Get detailed information about a specific area"""
    try:
        rows = db.query(CrimeORM).filter(CrimeORM.area_name == area_name).all()
        
        if not rows:
            return {'error': f'No data found for area: {area_name}'}
        
        area_crimes = [orm_to_pydantic(r) for r in rows]
        
        # Historical data
        hist_crime_types = Counter([str(r.crime_type) for r in rows if r.crime_type])
        
        # Time period analysis
        time_periods = {'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0}
        for r in rows:
            if r.timestamp and hasattr(r.timestamp, 'hour'):
                hour = r.timestamp.hour
                if 6 <= hour < 12:
                    time_periods['morning'] += 1
                elif 12 <= hour < 17:
                    time_periods['afternoon'] += 1
                elif 17 <= hour < 21:
                    time_periods['evening'] += 1
                else:
                    time_periods['night'] += 1
        
        # Real-time data (last 24 hours)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_rows = [r for r in rows if r.timestamp and r.timestamp >= cutoff]
        realtime_crime_types = Counter([str(r.crime_type) for r in recent_rows if r.crime_type])
        
        last_crime = None
        if recent_rows:
            latest = max(recent_rows, key=lambda x: x.timestamp if x.timestamp else datetime.min.replace(tzinfo=timezone.utc))
            last_crime = {
                'id': str(latest.id),
                'crime_type': str(latest.crime_type),
                'timestamp': latest.timestamp.isoformat() if latest.timestamp else datetime.now(timezone.utc).isoformat()
            }
        
        # Calculate risk
        risk_data = risk_analyzer.calculate_area_risk(area_crimes, area_name, time_window=24)
        
        # Population info defaults
        pop_info = {
            'population': 100000,
            'area_sq_km': 10.0,
            'police_stations': 2,
            'avg_response_time': 15
        }
        
        return {
            'area_name': area_name,
            'historical_data': {
                'total_crimes': len(rows),
                'crime_types': dict(hist_crime_types),
                'time_period_crimes': time_periods
            },
            'realtime_data': {
                'new_crimes_today': len(recent_rows),
                'new_crime_types': dict(realtime_crime_types),
                'last_crime': last_crime,
                'status': 'Active' if recent_rows else 'Inactive'
            },
            'population_info': pop_info,
            'risk_prediction': risk_data,
            'last_update': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        print(f"Error in area details: {e}")
        import traceback
        traceback.print_exc()
        return {'error': f'Error retrieving data for area: {area_name}'}


# === PATROL OPTIMIZATION ENDPOINTS ===

@app.post("/api/optimize-patrol")
async def optimize_patrol_route(
    start_point: Location,
    patrol_points: List[PatrolPoint],
    optimization_type: str = "balanced"
):
    """Optimize patrol route"""
    try:
        waypoints = [p.location for p in patrol_points]
        route, total_distance = nearest_neighbor_route(start_point, waypoints)
        
        # Calculate estimated duration (assuming average speed of 30 km/h)
        duration = int((total_distance / 30) * 60)  # convert to minutes
        
        # Get nearby hotspots
        nearby_hotspots = [
            h for h in hotspot_cache
            if any(calculate_distance(loc, h.center) <= h.radius for loc in route)
        ]
        
        patrol_route = PatrolRoute(
            route_id=str(uuid4()),
            waypoints=route,
            total_distance=total_distance,
            estimated_duration=duration,
            priority_zones=[f"hotspot_{i}" for i, _ in enumerate(nearby_hotspots)],
            route_type=optimization_type
        )
        
        return patrol_route
    except Exception as e:
        print(f"Error in optimize-patrol: {e}")
        raise HTTPException(status_code=500, detail=str(e))