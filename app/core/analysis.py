from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
from ..models.schemas import Location, CrimeIncident, HotspotZone
from enum import Enum
from sklearn.cluster import DBSCAN

def calculate_distance(loc1: Location, loc2: Location) -> float:
    """Calculate distance between two locations in km"""
    lat1, lng1 = np.radians([loc1.lat, loc1.lng])
    lat2, lng2 = np.radians([loc2.lat, loc2.lng])
    
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return 6371 * c  # Earth radius in km

def create_circle_polygon(lat: float, lng: float, radius_km: float, num_points: int = 16) -> List[Location]:
    """Create a circular polygon around a point"""
    polygon = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        dx = radius_km * np.cos(angle) / 111.0
        # guard against division by zero near poles
        cos_lat = np.cos(np.radians(lat))
        denom = 111.0 * cos_lat if abs(cos_lat) > 1e-6 else 111.0
        dy = radius_km * np.sin(angle) / denom
        polygon.append(Location(lat=lat + dx, lng=lng + dy))
    return polygon

def calculate_hotspots(crimes: List[CrimeIncident], eps_km: float = 0.5, min_samples: int = 3) -> List[HotspotZone]:
    """Calculate crime hotspots using DBSCAN clustering"""
    if len(crimes) < min_samples:
        return []
    
    # Extract coordinates
    coords = np.array([[c.location.lat, c.location.lng] for c in crimes])
    
    # DBSCAN clustering (eps in km converted to degrees approximately)
    eps_deg = eps_km / 111.0  # rough conversion
    clustering = DBSCAN(eps=eps_deg, min_samples=min_samples).fit(coords)
    
    hotspots = []
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:  # Skip noise
            continue
        
        cluster_mask = clustering.labels_ == cluster_id
        cluster_crimes = [c for i, c in enumerate(crimes) if cluster_mask[i]]
        cluster_coords = coords[cluster_mask]
        
        # Calculate center
        center_lat = float(np.mean(cluster_coords[:, 0]))
        center_lng = float(np.mean(cluster_coords[:, 1]))
        
        # Calculate radius (max distance from center)
        distances = np.sqrt(np.sum((cluster_coords - [center_lat, center_lng])**2, axis=1))
        radius = float(np.max(distances) * 111.0)  # Convert to km
        
        # Count crime types
        crime_type_counts: Dict[str, int] = {}
        for crime in cluster_crimes:
            # crime.crime_type may be an enum or a string; handle both
            if isinstance(crime.crime_type, Enum):
                key = crime.crime_type.value
            else:
                key = str(crime.crime_type)
            crime_type_counts[key] = crime_type_counts.get(key, 0) + 1
        
        # Calculate risk score
        severity_sum = sum(c.severity for c in cluster_crimes)
        risk_score = (len(cluster_crimes) * 0.5 + severity_sum * 0.5) / max(len(cluster_crimes), 1)
        
        # Create polygon
        polygon = create_circle_polygon(center_lat, center_lng, radius)
        
        hotspot = HotspotZone(
            center=Location(lat=center_lat, lng=center_lng),
            radius=radius,
            crime_count=len(cluster_crimes),
            crime_types=crime_type_counts,
            risk_score=min(risk_score, 10.0),
            polygon=polygon
        )
        hotspots.append(hotspot)
    
    return sorted(hotspots, key=lambda h: h.risk_score, reverse=True)

def nearest_neighbor_route(start: Location, waypoints: List[Location]) -> Tuple[List[Location], float]:
    """Simple nearest neighbor TSP heuristic"""
    if not waypoints:
        return [start], 0.0
    
    route = [start]
    remaining = waypoints.copy()
    total_distance = 0.0
    
    current = start
    while remaining:
        # Find nearest point
        nearest = min(remaining, key=lambda p: calculate_distance(current, p))
        distance = calculate_distance(current, nearest)
        
        total_distance += distance
        route.append(nearest)
        remaining.remove(nearest)
        current = nearest
    
    return route, total_distance