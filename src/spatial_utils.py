"""
Spatial and geometric utilities for map matching.
Provides projection from spherical GPS (WGS84) to local metric planar coordinates,
point-to-segment orthogonal projection, and geodesic distances.
"""

import math
import numpy as np

EARTH_RADIUS = 6371000.0  # meters

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Computes Great-Circle (Haversine) distance between two GPS points in meters.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS * c

def local_projection_factors(ref_lat):
    """
    Returns meters per degree of longitude and latitude at reference latitude.
    """
    lat_rad = math.radians(ref_lat)
    meters_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    meters_per_deg_lon = (math.pi / 180.0) * EARTH_RADIUS * math.cos(lat_rad)
    return meters_per_deg_lon, meters_per_deg_lat

def project_to_meters(lons, lats, ref_lon, ref_lat):
    """
    Project (lon, lat) arrays to local Cartesian (x, y) in meters relative to reference.
    """
    m_per_lon, m_per_lat = local_projection_factors(ref_lat)
    x = (np.asarray(lons) - ref_lon) * m_per_lon
    y = (np.asarray(lats) - ref_lat) * m_per_lat
    return x, y

def point_to_segment_project(px, py, x1, y1, x2, y2):
    """
    Computes orthogonal projection of point (px, py) onto line segment (x1, y1)-(x2, y2).
    Returns:
        dist: Distance in meters from (px, py) to the nearest point on the segment.
        t: Normalized fraction [0, 1] along the segment.
        proj_x, proj_y: Coordinates of the projected point.
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-12:
        dist = math.hypot(px - x1, py - y1)
        return dist, 0.0, x1, y1

    # Fractional projection parameter
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, t))
    
    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy
    dist = math.hypot(px - proj_x, py - proj_y)
    
    return dist, t_clamped, proj_x, proj_y
