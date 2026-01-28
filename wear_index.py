import argparse
import logging
import pathlib
import time
import requests
import polyline
import isodate
import io
import os

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageDraw, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from matplotlib import colors as mcolors
from typing import Tuple, Optional
from scipy.interpolate import make_smoothing_spline
from numba import njit
from surrogate import model_f
from shapely.geometry import Point

__version__ = 1 
logger = logging.getLogger(__name__)

GM_API_KEY = os.environ['GM_API_KEY']

# Layout constants
PAGE_WIDTH, PAGE_HEIGHT = A4  # Standard A4 size
CONTENT_WIDTH = 500
MARGIN = 50

dt = 0.01  # Time step (s)

# Vehicle and Tire Parameters
mu = 0.9  # Tire-road friction coefficient
mass = 21884  # Total vehicle mass on trailer axles (kg)
g = 9.81  # Gravity (m/s^2)
Fz = mass * g / 6  # Static vertical load per tire (N)
radius = 1  # Tire radius (m)
tyre_width = 0.3  # Tire width (m)
inertia_z = 1250925  # Yaw inertia (kg·m^2)
alpha_0 = 10  # Saturation angle in radians (10°)

# Tire wear model parameters
K_average = 1.2e-7 # s^2/m^3
K_law = 0.0125  # for wear law, not filtered parabolic function

# Fancher tyre force model
mu = 0.9
Ca0 = 3250
Ca1 = 9.57
Ca2 = 8.02e-5

# Vehicle Geometry
track_width = 2.14  # Distance between left and right wheels (m)
front_axle_distance = 6.335  # Distance from CG to front axle (m)
middle_axle_distance = 7.645  # From CG to middle axle (m)
rear_axle_distance = 8.955  # From CG to rear axle (m)
total_length = 12  # Distance from front axle to rear axle

# Wheel positions relative to Lead point (x, y) -
wheel_offsets = {
    'front_left': (front_axle_distance, +track_width / 2),
    'front_right': (front_axle_distance, -track_width / 2),
    'middle_left': (middle_axle_distance, +track_width / 2),
    'middle_right': (middle_axle_distance, -track_width / 2),
    'rear_left': (rear_axle_distance, +track_width / 2),
    'rear_right': (rear_axle_distance, -track_width / 2),
}


def wear_index(origin, destination, output_file):
    logger.info(f"Calculating wear index"
                f" from {origin} to {destination}")

    timer_start = time.perf_counter()
    # parse post codes
    start_coords = geocode(origin)
    end_coords = geocode(destination)
    # get route
    route_info = compute_route(start_coords,
                               end_coords,
                               travel_mode="DRIVE",
                               traffic=False)
    # extract coordinates
    x_coords_raw, y_coords_raw, distance = route_info_to_meters(route_info)
    # resample it as equal spacing (ds is distance in meters, lam is spline smoothing factor)
    x_coords, y_coords, kappa = resample_curve_fixed_step(
        x_coords_raw, y_coords_raw, ds=0.02, lam=100)
    # extract turns (start the turn when higher than start curvature and end when lower than end curvature)
    # add points before and after turn for full wear, merge turns with a smaller gap than min_gap
    sections = extract_sections(kappa,
                                thresh_start=0.01,
                                thresh_end=0.005,
                                pre=400,
                                post=800,
                                min_gap=1000)
    # use run_sim_fast to simulate each turn
    wheel_data = stitch_simulations(
        x_coords,
        y_coords,
        sections,
        run_sim_fast,
        dt,
        wheel_offsets=wheel_offsets,  # dict: name -> (dx, dy)
        Fz=Fz,
        K_average=K_average,
        inertia_z=inertia_z)

    logger.info(f"Calculated in {time.perf_counter() - timer_start:.3g} s")

    timer_start = time.perf_counter()
    generate_report(wheel_data, distance, x_coords, y_coords, origin,
                    destination, output_file)
    logger.info(
        f"Report generated in {time.perf_counter() - timer_start:.3g} s")


def geocode(address: str):
    """Forward geocode: address -> {lat, lng, formatted_address}"""

    GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

    params = {"address": address, "key": GM_API_KEY}
    r = requests.get(GEOCODE_URL, params=params)
    r.raise_for_status()
    data = r.json()

    if data["status"] != "OK" or not data.get("results"):
        logger.error("No geocode results for: %s", address)

    result = data["results"][0]
    return {
        "formatted_address": result["formatted_address"],
        "lat": result["geometry"]["location"]["lat"],
        "lng": result["geometry"]["location"]["lng"]
    }


def compute_route(origin: dict,
                  destination: dict,
                  travel_mode="DRIVE",
                  traffic=False):
    """
    Compute a route using Google Maps Routes API v2.

    Parameters:
        origin/destination = {"lat": float, "lng": float}
        travel_mode: "DRIVE", "WALK", "BICYCLE", "TWO_WHEELER"
        traffic: If True, uses traffic-aware routing

    Returns:
        dict with:
            - distance_meters: float
            - encoded_polyline: str
            - decoded_points: list of (lat, lng) tuples
    """

    # Add your API key here
    ROUTES_URL = f"https://routes.googleapis.com/directions/v2:computeRoutes?key={GM_API_KEY}"

    body = {
        "origin": {"location": {"latLng": {"latitude": origin["lat"], "longitude": origin["lng"]}}},
        "destination": {"location": {"latLng": {"latitude": destination["lat"], "longitude": destination["lng"]}}},
        "travelMode": travel_mode,
        "polylineQuality": "HIGH_QUALITY",
        "routingPreference": "TRAFFIC_AWARE" if traffic else "TRAFFIC_UNAWARE",
        "computeAlternativeRoutes": False,
        "routeModifiers": {
            "avoidTolls": False,
            "avoidHighways": False,
            "avoidFerries": False
        },
        "languageCode": "en-GB",
        "units": "METRIC",
    }

    headers = {
        "Content-Type": "application/json",
        # Only request what we need
        "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline"
    }

    # Send POST request
    r = requests.post(ROUTES_URL, headers=headers, json=body)
    
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e}, response: {r.text}")
        raise

    data = r.json()

    if not data.get("routes"):
        raise ValueError("No route found.")

    route = data["routes"][0]

    # Decode polyline
    encoded_poly = route["polyline"]["encodedPolyline"]
    decoded_points = polyline.decode(encoded_poly)  # [(lat, lng), ...]

    return {
        "distance_meters": route["distanceMeters"],
        "encoded_polyline": encoded_poly,
        "decoded_points": decoded_points
    }


def route_info_to_meters(route_info):
    """
    Convert route_info['decoded_points'] (list/array of [lat, lon])
    to absolute meter coordinates.
    """
    pts = np.asarray(route_info["decoded_points"], dtype=float)
    if pts.size == 0:
        return np.array([]), np.array([]), None

    # pts[:, 0] = lat, pts[:, 1] = lon
    latitudes = pts[:, 0]
    longitudes = pts[:, 1]

    # Build GeoSeries in WGS84 and reproject to an UTM zone (meters)
    gser = gpd.GeoSeries(gpd.points_from_xy(longitudes, latitudes),
                         crs="EPSG:4326")
    gser_m = gser.to_crs(crs="EPSG:3857")

    x_coords = gser_m.x.to_numpy()
    y_coords = gser_m.y.to_numpy()
    return x_coords, y_coords, route_info['distance_meters']


def resample_curve_fixed_step(x, y, ds, lam, include_endpoint=True):
    """
    Resample a 2D curve (x, y) at fixed arc-length spacing ds using
    shape-preserving parametric cubic interpolation (PCHIP).

    Notes
    -----
    - Consecutive duplicate points are removed to keep s strictly increasing.
    - The last segment may be shorter than ds if include_endpoint=True.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.size < 2:
        logger.error("x and y must have the same shape with length >= 2.")
    if not np.isfinite(ds) or ds <= 0:
        logger.error("ds must be a positive finite number.")

    # Remove NaN pairs
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.size < 2:
        logger.error("Insufficient finite points after removing NaNs.")

    # Chord-length parameter
    dx = np.diff(x)
    dy = np.diff(y)
    seg = np.hypot(dx, dy)

    # Drop consecutive duplicates (zero-length segments)
    keep = np.r_[True, seg > 0]
    x, y = x[keep], y[keep]
    if x.size < 2:
        logger.error("All points are identical after de-duplication.")

    s = np.concatenate(([0.0], np.cumsum(seg)))  # strictly increasing

    # Interpolators x(s), y(s)
    px = make_smoothing_spline(s, x, lam=lam)
    py = make_smoothing_spline(s, y, lam=lam)

    # Build fixed-step parameter grid
    s0, s1 = float(s[0]), float(s[-1])
    n = int(np.floor((s1 - s0) / ds))
    s_new = s0 + ds * np.arange(n + 1)
    if include_endpoint and (s1 - s_new[-1]) > 1e-12:
        s_new = np.append(s_new, s1)

    # Calculate curvature
    dx = px.derivative(1)(s_new)
    dy = py.derivative(1)(s_new)
    ddx = px.derivative(2)(s_new)
    ddy = py.derivative(2)(s_new)
    kappa = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy)**1.5

    # Evaluate
    x_new = px(s_new)
    y_new = py(s_new)
    return x_new, y_new, kappa


def extract_sections(kappa,
                     thresh_start=0.02,
                     thresh_end=0.01,
                     pre=400,
                     post=800,
                     min_gap=1000):
    """
    Extract (start,end) index ranges using hysteresis thresholds:
    - Start when kappa > thresh_start
    - End   when kappa < thresh_end
    Then expand by `pre` before and `post` after, merge close ranges.
    """
    n = len(kappa)
    in_section = False
    starts, ends = [], []

    for i, val in enumerate(kappa):
        if not in_section and val > thresh_start:
            in_section = True
            starts.append(i)
        elif in_section and val < thresh_end:
            in_section = False
            ends.append(i)

    # If still inside at end, close it
    if in_section:
        ends.append(n - 1)

    # Expand by pre/post
    expanded = []
    for s, e in zip(starts, ends):
        s_exp = max(0, s - pre)
        e_exp = min(n - 1, e + post)
        expanded.append((s_exp, e_exp))

    # Merge if gaps are small
    merged = []
    if expanded:
        cur_s, cur_e = expanded[0]
        for s, e in expanded[1:]:
            if s - cur_e <= min_gap:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
    return merged


def plot_sections(x, y, sections, base_alpha=0.3):
    """
    Plot the full (x,y) curve faintly, with extracted sections overlaid.

    """
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linewidth=1, alpha=base_alpha)
    for (s, e) in sections:
        plt.plot(x[s:e + 1], y[s:e + 1], linewidth=2)
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.title('High-curvature sections (overlay)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


@njit(cache=True, fastmath=True)
def rk4_step_scalar(yaw_t, yaw_rate_t, K, h):
    # dynamics for yaw angle, angular speed and acceleration
    k1_yaw, k1_rate = yaw_rate_t, K
    k2_yaw, k2_rate = yaw_rate_t + 0.5 * h * k1_rate, K
    k3_yaw, k3_rate = yaw_rate_t + 0.5 * h * k2_rate, K
    k4_yaw, k4_rate = yaw_rate_t + h * k3_rate, K

    yaw_next = yaw_t + (h / 6.0) * (k1_yaw + 2 * k2_yaw + 2 * k3_yaw + k4_yaw)
    rate_next = yaw_rate_t + (h / 6.0) * (k1_rate + 2 * k2_rate + 2 * k3_rate +
                                          k4_rate)
    return yaw_next, rate_next


def compute_wheel_velocities(u, v, yaw_rate, yaw, dx, dy):
    # Vehicle velocity at wheel location due to rotation
    ry = dx * np.cos(yaw) - dy * np.sin(yaw)
    rx = dx * np.sin(yaw) + dy * np.cos(yaw)
    vx_wheel = u + yaw_rate * ry
    vy_wheel = v + yaw_rate * rx
    return vx_wheel, vy_wheel


def compute_slip_angle(yaw, vx_wheel, vy_wheel):
    # using the defination of dot product to calculate angle and positive/negative
    v1 = -np.sin(yaw)
    v2 = np.cos(yaw)

    a = vx_wheel * v1 + vy_wheel * v2
    b = np.hypot(vx_wheel, vy_wheel)

    denom = np.maximum(b, 1e-12)
    cosang = np.clip(a / denom, -1.0, 1.0)
    alpha = np.arccos(cosang)

    cross = vx_wheel * v2 - vy_wheel * v1
    alpha = np.where(cross > 0, alpha, -alpha)

    return alpha


def fancher_lateral_force(alpha, Fz):
    if abs(alpha) < 1e-6:
        return 0.0

    Ca = Ca0 + Ca1 * Fz + Ca2 * (Fz**2)
    tan_a = np.tan(alpha)
    abs_tan = np.abs(tan_a)
    # avoid division by zero
    abs_tan = np.where(abs_tan < 1e-12, 1e-12, abs_tan)

    xval = mu * Fz / (2.0 * Ca * abs_tan)
    xval = np.clip(xval, 0.0, 1.0)
    Fy = -Ca * tan_a * (xval**2) - np.sign(alpha) * mu * Fz * (1.0 - xval)
    return Fy


def compute_tire_wear(alpha, Fz, wheel_speed, K_average, alpha_0):
    # currently the filtered parabolic function, replace with surrogate model from string model
    if wheel_speed < 0.01:
        return 0.0

    alpha_abs = np.abs(np.rad2deg(alpha))  # Use absolute value of slip angle
    wear_index = (alpha_abs)**2 * wheel_speed * tyre_width * (K_average * Fz) / (
        1 + alpha_abs / alpha_0)
    return wear_index


def run_sim_fast(x_coords, y_coords, dt, wheel_offsets, Fz, K_average,
                 inertia_z):
    # ---------- setup ----------
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)
    N = x_coords.size
    W = len(wheel_offsets)
    if N < 2:
        logger.error("Need at least 2 points.")

    # Finite-difference velocities (central; forward/backward at ends)
    u = np.empty(N, dtype=float)
    v = np.empty(N, dtype=float)
    u[0] = (x_coords[1] - x_coords[0]) / dt
    v[0] = (y_coords[1] - y_coords[0]) / dt
    u[-1] = (x_coords[-1] - x_coords[-2]) / dt
    v[-1] = (y_coords[-1] - y_coords[-2]) / dt
    if N > 2:
        u[1:-1] = (x_coords[2:] - x_coords[:-2]) / (2 * dt)
        v[1:-1] = (y_coords[2:] - y_coords[:-2]) / (2 * dt)

    # Storage (preallocated)
    wheel_spd = np.zeros((W, N), dtype=float)
    wheel_sa = np.zeros((W, N), dtype=float)  # slip angle
    wheel_wear = np.zeros((W, N), dtype=float)

    yaw = np.zeros(N, dtype=float)
    yaw_rate = np.zeros(N, dtype=float)

    # Initial yaw (replicating your logic)
    yaw0 = -np.arctan(
        (x_coords[1] - x_coords[0]) / (y_coords[1] - y_coords[0]))
    if (y_coords[1] - y_coords[0]) < 0:
        yaw0 += np.pi

    yaw[0] = yaw0
    yaw_rate[0] = 0

    # ---------- main loop over time ----------
    for i in range(N):
        total_torque = 0

        for idx, values in enumerate(wheel_offsets.values()):
            # Compute wheel velocities
            dx, dy = values
            vx_wheel, vy_wheel = compute_wheel_velocities(
                u[i], v[i], yaw_rate[i], yaw[i], dx, dy)
            wheel_speed = np.sqrt(vx_wheel**2 + vy_wheel**2)

            # Compute slip angle
            alpha = compute_slip_angle(yaw[i], vx_wheel, vy_wheel)

            # Compute lateral force using Fancher model
            Fy = fancher_lateral_force(alpha, Fz)

            # Compute wear index
            wear_index = compute_tire_wear(alpha, Fz, wheel_speed, K_average,
                                           alpha_0)
            # wear_index = model_f(K_law, Fz, wheel_speed, alpha)

            # Store wheel data
            wheel_spd[idx, i] = wheel_speed
            wheel_sa[idx, i] = alpha
            wheel_wear[idx, i] = wear_index * dt

            torque = dx * Fy
            total_torque += torque

        # Compute angular acceleration and update yaw states for NEXT timestep
        aa = total_torque / inertia_z

        # Integrate yaw,yaw_rate for next step (except at the last index)
        if i < N - 1:
            yaw[i + 1], yaw_rate[i + 1] = rk4_step_scalar(
                yaw[i], yaw_rate[i], aa, dt)

    return wheel_wear


def stitch_simulations(x_coords, y_coords, sections, run_sim_fast, dt,
                       wheel_offsets, Fz, K_average, inertia_z):
    """
    For each (start, end) in sections, run run_sim_fast on the segment
    and splice the results back into zero-initialized global containers.
    Returns (wheel_data_global, vehicle_states_global).
    """
    N = len(x_coords)
    sections = sorted(sections)  # ensure increasing order

    wheel_wear_global = np.zeros((6, N), dtype=float)

    # --- Iterate sections, run & splice ---
    for (s, e) in sections:
        # Defensive bounds (in case)
        s = max(0, int(s))
        e = min(N - 1, int(e))
        if e < s:
            continue

        x_seg = x_coords[s:e + 1]
        y_seg = y_coords[s:e + 1]

        # Run your fast simulator on the segment
        wheel_wear = run_sim_fast(
            x_seg,
            y_seg,
            dt,
            wheel_offsets=wheel_offsets,  # dict: name -> (dx, dy)
            Fz=Fz,
            K_average=K_average,
            inertia_z=inertia_z)

        # Sanity checks
        n_cols = wheel_wear.shape[1]
        Lseg = e - s + 1
        if n_cols != Lseg:
            logger.error(
                "Wheel wear length != segment length %s", Lseg)

        # Splice wheel data
        wheel_wear_global[:, s:e + 1] = wheel_wear

    return wheel_wear_global


def calculate_corner(geo: gpd.array.GeometryArray):
    """Calculate corners of the map area."""
    minlon, minlat, maxlon, maxlat = geo.total_bounds
    lat_diff = maxlat - minlat
    lon_diff = maxlon - minlon
    lat_c = (maxlat + minlat) / 2
    lon_c = (maxlon + minlon) / 2

    if lat_diff > lon_diff:
        extend = 0.6 * lat_diff
    else:
        extend = 0.6 * lon_diff

    return lon_c - extend, lat_c - extend, lon_c + extend, lat_c + extend


def lon_to_px(corners, lon, width):
    """
    Map longitudes to pixel x (0..width) given corners=(min_lon, min_lat, max_lon, max_lat).
    Returns a 1-D float array.
    """
    lon = np.asarray(lon, dtype=float).reshape(-1)
    minx, miny, maxx, maxy = corners
    span = max(maxx - minx, 1e-12)  # guard against zero width
    return (lon - minx) / span * width


def lat_to_py(corners, lat, height):
    """
    Map latitudes to pixel y (0..height), top-left origin (so y increases downward).
    Returns a 1-D float array.
    """
    lat = np.asarray(lat, dtype=float).reshape(-1)
    minx, miny, maxx, maxy = corners
    span = max(maxy - miny, 1e-12)  # guard against zero height
    return (maxy - lat) / span * height


def fetch_here_static_map(
    corners,
    size: Tuple[int, int] = (800, 800),
    image_format: str = "png8",
) -> Optional[Image.Image]:
    # Convert to geographic coordinates (WGS84)
    map_corner = gpd.points_from_xy([corners[0], corners[2]],
                                    [corners[1], corners[3]],
                                    crs="EPSG:3857").to_crs("EPSG:4326")

    # Construct HERE static image URL
    bbox = f"{map_corner.x[0]},{map_corner.y[0]},{map_corner.x[1]},{map_corner.y[1]}"
    w, h = size
    map_url = (f"https://image.maps.hereapi.com/mia/v3/base/mc"
               f"/bbox:{bbox}"
               f"/{w}x{h}/{image_format}"
               f"?apiKey={os.environ['HERE_API_KEY']}")

    try:
        response = requests.get(map_url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        logger.error("Failed to retrieve map image: %s", e)
        return None


def draw_wear(draw, xx, yy, wear, point_radius=1, step=1):
    """
    Draw the line and then add sections with wear on it.

    """
    # Ensure 1D arrays
    xx   = np.asarray(xx,   dtype=float).reshape(-1)
    yy   = np.asarray(yy,   dtype=float).reshape(-1)
    wear = np.asarray(wear, dtype=float).reshape(-1)
    step = max(int(step), 1)

    # Subsample
    xi = xx[::step].astype(int)
    yi = yy[::step].astype(int)
    wi = wear[::step]

    for i in range(len(xi) - 1):
        draw.line([(xi[i], yi[i]), (xi[i+1], yi[i+1])],
                    fill=(0,255,0), width=2)

    for x, y, w in zip(xi, yi, wi):
        if w > 0:
            draw.ellipse([x - point_radius, y - point_radius,
                        x + point_radius, y + point_radius],
                        fill=(255,0,0), outline=None)


def generate_report(wheel_data, s_total, x_coords, y_coords, origin,
                    destination, output_file):
    # ----------Initialise PDF----------

    page_number = 2
    c = canvas.Canvas(output_file, pagesize=A4)

    # ----------Function to initialize a new page----------
    def initialize_page():
        """Sets up each page"""

        # Start next page
        c.showPage()

        # Set font for header
        c.setFont("Helvetica-Bold", 12)

        # Draw 'Wear Index Report' on top-left
        c.drawString(10, PAGE_HEIGHT - 15, "Wear Index Report")
        c.drawString(10, PAGE_HEIGHT - 30, f"v{__version__}")

        # Draw page number at the bottom center
        nonlocal page_number
        c.setFont("Helvetica", 10)
        c.drawCentredString(PAGE_WIDTH / 2, 30, f"{page_number}")
        page_number += 1

    c.setTitle("Wear Index Report")
    c.setSubject(f"From {origin} to {destination}")
    c.setAuthor("Chen Liu")
    c.setCreator(f"{pathlib.Path(__file__).name} v{__version__}")

    # Generate title page
    title = "Wear Index Report"
    subtitle = f"From {origin} to {destination}"

    # Set font and draw title
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 100, title)

    c.setFont("Helvetica", 18)
    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - 130, subtitle)

    # Draw a horizontal line
    c.line(MARGIN, PAGE_HEIGHT - 150, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 150)

    # Move to the next page
    initialize_page()

    n_cols = wheel_data.shape[1]
    wear = np.zeros(n_cols)
    for i in range(n_cols):
        wear[i] = np.sum(wheel_data[:, i])

    wear_wheel = np.zeros(6)
    for i in range(6):
        wear_wheel[i] = np.sum(wheel_data[i, :])

    s_total = s_total / 1000.0

    front_wear = (wear_wheel[4] + wear_wheel[5])/2.0/s_total
    mid_wear = (wear_wheel[2] + wear_wheel[3])/2.0/s_total
    rear_wear = (wear_wheel[0] + wear_wheel[1])/2.0/s_total
    total_wear = np.sum(wear_wheel)

    av_wear = total_wear / s_total

    if av_wear < 0.6:
        wear_level = 'Low'
    elif av_wear < 1.8:
        wear_level = 'Mid'
    else:
        wear_level = 'High'

    mm_wear = av_wear/6.0

    distance = 15.0/mm_wear*1000
    distance_f = 15.0/front_wear*1000
    distance_m = 15.0/mid_wear*1000
    distance_r = 15.0/rear_wear*1000

    y_start = PAGE_HEIGHT - MARGIN - 13

    c.setFont("Helvetica", 12)
    c.drawString(MARGIN, y_start, f"Total distance: {s_total:.1f}km")

    y_start -= 13

    c.drawString(MARGIN, y_start, f"Total wear: {total_wear:.2f}g")

    y_start -= 17

    c.drawString(MARGIN, y_start, "Wear index (sum of 6 tyres):")

    c.setFont("Helvetica-Bold", 16)
    c.drawString(MARGIN + 150, y_start, f"{av_wear:.3f}g/km {wear_level}")

    y_start -= 15
    c.setFont("Helvetica", 12)
    c.drawString(MARGIN, y_start, "Equivalent wear per tyre per 1000km:")

    data = [
        ["Front", "Mid", "Rear", "Average"],
        [f"{front_wear:.3f}mm", f"{mid_wear:.3f}mm", f"{rear_wear:.3f}mm", f"{mm_wear:.3f}mm"],
    ]

    col_widths = [125, 125, 125, 125]  # total = 500

    table = Table(data, colWidths=col_widths, rowHeights=20)

    table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 12),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),

        # center all cells
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
    ]))

    # draw table
    y_start -= 45
    table.wrapOn(c, 500, 200)
    table.drawOn(c, MARGIN, y_start)

    y_start -= 13
    c.setFont("Helvetica", 12)
    c.drawString(MARGIN, y_start, "Estimated tyre life (16mm tread):")

    data = [
        ["Front", "Mid", "Rear", "Average"],
        [f"{distance_f:,.0f}km", f"{distance_m:,.0f}km", f"{distance_r:,.0f}km", f"{distance:,.0f}km"],
    ]

    col_widths = [125, 125, 125, 125]  # total = 500

    table = Table(data, colWidths=col_widths, rowHeights=20)

    table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 12),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),

        # center all cells
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
    ]))

    # draw table
    y_start -= 45
    table.wrapOn(c, 500, 200)
    table.drawOn(c, MARGIN, y_start)

    # Calculate map corner coordinates in km
    geo = gpd.GeoSeries(gpd.points_from_xy(x_coords, y_coords),
                        crs="EPSG:3857")

    corners = calculate_corner(geo)

    # Fetch map image
    map_image = fetch_here_static_map(corners)

    # Draw the lines to show the trips
    yy = lat_to_py(corners, geo.y, 800.0)
    xx = lon_to_px(corners, geo.x, 800.0)
    draw = ImageDraw.Draw(map_image)
    draw_wear(draw, xx, yy, wear, step=100)

    # ----------Save the map to PDF----------

    # Fit to page
    map_scale_height = int(map_image.height * CONTENT_WIDTH / map_image.width)

    # Add text annotation below image
    c.setFont("Helvetica", 14)
    c.drawString(MARGIN, MARGIN + map_scale_height + 15, "Map of the route and wear:")
    c.drawImage(ImageReader(map_image),
                MARGIN,
                MARGIN,
                width=CONTENT_WIDTH,
                height=map_scale_height)
    map_image.close()

    # ----------add explanation of wear index----------
    initialize_page()

    y_start = PAGE_HEIGHT - MARGIN - 13

    c.setFont("Helvetica", 12)
    c.drawString(MARGIN, y_start, "Explanation of wear index:")

    data = [
        ["", "Wear index", "Wear per tyre", "Tyre life"],
        ["Low wear (long-haul)", "<0.6 g/km", "<0.1 mm/1000km", "> 150,000 km"],
        ["Mid wear (mixed)", "0.6 - 1.8 g/km", "0.1 - 0.3 mm/1000km", "50,000 - 150,000 km"],
        ["High wear (urban)", "> 1.8 g/km", "> 0.3 mm/1000km", "< 50,000 km"],
    ]

    col_widths = [120, 80, 150, 150]  # total = 500

    table = Table(data, colWidths=col_widths, rowHeights=20)

    table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 12),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),

        # center all cells
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        # left-align first column only
        ("ALIGN", (0, 0), (0, -1), "LEFT"),

        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
    ]))

    y_start -= 10 + 20 * len(data)
    # draw table
    table.wrapOn(c, 500, 200)
    table.drawOn(c, MARGIN, y_start)

    img_path = "wear_range.jpg"
    img = ImageReader(img_path)

    orig_w, orig_h = img.getSize()
    target_w = 500
    scale = target_w / orig_w
    target_h = orig_h * scale

    c.drawImage(
        img,
        MARGIN,
        y_start - 10 - target_h,   # so the image starts *under* y_start
        width=target_w,
        height=target_h,
        preserveAspectRatio=True,
        mask='auto'
    )

    # ----------Save file----------
    c.save()
    output_file.flush()
    logger.info("Wear index report successfully written to the output file.")


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    pd.options.mode.copy_on_write = True

    parser = argparse.ArgumentParser(
        description="Generate a wear index report for a route.")
    parser.add_argument("--start",
                        type=str,
                        required=True,
                        help="Origin (post code).")
    parser.add_argument("--end",
                        type=str,
                        required=True,
                        help="Destination (post code).")
    parser.add_argument("--output",
                        type=argparse.FileType('wb'),
                        required=True,
                        help="Output file for the PDF report (eg. report.pdf)."
                        " Use '-' for stdout.")

    try:
        args = parser.parse_args()
        wear_index(origin=args.start,
                   destination=args.end,
                   output_file=args.output)
    except Exception:
        logger.exception(
            "An error occurred during the execution of the program.")
        raise


if __name__ == "__main__":
    main()
