# File: tools/pointing_planner.py
# Purpose: Compute spacecraft boresight pointings at specified angular distances from the Earth or Moon limb,
#          mapped to detector "cardinal" entry directions, for dates between 2026-01-25 and 2026-02-05.
#
# Usage:
#   In Jupyter or IDE:
#       import pointing_planner as pp
#       df = pp.load_ephemeris("your_ephemeris.csv")
#       pt = pp.compute_pointing(df, "2026-01-29 00:00Z", "earth", "up", 0.0)
#       print(pt)
#       pp.plot_sky_scene(df, "2026-01-29 00:00Z", "earth", mode="proj")
#       pp.plot_sky_scene(df, "2026-01-29 00:00Z", "earth", mode="radec")
#       pp.plot_sky_scene(df, "2026-01-29 00:00Z", "earth", mode="angdist")

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

R_EARTH_KM = 6378.137
R_MOON_KM = 1737.4
DEG = math.pi / 180.0

@dataclass(frozen=True)
class Pointing:
    body: str
    date: pd.Timestamp
    cardinal: str
    limb_offset_deg: float
    theta_deg: float  # angular separation between body center vector and pointing vector
    boresight_eci: np.ndarray
    ra_deg: float
    dec_deg: float

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def _hat(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n == 0:
        raise ValueError("Zero vector cannot be normalized")
    return v / n

def _radec_from_vec(v: np.ndarray) -> Tuple[float, float]:
    v = _hat(v)
    x, y, z = v
    ra = math.degrees(math.atan2(y, x)) % 360.0
    dec = math.degrees(math.asin(np.clip(z, -1.0, 1.0)))
    return ra, dec

def load_ephemeris(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["utc"] = pd.to_datetime(df["Earth.UTCGregorian"], utc=True, errors="coerce")
    df = df.set_index("utc").sort_index()
    return df

def row_at(df: pd.DataFrame, when: pd.Timestamp) -> pd.Series:
    when = pd.to_datetime(when, utc=True)
    i = df.index.get_indexer([when], method="nearest")[0]
    return df.iloc[i]

def body_center_vec_from_sc(row: pd.Series, body: str) -> np.ndarray:
    sc = np.array([
        row["Pandora.EarthMJ2000Eq.X"],
        row["Pandora.EarthMJ2000Eq.Y"],
        row["Pandora.EarthMJ2000Eq.Z"],
    ], dtype=float)
    if body.lower() == "earth":
        bc = np.zeros(3)
    elif body.lower() == "moon":
        bc = np.array([
            row["Luna.EarthMJ2000Eq.X"],
            row["Luna.EarthMJ2000Eq.Y"],
            row["Luna.EarthMJ2000Eq.Z"],
        ], dtype=float)
    else:
        raise ValueError("body must be 'earth' or 'moon'")
    return bc - sc

def body_radius_km(body: str) -> float:
    if body.lower() == "earth":
        return R_EARTH_KM
    if body.lower() == "moon":
        return R_MOON_KM
    raise ValueError("body must be 'earth' or 'moon'")

def apparent_angular_radius_rad(sc_to_body: np.ndarray, body: str) -> float:
    R = body_radius_km(body)
    d = _norm(sc_to_body)
    return math.asin(R / d)

def sky_basis(look_vec: np.ndarray, ref_axis: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = _hat(look_vec)
    if ref_axis is None:
        ref_axis = np.array([0.0, 0.0, 1.0])
    ref_proj = ref_axis - np.dot(ref_axis, k) * k
    if _norm(ref_proj) < 1e-12:
        ref_proj = np.array([0.0, 1.0, 0.0])
    u = _hat(ref_proj)
    r = _hat(np.cross(k, u))
    return k, u, r

_CARDINAL_TO_AZ: Dict[str, float] = {
    "up": 0.0,
    "ur": 45.0,
    "right": 90.0,
    "dr": 135.0,
    "down": 180.0,
    "dl": 225.0,
    "left": 270.0,
    "ul": 315.0,
}

def pointing_vector_for_cardinal(sc_to_body: np.ndarray, limb_offset_deg: float, cardinal: str, body: str, ref_axis: np.ndarray | None = None):
    k, u, r = sky_basis(sc_to_body, ref_axis)
    ang_rad = apparent_angular_radius_rad(sc_to_body, body)
    theta = ang_rad + limb_offset_deg * DEG
    az = _CARDINAL_TO_AZ[cardinal.lower()] * DEG
    off = math.cos(az) * u + math.sin(az) * r
    v = k * math.cos(theta) + off * math.sin(theta)
    return _hat(v), math.degrees(theta), math.degrees(ang_rad)

def compute_pointing(df: pd.DataFrame, when_utc: str | pd.Timestamp, body: str, cardinal: str, limb_offset_deg: float, ref_axis: np.ndarray | None = None) -> Pointing:
    row = row_at(df, pd.to_datetime(when_utc, utc=True))
    scb = body_center_vec_from_sc(row, body)
    v, theta_deg, _ = pointing_vector_for_cardinal(scb, limb_offset_deg, cardinal, body, ref_axis)
    ra, dec = _radec_from_vec(v)
    return Pointing(body=body.lower(), date=row.name, cardinal=cardinal.lower(), limb_offset_deg=limb_offset_deg, theta_deg=theta_deg, boresight_eci=v, ra_deg=ra, dec_deg=dec)

def plot_sky_scene(df: pd.DataFrame, when_utc: str | pd.Timestamp, body: str, cardinals: Iterable[str] = ("up","right","down","left","ur","dr","dl","ul"), limb_offset_deg: float = 0.0, ref_axis: np.ndarray | None = None, title: str | None = None, mode: str = "proj"):
    """
    mode options:
        - "proj": unit-vector projection in spacecraft sky-plane (dimensionless, with body disk overlay)
        - "radec": RA and Dec in degrees (with limb circle overlay)
        - "angdist": angular offsets from body center in degrees (with limb circle overlay)

    Note: The theta values annotated are the total angular separation between the body center vector and the pointing vector.
    """
    row = row_at(df, pd.to_datetime(when_utc, utc=True))
    scb = body_center_vec_from_sc(row, body)
    k, u, r = sky_basis(scb, ref_axis)
    ang = apparent_angular_radius_rad(scb, body)

    fig, ax = plt.subplots(figsize=(6, 6))

    xs, ys, labels = [], [], []
    for c in cardinals:
        v, theta_deg, _ = pointing_vector_for_cardinal(scb, limb_offset_deg, c, body, ref_axis)
        ra, dec = _radec_from_vec(v)
        if mode == "proj":
            x = np.dot(v, r)
            y = np.dot(v, u)
        elif mode == "radec":
            x = ra
            y = dec
        elif mode == "angdist":
            cosang = np.dot(_hat(scb), v)
            ang_sep = math.degrees(math.acos(np.clip(cosang, -1, 1)))
            az = _CARDINAL_TO_AZ[c] if c in _CARDINAL_TO_AZ else 0
            x = az
            y = ang_sep
        else:
            raise ValueError("mode must be 'proj', 'radec', or 'angdist'")

        xs.append(x)
        ys.append(y)
        labels.append(f"{c}\nθ={theta_deg:.2f}°")

    ax.scatter(xs, ys, c="tab:red")
    for x, y, lab in zip(xs, ys, labels):
        ax.text(x, y, lab, fontsize=9, ha="center", va="bottom")

    if mode == "proj":
        ax.set_aspect("equal")
        ax.set_xlabel("Right (unit)")
        ax.set_ylabel("Up (unit)")
        lim = math.sin(ang + abs(limb_offset_deg) * DEG) * 1.4
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # Overlay the body disk
        phi = np.linspace(0, 2*math.pi, 360)
        body_outline = []
        for p in phi:
            v = k * math.cos(ang) + (math.cos(p) * u + math.sin(p) * r) * math.sin(ang)
            x_c = np.dot(v, r)
            y_c = np.dot(v, u)
            body_outline.append((x_c, y_c))
        body_outline = np.array(body_outline)
        ax.plot(body_outline[:,0], body_outline[:,1], color="tab:blue" if body=="earth" else "tab:gray", lw=2)
    elif mode == "radec":
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.invert_xaxis()
        circle_points = []
        for phi in np.linspace(0, 2*math.pi, 360):
            v = k * math.cos(ang) + (math.cos(phi) * u + math.sin(phi) * r) * math.sin(ang)
            ra_c, dec_c = _radec_from_vec(v)
            circle_points.append((ra_c, dec_c))
        circle_points = np.array(circle_points)
        ax.plot(circle_points[:,0], circle_points[:,1], color="tab:blue" if body=="earth" else "tab:gray", lw=2)
    elif mode == "angdist":
        ax.set_xlabel("Azimuth (deg)")
        ax.set_ylabel("Angular separation from center (deg)")
        ax.axhline(y=math.degrees(ang), color="tab:blue" if body=="earth" else "tab:gray", lw=2)

    if title is None:
        title = f"{body.title()} scene @ {pd.to_datetime(when_utc, utc=True).strftime('%Y-%m-%d %H:%M UTC')}"
    ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)
    return fig, ax

def suggest_observations(df: pd.DataFrame, when_utc: str, body: str, offsets_deg: Iterable[float] = (0.0, +1.0, -1.0), cardinals: Iterable[str] = ("up","right","down","left","ur","dr","dl","ul"), ref_axis: np.ndarray | None = None) -> List[Pointing]:
    pts: List[Pointing] = []
    for off in offsets_deg:
        for c in cardinals:
            pts.append(compute_pointing(df, when_utc, body, c, off, ref_axis))
    return pts

def tabulate_pointings(points: List[Pointing]) -> pd.DataFrame:
    return pd.DataFrame([{ "UTC": p.date.strftime("%Y-%m-%d %H:%M:%S"), "Body": p.body, "Cardinal": p.cardinal, "Offset_deg": p.limb_offset_deg, "Theta_deg": p.theta_deg, "RA_deg": p.ra_deg, "Dec_deg": p.dec_deg, "ECI_x": p.boresight_eci[0], "ECI_y": p.boresight_eci[1], "ECI_z": p.boresight_eci[2]} for p in points])
