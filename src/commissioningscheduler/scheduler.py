"""
File: scheduler.py
Purpose: Merge Commissioning-task XML schedule fragments into a master ScienceCalendar XML.
Now includes real pandora-visibility integration and Visit/Observation_Sequence ID handling.
Adds support for dynamic Earth/Moon cardinal pointings (tasks 0341/0342) via external pointing_planner,
special keep-out overrides for those tasks, and visibility caching with full-window cache.
"""

import os
import re
import math
import json
import pickle
import hashlib
import copy
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional, Dict, Any
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict, Counter

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from pandoravisibility import Visibility

# optional external pointing planner (user-provided file)
try:
    import pointing_planner as pp
    HAVE_POINTING_PLANNER = True
except Exception:
    HAVE_POINTING_PLANNER = False

# -----------------------------
# Config / Constants
# -----------------------------
DOWNLINK_RATE_BPS = 5e6  # 5 Mbps
DOWNLINK_DURATION_S = 8 * 60  # 8 minutes
COMMISSIONING_START = datetime(2026, 1, 5, 0, 0, 0)
COMMISSIONING_END = datetime(2026, 2, 4, 23, 59, 59)
EXTENDED_SCHEDULING_PERIOD = timedelta(days=30)  # One month extension
TEN_MIN = 600
ONE_MIN = 60
BYTES_PER_PIXEL = 2
VIS_FRAME_OVERHEAD_BYTES = 1000
VIS_CACHE_DIR = '.vis_cache'
os.makedirs(VIS_CACHE_DIR, exist_ok=True)
FULL_MOON_TIME = datetime(2026, 2, 2, 22, 9, 0)  # 10:09 PM UTC on Feb 2, 2026

NS = "/pandora/calendar/"
ET.register_namespace("", NS)
NS = {"cal": "/pandora/calendar/"}

# def make_tag(name: str) -> str:
#     return f"{{{NS}}}{name}"

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class ObservationSequence:
    filename: str
    visit_id: str
    obs_id: str
    target: str
    ra: Optional[float]
    dec: Optional[float]
    xml_tree: ET.ElementTree
    xml_root: ET.Element


# -----------------------------
# XML Indent Function
# -----------------------------
def indent(elem, level: int = 0):
    """
    Pretty-print XML tree by adjusting .text and .tail for proper indentation.
    Ensures sibling tags align and closing tags are not over-indented.
    """
    i = "\n" + level * "    "
    if len(elem):  # if the element has children
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for child in elem:
            indent(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    else:  # leaf element
        if not elem.text or not elem.text.strip():
            elem.text = ""
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    elif level == 0:
        elem.tail = "\n"

def write_pretty_xml(root: ET.Element, output_path: str):
    """
    Serialize an ElementTree root to a file with proper indentation
    using minidom for formatting, while removing extra blank lines.
    """
    rough_string = ET.tostring(root, encoding="utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ", encoding="utf-8")

    # Decode to string, strip extra blank lines, re-encode
    pretty_str = pretty_xml.decode("utf-8")
    pretty_str = "\n".join([line for line in pretty_str.splitlines() if line.strip()])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_str)


# -----------------------------
# Visibility caching
# -----------------------------
def _vis_cache_key(ra: float, dec: float, tle1: str, tle2: str, key_extra: str = "") -> str:
    s = f"{ra}:{dec}:{tle1}:{tle2}:{key_extra}"
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def load_visibility_cache(ra: float, dec: float, tle1: str, tle2: str, key_extra: str = "") -> Optional[List[Tuple[datetime, datetime]]]:
    key = _vis_cache_key(ra, dec, tle1, tle2, key_extra)
    path = os.path.join(VIS_CACHE_DIR, key + '.pkl')
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def save_visibility_cache(ra: float, dec: float, tle1: str, tle2: str, windows: List[Tuple[datetime, datetime]], key_extra: str = ""):
    key = _vis_cache_key(ra, dec, tle1, tle2, key_extra)
    path = os.path.join(VIS_CACHE_DIR, key + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(windows, f)

def compute_and_cache_visibility(ra: float, dec: float, start: datetime, end: datetime,
                                 tle1: str, tle2: str,
                                 *, moon_min: Optional[u.Quantity] = None,
                                 earthlimb_min: Optional[u.Quantity] = None) -> List[Tuple[datetime, datetime]]:
    """
    Compute visibility windows for a target (ra,dec) over [start,end].
    Cache full commissioning window grid (1-min cadence) per target and slice as needed.
    moon_min or earthlimb_min can override keep-out for special tasks.
    """
    key_extra = f"moon_min={moon_min}_earthlimb_min={earthlimb_min}"
    cached = load_visibility_cache(ra, dec, tle1, tle2, key_extra)
    if cached is None:
        kwargs = {}
        if moon_min is not None:
            kwargs['moon_min'] = moon_min
        if earthlimb_min is not None:
            kwargs['earthlimb_min'] = earthlimb_min
        vis = Visibility(tle1, tle2, **kwargs) if kwargs else Visibility(tle1, tle2)
        tstart = Time(COMMISSIONING_START)
        tstop = Time(COMMISSIONING_END + EXTENDED_SCHEDULING_PERIOD)
        deltas = np.arange(0, (tstop - tstart).to_value(u.min), 1) * u.min
        times = tstart + TimeDelta(deltas)
        target_coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
        targ_vis = vis.get_visibility(target_coord, times)
        windows: List[Tuple[datetime, datetime]] = []
        in_window = False
        win_start = None
        for i, visible in enumerate(targ_vis):
            if visible and not in_window:
                in_window = True
                win_start = times[i].to_datetime()
            elif not visible and in_window:
                in_window = False
                win_stop = times[i].to_datetime()
                windows.append((win_start, win_stop))
        if in_window:
            windows.append((win_start, times[-1].to_datetime()))
        save_visibility_cache(ra, dec, tle1, tle2, windows, key_extra)
        cached = windows
    sliced = []
    for ws, we in cached:
        if we < start or ws > end:
            continue
        sliced.append((max(ws, start), min(we, end)))
    return sliced

# -----------------------------
# Helper: Find visibility window
# -----------------------------
def find_next_visibility_window(ra, dec, start_time, end_time, tle_data, min_duration_s=120):
    """Find the next visibility window that meets minimum duration requirement"""
    windows = compute_and_cache_visibility(ra, dec, start_time, end_time, 
                                         tle_data[0], tle_data[1])
    
    for window_start, window_end in windows:
        if (window_end - window_start).total_seconds() >= min_duration_s:
            return (window_start, window_end)
    
    return None

# -----------------------------
# Helper: parse xml into ObservationSequence
# -----------------------------
def parse_task_xml(path: str) -> ObservationSequence:
    # print(path)
    tree = ET.parse(path)
    root = tree.getroot()
    NS = {"cal": "/pandora/calendar/"}
    fn = os.path.basename(path)
    m = re.match(r"(\d{4})_(\d{3})_.*\.xml", fn)
    if m:
        visit_id, obs_id = m.group(1), m.group(2)
    else:
        visit_id, obs_id = "0000", "000"

    tgt_elem = root.find('.//cal:Observational_Parameters/cal:Target', namespaces=NS)
    # print(tgt_elem)
    target = tgt_elem.text if tgt_elem is not None else ''
    ra_elem = root.find('.//cal:Observational_Parameters/cal:Boresight/cal:RA', namespaces=NS)
    # print(ra_elem)
    dec_elem = root.find('.//cal:Observational_Parameters/cal:Boresight/cal:DEC', namespaces=NS)
    def parse_angle(elem):
        if elem is None or elem.text is None:
            return None
        try:
            return float(elem.text.strip().split()[0])
        except Exception:
            return None
    ra = parse_angle(ra_elem)
    dec = parse_angle(dec_elem)
    return ObservationSequence(filename=fn, visit_id=visit_id, obs_id=obs_id,
                               target=target, ra=ra, dec=dec,
                               xml_tree=tree, xml_root=root)

# -----------------------------
# Helper: compute durations from payload parameters
# -----------------------------
def compute_instrument_durations(obs: ObservationSequence) -> Tuple[float, float, float, float]:
    root = obs.xml_root
    nir_total_s = nir_integration_s = 0.0
    nir = root.find('.//cal:AcquireInfCamImages', namespaces=NS)
    if nir is not None:
        ROI_SizeX = int(nir.findtext('cal:ROI_SizeX', '0', namespaces=NS))
        ROI_SizeY = int(nir.findtext('cal:ROI_SizeY', '0', namespaces=NS))
        SC_Resets1 = int(nir.findtext('cal:SC_Resets1', '0', namespaces=NS))
        SC_Resets2 = int(nir.findtext('cal:SC_Resets2', '0', namespaces=NS))
        SC_DropFrames1 = int(nir.findtext('cal:SC_DropFrames1', '0', namespaces=NS))
        SC_DropFrames2 = int(nir.findtext('cal:SC_DropFrames2', '0', namespaces=NS))
        SC_DropFrames3 = int(nir.findtext('cal:SC_DropFrames3', '0', namespaces=NS))
        SC_ReadFrames = int(nir.findtext('cal:SC_ReadFrames', '0', namespaces=NS))
        SC_Integrations = int(nir.findtext('cal:SC_Integrations', '0', namespaces=NS))
        group_sum = (SC_Resets1 + SC_Resets2 + SC_DropFrames1 +
                     SC_DropFrames2 + SC_DropFrames3 + SC_ReadFrames + 1)
        nir_integration_s = (ROI_SizeX * ROI_SizeY + (ROI_SizeY * 12)) * 0.00001 * group_sum
        nir_total_s = nir_integration_s * SC_Integrations

    vda_total_s = vda_integration_s = 0.0
    vda = root.find('.//cal:AcquireVisCamImages', namespaces=NS)
    vda_science = root.find('.//cal:AcquireVisCamScienceData', namespaces=NS)
    if vda is not None:
        ExposureTime_us = float(vda.findtext('cal:ExposureTime_us', '0', namespaces=NS))
        NumExposures = int(vda.findtext('cal:NumExposures', '0', namespaces=NS))
        vda_integration_s = ExposureTime_us / 1e6
        vda_total_s = vda_integration_s * NumExposures
    elif vda_science is not None:
        ExposureTime_us = float(vda_science.findtext('cal:ExposureTime_us', '0', namespaces=NS))
        NumTotalFramesRequested = int(vda_science.findtext('cal:NumTotalFramesRequested', '0', namespaces=NS))
        FramesPerCoadd = int(vda_science.findtext('cal:FramesPerCoadd', '1', namespaces=NS))
        vda_integration_s = (ExposureTime_us / 1e6) * FramesPerCoadd
        vda_total_s = (ExposureTime_us / 1e6) * NumTotalFramesRequested

    return nir_total_s, vda_total_s, nir_integration_s, vda_integration_s

# -----------------------------
# Helper: adjust payload for chunk
# -----------------------------
def adjust_payload_for_chunk(obs: ObservationSequence, chunk_seconds: int,
                             nir_integr_s: float, vda_integr_s: float) -> ObservationSequence:
    """
    Modify the obs.xml_tree in-place to reflect the number of integrations/exposures
    that fit in chunk_seconds. Does not touch sequence IDs.
    """
    # root = obs.xml_root
    new_tree = ET.ElementTree(ET.fromstring(ET.tostring(obs.xml_root)))
    new_root = new_tree.getroot()
    obs_seq = new_root.find('.//cal:Observation_Sequence', namespaces=NS)

    # NIRDA camera
    nir = obs_seq.find('.//cal:AcquireInfCamImages', namespaces=NS)
    if nir is not None and nir_integr_s > 0:
        max_integrations = int(chunk_seconds // nir_integr_s)
        if max_integrations < 1:
            max_integrations = 1
        nir.find('cal:SC_Integrations', namespaces=NS).text = str(max_integrations)

    # VIS camera — AcquireVisCamImages
    vda_img = obs_seq.find('.//cal:AcquireVisCamImages', namespaces=NS)
    if vda_img is not None and vda_integr_s > 0:
        max_exposures = int(chunk_seconds // vda_integr_s)
        if max_exposures < 1:
            max_exposures = 1
        vda_img.find('cal:NumExposures', namespaces=NS).text = str(max_exposures)

    # VIS camera — AcquireVisCamScienceData
    vda_sci = obs_seq.find('.//cal:AcquireVisCamScienceData', namespaces=NS)
    if vda_sci is not None and vda_integr_s > 0:
        max_exposures = int(chunk_seconds // vda_integr_s)
        if max_exposures < 1:
            max_exposures = 1
        vda_sci.find('cal:NumExposuresMax', namespaces=NS).text = str(max_exposures)

    return ObservationSequence(
        filename=obs.filename,
        visit_id=obs.visit_id,
        obs_id=obs.obs_id,
        target=obs.target,
        ra=obs.ra,
        dec=obs.dec,
        xml_tree=new_tree,
        xml_root=new_root
    )

# -----------------------------
# Helper: create visit element
# -----------------------------
def create_visit_element(obs, start_time, end_time, visit_id, obs_seq_id):
    """Create a visit element for the XML with proper order of elements"""
    visit = ET.Element('cal:Visit', namespace=NS)
    ET.SubElement(visit, 'ID').text = f"{visit_id:04d}"

    obs_seq = ET.SubElement(visit, 'Observation_Sequence')
    ET.SubElement(obs_seq, 'ID').text = f"{obs_seq_id:03d}"

    # Observational Parameters
    obs_params = ET.SubElement(obs_seq, 'Observational_Parameters')
    ET.SubElement(obs_params, 'Target').text = obs.target
    ET.SubElement(obs_params, 'Priority').text = "1"
    
    # Timing MUST come before Boresight
    timing = ET.SubElement(obs_params, 'Timing')
    ET.SubElement(timing, 'Start').text = format_utc_time(start_time)
    ET.SubElement(timing, 'Stop').text = format_utc_time(end_time)

    # Boresight comes after Timing
    boresight = ET.SubElement(obs_params, 'Boresight')
    ET.SubElement(boresight, 'RA').text = str(obs.ra)
    ET.SubElement(boresight, 'DEC').text = str(obs.dec)

    # Copy Payload_Parameters from original XML
    original_payload = obs.xml_root.find('.//cal:Payload_Parameters', namespaces=NS)
    if original_payload is not None:
        payload_params = ET.SubElement(obs_seq, 'Payload_Parameters')
        # Copy all children from original payload
        for child in original_payload:
            payload_params.append(copy.deepcopy(child))

    return visit

# -----------------------------
# Helper: update observation sequence
# -----------------------------
def update_observation_sequence(obs: ObservationSequence, duration_s: float) -> ObservationSequence:
    """Update observation sequence with new duration and frame counts"""
    
    # Create a deep copy of the observation
    new_tree = copy.deepcopy(obs.xml_tree)
    new_root = new_tree.getroot()
    
    # Calculate exposure parameters from original XML
    nir_total, vda_total, nir_int_s, vda_int_s = compute_instrument_durations(obs)
    
    # Determine which instrument takes longer (limiting factor)
    if nir_total >= vda_total:
        # NIR is limiting - scale based on NIR timing
        scale_factor = duration_s / nir_total if nir_total > 0 else 1
        
        # Update AcquireInfCamImages
        nir_cmd = new_root.find('.//cal:AcquireInfCamImages', namespaces=NS)
        if nir_cmd is not None:
            sc_integrations = nir_cmd.find('cal:SC_Integrations', namespaces=NS)
            if sc_integrations is not None:
                original_integrations = int(sc_integrations.text)
                new_integrations = max(1, int(original_integrations * scale_factor))
                sc_integrations.text = str(new_integrations)
    else:
        # VDA is limiting - scale based on VDA timing
        scale_factor = duration_s / vda_total if vda_total > 0 else 1
    
    # Update VDA commands
    # AcquireVisCamImages
    vda_images = new_root.find('.//cal:AcquireVisCamImages', namespaces=NS)
    if vda_images is not None:
        num_exposures = vda_images.find('cal:NumExposures', namespaces=NS)
        if num_exposures is not None:
            original_exposures = int(num_exposures.text)
            new_exposures = max(1, int(original_exposures * scale_factor))
            num_exposures.text = str(new_exposures)
    
    # AcquireVisCamScienceData
    vda_science = new_root.find('.//cal:AcquireVisCamScienceData', namespaces=NS)
    if vda_science is not None:
        num_frames = vda_science.find('cal:NumTotalFramesRequested', namespaces=NS)
        if num_frames is not None:
            original_frames = int(num_frames.text)
            new_frames = max(1, int(original_frames * scale_factor))
            num_frames.text = str(new_frames)
    
    return ObservationSequence(
        filename=obs.filename,
        visit_id=obs.visit_id,
        obs_id=obs.obs_id,
        target=obs.target,
        ra=obs.ra,
        dec=obs.dec,
        xml_tree=new_tree,
        xml_root=new_root
    )

def process_normal_task(obs: ObservationSequence, 
                       current_time: datetime, 
                       cvz_coords: Tuple[float, float], 
                       master_root: ET.Element, 
                       visit_id: int,
                       tle_line1: str, 
                       tle_line2: str) -> Optional[Tuple[int, datetime]]:
    """Process a normal observation task"""
    
    # Extract task number
    task_match = re.match(r"(\d{4})", obs.filename)
    task_num = task_match.group(1) if task_match else None

    # Calculate observation requirements
    nir_total, vda_total, nir_int_s, vda_int_s = compute_instrument_durations(obs)
    remaining_seconds = int(math.ceil(max(nir_total, vda_total)))
    
    print(f"Observation needs {remaining_seconds/60:.1f} minutes")
    
    # Get visibility windows
    vis_windows = compute_and_cache_visibility(obs.ra, obs.dec, current_time, 
                                             current_time + timedelta(days=7), 
                                             tle_line1, tle_line2)
    
    if not vis_windows:
        print(f"No visibility windows found for {obs.filename}")
        return None
    
    # Create new visit for this task
    visit_el = ET.Element('Visit')
    ET.SubElement(visit_el, 'ID').text = f"{visit_id:04d}"
    obs_seq_id = 1
    
    scheduled_any = False
    
    # Process visibility windows
    for window_start, window_end in vis_windows:
        if remaining_seconds <= 0:
            break
            
        if window_start < current_time:
            window_start = current_time
        
        window_duration_s = (window_end - window_start).total_seconds()
        
        # Fill gap with CVZ if needed
        if window_start > current_time:
            gap_duration = (window_start - current_time).total_seconds()
            if gap_duration >= 120:  # At least 2 minutes
                # Split CVZ gap into 90-minute chunks
                cvz_segments = split_long_observation_time(gap_duration, 5400)
                
                for segment_duration in cvz_segments:
                    segment_end = current_time + timedelta(seconds=segment_duration)
                    
                    # Add CVZ to current visit
                    cvz_seq = ET.SubElement(visit_el, 'Observation_Sequence')
                    ET.SubElement(cvz_seq, 'ID').text = f"{obs_seq_id:03d}"
                    obs_seq_id += 1
                    
                    cvz_params = ET.SubElement(cvz_seq, 'Observational_Parameters')
                    ET.SubElement(cvz_params, 'Target').text = "CVZ_IDLE"
                    ET.SubElement(cvz_params, 'Priority').text = "2"
                    
                    cvz_timing = ET.SubElement(cvz_params, 'Timing')
                    ET.SubElement(cvz_timing, 'Start').text = format_utc_time(current_time)
                    ET.SubElement(cvz_timing, 'Stop').text = format_utc_time(segment_end)
                    
                    cvz_bore = ET.SubElement(cvz_params, 'Boresight')
                    ET.SubElement(cvz_bore, 'RA').text = f"{cvz_coords[0]}"
                    ET.SubElement(cvz_bore, 'DEC').text = f"{cvz_coords[1]}"
                    
                    current_time = segment_end
            
            current_time = align_to_minute_boundary(window_start)
        
        if window_duration_s < 120:  # Skip short windows
            continue
            
        # Calculate observation time for this window
        obs_duration_needed = min(remaining_seconds, window_duration_s)
        obs_duration_minutes = max(2, math.ceil(obs_duration_needed / 60))
        obs_duration_s = obs_duration_minutes * 60
        
        # Split into 90-minute segments if needed
        obs_segments = split_long_observation_time(obs_duration_s, 5400)
        
        for segment_duration in obs_segments:
            if remaining_seconds <= 0:
                break
                
            # Create observation sequence
            obs_end_time = current_time + timedelta(seconds=segment_duration)
            updated_obs = update_observation_sequence(obs, segment_duration)
            
            obs_seq = ET.SubElement(visit_el, 'Observation_Sequence')
            ET.SubElement(obs_seq, 'ID').text = f"{obs_seq_id:03d}"
            obs_seq_id += 1
            
            # Special handling for task 0319 - copy Bus_Parameters
            if task_num == "0319":
                bus_params = obs.xml_root.find('.//cal:Bus_Parameters', namespaces=NS)
                if bus_params is not None:
                    # Copy Bus_Parameters before Observational_Parameters
                    new_bus_params = ET.SubElement(obs_seq, 'Bus_Parameters')
                    for child in bus_params:
                        new_bus_params.append(copy.deepcopy(child))

            obs_params = ET.SubElement(obs_seq, 'Observational_Parameters')
            ET.SubElement(obs_params, 'Target').text = updated_obs.target
            ET.SubElement(obs_params, 'Priority').text = "1"
            
            timing = ET.SubElement(obs_params, 'Timing')
            ET.SubElement(timing, 'Start').text = format_utc_time(current_time)
            ET.SubElement(timing, 'Stop').text = format_utc_time(obs_end_time)
            
            boresight = ET.SubElement(obs_params, 'Boresight')
            ET.SubElement(boresight, 'RA').text = str(updated_obs.ra)
            ET.SubElement(boresight, 'DEC').text = str(updated_obs.dec)
            
            # Copy payload parameters
            payload = updated_obs.xml_root.find('.//cal:Payload_Parameters', namespaces=NS)
            if payload is not None:
                new_payload = ET.SubElement(obs_seq, 'Payload_Parameters')
                for child in payload:
                    new_payload.append(copy.deepcopy(child))
            
            current_time = obs_end_time
            remaining_seconds -= segment_duration
            scheduled_any = True
    
    # Add visit to master root if we scheduled anything
    if scheduled_any:
        master_root.append(visit_el)
        return visit_id + 1, current_time
    
    return None

# -----------------------------
# Helper: make cardinal direction obs file
# -----------------------------
def create_cardinal_observation(template_obs, ra, dec, cardinal_direction, target_body):
    """Create a synthetic observation for cardinal pointings with proper naming"""
    new_tree = copy.deepcopy(template_obs.xml_tree)
    new_root = new_tree.getroot()
    
    # Create target name as Body_Cardinal (e.g., "Earth_up", "Moon_down")
    target_name = f"{target_body.capitalize()}_{cardinal_direction}"
    
    # Update the target name to indicate cardinal pointing
    target_elements = new_root.findall('.//cal:Target', namespaces=NS)
    for target_el in target_elements:
        target_el.text = target_name
    
    # Update RA/DEC in boresight elements
    ra_elements = new_root.findall('.//cal:RA', namespaces=NS)
    dec_elements = new_root.findall('.//cal:DEC', namespaces=NS)
    
    for ra_el in ra_elements:
        ra_el.text = str(ra)
    for dec_el in dec_elements:
        dec_el.text = str(dec)
    
    return ObservationSequence(
        filename=f"{template_obs.filename}_cardinal_{cardinal_direction}",
        visit_id=template_obs.visit_id,
        obs_id=template_obs.obs_id,
        target=target_name,
        ra=ra,
        dec=dec,
        xml_tree=new_tree,
        xml_root=new_root
    )

# -----------------------------
# Helper: Align time to minute boundaries
# -----------------------------
def align_to_minute_boundary(dt):
    """Align datetime to the next minute boundary, ensuring UTC"""
    utc_dt = ensure_utc_time(dt)
    
    if utc_dt.second == 0 and utc_dt.microsecond == 0:
        return utc_dt
    return utc_dt.replace(second=0, microsecond=0) + timedelta(minutes=1)

# -----------------------------
# Helper: format UTC time
# -----------------------------
def ensure_utc_time(dt):
    """Convert datetime to UTC using astropy.time"""
    if isinstance(dt, datetime):
        # Convert to astropy Time object and ensure UTC
        t = Time(dt, format='datetime', scale='utc')
        return t.datetime
    return dt

def format_utc_time(dt):
    """Format datetime to UTC string with Z suffix, ensuring UTC frame"""
    utc_dt = ensure_utc_time(dt)
    return utc_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# -----------------------------
# Helper: create CVZ visit
# -----------------------------
def create_cvz_visit(cvz_ra: float, cvz_dec: float,
                     start: datetime, stop: datetime,
                     visit_id: int, obs_seq_id: int) -> ET.Element:
    
    # Ensure UTC times
    start_utc = ensure_utc_time(start)
    stop_utc = ensure_utc_time(stop)
    
    visit = ET.Element('Visit')
    ET.SubElement(visit, 'ID').text = f"{visit_id:04d}"

    obs = ET.SubElement(visit, 'Observation_Sequence')
    ET.SubElement(obs, 'ID').text = f"{obs_seq_id:03d}"

    op = ET.SubElement(obs, 'Observational_Parameters')
    ET.SubElement(op, 'Target').text = 'CVZ_IDLE'
    
    # Add Priority=2 after Target, before Timing
    ET.SubElement(op, 'Priority').text = '2'

    timing = ET.SubElement(op, 'Timing')
    ET.SubElement(timing, 'Start').text = format_utc_time(start_utc)
    ET.SubElement(timing, 'Stop').text = format_utc_time(stop_utc)

    bore = ET.SubElement(op, 'Boresight')
    ET.SubElement(bore, 'RA').text = f"{cvz_ra}"
    ET.SubElement(bore, 'DEC').text = f"{cvz_dec}"
    
    # No Payload_Parameters for CVZ observations
    
    return visit

# -----------------------------
# Helper: Make blank stare
# -----------------------------
def modify_for_staring_only(obs: ObservationSequence) -> ObservationSequence:
    """Modify observation to stare without collecting data - remove payload parameters entirely"""
    
    # Create a copy with modified XML
    new_tree = copy.deepcopy(obs.xml_tree)
    new_root = new_tree.getroot()
    
    # Find and remove all Payload_Parameters
    payload_params = new_root.find('.//cal:Payload_Parameters', namespaces=NS)
    if payload_params is not None:
        parent = new_root.find('.//cal:Observation_Sequence', namespaces=NS)
        if parent is not None and payload_params in parent:
            parent.remove(payload_params)
    
    return ObservationSequence(
        filename=obs.filename,
        visit_id=obs.visit_id,
        obs_id=obs.obs_id,
        target=obs.target,
        ra=obs.ra,
        dec=obs.dec,
        xml_tree=new_tree,
        xml_root=new_root
    )

# -----------------------------
# Helper: CVZ visibility check
# -----------------------------
def verify_cvz_visibility(cvz_ra, cvz_dec, start_time, end_time, tle_data):
    """Verify that CVZ coordinates are visible during the specified time window"""
    vis_windows = compute_and_cache_visibility(cvz_ra, cvz_dec, start_time, end_time, 
                                             tle_data[0], tle_data[1])
    total_visible_time = sum((end - start).total_seconds() for start, end in vis_windows)
    total_window_time = (end_time - start_time).total_seconds()
    
    visibility_fraction = total_visible_time / total_window_time if total_window_time > 0 else 0
    
    if visibility_fraction < 0.9:  # Warn if less than 90% visible
        print(f"Warning: CVZ coordinates ({cvz_ra}, {cvz_dec}) only {visibility_fraction:.2%} visible")
    
    return visibility_fraction > 0.5  # Return True if more than 50% visible

# -----------------------------
# Helper: Split long observations
# -----------------------------
def split_long_observation_time(total_duration_s: float, max_duration_s: int = 5400) -> List[float]:
    """Split a duration into segments of max_duration_s or less"""
    if total_duration_s <= max_duration_s:
        return [total_duration_s]
    
    segments = []
    remaining = total_duration_s
    
    while remaining > 0:
        segment = min(remaining, max_duration_s)
        segments.append(segment)
        remaining -= segment
    
    return segments

# -----------------------------
# Helper: get schedulable observations
# -----------------------------
def get_schedulable_observations(obs_list: List[ObservationSequence], 
                               completed_obs_files: set, 
                               completed_tasks: set, 
                               dep_map: Dict) -> List[ObservationSequence]:
    """Get observations that can be scheduled based on dependencies"""
    schedulable = []
    
    # Group observations by task
    remaining_by_task = defaultdict(list)
    for obs in obs_list:
        task_match = re.match(r"(\d{4})", obs.filename)
        if task_match:
            task_num = task_match.group(1)
            obs_file_key = obs.filename[:8]
            if obs_file_key not in completed_obs_files:
                remaining_by_task[task_num].append(obs)
    
    # Find tasks whose dependencies are satisfied
    ready_tasks = []
    for task_num, task_obs in remaining_by_task.items():
        if check_dependencies_satisfied(task_num, completed_tasks, dep_map):
            ready_tasks.append((task_num, task_obs))
    
    # Sort ready tasks by task number (maintaining order within dependencies)
    ready_tasks.sort(key=lambda x: x[0])
    
    # Return observations in proper order
    for task_num, task_obs in ready_tasks:
        for obs in task_obs:
            schedulable.append(obs)
    
    return schedulable

# -----------------------------
# Helper: handle task 0312
# -----------------------------
def create_task_0312_sequences(obs: ObservationSequence) -> List[Tuple[ObservationSequence, int, str]]:
    """
    Create alternating 5-minute data collection and 15-minute staring sequences for task 0312
    Returns list of (observation_sequence, duration_s, sequence_type) tuples
    """
    # Get total observation time from XML
    nir_total, vda_total, nir_int_s, vda_int_s = compute_instrument_durations(obs)
    total_duration_s = int(math.ceil(max(nir_total, vda_total)))
    
    print(f"Task 0312 total observation time: {total_duration_s} seconds ({total_duration_s/60:.1f} minutes)")
    
    sequences = []
    remaining_time = total_duration_s
    is_data_collection = True  # Start with data collection
    
    while remaining_time > 0:
        if is_data_collection:
            # 5-minute data collection sequence
            if remaining_time >= 300:  # 5 minutes
                data_obs = copy.deepcopy(obs)
                sequences.append((data_obs, 300, "data"))
                remaining_time -= 300
            else:
                # Less than 5 minutes left, make final data collection sequence
                data_obs = copy.deepcopy(obs)
                sequences.append((data_obs, remaining_time, "data"))
                remaining_time = 0
        else:
            # 15-minute staring sequence
            if remaining_time >= 900:  # 15 minutes
                stare_obs = copy.deepcopy(obs)
                stare_obs = modify_for_staring_only(stare_obs)
                sequences.append((stare_obs, 900, "stare"))
                remaining_time -= 900
            else:
                # Less than 15 minutes left, make final staring sequence
                stare_obs = copy.deepcopy(obs)
                stare_obs = modify_for_staring_only(stare_obs)
                sequences.append((stare_obs, remaining_time, "stare"))
                remaining_time = 0
        
        # Alternate between data collection and staring
        is_data_collection = not is_data_collection
    
    return sequences

def process_task_0312(obs: ObservationSequence, 
                     current_time: datetime, 
                     cvz_coords: Tuple[float, float], 
                     master_root: ET.Element, 
                     visit_id: int,
                     tle_line1: str, 
                     tle_line2: str) -> Optional[Tuple[int, datetime]]:
    """Process Task 0312 with alternating data collection and staring"""
    
    # Create alternating sequences
    task_0312_sequences = create_task_0312_sequences(obs)
    
    # Create new visit for this task
    visit_el = ET.Element('Visit')
    ET.SubElement(visit_el, 'ID').text = f"{visit_id:04d}"
    obs_seq_id = 1
    scheduled_any = False
    
    for seq_obs, seq_duration, seq_type in task_0312_sequences:
        # Get visibility windows
        vis_windows = compute_and_cache_visibility(obs.ra, obs.dec, current_time, 
                                                 current_time + timedelta(days=7), 
                                                 tle_line1, tle_line2)
        
        # Find suitable visibility window
        scheduled_this_sequence = False
        for window_start, window_end in vis_windows:
            if window_start < current_time:
                window_start = current_time
            
            window_duration_s = (window_end - window_start).total_seconds()
            
            # Fill gap with CVZ if needed
            if window_start > current_time:
                gap_duration = (window_start - current_time).total_seconds()
                if gap_duration >= 120:
                    cvz_segments = split_long_observation_time(gap_duration, 5400)
                    
                    for segment_duration in cvz_segments:
                        segment_end = current_time + timedelta(seconds=segment_duration)
                        
                        cvz_seq = ET.SubElement(visit_el, 'Observation_Sequence')
                        ET.SubElement(cvz_seq, 'ID').text = f"{obs_seq_id:03d}"
                        obs_seq_id += 1
                        
                        cvz_params = ET.SubElement(cvz_seq, 'Observational_Parameters')
                        ET.SubElement(cvz_params, 'Target').text = "CVZ_IDLE"
                        ET.SubElement(cvz_params, 'Priority').text = "2"
                        
                        cvz_timing = ET.SubElement(cvz_params, 'Timing')
                        ET.SubElement(cvz_timing, 'Start').text = format_utc_time(current_time)
                        ET.SubElement(cvz_timing, 'Stop').text = format_utc_time(segment_end)
                        
                        cvz_bore = ET.SubElement(cvz_params, 'Boresight')
                        ET.SubElement(cvz_bore, 'RA').text = f"{cvz_coords[0]}"
                        ET.SubElement(cvz_bore, 'DEC').text = f"{cvz_coords[1]}"
                        
                        current_time = segment_end
                
                current_time = align_to_minute_boundary(window_start)
            
            if window_duration_s >= seq_duration:
                # Schedule this sequence
                obs_duration_minutes = max(2, math.ceil(seq_duration / 60))
                obs_duration_s = obs_duration_minutes * 60
                
                obs_end_time = current_time + timedelta(seconds=obs_duration_s)
                
                obs_seq = ET.SubElement(visit_el, 'Observation_Sequence')
                ET.SubElement(obs_seq, 'ID').text = f"{obs_seq_id:03d}"
                obs_seq_id += 1
                
                obs_params = ET.SubElement(obs_seq, 'Observational_Parameters')
                ET.SubElement(obs_params, 'Target').text = seq_obs.target
                ET.SubElement(obs_params, 'Priority').text = "1"
                
                timing = ET.SubElement(obs_params, 'Timing')
                ET.SubElement(timing, 'Start').text = format_utc_time(current_time)
                ET.SubElement(timing, 'Stop').text = format_utc_time(obs_end_time)
                
                boresight = ET.SubElement(obs_params, 'Boresight')
                ET.SubElement(boresight, 'RA').text = str(seq_obs.ra)
                ET.SubElement(boresight, 'DEC').text = str(seq_obs.dec)
                
                # Copy payload parameters (will be None for staring sequences)
                payload = seq_obs.xml_root.find('.//cal:Payload_Parameters', namespaces=NS)
                if payload is not None:
                    new_payload = ET.SubElement(obs_seq, 'Payload_Parameters')
                    for child in payload:
                        new_payload.append(copy.deepcopy(child))
                
                current_time = obs_end_time
                scheduled_this_sequence = True
                scheduled_any = True
                
                print(f"Scheduled Task 0312 {seq_type} sequence: {seq_duration}s")
                break
        
        if not scheduled_this_sequence:
            print(f"Warning: Could not schedule Task 0312 {seq_type} sequence of {seq_duration}s")
    
    if scheduled_any:
        master_root.append(visit_el)
        return visit_id + 1, current_time
    
    return None

# -----------------------------
# Helper: handle tasks 0341 and 0342
# -----------------------------
def process_cardinal_pointing_task(obs: ObservationSequence, 
                                 task_num: str, 
                                 current_time: datetime, 
                                 pointing_ephem_file: Optional[str],
                                 cvz_coords: Tuple[float, float], 
                                 master_root: ET.Element, 
                                 visit_id: int, 
                                 tle_line1: str, 
                                 tle_line2: str) -> Optional[int]:
    """Process cardinal pointing tasks (0341, 0342)"""
    
    target_body = "earth" if task_num == "0341" else "moon"
    
    if not HAVE_POINTING_PLANNER:
        print(f"Warning: Task {task_num} ({target_body} cardinal pointings) requires pointing_planner module")
        return None
        
    if not pointing_ephem_file or not os.path.exists(pointing_ephem_file):
        print(f"Warning: No ephemeris file provided for task {task_num}")
        return None
    
    try:
        df = pp.load_ephemeris(pointing_ephem_file)
        cardinals = ["up", "ur", "right", "dr", "down", "dl", "left", "ul"]
        
        # Create new visit for this task
        visit_el = ET.Element('Visit')
        ET.SubElement(visit_el, 'ID').text = f"{visit_id:04d}"
        obs_seq_id = 1
        scheduled_any = False
        
        for cardinal in cardinals:
            # Try up to 5 times to find a valid pointing for this cardinal direction
            scheduled_this_cardinal = False
            attempts = 0
            
            while attempts < 2 and not scheduled_this_cardinal:
                attempts += 1
                
                try:
                    # If this is a retry, offset the time slightly to get a different pointing
                    pointing_time = current_time + timedelta(minutes=attempts-1) if attempts > 1 else current_time
                    
                    pointing = pp.compute_pointing(df, pointing_time.strftime('%Y-%m-%d %H:%M:%S'), 
                                                  target_body, cardinal, 15.0)
                    ra, dec = pointing.ra_deg, pointing.dec_deg
                    
                    # Create cardinal observation with proper target naming
                    cardinal_obs = create_cardinal_observation(obs, ra, dec, cardinal, target_body)
                    
                    # Create proper target name for TargetID fields
                    target_name = f"{target_body.capitalize()}_{cardinal}"
                    
                    # Calculate durations
                    nir_total, vda_total, nir_int_s, vda_int_s = compute_instrument_durations(cardinal_obs)
                    remaining_seconds = int(math.ceil(max(nir_total, vda_total)))
                    
                    # Set special keep-out angles
                    kwargs = {}
                    if task_num == "0341":
                        kwargs['earthlimb_min'] = 0*u.deg
                    elif task_num == "0342":
                        kwargs['moon_min'] = 0*u.deg
                    
                    # Get visibility windows
                    vis_windows = compute_and_cache_visibility(ra, dec, current_time, 
                                                             current_time + timedelta(days=7), 
                                                             tle_line1, tle_line2, **kwargs)
                    
                    if not vis_windows:
                        print(f"  Attempt {attempts}/2: No visibility windows found for {target_body}_{cardinal} pointing at RA={ra:.3f}, DEC={dec:.3f}")
                        continue
                    
                    # Schedule this cardinal pointing
                    for window_start, window_end in vis_windows:
                        if remaining_seconds <= 0:
                            break
                            
                        if window_start < current_time:
                            window_start = current_time
                        
                        window_duration_s = (window_end - window_start).total_seconds()
                        
                        if window_duration_s < 120:
                            continue
                            
                        obs_duration_needed = min(remaining_seconds, window_duration_s)
                        obs_duration_minutes = max(2, math.ceil(obs_duration_needed / 60))
                        obs_duration_s = obs_duration_minutes * 60
                        
                        # Split into 90-minute segments if needed
                        obs_segments = split_long_observation_time(obs_duration_s, 5400)
                        
                        for segment_duration in obs_segments:
                            if remaining_seconds <= 0:
                                break
                                
                            obs_end_time = current_time + timedelta(seconds=segment_duration)
                            updated_obs = update_observation_sequence(cardinal_obs, segment_duration)
                            
                            obs_seq = ET.SubElement(visit_el, 'Observation_Sequence')
                            ET.SubElement(obs_seq, 'ID').text = f"{obs_seq_id:03d}"
                            obs_seq_id += 1
                            
                            obs_params = ET.SubElement(obs_seq, 'Observational_Parameters')
                            ET.SubElement(obs_params, 'Target').text = target_name
                            ET.SubElement(obs_params, 'Priority').text = "1"
                            
                            timing = ET.SubElement(obs_params, 'Timing')
                            ET.SubElement(timing, 'Start').text = format_utc_time(current_time)
                            ET.SubElement(timing, 'Stop').text = format_utc_time(obs_end_time)
                            
                            boresight = ET.SubElement(obs_params, 'Boresight')
                            ET.SubElement(boresight, 'RA').text = str(ra)
                            ET.SubElement(boresight, 'DEC').text = str(dec)
                            
                            # Copy payload parameters and update TargetID values
                            payload = updated_obs.xml_root.find('.//cal:Payload_Parameters', namespaces=NS)
                            if payload is not None:
                                new_payload = ET.SubElement(obs_seq, 'Payload_Parameters')
                                for child in payload:
                                    new_child = copy.deepcopy(child)
                                    
                                    # Update all TargetID fields to match the cardinal pointing target name
                                    target_id_elems = new_child.findall('.//cal:TargetID', namespaces=NS)
                                    for target_id_elem in target_id_elems:
                                        target_id_elem.text = target_name
                                    
                                    new_payload.append(new_child)
                            
                            current_time = obs_end_time
                            remaining_seconds -= segment_duration
                            scheduled_any = True
                            scheduled_this_cardinal = True
                        
                        if remaining_seconds <= 0 or scheduled_this_cardinal:
                            break
                    
                except Exception as e:
                    print(f"  Error processing {target_body}_{cardinal} pointing (attempt {attempts}/5): {e}")
            
            if not scheduled_this_cardinal:
                print(f"WARNING: Failed to schedule {target_body}_{cardinal} pointing after 5 attempts")
        
        if scheduled_any:
            master_root.append(visit_el)
            return visit_id + 1
        else:
            print(f"ERROR: Could not schedule any cardinal pointings for task {task_num}")
        
    except Exception as e:
        print(f"Error processing task {task_num}: {e}")
    
    return None

# -----------------------------
# Helper: estimate data volume
# -----------------------------
def estimate_data_volume_bytes(obs: ObservationSequence) -> int:
    root = obs.xml_root
    total_bytes = 0
    nir = root.find('.//cal:AcquireInfCamImages', namespaces=NS)
    if nir is not None:
        ROI_SizeX = int(nir.findtext('cal:ROI_SizeX', '0', namespaces=NS))
        ROI_SizeY = int(nir.findtext('cal:ROI_SizeY', '0', namespaces=NS))
        SC_Integrations = int(nir.findtext('cal:SC_Integrations', '0', namespaces=NS))
        frames = SC_Integrations
        total_bytes += ROI_SizeX * ROI_SizeY * frames * BYTES_PER_PIXEL
    vda = root.find('.//cal:AcquireVisCamImages', namespaces=NS)
    vda_science = root.find('.//cal:AcquireVisCamScienceData', namespaces=NS)
    if vda is not None:
        NumExposures = int(vda.findtext('cal:NumExposures', '0', namespaces=NS))
        ROI_SizeX = int(vda.findtext('cal:ROI_SizeX', '1024', namespaces=NS))
        ROI_SizeY = int(vda.findtext('cal:ROI_SizeY', '1024', namespaces=NS))
        total_bytes += (ROI_SizeX * ROI_SizeY * BYTES_PER_PIXEL + VIS_FRAME_OVERHEAD_BYTES) * NumExposures
    elif vda_science is not None:
        NumTotalFramesRequested = int(vda_science.findtext('cal:NumTotalFramesRequested', '0', namespaces=NS))
        ROI_SizeX = int(vda_science.findtext('cal:ROI_SizeX', '1024', namespaces=NS))
        ROI_SizeY = int(vda_science.findtext('cal:ROI_SizeY', '1024', namespaces=NS))
        total_bytes += (ROI_SizeX * ROI_SizeY * BYTES_PER_PIXEL + VIS_FRAME_OVERHEAD_BYTES) * NumTotalFramesRequested
    return int(total_bytes)

# -----------------------------
# File gatherer
# -----------------------------
def gather_task_xmls(xml_dir: str) -> List[str]:
    files = [f for f in os.listdir(xml_dir) if f.lower().endswith('.xml')]
    def key(fn: str):
        m = re.match(r"(\d{4})_(\d{3})", fn)
        return (int(m.group(1)), int(m.group(2))) if m else (9999, 999)
    return [os.path.join(xml_dir, f) for f in sorted(files, key=key)]

# -----------------------------
# Task dependency / ordering utilities
# -----------------------------

def load_dependencies(dep_file: Optional[str]) -> Dict[str, List[str]]:
    """Load JSON mapping task numbers (as strings) to list of prerequisite tasks."""
    if not dep_file:
        return {}
    with open(dep_file, 'r') as f:
        return json.load(f)

# -----------------------------
# Determine dependency bottlenecks
# -----------------------------
def calculate_dependency_bottlenecks(obs_list: List[ObservationSequence], 
                                   dependency_map: Dict) -> Dict[str, int]:
    """Calculate how many tasks depend on each task (bottleneck score including indirect dependencies)"""
    bottleneck_scores = defaultdict(int)
    
    def count_all_dependents(task_num: str, visited: set = None) -> int:
        if visited is None:
            visited = set()
        if task_num in visited:
            return 0
        visited.add(task_num)
        
        # Find direct dependents
        direct_dependents = [t for t, deps in dependency_map.items() if task_num in deps]
        total_dependents = len(direct_dependents)
        
        # Add indirect dependents
        for dependent in direct_dependents:
            total_dependents += count_all_dependents(dependent, visited.copy())
        
        return total_dependents
    
    # Calculate bottleneck scores for all tasks
    for obs in obs_list:
        task_num = re.match(r"(\d{4})", obs.filename).group(1)
        bottleneck_scores[task_num] = count_all_dependents(task_num)
    
    return dict(bottleneck_scores)

# -----------------------------
# Dependency handling
# -----------------------------
def enforce_dependencies(obs_list: List[ObservationSequence], dep_map: Dict[str, List[str]]) -> List[ObservationSequence]:
    # simple topological sort by task number strings
    graph = {o.filename[:4]: set() for o in obs_list}
    for task, preds in dep_map.items():
        graph.setdefault(task, set()).update(preds)
    result = []
    visited = {}
    def dfs(node):
        if visited.get(node) == 'temp':
            raise RuntimeError("Cycle in dependency graph")
        if visited.get(node) == 'perm':
            return
        visited[node] = 'temp'
        for p in graph.get(node, []):
            dfs(p)
        visited[node] = 'perm'
        for o in obs_list:
            if o.filename.startswith(node) and o not in result:
                result.append(o)
    for node in graph:
        dfs(node)
    return result

def check_dependencies_satisfied(task_num: str, completed_tasks: set, dep_map: Dict) -> bool:
    """Check if all dependencies for a task are satisfied"""
    if task_num not in dep_map:
        return True
    
    dependencies = dep_map[task_num]
    return all(dep in completed_tasks for dep in dependencies)

# -----------------------------
# Find best task for schedule gap
# -----------------------------
def find_best_task_for_gap(available_tasks: List[ObservationSequence],
                          current_time: datetime,
                          gap_duration: float,
                          bottleneck_scores: Dict,
                          tle_line1: str, tle_line2: str) -> Optional[ObservationSequence]:
    """Find the best task to fill a visibility gap based on priority criteria"""
    viable_tasks = []
    
    for obs in available_tasks:
        task_num = re.match(r"(\d{4})", obs.filename).group(1)
        
        # Check if task is visible during this gap
        gap_end = current_time + timedelta(seconds=gap_duration)
        vis_windows = compute_and_cache_visibility(obs.ra, obs.dec, current_time, gap_end,
                                                 tle_line1, tle_line2)
        
        if not vis_windows:
            continue
            
        # Check if there's sufficient visibility during the gap
        has_sufficient_visibility = False
        for window_start, window_end in vis_windows:
            obs_start = max(window_start, current_time)
            obs_end = min(window_end, gap_end)
            available_time = (obs_end - obs_start).total_seconds()
            
            if available_time >= 120:  # Minimum 2 minutes
                has_sufficient_visibility = True
                break
        
        if not has_sufficient_visibility:
            continue
            
        # Calculate scoring criteria
        bottleneck_score = bottleneck_scores.get(task_num, 0)
        
        # Calculate future visibility scarcity
        future_vis_windows = compute_and_cache_visibility(obs.ra, obs.dec, gap_end,
                                                        gap_end + timedelta(days=7),
                                                        tle_line1, tle_line2)
        future_visibility = sum((end-start).total_seconds() for start, end in future_vis_windows)
        scarcity_score = 1.0 / (future_visibility + 1)
        
        # Calculate observation requirement (shorter is better for gap filling)
        nir_total, vda_total, nir_int_s, vda_int_s = compute_instrument_durations(obs)
        remaining_time = max(nir_total, vda_total)
        brevity_score = 1.0 / (remaining_time + 1)
        
        # Combined priority score (weighted according to your priorities)
        priority_score = (bottleneck_score * 3.0 +  # Highest priority: dependency bottlenecks
                         scarcity_score * 2.0 +     # Medium priority: visibility scarcity
                         brevity_score * 1.0)       # Lowest priority: brevity
        
        viable_tasks.append((obs, priority_score, task_num))
    
    if viable_tasks:
        # Sort by priority score (descending) and return best
        viable_tasks.sort(key=lambda x: x[1], reverse=True)
        best_task, score, task_num = viable_tasks[0]
        print(f"Selected task {task_num} for gap filling (bottleneck: {bottleneck_scores.get(task_num, 0)}, score: {score:.3f})")
        return best_task
    
    return None

def schedule_gap_filling_task(gap_task: ObservationSequence, gap_start: datetime, gap_end: datetime,
                             remaining_obs_time: Dict, master_root: ET.Element, visit_id: int,
                             obs_seq_id: int, visits_written: int, 
                             tle_line1: str, tle_line2: str) -> Optional[Tuple[datetime, int, int]]:
    """Schedule a gap-filling task and return updated time and counters"""
    
    gap_task_num = re.match(r"(\d{4})", gap_task.filename).group(1)
    
    # Get visibility windows during the gap
    vis_windows = compute_and_cache_visibility(gap_task.ra, gap_task.dec, gap_start, gap_end,
                                             tle_line1, tle_line2)
    
    current_gap_time = gap_start
    
    for window_start, window_end in vis_windows:
        if window_start < current_gap_time:
            window_start = current_gap_time
        if window_end > gap_end:
            window_end = gap_end
            
        window_duration = (window_end - window_start).total_seconds()
        
        if window_duration >= 120:  # Minimum 2 minutes
            obs_duration_needed = min(remaining_obs_time[gap_task_num], window_duration)
            obs_duration_minutes = max(2, math.ceil(obs_duration_needed / 60))
            obs_duration_s = obs_duration_minutes * 60
            
            gap_obs_end_time = window_start + timedelta(seconds=obs_duration_s)
            updated_gap_obs = update_observation_sequence(gap_task, obs_duration_s)
            
            gap_visit_el = create_visit_element(updated_gap_obs, window_start, 
                                              gap_obs_end_time, visit_id, obs_seq_id)
            master_root.append(gap_visit_el)
            
            remaining_obs_time[gap_task_num] -= obs_duration_needed
            if remaining_obs_time[gap_task_num] <= 0:
                del remaining_obs_time[gap_task_num]
            
            current_gap_time = gap_obs_end_time
            visit_id += 1
            visits_written += 1
            
            print(f"Scheduled gap-filling task {gap_task_num}: {obs_duration_s}s at {format_utc_time(window_start)}")
            return (current_gap_time, visit_id, visits_written)
    
    return None

def find_schedulable_tasks_for_gap(obs_list: List[ObservationSequence], 
                                 current_time: datetime,
                                 gap_end_time: datetime,
                                 completed_tasks: set,
                                 task_progress: Dict[str, Dict],
                                 dep_map: Dict,
                                 tle_line1: str, tle_line2: str) -> Optional[Tuple[ObservationSequence, float]]:
    """
    Find tasks that can be scheduled during a gap
    Returns (observation, available_time) or None
    """
    
    gap_duration = (gap_end_time - current_time).total_seconds()
    if gap_duration < 120:  # Minimum 2 minutes needed
        return None
    
    candidate_tasks = []
    
    for obs in obs_list:
        task_num = re.match(r"(\d{4})", obs.filename).group(1)
        
        # Skip if already completed
        if task_num in completed_tasks:
            continue
            
        # Skip special tasks that have specific scheduling requirements
        if task_num == "0342":  # Moon cardinal pointing task
            continue
            
        # Check dependencies
        if not check_dependencies_satisfied(task_num, completed_tasks, dep_map):
            continue
        
        # Check if task is visible during the gap
        vis_windows = compute_and_cache_visibility(obs.ra, obs.dec, current_time, gap_end_time,
                                                 tle_line1, tle_line2)
        
        if vis_windows:
            # Calculate available observation time
            available_time = 0
            for window_start, window_end in vis_windows:
                overlap_start = max(window_start, current_time)
                overlap_end = min(window_end, gap_end_time)
                if overlap_end > overlap_start:
                    available_time += (overlap_end - overlap_start).total_seconds()
            
            if available_time >= 120:  # At least 2 minutes available
                # Calculate remaining time for this task
                if task_num in task_progress:
                    remaining_seconds = task_progress[task_num]
                else:
                    nir_total, vda_total, nir_int_s, vda_int_s = compute_instrument_durations(obs)
                    remaining_seconds = int(math.ceil(max(nir_total, vda_total)))
                    task_progress[task_num] = remaining_seconds
                
                # Calculate priority score - higher is better
                # Consider: dependency bottleneck factor, time remaining, fit to gap
                dependency_factor = sum(1 for t, deps in dep_map.items() if task_num in deps)
                fit_factor = 1.0 - abs(remaining_seconds - available_time) / max(remaining_seconds, available_time)
                priority_score = dependency_factor * 10 + fit_factor * 5
                
                candidate_tasks.append((obs, priority_score, available_time, task_num))
    
    if candidate_tasks:
        # Sort by priority score (highest first) and return best candidate
        candidate_tasks.sort(key=lambda x: x[1], reverse=True)
        best_task, _, available_time, task_num = candidate_tasks[0]
        return best_task, available_time
    
    return None

# -----------------------------
# Core merging + scheduling loop
# -----------------------------
def merge_schedules(input_paths: List[str], output_path: str,
                    cvz_coords: Tuple[float, float],
                    tle_line1: str, tle_line2: str,
                    commissioning_start: datetime = COMMISSIONING_START,
                    commissioning_end: datetime = COMMISSIONING_END,
                    pointing_ephem_file: Optional[str] = None,
                    dependency_json: Optional[str] = None,
                    progress_json: Optional[str] = None,
                    extra_cvz_json: Optional[str] = None,
                    enable_gap_filling: bool = True) -> Dict:
    
    # Ensure all input times are in UTC
    commissioning_start = ensure_utc_time(commissioning_start)
    commissioning_end = ensure_utc_time(commissioning_end)
    extended_end = ensure_utc_time(commissioning_end + EXTENDED_SCHEDULING_PERIOD)
    
    obs_list = [parse_task_xml(p) for p in input_paths]

    # Load dependency map
    dep_map = {}
    if dependency_json and os.path.exists(dependency_json):
        with open(dependency_json) as f:
            dep_map = json.load(f)
        obs_list = enforce_dependencies(obs_list, dep_map)
    else:
        obs_list.sort(key=lambda o: (int(o.visit_id), int(o.obs_id)))

    # Create task mapping: task_num -> list of observations
    task_to_obs = defaultdict(list)
    for obs in obs_list:
        task_match = re.match(r"(\d{4})", obs.filename)
        if task_match:
            task_num = task_match.group(1)
            task_to_obs[task_num].append(obs)

    # Load progress info and track completion properly
    completed = {}
    completed_obs_files = set()
    completed_tasks = set()
    
    if progress_json and os.path.exists(progress_json):
        with open(progress_json) as f:
            completed = json.load(f)

    # Filter out completed individual observation files
    obs_list_filtered = []
    for obs in obs_list:
        obs_file_key = obs.filename[:8]
        if obs_file_key in completed and completed[obs_file_key] == "done":
            completed_obs_files.add(obs_file_key)
        else:
            obs_list_filtered.append(obs)
    
    obs_list = obs_list_filtered

    # Determine which tasks are completely done
    for task_num, task_obs_list in task_to_obs.items():
        all_completed = True
        for obs in task_obs_list:
            obs_file_key = obs.filename[:8]
            if obs_file_key not in completed_obs_files:
                all_completed = False
                break
        if all_completed:
            completed_tasks.add(task_num)

    # Load extra CVZ blocks
    extra_cvz_blocks = []
    if extra_cvz_json and os.path.exists(extra_cvz_json):
        with open(extra_cvz_json) as f:
            cvz_blocks = json.load(f)
        for block in cvz_blocks:
            s = datetime.fromisoformat(block["start"])
            e = datetime.fromisoformat(block["stop"])
            extra_cvz_blocks.append((s, e))
        extra_cvz_blocks.sort(key=lambda x: x[0])

    master_root = ET.Element('ScienceCalendar')
    if obs_list:
        meta = obs_list[0].xml_root.find('cal:Meta', namespaces=NS)
        if meta is not None:
            master_root.append(meta)

    # Initialize scheduling variables
    current_time = align_to_minute_boundary(commissioning_start)
    tle_data = (tle_line1, tle_line2)
    visit_id = 0
    visits_written = 0
    exceeded_commissioning = False
    
    # For task 0342 (Moon) timing
    full_moon_time = ensure_utc_time(FULL_MOON_TIME)
    moon_task_window = timedelta(days=2)  # Window around full moon
    skipped_0342_observations = set()  # Track which 0342 observations we've skipped

    # Verify CVZ visibility
    verify_cvz_visibility(cvz_coords[0], cvz_coords[1], current_time, extended_end, tle_data)

    # Main scheduling loop - process observations in dependency order
    while True:
        # Check if we've exceeded commissioning period
        if current_time > commissioning_end and not exceeded_commissioning:
            exceeded_commissioning = True
            print(f"Warning: Scheduling has exceeded commissioning period at {format_utc_time(current_time)}")

        # Handle extra CVZ blocks if scheduled for this time
        while extra_cvz_blocks and extra_cvz_blocks[0][0] <= current_time:
            cvz_start, cvz_end = extra_cvz_blocks.pop(0)
            if cvz_end > current_time:
                actual_start = max(cvz_start, current_time)
                cvz_duration = (cvz_end - actual_start).total_seconds()
                
                # Split CVZ into 90-minute chunks if needed
                cvz_segments = split_long_observation_time(cvz_duration, 5400)
                
                for segment_duration in cvz_segments:
                    segment_end = actual_start + timedelta(seconds=segment_duration)
                    cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], actual_start, segment_end, visit_id, 1)
                    master_root.append(cvz_visit)
                    visit_id += 1
                    visits_written += 1
                    actual_start = segment_end
                
                current_time = align_to_minute_boundary(cvz_end)

        # Get next schedulable observations
        schedulable_obs = get_schedulable_observations(obs_list, completed_obs_files, completed_tasks, dep_map)
        
        if not schedulable_obs:
            print("No more schedulable observations - ending schedule")
            break
        
        # Check if we need to process any skipped 0342 observations
        moon_window_start = full_moon_time - moon_task_window
        moon_window_end = full_moon_time + moon_task_window
        near_full_moon = moon_window_start <= current_time <= moon_window_end
        
        # Find non-0342 observations first
        non_moon_obs = []
        moon_obs = []
        
        for obs in schedulable_obs:
            task_match = re.match(r"(\d{4})", obs.filename)
            if task_match:
                task_num = task_match.group(1)
                obs_file_key = obs.filename[:8]
                
                if task_num == "0342":
                    moon_obs.append(obs)
                else:
                    non_moon_obs.append(obs)
        
        # Decide which observation to process
        obs_to_process = None
        
        # If we're near full moon, or there are no other tasks, process 0342
        if (near_full_moon or not non_moon_obs) and moon_obs:
            obs_to_process = moon_obs[0]
            print(f"Processing task 0342 (Moon) - near full moon: {near_full_moon}, no other tasks: {not non_moon_obs}")
        # If we're not near full moon and have other tasks, process non-0342 tasks
        elif non_moon_obs:
            obs_to_process = non_moon_obs[0]
            # Mark any 0342 observations as temporarily skipped (but don't mark them as completed)
            for moon_ob in moon_obs:
                obs_file_key = moon_ob.filename[:8]
                if obs_file_key not in skipped_0342_observations:
                    skipped_0342_observations.add(obs_file_key)
                    print(f"Skipping task 0342 observation {moon_ob.filename} until closer to full moon (current: {current_time.strftime('%Y-%m-%d')}, target: {full_moon_time.strftime('%Y-%m-%d')})")
        # If we only have 0342 observations left and we're not near full moon, process them anyway
        elif moon_obs:
            obs_to_process = moon_obs[0]
            print(f"Processing task 0342 anyway - no other tasks remaining")
        
        if not obs_to_process:
            print("No observations to process - ending schedule")
            break
        
        task_match = re.match(r"(\d{4})", obs_to_process.filename)
        if not task_match:
            # Remove this observation and continue
            obs_file_key = obs_to_process.filename[:8]
            completed_obs_files.add(obs_file_key)
            continue
            
        task_num = task_match.group(1)
        obs_file_key = obs_to_process.filename[:8]
        
        print(f"Processing task {task_num}: {obs_to_process.filename}, RA: {obs_to_process.ra}, Dec: {obs_to_process.dec}")
        
        # Double-check dependencies before processing
        if not check_dependencies_satisfied(task_num, completed_tasks, dep_map):
            missing_deps = [d for d in dep_map.get(task_num, []) if d not in completed_tasks]
            print(f"Task {task_num} dependencies not satisfied: {missing_deps}, cannot schedule yet.")
            break  # Wait for dependencies to be satisfied
        
        # Process different task types
        success = None
        
        if task_num in ["0341", "0342"]:
            # Cardinal pointing tasks
            success = process_cardinal_pointing_task(obs_to_process, task_num, current_time, pointing_ephem_file,
                                                   cvz_coords, master_root, visit_id, tle_line1, tle_line2)
            if success:
                visit_id = success
                visits_written += 1
                completed_obs_files.add(obs_file_key)
                
        elif task_num == "0312":
            # Special alternating task
            success = process_task_0312(obs_to_process, current_time, cvz_coords, master_root, visit_id, 
                                      tle_line1, tle_line2)
            if success:
                visit_id, current_time = success
                visits_written += 1
                completed_obs_files.add(obs_file_key)
                
        else:
            # Normal task processing
            success = process_normal_task(obs_to_process, current_time, cvz_coords, master_root, visit_id,
                                        tle_line1, tle_line2)
            if success:
                visit_id, current_time = success
                visits_written += 1
                completed_obs_files.add(obs_file_key)
        
        if not success:
            print(f"Failed to schedule {obs_to_process.filename}, marking as completed to avoid infinite loop")
            completed_obs_files.add(obs_file_key)
        
        # Check if task is now complete
        task_obs_remaining = [o for o in task_to_obs[task_num] 
                             if o.filename[:8] not in completed_obs_files]
        if not task_obs_remaining:
            completed_tasks.add(task_num)
            print(f"All observations for task {task_num} completed")

    # Update Meta values in the XML
    meta = master_root.find('cal:Meta', namespaces=NS)
    if meta is not None:
        # Find first and last observation times
        all_start_times = []
        all_stop_times = []
        
        for visit in master_root.findall('.//cal:Visit', namespaces=NS):
            for obs_seq in visit.findall('.//cal:Observation_Sequence', namespaces=NS):
                timing = obs_seq.find('.//cal:Timing', namespaces=NS)
                if timing is not None:
                    start = timing.find('cal:Start', namespaces=NS)
                    stop = timing.find('cal:Stop', namespaces=NS)
                    
                    if start is not None and start.text:
                        all_start_times.append(start.text)
                    
                    if stop is not None and stop.text:
                        all_stop_times.append(stop.text)
        
        # Update Meta values if we found observation times
        if all_start_times and all_stop_times:
            # Use attrib dictionary to update attributes directly
            meta.attrib['Valid_From'] = min(all_start_times)
            meta.attrib['Expires'] = max(all_stop_times)
            
            # Update Created time to current UTC time
            now = datetime.now(timezone.utc)
            meta.attrib['Created'] = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Update Keepout_Angles
            meta.attrib['Keepout_Angles'] = "90.0, 25.0, 63.0"

    # Write output
    tree = ET.ElementTree(master_root)
    indent(tree.getroot(), 0)
    tree.write(output_path, encoding='unicode', xml_declaration=True)
    
    return {
        "visits_written": visits_written,
        "total_observations": len([obs for obs in obs_list if obs.filename[:8] not in completed_obs_files]),
        "schedule_end": current_time.isoformat(),
        "exceeded_commissioning": exceeded_commissioning
    }

# def merge_schedules(input_paths: List[str], output_path: str,
#                     cvz_coords: Tuple[float, float],
#                     tle_line1: str, tle_line2: str,
#                     commissioning_start: datetime = COMMISSIONING_START,
#                     commissioning_end: datetime = COMMISSIONING_END,
#                     pointing_ephem_file: Optional[str] = None,
#                     dependency_json: Optional[str] = None,
#                     progress_json: Optional[str] = None,
#                     extra_cvz_json: Optional[str] = None,
#                     enable_gap_filling: bool = False) -> Dict:
   
#    # Ensure all input times are in UTC
#     commissioning_start = ensure_utc_time(commissioning_start)
#     commissioning_end = ensure_utc_time(commissioning_end)

#     # Calculate extended end time for scheduling
#     extended_end = ensure_utc_time(commissioning_end + EXTENDED_SCHEDULING_PERIOD)
    
#     obs_list = [parse_task_xml(p) for p in input_paths]

#     # load dependency / progress / cvz files
#     dep_map = {}
#     if dependency_json and os.path.exists(dependency_json):
#         with open(dependency_json) as f:
#             dep_map = json.load(f)
#         obs_list = enforce_dependencies(obs_list, dep_map)
#     else:
#         obs_list.sort(key=lambda o: (int(o.visit_id), int(o.obs_id)))

#     completed = {}
#     if progress_json and os.path.exists(progress_json):
#         with open(progress_json) as f:
#             completed = json.load(f)

#     obs_list = [o for o in obs_list if not (o.filename[:4] in completed and completed[o.filename[:4]] == "done")]

#    # Load extra CVZ blocks for timeline integration
#     extra_cvz_blocks = []
#     if extra_cvz_json and os.path.exists(extra_cvz_json):
#         with open(extra_cvz_json) as f:
#             cvz_blocks = json.load(f)
#         for block in cvz_blocks:
#             s = datetime.fromisoformat(block["start"])
#             e = datetime.fromisoformat(block["stop"])
#             extra_cvz_blocks.append((s, e))
#         # Sort CVZ blocks by start time
#         extra_cvz_blocks.sort(key=lambda x: x[0])

#     master_root = ET.Element('ScienceCalendar')
#     if obs_list:
#         meta = obs_list[0].xml_root.find('cal:Meta', namespaces=NS)
#         if meta is not None:
#             master_root.append(meta)

#     # Initialize scheduling variables
#     current_time = align_to_minute_boundary(commissioning_start)
#     tle_data = (tle_line1, tle_line2)
#     visit_id = 0
#     obs_seq_id = 1
#     total_bytes = 0
#     visits_written = 0
#     exceeded_commissioning = False

#     last_target = None
#     visit_el = None

#     # Calculate dependency bottlenecks for gap-filling optimization (if enabled)
#     bottleneck_scores = {}
#     remaining_obs_time = {}
    
#     if enable_gap_filling:
#         bottleneck_scores = calculate_dependency_bottlenecks(obs_list, dep_map)
        
#         # Track remaining observation time for each task
#         for obs in obs_list:
#             task_num = re.match(r"(\d{4})", obs.filename).group(1)
#             if task_num not in completed or (isinstance(completed[task_num], dict) and "remaining" in completed[task_num]):
#                 if task_num in completed and isinstance(completed[task_num], dict):
#                     remaining_obs_time[task_num] = completed[task_num]["remaining"]
#                 else:
#                     nir_total, vda_total, nir_int_s, vda_int_s = compute_instrument_durations(obs)
#                     remaining_obs_time[task_num] = int(math.ceil(max(nir_total, vda_total)))

#     # Verify CVZ visibility
#     verify_cvz_visibility(cvz_coords[0], cvz_coords[1], current_time, commissioning_end, tle_data)

#     print(obs_list)
#     for obs in obs_list:
#         print("Parsed: ", obs.filename, "RA: ", obs.ra, "Dec: ", obs.dec)
#         tasknum = re.match(r"(\d{4})", obs.filename).group(1)

#         if current_time > commissioning_end and not exceeded_commissioning:
#             exceeded_commissioning = True
#             print(f"Warning: Scheduling has exceeded commissioning period at {format_utc_time(current_time)}")

#         # Check if any extra CVZ blocks should be scheduled before this observation
#         while extra_cvz_blocks and extra_cvz_blocks[0][0] <= current_time:
#             cvz_start, cvz_end = extra_cvz_blocks.pop(0)
#             if cvz_end > current_time:
#                 # Schedule this CVZ block
#                 actual_start = max(cvz_start, current_time)
#                 cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], actual_start, cvz_end, visit_id, obs_seq_id)
#                 master_root.append(cvz_visit)
#                 visit_id += 1
#                 visits_written += 1
#                 current_time = align_to_minute_boundary(cvz_end)

#         # Handle special tasks 0341 and 0342 (Earth/Moon cardinal pointings)
#         if tasknum in ["0341", "0342"]:
#             target_body = "earth" if tasknum == "0341" else "moon"
            
#             # For task 0342 (Moon), prioritize scheduling near full moon
#             if tasknum == "0342":
#                 # Calculate optimal scheduling window around full moon (±12 hours)
#                 optimal_start = ensure_utc_time(FULL_MOON_TIME) - timedelta(hours=12)
#                 optimal_end = ensure_utc_time(FULL_MOON_TIME) + timedelta(hours=12)
                
#                 # If we're not yet at the optimal window, schedule CVZ until then
#                 if current_time < optimal_start:
#                     cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                                optimal_start, visit_id, obs_seq_id)
#                     master_root.append(cvz_visit)
#                     visit_id += 1
#                     visits_written += 1
#                     current_time = align_to_minute_boundary(optimal_start)
#                     print(f"Scheduled CVZ until full moon optimal window for task 0342")

#             if not HAVE_POINTING_PLANNER:
#                 print(f"Warning: Task {tasknum} ({target_body} cardinal pointings) requires pointing_planner module")
#                 continue
                
#             # Get cardinal pointings for this body at current time
#             try:
#                 # Load ephemeris data for pointing planner
#                 if pointing_ephem_file and os.path.exists(pointing_ephem_file):
#                     df = pp.load_ephemeris(pointing_ephem_file)
                    
#                     # Get 8 cardinal pointings for this body
#                     cardinals = ["up", "ur", "right", "dr", "down", "dl", "left", "ul"]
#                     cardinal_coords = []
                    
#                     for cardinal in cardinals:
#                         try:
#                             pointing = pp.compute_pointing(df, current_time.strftime('%Y-%m-%d %H:%M:%S'), 
#                                                           target_body, cardinal, 15.0)
#                             cardinal_coords.append((pointing.ra_deg, pointing.dec_deg))
#                         except Exception as e:
#                             print(f"Warning: Could not compute {cardinal} pointing for {target_body}: {e}")
#                             continue
                    
#                     for i, (ra, dec) in enumerate(cardinal_coords):
#                         # Create a synthetic observation for this pointing
#                         cardinal_obs = create_cardinal_observation(obs, ra, dec, i+1)
                        
#                         # Process this cardinal pointing like a regular observation
#                         nir_total, vda_total, nir_int_s, vda_int_s = compute_instrument_durations(cardinal_obs)
#                         remaining_seconds = int(math.ceil(max(nir_total, vda_total)))
                        
#                         # Set special keep-out angles for cardinal pointings
#                         kwargs = {}
#                         if tasknum == "0341":  # Earth cardinal pointings
#                             kwargs['earthlimb_min'] = 0*u.deg
#                         elif tasknum == "0342":  # Moon cardinal pointings
#                             kwargs['moon_min'] = 0*u.deg
                        
#                         # Calculate visibility windows for this cardinal pointing with special keep-out
#                         vis_windows = compute_and_cache_visibility(ra, dec, current_time, 
#                                                                  current_time + timedelta(days=7), 
#                                                                  tle_line1, tle_line2, **kwargs)
                        
#                         # Check if target changed (for slew time)
#                         target_changed = (last_target is None or 
#                                          abs(ra - last_target[0]) > 1e-6 or 
#                                          abs(dec - last_target[1]) > 1e-6)
                        
#                         # Add slew time if target changed
#                         if target_changed and last_target is not None:
#                             current_time = align_to_minute_boundary(current_time + timedelta(minutes=1))
                        
#                         # Process visibility windows for this cardinal pointing
#                         scheduled_this_pointing = False
#                         for window_start, window_end in vis_windows:
#                             if window_start < current_time:
#                                 window_start = current_time
                            
#                             window_duration_s = (window_end - window_start).total_seconds()
                            
#                             # Fill gap with CVZ if there's time between current_time and window_start
#                             if window_start > current_time:
#                                 cvz_duration = (window_start - current_time).total_seconds()
#                                 if cvz_duration >= 120:  # At least 2 minutes

#                                     if enable_gap_filling:
#                                         # Try to find another task to fill this gap
#                                         available_tasks = [o for o in obs_list 
#                                                          if re.match(r"(\d{4})", o.filename).group(1) in remaining_obs_time
#                                                          and re.match(r"(\d{4})", o.filename).group(1) != tasknum]
                                        
#                                         gap_fill_task = find_best_task_for_gap(available_tasks, current_time, gap_duration,
#                                                                              bottleneck_scores, tle_line1, tle_line2)
                                        
#                                         if gap_fill_task:
#                                             gap_task_num = re.match(r"(\d{4})", gap_fill_task.filename).group(1)
#                                             print(f"Filling gap with task {gap_task_num} instead of CVZ")
                                            
#                                             # Schedule this gap-filling task
#                                             gap_scheduled = schedule_gap_filling_task(gap_fill_task, current_time, window_start,
#                                                                                     remaining_obs_time, master_root, visit_id,
#                                                                                     obs_seq_id, visits_written, tle_line1, tle_line2)
                                            
#                                             if gap_scheduled:
#                                                 current_time, visit_id, visits_written = gap_scheduled
#                                             else:
#                                                 # Fall back to CVZ
#                                                 cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                                                            window_start, visit_id, obs_seq_id)
#                                                 master_root.append(cvz_visit)
#                                                 visit_id += 1
#                                                 visits_written += 1
#                                         else:
#                                             # Fall back to CVZ if no suitable task found
#                                             cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                                                        window_start, visit_id, obs_seq_id)
#                                             master_root.append(cvz_visit)
#                                             visit_id += 1
#                                             visits_written += 1
#                                     else:
#                                         # Original behavior - just schedule CVZ
#                                         cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                                                    window_start, visit_id, obs_seq_id)
#                                         master_root.append(cvz_visit)
#                                         visit_id += 1
#                                         visits_written += 1
                                
#                                 current_time = align_to_minute_boundary(window_start)
                            
#                             # Skip windows shorter than 2 minutes
#                             if window_duration_s < 120:
#                                 continue
                                
#                             # Calculate how much observation time we need
#                             obs_duration_needed = min(remaining_seconds, window_duration_s)
                            
#                             # Round up to nearest minute, minimum 2 minutes
#                             obs_duration_minutes = max(2, math.ceil(obs_duration_needed / 60))
#                             obs_duration_s = obs_duration_minutes * 60
                            
#                             # Create observation sequence
#                             obs_end_time = current_time + timedelta(seconds=obs_duration_s)
                            
#                             # Update XML with new duration and frame counts
#                             updated_obs = update_observation_sequence(cardinal_obs, obs_duration_s)
                            
#                             # Create visit XML
#                             visit_el = create_visit_element(updated_obs, current_time, obs_end_time, 
#                                                           visit_id, obs_seq_id)
#                             master_root.append(visit_el)
                            
#                             # Update counters
#                             remaining_seconds -= obs_duration_needed
#                             current_time = obs_end_time
#                             visit_id += 1
#                             visits_written += 1
#                             scheduled_this_pointing = True
                            
#                             if remaining_seconds <= 0:
#                                 break
                        
#                         # If we couldn't schedule this cardinal pointing, fill with CVZ
#                         if not scheduled_this_pointing and remaining_seconds > 0:
#                             cvz_end_time = align_to_minute_boundary(current_time + timedelta(seconds=remaining_seconds))
#                             if cvz_end_time > current_time:
#                                 cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                                            cvz_end_time, visit_id, obs_seq_id)
#                                 master_root.append(cvz_visit)
#                                 visit_id += 1
#                                 visits_written += 1
#                                 current_time = cvz_end_time
                        
#                         last_target = (ra, dec)
                        
#             except Exception as e:
#                 print(f"Error processing task {tasknum}: {e}")
#                 continue
#             else:
#                 print(f"Warning: No ephemeris file provided for task {tasknum}")
#                 continue
            
#             continue  # Skip normal processing for cardinal pointing tasks

#         # Special handling for task 0312
#         if tasknum == "0312":
#             # Create alternating sequences using existing function
#             task_0312_sequences = create_task_0312_sequences(obs)
            
#             # Create single Visit element for all Task 0312 sequences
#             if obs.target != last_target:
#                 if visit_el is not None:
#                     master_root.append(visit_el)
#                     visits_written += 1
#                 visit_el = ET.Element('Visit')
#                 vid = ET.SubElement(visit_el, 'ID'); vid.text = f"{visit_id:04d}"
#                 visit_id += 1
#                 obs_seq_id = 1
#                 # last_target = obs.target
#                 last_target = (obs.ra, obs.dec)
            
#             for seq_obs, seq_duration, seq_type in task_0312_sequences:
#                 # Calculate visibility windows for this target
#                 vis_windows = compute_and_cache_visibility(obs.ra, obs.dec, current_time, 
#                                                          current_time + timedelta(days=7), 
#                                                          tle_line1, tle_line2)
                
#                 # Find a visibility window that can accommodate the full sequence duration
#                 scheduled_this_sequence = False
#                 for window_start, window_end in vis_windows:
#                     if window_start < current_time:
#                         window_start = current_time
                    
#                     window_duration_s = (window_end - window_start).total_seconds()
                    
#                     # Fill gap with CVZ if there's time between current_time and window_start
#                     if window_start > current_time:
#                         cvz_duration = (window_start - current_time).total_seconds()
#                         if cvz_duration >= 120:  # At least 2 minutes

#                             if enable_gap_filling:
#                                 # Try to find another task to fill this gap
#                                 available_tasks = [o for o in obs_list 
#                                                  if re.match(r"(\d{4})", o.filename).group(1) in remaining_obs_time
#                                                  and re.match(r"(\d{4})", o.filename).group(1) != tasknum]
                                
#                                 gap_fill_task = find_best_task_for_gap(available_tasks, current_time, gap_duration,
#                                                                      bottleneck_scores, tle_line1, tle_line2)
                                
#                                 if gap_fill_task:
#                                     gap_task_num = re.match(r"(\d{4})", gap_fill_task.filename).group(1)
#                                     print(f"Filling gap with task {gap_task_num} instead of CVZ")
                                    
#                                     # Schedule this gap-filling task
#                                     gap_scheduled = schedule_gap_filling_task(gap_fill_task, current_time, window_start,
#                                                                             remaining_obs_time, master_root, visit_id,
#                                                                             obs_seq_id, visits_written, tle_line1, tle_line2)
                                    
#                                     if gap_scheduled:
#                                         current_time, visit_id, visits_written = gap_scheduled
#                                     else:
#                                         # Fall back to CVZ
#                                         cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                                                    window_start, visit_id, obs_seq_id)
#                                         master_root.append(cvz_visit)
#                                         visit_id += 1
#                                         visits_written += 1
#                                 else:
#                                     # Fall back to CVZ if no suitable task found
#                                     cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                                                window_start, visit_id, obs_seq_id)
#                                     master_root.append(cvz_visit)
#                                     visit_id += 1
#                                     visits_written += 1
#                             else:
#                                 # Original behavior - just schedule CVZ
#                                 cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                                            window_start, visit_id, obs_seq_id)
#                                 master_root.append(cvz_visit)
#                                 visit_id += 1
#                                 visits_written += 1
                        
#                         current_time = align_to_minute_boundary(window_start)
                    
#                     # Check if this window can accommodate the full sequence duration
#                     if window_duration_s >= seq_duration:
#                         # Round up to nearest minute, minimum 2 minutes
#                         obs_duration_minutes = max(2, math.ceil(seq_duration / 60))
#                         obs_duration_s = obs_duration_minutes * 60
                        
#                         # Create observation sequence within the same Visit
#                         obs_seq = ET.SubElement(visit_el, 'Observation_Sequence')
#                         oid = ET.SubElement(obs_seq, 'ID'); oid.text = f"{obs_seq_id:03d}"
#                         obs_seq_id += 1
                        
#                         # Add observational parameters
#                         obs_params = ET.SubElement(obs_seq, 'Observational_Parameters')
#                         target_el = ET.SubElement(obs_params, 'Target'); target_el.text = seq_obs.target
#                         priority_el = ET.SubElement(obs_params, 'Priority'); priority_el.text = "1"
                        
#                         # Timing comes before Boresight
#                         timing = ET.SubElement(obs_params, 'Timing')
#                         start_el = ET.SubElement(timing, 'Start')
#                         stop_el = ET.SubElement(timing, 'Stop')
#                         obs_end_time = current_time + timedelta(seconds=obs_duration_s)
#                         start_el.text = format_utc_time(current_time)
#                         stop_el.text = format_utc_time(obs_end_time)
                        
#                         # Boresight
#                         boresight = ET.SubElement(obs_params, 'Boresight')
#                         ra_el = ET.SubElement(boresight, 'RA'); ra_el.text = str(seq_obs.ra)
#                         dec_el = ET.SubElement(boresight, 'DEC'); dec_el.text = str(seq_obs.dec)
                        
#                         # Update XML with new duration and frame counts
#                         updated_obs = update_observation_sequence(seq_obs, obs_duration_s)
                        
#                         # Copy payload parameters from updated observation
#                         original_payload = updated_obs.xml_root.find('.//cal:Payload_Parameters', namespaces=NS)
#                         if original_payload is not None:
#                             for child in original_payload:
#                                 obs_seq.append(copy.deepcopy(child))
                        
#                         # Update counters
#                         current_time = obs_end_time
#                         scheduled_this_sequence = True
                        
#                         print(f"Scheduled Task 0312 {seq_type} sequence {obs_seq_id-1:03d}: {seq_duration}s at {format_utc_time(current_time - timedelta(seconds=obs_duration_s))}")
#                         break
                
#                 # If we couldn't schedule this sequence, fill with CVZ and try to continue
#                 if not scheduled_this_sequence:
#                     print(f"Warning: Could not schedule Task 0312 {seq_type} sequence of {seq_duration}s")
#                     # Add some CVZ time and continue
#                     cvz_end_time = align_to_minute_boundary(current_time + timedelta(minutes=10))
#                     cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                                cvz_end_time, visit_id, obs_seq_id)
#                     master_root.append(cvz_visit)
#                     visit_id += 1
#                     visits_written += 1
#                     current_time = cvz_end_time

#             # Append the Task 0312 visit to master_root
#             if visit_el is not None:
#                 master_root.append(visit_el)
#                 visits_written += 1
#                 visit_el = None
            
#             # No slew time needed since we're staying on the same target
#             # Don't update last_target here since we want to keep the same Visit
#             continue  # Skip normal processing for task 0312

#         # Normal task processing (not 0341/0342/0312)
#         # check progress
#         if tasknum in completed and isinstance(completed[tasknum], dict):
#             if "remaining" in completed[tasknum]:
#                 remaining_seconds = completed[tasknum]["remaining"]
#             else:
#                 continue
#         else:
#             nir_total, vda_total, nir_int_s, vda_int_s = compute_instrument_durations(obs)
#             remaining_seconds = int(math.ceil(max(nir_total, vda_total)))
#         print(nir_total, vda_total, nir_int_s, vda_int_s)

#         vis_windows = compute_and_cache_visibility(
#             obs.ra, obs.dec, current_time, commissioning_end,
#             tle_line1, tle_line2,
#         )

#         print(obs.target)
#         print(obs.ra, obs.dec, current_time, commissioning_end)
#         # print(tle_line1, tle_line2)
#         print(vis_windows)

#         # Check if target changed (for slew time)
#         print(last_target)
#         # print(obs.ra, last_target[0], obs.dec, last_target[1])
#         target_changed = (
#             last_target is None or
#             abs(obs.ra - last_target[0]) > 1e-6 or
#             abs(obs.dec - last_target[1]) > 1e-6
#         )

#         # Add slew time if target changed
#         if target_changed and last_target is not None:
#             current_time = align_to_minute_boundary(current_time + timedelta(minutes=1))

#         # Process each visibility window
#         for window_start, window_end in vis_windows:
#             if window_start < current_time:
#                 window_start = current_time
            
#             window_duration_s = (window_end - window_start).total_seconds()
            
#             # Fill gap with CVZ if there's time between current_time and window_start
#             if window_start > current_time:
#                 gap_duration = (window_start - current_time).total_seconds()
#                 if gap_duration >= 120:  # At least 2 minutes

#                     if enable_gap_filling:
#                         # Try to find another task to fill this gap
#                         available_tasks = [o for o in obs_list 
#                                          if re.match(r"(\d{4})", o.filename).group(1) in remaining_obs_time
#                                          and re.match(r"(\d{4})", o.filename).group(1) != tasknum]
                        
#                         gap_fill_task = find_best_task_for_gap(available_tasks, current_time, gap_duration,
#                                                              bottleneck_scores, tle_line1, tle_line2)
                        
#                         if gap_fill_task:
#                             gap_task_num = re.match(r"(\d{4})", gap_fill_task.filename).group(1)
#                             print(f"Filling gap with task {gap_task_num} instead of CVZ")
                            
#                             # Schedule this gap-filling task
#                             gap_scheduled = schedule_gap_filling_task(gap_fill_task, current_time, window_start,
#                                                                     remaining_obs_time, master_root, visit_id,
#                                                                     obs_seq_id, visits_written, tle_line1, tle_line2)
                            
#                             if gap_scheduled:
#                                 current_time, visit_id, visits_written = gap_scheduled
#                             else:
#                                 # Fall back to CVZ
#                                 cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                                            window_start, visit_id, obs_seq_id)
#                                 master_root.append(cvz_visit)
#                                 visit_id += 1
#                                 visits_written += 1
#                         else:
#                             # Fall back to CVZ if no suitable task found
#                             cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                                        window_start, visit_id, obs_seq_id)
#                             master_root.append(cvz_visit)
#                             visit_id += 1
#                             visits_written += 1
#                     else:
#                         # Original behavior - just schedule CVZ
#                         cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                                    window_start, visit_id, obs_seq_id)
#                         master_root.append(cvz_visit)
#                         visit_id += 1
#                         visits_written += 1
                
#                 current_time = align_to_minute_boundary(window_start)
            
#             # Skip windows shorter than 2 minutes
#             if window_duration_s < 120:
#                 continue
                
#             # Calculate how much observation time we need
#             obs_duration_needed = min(remaining_seconds, window_duration_s)
            
#             # Round up to nearest minute, minimum 2 minutes
#             obs_duration_minutes = max(2, math.ceil(obs_duration_needed / 60))
#             obs_duration_s = obs_duration_minutes * 60
            
#             # Create observation sequence
#             obs_end_time = current_time + timedelta(seconds=obs_duration_s)
            
#             # Update XML with new duration and frame counts
#             updated_obs = update_observation_sequence(obs, obs_duration_s)
            
#             # Create visit XML
#             visit_el = create_visit_element(updated_obs, current_time, obs_end_time, 
#                                           visit_id, obs_seq_id)
#             master_root.append(visit_el)
            
#             # Update counters
#             remaining_seconds -= obs_duration_needed
#             current_time = obs_end_time
#             visit_id += 1
#             visits_written += 1

#             # Update remaining observation time for gap filling
#             if enable_gap_filling and tasknum in remaining_obs_time:
#                 remaining_obs_time[tasknum] -= obs_duration_needed
#                 if remaining_obs_time[tasknum] <= 0:
#                     del remaining_obs_time[tasknum]
            
#             if remaining_seconds <= 0:
#                 break
        
#         # If we still have remaining time, schedule CVZ until next target or end
#         if remaining_seconds > 0:
#             # Schedule CVZ for remaining time or until next suitable visibility window
#             next_good_window = find_next_visibility_window(obs.ra, obs.dec, current_time, 
#                                                          current_time + timedelta(days=7), 
#                                                          (tle_line1, tle_line2), min_duration_s=120)
#             if next_good_window:
#                 cvz_end_time = next_good_window[0]
#             else:
#                 cvz_end_time = align_to_minute_boundary(current_time + timedelta(seconds=remaining_seconds))
            
#             if cvz_end_time > current_time:
#                 cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, 
#                                            cvz_end_time, visit_id, obs_seq_id)
#                 master_root.append(cvz_visit)
#                 visit_id += 1
#                 visits_written += 1
#                 current_time = cvz_end_time
        
#         last_target = (obs.ra, obs.dec)

#     # Schedule any remaining extra CVZ blocks at the end
#     while extra_cvz_blocks:
#         cvz_start, cvz_end = extra_cvz_blocks.pop(0)
#         actual_start = max(cvz_start, current_time)
#         if cvz_end > actual_start:
#             cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], actual_start, cvz_end, visit_id, obs_seq_id)
#             master_root.append(cvz_visit)
#             visit_id += 1
#             visits_written += 1
#             current_time = align_to_minute_boundary(cvz_end)

#         # wi = 0
#         # while remaining_seconds > 0 and current_time < commissioning_end:
#         #     if wi >= len(vis_windows):
#         #         break
#         #     win_start, win_stop = vis_windows[wi]

#         #     # Seconds available in this visibility window
#         #     usable = int((win_stop - current_time).total_seconds())
#         #     if usable <= 0:
#         #         wi += 1
#         #         current_time = win_stop
#         #         continue
#         #     # The chunk is the smaller if (time left in window, time left in task)
#         #     chunk = min(usable, remaining_seconds)

#         #     # If chunk >= ONE_MIN, snap down to nearest multiple of ONE_MIN
#         #     # Otherwise, allow the short chunk (don't round to 0!)
#         #     print(chunk)
#         #     if chunk >= ONE_MIN:
#         #         chunk = (chunk // ONE_MIN) * ONE_MIN
#         #         if chunk == 0:
#         #             chunk = ONE_MIN
#         #     else:
#         #         chunk = chunk
#         #     print(chunk)

#         #     # Create/reuse Visit
#         #     if obs.target != last_target:
#         #         if visit_el is not None:
#         #             master_root.append(visit_el)
#         #             visits_written += 1
#         #         visit_el = ET.Element('Visit')
#         #         vid = ET.SubElement(visit_el, 'ID'); vid.text = f"{visit_id:04d}"
#         #         visit_id += 1
#         #         obs_seq_id = 1
#         #         last_target = obs.target

#         #     # Add Observation_Sequence
#         #     obs_seq = ET.SubElement(visit_el, 'Observation_Sequence')
#         #     oid = ET.SubElement(obs_seq, 'ID'); oid.text = f"{obs_seq_id:03d}"
#         #     obs_seq_id += 1

#         #     # Clone payload, adjusted for chunk
#         #     chunk_obs = adjust_payload_for_chunk(obs, chunk, nir_int_s, vda_int_s)
#         #     template_obs = chunk_obs.xml_root.find('.//cal:Observation_Sequence', namespaces=NS)
#         #     if template_obs is not None:
#         #         for child in template_obs:
#         #             if child.tag.endswith('ID'):
#         #                 continue  # Skip duplicate ID
#         #             obs_seq.append(ET.fromstring(ET.tostring(child)))

#         #     # Update timing
#         #     op = obs_seq.find('cal:Observational_Parameters', namespaces=NS)
#         #     old_timing = op.find('cal:Timing', namespaces=NS)
#         #     if old_timing is not None:
#         #         op.remove(old_timing)
#         #     timing_el = ET.SubElement(obs_seq.find('cal:Observational_Parameters', namespaces=NS), 'Timing')

#         #     chunk_start = current_time
#         #     chunk_stop = current_time + timedelta(seconds=chunk)
#         #     chunk_start_str = Time(chunk_start, scale="utc").isot + "Z"
#         #     chunk_stop_str = Time(chunk_stop, scale="utc").isot + "Z"
#         #     ET.SubElement(timing_el, 'Start').text = chunk_start_str
#         #     ET.SubElement(timing_el, 'Stop').text = chunk_stop_str

#         #     children = list(op)
#         #     insert_index = 2 if len(children) >= 2 else len(children)
#         #     op.insert(insert_index, timing_el)

#         #     # Estimate data volume
#         #     chunk_bytes = estimate_data_volume_bytes(chunk_obs)
#         #     total_bytes += chunk_bytes

#         #     remaining_seconds -= chunk
#         #     current_time = chunk_stop
#         #     if current_time >= win_stop:
#         #         wi += 1

#     # if visit_el is not None:
#     #     master_root.append(visit_el)
#     #     visits_written += 1

#     downlinks = math.ceil(total_bytes * 8 / (DOWNLINK_RATE_BPS * DOWNLINK_DURATION_S))
#     # summary = {
#     #     "total_bytes": total_bytes,
#     #     "downlinks_required": downlinks,
#     #     "visits_written": visits_written
#     # }
#     # indent(master_root)
#     # ET.ElementTree(master_root).write(output_path, encoding='utf-8', xml_declaration=True)
#     # write_pretty_xml(master_root, output_path)
#     # return summary

#     # Write output
#     tree = ET.ElementTree(master_root)
#     ET.indent(tree, space="    ", level=0)
#     tree.write(output_path, encoding='unicode', xml_declaration=True)
    
#     return {
#         "visits_written": visits_written,
#         "total_observations": len(obs_list),
#         "total_bytes": total_bytes,
#         "schedule_end": current_time.isoformat()
#     }

# -----------------------------
# Diagnostic functions
# -----------------------------
def analyze_schedule_diagnostics(xml_file_path: str, commissioning_end: datetime, 
                               input_xml_dir: str, debug=False, detailed_sequences=False) -> Dict:
    """
    Analyze the generated schedule XML and provide comprehensive diagnostics
    
    Parameters:
    - xml_file_path: Path to the output XML schedule
    - commissioning_end: End time of commissioning period
    - input_xml_dir: Directory containing original task XML files (for task number mapping)
    - debug: Print debugging information
    - detailed_sequences: Track and report every observation sequence with duration
    """
    import xml.etree.ElementTree as ET
    from collections import defaultdict, Counter
    import glob
    import os
    
    # Ensure commissioning_end is timezone-aware (UTC)
    commissioning_end = ensure_utc_time(commissioning_end)
    
    # Build mapping of target names/coordinates to task numbers from input files
    target_to_task = {}
    coord_to_task = {}
    task_info = {}  # Store detailed info about each task
    
    if debug:
        print("Building task mappings from input XML files...")
    
    if input_xml_dir and os.path.exists(input_xml_dir):
        input_files = glob.glob(os.path.join(input_xml_dir, "*.xml"))
        for file_path in input_files:
            filename = os.path.basename(file_path)
            task_match = re.match(r"(\d{4})_", filename)
            if task_match:
                task_num = task_match.group(1)
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    
                    # Get target name
                    target_elem = root.find('.//cal:Observational_Parameters/cal:Target', namespaces=NS)
                    target_name = target_elem.text if target_elem is not None and target_elem.text else None
                    
                    # Get coordinates
                    ra_elem = root.find('.//cal:Observational_Parameters/cal:Boresight/cal:RA', namespaces=NS)
                    dec_elem = root.find('.//cal:Observational_Parameters/cal:Boresight/cal:DEC', namespaces=NS)
                    
                    ra = None
                    dec = None
                    if ra_elem is not None and dec_elem is not None:
                        try:
                            ra_text = ra_elem.text.strip().split()[0] if ra_elem.text else '0'
                            dec_text = dec_elem.text.strip().split()[0] if dec_elem.text else '0'
                            ra = float(ra_text)
                            dec = float(dec_text)
                        except (ValueError, IndexError):
                            if debug:
                                print(f"  Warning: Could not parse coordinates from {filename}")
                    
                    # Store mappings
                    if target_name:
                        target_to_task[target_name] = task_num
                        if debug:
                            print(f"  Task {task_num}: Target '{target_name}' at RA={ra}, DEC={dec}")
                    
                    if ra is not None and dec is not None:
                        coord_to_task[(ra, dec)] = task_num
                    
                    # Store comprehensive task info
                    task_info[task_num] = {
                        'filename': filename,
                        'target': target_name,
                        'ra': ra,
                        'dec': dec
                    }
                            
                except Exception as e:
                    print(f"Error reading input file {filename}: {e}")
    
    if debug:
        print(f"\nBuilt mappings for {len(target_to_task)} targets and {len(coord_to_task)} coordinate pairs")
        print("Target to Task mapping:")
        for target, task in sorted(target_to_task.items()):
            print(f"  '{target}' -> Task {task}")
    
    # Initialize diagnostics
    diagnostics = {
        'task_durations': defaultdict(float),  # commissioning task_num -> total_seconds
        'target_info': {},  # (ra, dec) -> {'name': target_name, 'duration': seconds, 'task': task_num}
        'cvz_time': 0.0,
        'total_schedule_time': 0.0,
        'time_beyond_commissioning': 0.0,
        'unscheduled_tasks': [],
        'visibility_issues': [],
        'observation_counts': Counter(),  # commissioning task_num -> count
        'visit_timeline': [],
        'data_collection_efficiency': {},
        'target_observations': defaultdict(list),  # target_name -> list of observations
        'scheduling_order': [],  # Track scheduling order
        'task_info': task_info,  # Include task info for debugging
        'matching_issues': [],  # Track targets that couldn't be matched
        'detailed_sequences': [] if detailed_sequences else None  # All observation sequences in order
    }
    
    # Define expected commissioning tasks
    expected_tasks = {
        "0310", "0312", "0315", "0317", "0318", "0319", "0320",
        "0330", "0341", "0342", "0343", "0350", "0355", "0360"
    }
    
    # Parse the output XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    visits = root.findall('.//cal:Visit', namespaces=NS)
    
    if debug:
        print(f"\nAnalyzing {len(visits)} visits in output XML...")
    
    sequence_counter = 0  # Counter for all observation sequences
    
    for visit in visits:
        visit_id = visit.find('cal:ID', namespaces=NS).text
        
        # Handle multiple observation sequences within a visit
        obs_sequences = visit.findall('cal:Observation_Sequence', namespaces=NS)
        
        for obs_seq in obs_sequences:
            sequence_counter += 1
            obs_seq_id = obs_seq.find('cal:ID', namespaces=NS).text if obs_seq.find('cal:ID', namespaces=NS) is not None else "001"
            
            # Get timing
            timing = obs_seq.find('.//cal:Timing', namespaces=NS)
            start_str = timing.find('cal:Start', namespaces=NS).text
            stop_str = timing.find('cal:Stop', namespaces=NS).text
            
            # Parse times and ensure they're timezone-aware
            start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
            stop_time = datetime.fromisoformat(stop_str.replace('Z', '+00:00'))
            
            # Convert to UTC timezone-aware datetimes
            start_time = ensure_utc_time(start_time)
            stop_time = ensure_utc_time(stop_time)
            
            duration = (stop_time - start_time).total_seconds()
            
            # Get target info
            target_element = obs_seq.find('.//cal:Target', namespaces=NS)
            target = target_element.text if target_element is not None else 'Unknown'
            
            # Get coordinates
            boresight = obs_seq.find('.//cal:Boresight', namespaces=NS)
            ra = None
            dec = None
            if boresight is not None:
                ra_elem = boresight.find('cal:RA', namespaces=NS)
                dec_elem = boresight.find('cal:DEC', namespaces=NS)
                if ra_elem is not None and dec_elem is not None:
                    ra_text = ra_elem.text.replace(' deg', '') if ra_elem.text else '0'
                    dec_text = dec_elem.text.replace(' deg', '') if dec_elem.text else '0'
                    try:
                        ra = float(ra_text)
                        dec = float(dec_text)
                    except ValueError:
                        ra = 0.0
                        dec = 0.0
            
            # Determine commissioning task number
            task_num = None
            match_method = None
            
            # First try exact target name match
            if target in target_to_task:
                task_num = target_to_task[target]
                match_method = "exact_target_name"
            
            # Then try coordinate matching
            elif ra is not None and dec is not None:
                # Look for exact coordinate match
                coord_key = (ra, dec)
                if coord_key in coord_to_task:
                    task_num = coord_to_task[coord_key]
                    match_method = "exact_coordinates"
                else:
                    # Look for close coordinate match (within small tolerance)
                    for (input_ra, input_dec), input_task in coord_to_task.items():
                        if abs(ra - input_ra) < 0.001 and abs(dec - input_dec) < 0.001:
                            task_num = input_task
                            match_method = "approximate_coordinates"
                            break
            
            # Special handling for known task patterns
            if not task_num:
                if target == 'CVZ_IDLE':
                    task_num = None
                    match_method = "cvz"
                elif "Cardinal_" in target:
                    # For cardinal pointings, extract base target or use pattern matching
                    base_target = target.split('_Cardinal_')[0]
                    if base_target in target_to_task:
                        base_task = target_to_task[base_target]
                        if base_task in ["0341", "0342"]:
                            task_num = base_task
                            match_method = "cardinal_base_target"
                    else:
                        # Try to determine from visit ID or other patterns
                        if "0341" in visit_id or "Earth" in target:
                            task_num = "0341"
                            match_method = "cardinal_pattern_earth"
                        elif "0342" in visit_id or "Moon" in target:
                            task_num = "0342"
                            match_method = "cardinal_pattern_moon"
            
            # Debug output for problematic cases
            if debug and task_num is None and target != 'CVZ_IDLE':
                print(f"  Could not match: Visit {visit_id}, Target '{target}', RA={ra}, DEC={dec}")
                diagnostics['matching_issues'].append({
                    'visit_id': visit_id,
                    'obs_seq_id': obs_seq_id,
                    'target': target,
                    'ra': ra,
                    'dec': dec
                })
            elif debug and task_num:
                print(f"  Matched: Visit {visit_id}, Target '{target}' -> Task {task_num} ({match_method})")
            
            # Record detailed sequence information if requested
            if detailed_sequences:
                diagnostics['detailed_sequences'].append({
                    'sequence_number': sequence_counter,
                    'visit_id': visit_id,
                    'obs_seq_id': obs_seq_id,
                    'task': task_num if task_num else 'CVZ',
                    'target': target,
                    'duration_minutes': duration / 60.0,
                    'start_time': start_time,
                    'stop_time': stop_time,
                    'ra': ra,
                    'dec': dec,
                    'match_method': match_method
                })
            
            # Record time for the appropriate category
            if target == 'CVZ_IDLE':
                diagnostics['cvz_time'] += duration
            elif task_num:
                diagnostics['task_durations'][task_num] += duration
                diagnostics['observation_counts'][task_num] += 1
                
                # Add to scheduling order
                diagnostics['scheduling_order'].append({
                    'task': task_num,
                    'duration': duration,
                    'start_time': start_time,
                    'target': target,
                    'visit_id': visit_id,
                    'obs_seq_id': obs_seq_id,
                    'match_method': match_method
                })
            
            # Record target info
            if ra is not None and dec is not None:
                target_key = (ra, dec)
                if target_key not in diagnostics['target_info']:
                    diagnostics['target_info'][target_key] = {
                        'name': target,
                        'duration': 0.0,
                        'task': task_num
                    }
                diagnostics['target_info'][target_key]['duration'] += duration
                
                # Also track observations by target name
                diagnostics['target_observations'][target].append({
                    'start': start_time,
                    'stop': stop_time,
                    'duration': duration,
                    'ra': ra,
                    'dec': dec,
                    'task': task_num
                })
            
            # Track timeline
            diagnostics['visit_timeline'].append({
                'visit_id': visit_id,
                'obs_seq_id': obs_seq_id,
                'start': start_time,
                'stop': stop_time,
                'duration': duration,
                'target': target,
                'task': task_num,
                'coords': (ra, dec) if ra is not None and dec is not None else None
            })
            
            # Check if beyond commissioning period
            if start_time > commissioning_end:
                diagnostics['time_beyond_commissioning'] += duration
    
    # Calculate total schedule time
    if diagnostics['visit_timeline']:
        first_start = min(v['start'] for v in diagnostics['visit_timeline'])
        last_stop = max(v['stop'] for v in diagnostics['visit_timeline'])
        diagnostics['total_schedule_time'] = (last_stop - first_start).total_seconds()
        diagnostics['schedule_start_time'] = first_start
        diagnostics['schedule_end_time'] = last_stop
        
        # Calculate observation efficiency
        total_task_time = sum(diagnostics['task_durations'].values())
        diagnostics['task_observation_efficiency'] = total_task_time / diagnostics['total_schedule_time']
    
    # Identify unscheduled tasks
    observed_tasks = set(diagnostics['task_durations'].keys())
    diagnostics['unscheduled_tasks'] = [task for task in expected_tasks if task not in observed_tasks]
    
    return diagnostics

def print_diagnostics_report(diagnostics: Dict, output_format="console", output_file=None, show_detailed_sequences=False):
    """
    Print a formatted diagnostics report
    
    Parameters:
    - diagnostics: Dictionary of diagnostic information
    - output_format: "console" or "csv"
    - output_file: Path to output file (if CSV format)
    - show_detailed_sequences: Print every observation sequence with duration
    """
    if output_format == "console":
        print("\n" + "="*60)
        print("SCHEDULE DIAGNOSTICS REPORT")
        print("="*60)
        
        if 'schedule_start_time' in diagnostics and 'schedule_end_time' in diagnostics:
            print(f"\nSchedule Start: {diagnostics['schedule_start_time']}")
            print(f"Schedule End: {diagnostics['schedule_end_time']}")
        
        print(f"\nTOTAL SCHEDULE DURATION: {diagnostics['total_schedule_time']/60:.1f} minutes")
        print(f"TIME BEYOND COMMISSIONING: {diagnostics['time_beyond_commissioning']/60:.1f} minutes")
        print(f"CVZ OBSERVING TIME: {diagnostics['cvz_time']/60:.1f} minutes")
        
        if 'task_observation_efficiency' in diagnostics:
            print(f"\nTASK OBSERVATION EFFICIENCY: {diagnostics['task_observation_efficiency']:.1%}")
            print(f"(Time spent on tasks vs. total schedule time)")
        
        print(f"\nTASK DURATIONS:")
        for task, duration in sorted(diagnostics['task_durations'].items()):
            obs_count = diagnostics['observation_counts'].get(task, 0)
            print(f"  Task {task}: {duration/60:.1f} minutes ({obs_count} observations)")
        
        print(f"\nSCHEDULING ORDER:")
        for i, obs in enumerate(diagnostics['scheduling_order'], 1):
            print(f"  {i:3d}. Task {obs['task']}: {obs['duration']/60:.1f} minutes "
                  f"at {obs['start_time'].strftime('%Y-%m-%d %H:%M:%S')} - {obs['target']}")
        
        # Show detailed sequences if requested and available
        if show_detailed_sequences and diagnostics.get('detailed_sequences'):
            print(f"\nDETAILED OBSERVATION SEQUENCES:")
            print(f"{'Seq#':>4} {'Visit':>6} {'ObsSeq':>6} {'Task':>6} {'Duration':>10} {'Start Time':>19} {'Target'}")
            print("-" * 80)
            for seq in diagnostics['detailed_sequences']:
                print(f"{seq['sequence_number']:>4} "
                      f"{seq['visit_id']:>6} "
                      f"{seq['obs_seq_id']:>6} "
                      f"{seq['task']:>6} "
                      f"{seq['duration_minutes']:>8.1f}m "
                      f"{seq['start_time'].strftime('%Y-%m-%d %H:%M:%S')} "
                      f"{seq['target']}")
        
        print(f"\nTARGET INFORMATION:")
        for (ra, dec), info in sorted(diagnostics['target_info'].items(), 
                                     key=lambda x: x[1]['duration'], reverse=True):
            print(f"  {info['name']:<30} (RA={ra:.3f}, DEC={dec:.3f}): "
                  f"{info['duration']/60:.1f} minutes, Task={info['task']}")
        
        if diagnostics.get('matching_issues'):
            print(f"\nMATCHING ISSUES ({len(diagnostics['matching_issues'])} targets could not be matched):")
            for issue in diagnostics['matching_issues']:
                print(f"  Visit {issue['visit_id']}: '{issue['target']}' at RA={issue['ra']}, DEC={issue['dec']}")
        
        if diagnostics['unscheduled_tasks']:
            print(f"\nUNSCHEDULED TASKS: {', '.join(diagnostics['unscheduled_tasks'])}")
        
        if diagnostics['visibility_issues']:
            print(f"\nVISIBILITY ISSUES:")
            for issue in diagnostics['visibility_issues']:
                print(f"  {issue}")
    
    elif output_format == "csv" and output_file:
        import csv
        
        # Write comprehensive CSV report
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Schedule Summary
            writer.writerow(['Schedule Summary'])
            writer.writerow(['Total Schedule Duration (minutes)', f"{diagnostics['total_schedule_time']/60:.1f}"])
            writer.writerow(['Time Beyond Commissioning (minutes)', f"{diagnostics['time_beyond_commissioning']/60:.1f}"])
            writer.writerow(['CVZ Time (minutes)', f"{diagnostics['cvz_time']/60:.1f}"])
            writer.writerow(['Task Observation Efficiency', f"{diagnostics.get('task_observation_efficiency', 0):.1%}"])
            writer.writerow([])
            
            # Task Durations
            writer.writerow(['Task Durations'])
            writer.writerow(['Task', 'Duration (minutes)', 'Observations'])
            for task, duration in sorted(diagnostics['task_durations'].items()):
                obs_count = diagnostics['observation_counts'].get(task, 0)
                writer.writerow([task, f"{duration/60:.1f}", obs_count])
            writer.writerow([])
            
            # Scheduling Order
            writer.writerow(['Scheduling Order'])
            writer.writerow(['Order', 'Task', 'Duration (minutes)', 'Start Time', 'Target', 'Visit ID', 'Obs Seq ID'])
            for i, obs in enumerate(diagnostics['scheduling_order'], 1):
                writer.writerow([
                    i,
                    obs['task'],
                    f"{obs['duration']/60:.1f}",
                    obs['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    obs['target'],
                    obs['visit_id'],
                    obs.get('obs_seq_id', 'N/A')
                ])
            writer.writerow([])
            
            # Detailed Sequences (if available)
            if diagnostics.get('detailed_sequences'):
                writer.writerow(['Detailed Observation Sequences'])
                writer.writerow(['Sequence #', 'Visit ID', 'Obs Seq ID', 'Task', 'Duration (minutes)', 
                               'Start Time', 'Stop Time', 'Target', 'RA', 'DEC'])
                for seq in diagnostics['detailed_sequences']:
                    writer.writerow([
                        seq['sequence_number'],
                        seq['visit_id'],
                        seq['obs_seq_id'],
                        seq['task'],
                        f"{seq['duration_minutes']:.1f}",
                        seq['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        seq['stop_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        seq['target'],
                        seq['ra'],
                        seq['dec']
                    ])
                writer.writerow([])
            
            # Target Information
            writer.writerow(['Target Information'])
            writer.writerow(['Target', 'RA', 'DEC', 'Duration (minutes)', 'Task'])
            for (ra, dec), info in sorted(diagnostics['target_info'].items(), 
                                         key=lambda x: x[1]['duration'], reverse=True):
                writer.writerow([
                    info['name'], 
                    f"{ra:.3f}", 
                    f"{dec:.3f}", 
                    f"{info['duration']/60:.1f}",
                    info['task']
                ])
            writer.writerow([])
            
            # Matching Issues
            if diagnostics.get('matching_issues'):
                writer.writerow(['Matching Issues'])
                writer.writerow(['Visit ID', 'Obs Seq ID', 'Target', 'RA', 'DEC'])
                for issue in diagnostics['matching_issues']:
                    writer.writerow([
                        issue['visit_id'],
                        issue.get('obs_seq_id', 'N/A'),
                        issue['target'],
                        issue['ra'],
                        issue['dec']
                    ])
                writer.writerow([])
            
            # Unscheduled Tasks
            if diagnostics['unscheduled_tasks']:
                writer.writerow(['Unscheduled Tasks'])
                for task in diagnostics['unscheduled_tasks']:
                    writer.writerow([task])
        
        print(f"CSV diagnostic report written to {output_file}")
    else:
        print("Invalid output format or missing output file for CSV format")

# -----------------------------
# CLI entrypoint
# -----------------------------
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Combine commissioning XML files into a master schedule.")
    parser.add_argument("xml_dir",help="Directory containing task XML files")
    parser.add_argument("output",help="Output master XML path")
    parser.add_argument("--cvz-ra",type=float,required=True,help="CVZ pointing RA (deg)")
    parser.add_argument("--cvz-dec",type=float,required=True,help="CVZ pointing DEC (deg)")
    parser.add_argument("--tle1",type=str,required=True,help="TLE line 1")
    parser.add_argument("--tle2",type=str,required=True,help="TLE line 2")
    parser.add_argument("--start",type=str,default="2026-01-05T00:00:00",help="Commissioning start UTC")
    parser.add_argument("--end",type=str,default="2026-02-05T00:00:00",help="Commissioning end UTC")
    parser.add_argument("--ephem",type=str,default=None,help="Ephemeris file for cardinal pointings")
    parser.add_argument("--dep",type=str,default=None,help="Dependency JSON file")
    parser.add_argument("--progress",type=str,default=None,help="Progress JSON file")
    parser.add_argument("--cvz",type=str,default=None,help="Extra CVZ JSON file")
    args=parser.parse_args()

    xml_paths=gather_task_xmls(args.xml_dir)
    result=merge_schedules(xml_paths,args.output,(args.cvz_ra,args.cvz_dec),args.tle1,args.tle2,
                           commissioning_start=datetime.fromisoformat(args.start),
                           commissioning_end=datetime.fromisoformat(args.end),
                           pointing_ephem_file=args.ephem,
                           dependency_json=args.dep,
                           progress_json=args.progress,
                           extra_cvz_json=args.cvz)
    print("Merge complete:",result)
