"""
File: pandora_schedule_merger_v2.py
Purpose: Merge Commissioning-task XML schedule fragments into a master ScienceCalendar XML.
Features:
 - pandora-visibility integration (10-minute sampling) with caching
 - ordering with task dependency support (topological sort via JSON)
 - ability to run/resume from arbitrary start time and include partial tasks
 - ability to insert extra CVZ padding intervals
 - Visit/Observation_Sequence numbering: Visit IDs increment globally; Observation_Sequence IDs increment within each Visit

Usage:
 - Import functions in a notebook or run as CLI.
"""

import os
import re
import math
import json
import pickle
import hashlib
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import xml.etree.ElementTree as ET

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from pandoravisibility import Visibility

# -----------------------------
# Config / Constants
# -----------------------------
DOWNLINK_RATE_BPS = 5e6  # 5 Mbps
DOWNLINK_DURATION_S = 8 * 60  # 8 minutes
COMMISSIONING_START = datetime(2025, 12, 15, 0, 0, 0)
COMMISSIONING_END = datetime(2026, 1, 14, 23, 59, 59)
TEN_MIN = 600
BYTES_PER_PIXEL = 2
VIS_FRAME_OVERHEAD_BYTES = 1000
VIS_CACHE_DIR = '.vis_cache'
os.makedirs(VIS_CACHE_DIR, exist_ok=True)

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
# Helper: parse xml into ObservationSequence
# -----------------------------
def parse_task_xml(path: str) -> ObservationSequence:
    tree = ET.parse(path)
    root = tree.getroot()
    fn = os.path.basename(path)
    m = re.match(r"(\d{4})_(\d{3})_.*\.xml", fn)
    if m:
        visit_id, obs_id = m.group(1), m.group(2)
    else:
        visit_id, obs_id = "0000", "000"

    tgt_elem = root.find('.//Observational_Parameters/Target')
    target = tgt_elem.text if tgt_elem is not None else ''
    ra_elem = root.find('.//Observational_Parameters/Boresight/RA')
    dec_elem = root.find('.//Observational_Parameters/Boresight/DEC')
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
    nir = root.find('.//AcquireInfCamImages')
    if nir is not None:
        ROI_SizeX = int(nir.findtext('ROI_SizeX', '0'))
        ROI_SizeY = int(nir.findtext('ROI_SizeY', '0'))
        SC_Resets1 = int(nir.findtext('SC_Resets1', '0'))
        SC_Resets2 = int(nir.findtext('SC_Resets2', '0'))
        SC_DropFrames1 = int(nir.findtext('SC_DropFrames1', '0'))
        SC_DropFrames2 = int(nir.findtext('SC_DropFrames2', '0'))
        SC_DropFrames3 = int(nir.findtext('SC_DropFrames3', '0'))
        SC_ReadFrames = int(nir.findtext('SC_ReadFrames', '0'))
        SC_Integrations = int(nir.findtext('SC_Integrations', '0'))
        group_sum = (SC_Resets1 + SC_Resets2 + SC_DropFrames1 +
                     SC_DropFrames2 + SC_DropFrames3 + SC_ReadFrames + 1)
        nir_integration_s = (ROI_SizeX * ROI_SizeY + (ROI_SizeY * 12)) * 0.00001 * group_sum
        nir_total_s = nir_integration_s * SC_Integrations

    vda_total_s = vda_integration_s = 0.0
    vda = root.find('.//AcquireVisCamImages')
    vda_science = root.find('.//AcquireVisCamScienceData')
    if vda is not None:
        ExposureTime_us = float(vda.findtext('ExposureTime_us', '0'))
        NumExposures = int(vda.findtext('NumExposures', '0'))
        vda_integration_s = ExposureTime_us / 1e6
        vda_total_s = vda_integration_s * NumExposures
    elif vda_science is not None:
        ExposureTime_us = float(vda_science.findtext('ExposureTime_us', '0'))
        NumTotalFramesRequested = int(vda_science.findtext('NumTotalFramesRequested', '0'))
        FramesPerCoadd = int(vda_science.findtext('FramesPerCoadd', '1'))
        vda_integration_s = (ExposureTime_us / 1e6) * FramesPerCoadd
        vda_total_s = (ExposureTime_us / 1e6) * NumTotalFramesRequested

    return nir_total_s, vda_total_s, nir_integration_s, vda_integration_s

# -----------------------------
# Helper: estimate data volume
# -----------------------------
def estimate_data_volume_bytes(obs: ObservationSequence) -> int:
    root = obs.xml_root
    total_bytes = 0
    nir = root.find('.//AcquireInfCamImages')
    if nir is not None:
        ROI_SizeX = int(nir.findtext('ROI_SizeX', '0'))
        ROI_SizeY = int(nir.findtext('ROI_SizeY', '0'))
        SC_Integrations = int(nir.findtext('SC_Integrations', '0'))
        frames = SC_Integrations
        total_bytes += ROI_SizeX * ROI_SizeY * frames * BYTES_PER_PIXEL
    vda = root.find('.//AcquireVisCamImages')
    vda_science = root.find('.//AcquireVisCamScienceData')
    if vda is not None:
        NumExposures = int(vda.findtext('NumExposures', '0'))
        ROI_SizeX = int(vda.findtext('ROI_SizeX', '1024'))
        ROI_SizeY = int(vda.findtext('ROI_SizeY', '1024'))
        total_bytes += (ROI_SizeX * ROI_SizeY * BYTES_PER_PIXEL + VIS_FRAME_OVERHEAD_BYTES) * NumExposures
    elif vda_science is not None:
        NumTotalFramesRequested = int(vda_science.findtext('NumTotalFramesRequested', '0'))
        ROI_SizeX = int(vda_science.findtext('ROI_SizeX', '1024'))
        ROI_SizeY = int(vda_science.findtext('ROI_SizeY', '1024'))
        total_bytes += (ROI_SizeX * ROI_SizeY * BYTES_PER_PIXEL + VIS_FRAME_OVERHEAD_BYTES) * NumTotalFramesRequested
    return int(total_bytes)

# -----------------------------
# Helper: adjust payload for chunk
# -----------------------------
def adjust_payload_for_chunk(obs: ObservationSequence, chunk_seconds: int,
                             nir_integr_s: float, vda_integr_s: float):
    """
    Modify the obs.xml_tree in-place to reflect the number of integrations/exposures
    that fit in chunk_seconds. Does not touch sequence IDs.
    """
    root = obs.xml_root

    # NIRDA camera
    nir = root.find('.//AcquireInfCamImages')
    if nir is not None and nir_integr_s > 0:
        SC_Integrations = int(nir.findtext('SC_Integrations', '0'))
        max_integr_in_chunk = int(math.floor(chunk_seconds / nir_integr_s))
        new_integr = min(SC_Integrations, max_integr_in_chunk)
        if new_integr < 0:
            new_integr = 0
        integr_el = nir.find('SC_Integrations')
        if integr_el is not None:
            integr_el.text = str(new_integr)

    # VIS camera — AcquireVisCamImages
    vda = root.find('.//AcquireVisCamImages')
    if vda is not None and vda_integr_s > 0:
        NumExposures = int(vda.findtext('NumExposures', '0'))
        max_exposures = int(math.floor(chunk_seconds / vda_integr_s))
        new_exp = min(NumExposures, max_exposures)
        exp_el = vda.find('NumExposures')
        if exp_el is not None:
            exp_el.text = str(new_exp)

    # VIS camera — AcquireVisCamScienceData
    vda_science = root.find('.//AcquireVisCamScienceData')
    if vda_science is not None and vda_integr_s > 0:
        NumTotalFramesRequested = int(vda_science.findtext('NumTotalFramesRequested', '0'))
        FramesPerCoadd = int(vda_science.findtext('FramesPerCoadd', '1'))
        max_frames = int(math.floor(chunk_seconds / vda_integr_s))
        new_frames = min(NumTotalFramesRequested, max_frames)
        frames_el = vda_science.find('NumTotalFramesRequested')
        if frames_el is not None:
            frames_el.text = str(new_frames)

        nem = vda_science.find('NumExposuresMax')
        if nem is not None:
            nem_val = int(nem.text) if nem.text is not None else 0
            if nem_val > new_frames:
                nem.text = str(new_frames)

# -----------------------------
# Helper: create CVZ visit
# -----------------------------
def create_cvz_visit(cvz_ra: float, cvz_dec: float,
                     start: datetime, stop: datetime,
                     visit_id: int, obs_seq_id: int) -> ET.Element:
    visit = ET.Element('Visit')
    ET.SubElement(visit, 'ID').text = f"{visit_id:04d}"

    obs = ET.SubElement(visit, 'Observation_Sequence')
    ET.SubElement(obs, 'ID').text = f"{obs_seq_id:03d}"

    op = ET.SubElement(obs, 'Observational_Parameters')
    ET.SubElement(op, 'Target').text = 'CVZ_IDLE'

    timing = ET.SubElement(op, 'Timing')
    ET.SubElement(timing, 'Start').text = start.strftime('%Y-%m-%d %H:%M:%S')
    ET.SubElement(timing, 'Stop').text = stop.strftime('%Y-%m-%d %H:%M:%S')

    bore = ET.SubElement(op, 'Boresight')
    ET.SubElement(bore, 'RA').text = f"{cvz_ra} deg"
    ET.SubElement(bore, 'DEC').text = f"{cvz_dec} deg"

    pp = ET.SubElement(obs, 'Payload_Parameters')
    vis = ET.SubElement(pp, 'AcquireVisCamImages')
    ET.SubElement(vis, 'TargetID').text = 'CVZ'
    ET.SubElement(vis, 'ROI_StartX').text = '512'
    ET.SubElement(vis, 'ROI_StartY').text = '512'
    ET.SubElement(vis, 'ROI_SizeX').text = '1024'
    ET.SubElement(vis, 'ROI_SizeY').text = '1024'
    ET.SubElement(vis, 'NumExposures').text = '1'
    ET.SubElement(vis, 'ExposureTime_us').text = '100000.0'

    return visit

# -----------------------------
# Visibility caching utilities
# -----------------------------

def _vis_cache_key(ra: float, dec: float, start: datetime, end: datetime, tle1: str, tle2: str) -> str:
    s = f"{ra}:{dec}:{start.isoformat()}:{end.isoformat()}:{tle1}:{tle2}"
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def load_visibility_cache(ra: float, dec: float, start: datetime, end: datetime, tle1: str, tle2: str) -> Optional[List[Tuple[datetime, datetime]]]:
    key = _vis_cache_key(ra, dec, start, end, tle1, tle2)
    path = os.path.join(VIS_CACHE_DIR, key + '.pkl')
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def save_visibility_cache(ra: float, dec: float, start: datetime, end: datetime, tle1: str, tle2: str, windows: List[Tuple[datetime, datetime]]):
    key = _vis_cache_key(ra, dec, start, end, tle1, tle2)
    path = os.path.join(VIS_CACHE_DIR, key + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(windows, f)

# -----------------------------
# Visibility wrapper (uses caching)
# -----------------------------

def compute_and_cache_visibility(ra: float, dec: float, start: datetime, end: datetime, tle1: str, tle2: str) -> List[Tuple[datetime, datetime]]:
    cached = load_visibility_cache(ra, dec, start, end, tle1, tle2)
    if cached is not None:
        return cached
    vis = Visibility(tle1, tle2)
    tstart = Time(start)
    tstop = Time(end)
    deltas = np.arange(0, (tstop - tstart).to_value(u.min), 10) * u.min
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

    save_visibility_cache(ra, dec, start, end, tle1, tle2, windows)
    return windows

# -----------------------------
# Task dependency / ordering utilities
# -----------------------------

def load_dependencies(dep_file: Optional[str]) -> Dict[str, List[str]]:
    """Load JSON mapping task numbers (as strings) to list of prerequisite tasks."""
    if not dep_file:
        return {}
    with open(dep_file, 'r') as f:
        return json.load(f)


def topo_sort_tasks(file_paths: List[str], deps: Dict[str, List[str]]) -> List[str]:
    """Topologically sort task filenames based on deps. Filenames expected to start with NNNN_."""
    # Build mapping from task_num -> list of file paths
    task_map: Dict[str, List[str]] = {}
    for p in file_paths:
        base = os.path.basename(p)
        m = re.match(r"(\d{4})_(\d{3})_.*\.xml", base)
        tasknum = m.group(1) if m else '9999'
        task_map.setdefault(tasknum, []).append(p)

    # Nodes are task numbers (strings). Use Kahn's algorithm.
    nodes = set(task_map.keys())
    for k in deps.keys():
        nodes.add(k)
    indeg = {n: 0 for n in nodes}
    adj = {n: [] for n in nodes}
    for t, pres in deps.items():
        for p in pres:
            adj[p].append(t)
            indeg[t] = indeg.get(t, 0) + 1
    # start with zero indeg nodes sorted by numeric task number
    zero = sorted([n for n, d in indeg.items() if d == 0], key=lambda x: int(x))
    order: List[str] = []
    while zero:
        n = zero.pop(0)
        # append all file paths for this task (sorted by obs number)
        files = sorted(task_map.get(n, []), key=lambda fn: int(re.match(r"(\d{4})_(\d{3})",(os.path.basename(fn))).group(2)))
        order.extend(files)
        for mnode in adj.get(n, []):
            indeg[mnode] -= 1
            if indeg[mnode] == 0:
                zero.append(mnode)
                zero.sort(key=lambda x: int(x))
    # If any nodes remain with indeg >0, there's a cycle; fallback to numeric sort of files
    remaining_files = []
    for tnum, files in task_map.items():
        for f in files:
            if f not in order:
                remaining_files.append(f)
    remaining_files.sort(key=lambda fn: (int(re.match(r"(\d{4})", os.path.basename(fn)).group(1)), int(re.match(r"(\d{4})_(\d{3})", os.path.basename(fn)).group(2))))
    order.extend(remaining_files)
    return order

# -----------------------------
# Core merging + scheduling loop (enhanced)
# -----------------------------

def merge_schedules(input_paths: List[str], output_path: str,
                    cvz_coords: Tuple[float, float],
                    tle_line1: str, tle_line2: str,
                    commissioning_start: datetime = COMMISSIONING_START,
                    commissioning_end: datetime = COMMISSIONING_END,
                    dep_file: Optional[str] = None,
                    include_tasks: Optional[List[str]] = None,
                    partial_progress: Optional[Dict[str, int]] = None,
                    extra_cvz_slots: Optional[List[Tuple[datetime, datetime]]] = None,
                    vis_cache: bool = True) -> Dict[str, Any]:
    """
    Merge a list of commissioning XMLs into a master schedule with advanced features:
      - dep_file: JSON mapping task nums -> list of prerequisite task nums (strings)
      - include_tasks: list of task numbers (as strings) to include; others omitted
      - partial_progress: mapping filename -> remaining_seconds to schedule
      - extra_cvz_slots: list of (start, stop) datetimes to reserve for CVZ
      - vis_cache: whether to use and populate visibility cache

    Returns bookkeeping dict.
    """
    # Apply dependency ordering if provided
    deps = load_dependencies(dep_file)
    if deps:
        ordered_paths = topo_sort_tasks(input_paths, deps)
    else:
        ordered_paths = sorted(input_paths, key=lambda fn: (int(re.match(r"(\d{4})", os.path.basename(fn)).group(1)), int(re.match(r"(\d{4})_(\d{3})", os.path.basename(fn)).group(2))))

    # Filter by include_tasks if provided (task numbers strings)
    if include_tasks is not None:
        filtered = []
        for p in ordered_paths:
            tnum = re.match(r"(\d{4})", os.path.basename(p)).group(1)
            if tnum in include_tasks:
                filtered.append(p)
        ordered_paths = filtered

    # Parse Observations
    obs_list = [parse_task_xml(p) for p in ordered_paths]

    master_root = ET.Element('ScienceCalendar', attrib={'xmlns': '/pandora/calendar/'})
    if obs_list:
        meta = obs_list[0].xml_root.find('Meta')
        if meta is not None:
            master_root.append(meta)

    current_time = commissioning_start
    # if resuming mid-commissioning, use provided start
    next_visit_id = 1
    total_bytes = 0
    visits_written = 0

    # normalize extra_cvz_slots
    extra_slots = sorted(extra_cvz_slots or [], key=lambda x: x[0])
    extra_i = 0

    last_target = None
    visit_el = None
    obs_seq_id = 1

    for obs in obs_list:
        # allow override of remaining via partial_progress keyed by filename
        if partial_progress and obs.filename in partial_progress:
            remaining = int(partial_progress[obs.filename])
        else:
            nir_total, vda_total, nir_int_s, vda_int_s = compute_instrument_durations(obs)
            remaining = int(math.ceil(max(nir_total, vda_total)))

        # get visibility windows (from cache if requested)
        if vis_cache:
            vis_windows = load_visibility_cache(obs.ra, obs.dec, current_time, commissioning_end, tle_line1, tle_line2) or compute_and_cache_visibility(obs.ra, obs.dec, current_time, commissioning_end, tle_line1, tle_line2)
        else:
            vis_windows = compute_and_cache_visibility(obs.ra, obs.dec, current_time, commissioning_end, tle_line1, tle_line2)

        wi = 0
        while remaining > 0 and current_time < commissioning_end:
            # Handle extra CVZ slots that start now or before next window
            while extra_i < len(extra_slots) and extra_slots[extra_i][0] <= current_time:
                es = extra_slots[extra_i]
                if es[1] <= current_time:
                    extra_i += 1
                    continue
                # insert CVZ slot from max(current_time, es.start) to es.stop
                cvz_start = max(current_time, es[0])
                cvz_stop = es[1]
                # close previous visit if any
                if visit_el is not None:
                    master_root.append(visit_el)
                    visits_written += 1
                    visit_el = None
                    last_target = None
                    obs_seq_id = 1
                cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], cvz_start, cvz_stop, next_visit_id, 1)
                next_visit_id += 1
                master_root.append(cvz_visit)
                visits_written += 1
                current_time = cvz_stop
                extra_i += 1

            if wi >= len(vis_windows):
                # no more visibility windows for this obs -> insert CVZ until commissioning_end and break
                if visit_el is not None:
                    master_root.append(visit_el)
                    visits_written += 1
                    visit_el = None
                    last_target = None
                    obs_seq_id = 1
                cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], current_time, commissioning_end, next_visit_id, 1)
                next_visit_id += 1
                master_root.append(cvz_visit)
                visits_written += 1
                current_time = commissioning_end
                break

            win_start, win_stop = vis_windows[wi]
            # if there's a gap before the window, insert CVZ idle
            if win_start > current_time:
                gap_start = current_time
                gap_stop = min(win_start, commissioning_end)
                if gap_stop > gap_start:
                    if visit_el is not None:
                        master_root.append(visit_el)
                        visits_written += 1
                        visit_el = None
                        last_target = None
                        obs_seq_id = 1
                    cvz_visit = create_cvz_visit(cvz_coords[0], cvz_coords[1], gap_start, gap_stop, next_visit_id, 1)
                    next_visit_id += 1
                    master_root.append(cvz_visit)
                    visits_written += 1
                    current_time = gap_stop
                    continue

            # Now current_time is within or at start of visibility window
            usable = int((win_stop - current_time).total_seconds())
            usable_blocks = (usable // TEN_MIN) * TEN_MIN
            if usable_blocks <= 0:
                wi += 1
                current_time = win_stop
                continue

            # decide chunk size: multiples of TEN_MIN, but don't exceed remaining
            if remaining >= TEN_MIN:
                chunk = min(usable_blocks, (remaining // TEN_MIN) * TEN_MIN)
            else:
                # remaining < TEN_MIN: we need a full TEN_MIN block to schedule (per spec)
                if usable_blocks >= TEN_MIN:
                    chunk = TEN_MIN
                else:
                    wi += 1
                    current_time = win_stop
                    continue
            if chunk <= 0:
                wi += 1
                current_time = win_stop
                continue

            # Create or reuse Visit for this target
            if obs.target != last_target:
                # close previous visit
                if visit_el is not None:
                    master_root.append(visit_el)
                    visits_written += 1
                visit_el = ET.Element('Visit')
                ET.SubElement(visit_el, 'ID').text = f"{next_visit_id:04d}"
                next_visit_id += 1
                obs_seq_id = 1
                last_target = obs.target

            # Create Observation_Sequence within current Visit
            obs_seq = ET.SubElement(visit_el, 'Observation_Sequence')
            ET.SubElement(obs_seq, 'ID').text = f"{obs_seq_id:03d}"
            obs_seq_id += 1

            # Copy template Observation_Sequence children
            template = obs.xml_root.find('.//Observation_Sequence')
            if template is not None:
                for child in list(template):
                    # deep copy
                    obs_seq.append(ET.fromstring(ET.tostring(child)))

            # Update timing for this obs_seq
            timing_el = obs_seq.find('Observational_Parameters/Timing')
            if timing_el is None:
                ops = obs_seq.find('Observational_Parameters')
                timing_el = ET.SubElement(ops, 'Timing')
            start_el = timing_el.find('Start') or ET.SubElement(timing_el, 'Start')
            stop_el = timing_el.find('Stop') or ET.SubElement(timing_el, 'Stop')
            chunk_start = current_time
            chunk_stop = current_time + timedelta(seconds=chunk)
            start_el.text = chunk_start.strftime('%Y-%m-%d %H:%M:%S')
            stop_el.text = chunk_stop.strftime('%Y-%m-%d %H:%M:%S')

            # Adjust payload numbers for this chunk by creating a temporary wrapper for adjust_payload_for_chunk
            temp_tree = ET.ElementTree(obs_seq)
            temp_obs = ObservationSequence(filename=obs.filename, visit_id=str(next_visit_id-1), obs_id=str(obs_seq_id-1), target=obs.target, ra=obs.ra, dec=obs.dec, xml_tree=temp_tree, xml_root=obs_seq)
            adjust_payload_for_chunk(temp_obs, chunk, *compute_instrument_durations(obs)[2:])

            # Estimate bytes for this chunk
            chunk_bytes = estimate_data_volume_bytes(temp_obs)
            total_bytes += chunk_bytes

            # Advance time
            remaining -= chunk
            current_time = chunk_stop
            if current_time >= win_stop:
                wi += 1

    # finalize
    if visit_el is not None:
        master_root.append(visit_el)
        visits_written += 1

    total_bits = total_bytes * 8
    downlinks_required = math.ceil(total_bits / (DOWNLINK_RATE_BPS * DOWNLINK_DURATION_S))
    downlink_time_s = downlinks_required * DOWNLINK_DURATION_S

    ET.ElementTree(master_root).write(output_path, encoding='utf-8', xml_declaration=True)

    return {
        'total_bytes': total_bytes,
        'total_bits': total_bits,
        'downlinks_required': downlinks_required,
        'downlink_time_s': downlink_time_s,
        'visits_written': visits_written
    }

# -----------------------------
# CLI entrypoint
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combine commissioning XML files into a master schedule.")
    parser.add_argument("xml_dir", help="Directory containing task XML files")
    parser.add_argument("output", help="Output master XML path")
    parser.add_argument("--cvz-ra", type=float, required=True, help="CVZ pointing RA (deg)")
    parser.add_argument("--cvz-dec", type=float, required=True, help="CVZ pointing DEC (deg)")
    parser.add_argument("--tle1", type=str, required=True, help="TLE line 1")
    parser.add_argument("--tle2", type=str, required=True, help="TLE line 2")
    parser.add_argument("--start", type=str, default="2026-01-05T00:00:00", help="Commissioning start UTC")
    parser.add_argument("--end", type=str, default="2026-02-05T00:00:00", help="Commissioning end UTC")
    parser.add_argument("--deps", type=str, default=None, help="Path to JSON file with task dependencies")
    parser.add_argument("--include", type=str, default=None, help="Comma-separated task numbers to include, e.g. '0330,0343'")
    parser.add_argument("--partial", type=str, default=None, help="Path to JSON file mapping filename -> remaining_seconds for partially-done tasks")
    parser.add_argument("--extra-cvz", type=str, default=None, help="Path to JSON file with list of extra CVZ slots [{'start':'2026-01-10T00:00:00','stop':'2026-01-10T00:10:00'}]")
    args = parser.parse_args()

    xml_paths = gather_task_xmls(args.xml_dir)
    include = args.include.split(',') if args.include else None
    partial = json.load(open(args.partial)) if args.partial else None
    extra_slots = None
    if args.extra_cvz:
        arr = json.load(open(args.extra_cvz))
        extra_slots = [(datetime.fromisoformat(a['start']), datetime.fromisoformat(a['stop'])) for a in arr]

    result = merge_schedules(
        xml_paths,
        args.output,
        (args.cvz_ra, args.cvz_dec),
        args.tle1, args.tle2,
        commissioning_start=datetime.fromisoformat(args.start),
        commissioning_end=datetime.fromisoformat(args.end),
        dep_file=args.deps,
        include_tasks=include,
        partial_progress=partial,
        extra_cvz_slots=extra_slots,
        vis_cache=True
    )
    print('Merge complete:', result)
