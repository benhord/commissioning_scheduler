# File: scheduler.py

"""
Pandora Commissioning Schedule Merger

This module merges individual commissioning XML files into a master ScienceCalendar,
schedules observations considering target visibility via pandoravisibility,
inserts CVZ idle periods when needed, and tracks data volume and downlinks.

Public API:
- gather_task_xmls(xml_dir: str) -> List[str]
- merge_schedules(input_paths: List[str], output_path: str, cvz_coords: Tuple[float,float],
                  commissioning_start: datetime, commissioning_end: datetime) -> Dict
"""

"""
File: pandora_schedule_merger.py
Purpose: Merge Commissioning-task XML schedule fragments into a master ScienceCalendar XML.
Now includes real pandora-visibility integration and Visit/Observation_Sequence ID handling.
"""

import os
import re
import math
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
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
# Visibility wrapper
# -----------------------------
def find_visibility_windows(ra: float, dec: float,
                           start: datetime, end: datetime,
                           tle_line1: str, tle_line2: str) -> List[Tuple[datetime, datetime]]:
    vis = Visibility(tle_line1, tle_line2)
    tstart = Time(start)
    tstop = Time(end)
    deltas = np.arange(0, (tstop - tstart).to_value(u.min), 10) * u.min
    times = tstart + TimeDelta(deltas)
    target_coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
    targ_vis = vis.get_visibility(target_coord, times)

    windows = []
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
    return windows

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

def adjust_payload_for_chunk(obs: ObservationSequence, chunk_seconds: int,
                             nir_integr_s: float, vda_integr_s: float):
    """
    Modify the obs.xml_tree in-place to reflect how many integrations/exposures
    can actually fit in the given chunk_seconds duration.

    - NIR: adjust SC_Integrations
    - VDA: adjust NumExposures (VisCamImages) or NumTotalFramesRequested (VisCamScienceData)
    - NumExposuresMax is capped not to exceed NumTotalFramesRequested

    IDs are not touched here (those are handled by the merge loop).
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

def create_cvz_visit(cvz_ra: float, cvz_dec: float,
                     start: datetime, stop: datetime,
                     visit_id: int, obs_seq_id: int) -> ET.Element:
    """
    Create a Visit element with a single Observation_Sequence pointing to the CVZ.
    Both Visit ID and Observation_Sequence ID are passed in explicitly
    (they are controlled by the merge loop).
    """
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

    # Minimal payload so spacecraft points and collects a "keep-alive" exposure
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
# Core merging + scheduling loop
# -----------------------------
def merge_schedules(input_paths: List[str], output_path: str,
                    cvz_coords: Tuple[float, float],
                    tle_line1: str, tle_line2: str,
                    commissioning_start: datetime = COMMISSIONING_START,
                    commissioning_end: datetime = COMMISSIONING_END) -> Dict:
    obs_list = [parse_task_xml(p) for p in input_paths]
    obs_list.sort(key=lambda o: (int(o.visit_id), int(o.obs_id)))

    master_root = ET.Element('ScienceCalendar', attrib={'xmlns': '/pandora/calendar/'})
    if obs_list:
        meta = obs_list[0].xml_root.find('Meta')
        if meta is not None:
            master_root.append(meta)

    current_time = commissioning_start
    visit_id = 1
    obs_seq_id = 1
    total_bytes = 0
    visits_written = 0

    last_target = None
    visit_el = None

    for obs in obs_list:
        nir_total, vda_total, nir_int_s, vda_int_s = compute_instrument_durations(obs)
        remaining = int(math.ceil(max(nir_total, vda_total)))
        vis_windows = find_visibility_windows(obs.ra, obs.dec, current_time, commissioning_end, tle_line1, tle_line2)

        wi = 0
        while remaining > 0 and current_time < commissioning_end:
            if wi >= len(vis_windows):
                break
            win_start, win_stop = vis_windows[wi]
            usable = int((win_stop - current_time).total_seconds())
            usable_blocks = (usable // TEN_MIN) * TEN_MIN
            if usable_blocks <= 0:
                wi += 1
                current_time = win_stop
                continue
            chunk = min(usable_blocks, remaining)
            chunk = (chunk // TEN_MIN) * TEN_MIN
            if chunk <= 0:
                wi += 1
                current_time = win_stop
                continue

            # Create or reuse Visit
            if obs.target != last_target:
                if visit_el is not None:
                    master_root.append(visit_el)
                    visits_written += 1
                visit_el = ET.Element('Visit')
                vid = ET.SubElement(visit_el, 'ID'); vid.text = f"{visit_id:04d}"
                visit_id += 1
                obs_seq_id = 1
                last_target = obs.target

            # Add new Observation_Sequence
            obs_seq = ET.SubElement(visit_el, 'Observation_Sequence')
            oid = ET.SubElement(obs_seq, 'ID'); oid.text = f"{obs_seq_id:03d}"
            obs_seq_id += 1

            # Copy Observational_Parameters and Payload_Parameters from template obs
            template_obs = obs.xml_root.find('.//Observation_Sequence')
            if template_obs is not None:
                for child in template_obs:
                    obs_seq.append(ET.fromstring(ET.tostring(child)))

            # Update timing
            timing_el = obs_seq.find('Observational_Parameters/Timing')
            if timing_el is None:
                timing_el = ET.SubElement(obs_seq.find('Observational_Parameters'), 'Timing')
            start_el = timing_el.find('Start') or ET.SubElement(timing_el, 'Start')
            stop_el = timing_el.find('Stop') or ET.SubElement(timing_el, 'Stop')
            chunk_start = current_time
            chunk_stop = current_time + timedelta(seconds=chunk)
            start_el.text = chunk_start.strftime('%Y-%m-%d %H:%M:%S')
            stop_el.text = chunk_stop.strftime('%Y-%m-%d %H:%M:%S')

            # Estimate data
            chunk_bytes = estimate_data_volume_bytes(obs)
            total_bytes += chunk_bytes

            remaining -= chunk
            current_time = chunk_stop
            if current_time >= win_stop:
                wi += 1

    if visit_el is not None:
        master_root.append(visit_el)
        visits_written += 1

    downlinks = math.ceil(total_bytes * 8 / (DOWNLINK_RATE_BPS * DOWNLINK_DURATION_S))
    book = {"total_bytes": total_bytes, "downlinks_required": downlinks, "visits_written": visits_written}
    ET.ElementTree(master_root).write(output_path, encoding='utf-8', xml_declaration=True)
    return book

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
    args = parser.parse_args()

    xml_paths = gather_task_xmls(args.xml_dir)
    result = merge_schedules(
        xml_paths,
        args.output,
        (args.cvz_ra, args.cvz_dec),
        args.tle1, args.tle2,
        commissioning_start=datetime.fromisoformat(args.start),
        commissioning_end=datetime.fromisoformat(args.end)
    )
    print("Merge complete:", result)
