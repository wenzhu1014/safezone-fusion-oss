# -*- coding: utf-8 -*-
"""Marker coasting: a transient single-frame detection miss must NOT reset the
stability counter; only a sustained miss (> budget) declares the marker gone."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from safezone_fusion.src.vision.aruco_detector import OpenCVArucoMarkerDetector
from safezone_fusion.src.vision.interfaces import CameraPacket

def _packet_with_marker(det, ts):
    """Build a frame containing a real ArUco marker the detector can see."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    img = np.full((480, 640, 3), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(d, 23, 200)
    img[140:340, 220:420] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return CameraPacket(rgb=img, timestamp=ts)

def _blank_packet(ts):
    return CameraPacket(rgb=np.full((480, 640, 3), 255, np.uint8), timestamp=ts)

def test_single_dropped_frame_does_not_reset_stability():
    det = OpenCVArucoMarkerDetector(dictionary_name="DICT_6X6_250", marker_size_m=0.05, target_marker_ids=None)
    # build up stability
    for i in range(5):
        m = det.detect(_packet_with_marker(det, float(i)))
        assert m is not None and m.visible
    assert det._stable_frames >= 5
    stable_before = det._stable_frames
    # one dropped frame -> coasting, NOT a reset
    m = det.detect(_blank_packet(5.0))
    assert m is not None, "single miss should coast, not return None"
    assert m.coasting is True
    assert m.visible is False
    assert det._stable_frames == stable_before, "stability must be preserved while coasting"
    # marker comes back -> stability keeps climbing (no restart from 1)
    m = det.detect(_packet_with_marker(det, 6.0))
    assert m is not None and m.visible
    assert m.stable_frames > stable_before

def test_sustained_miss_declares_marker_gone():
    det = OpenCVArucoMarkerDetector(dictionary_name="DICT_6X6_250", marker_size_m=0.05, target_marker_ids=None)
    for i in range(5):
        det.detect(_packet_with_marker(det, float(i)))
    # exceed the coast budget
    last = None
    for i in range(det._MAX_COAST_FRAMES + 1):
        last = det.detect(_blank_packet(10.0 + i))
    assert last is None, "after coast budget is exhausted the marker is truly gone"
    assert det._stable_frames == 0
