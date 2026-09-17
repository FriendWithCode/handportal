"""
Unit & Integration test for updated high-performance Hand Portal FX.
Tests ROI compositing, adaptive smoother, palm-relative pinch detector, and all 9 filters.
"""
import time
import numpy as np
import cv2
from hand_portal import (
    FilterEngine, Compositor, PortalGeometry,
    AdaptiveLandmarkSmoother, GestureDetector, HandSkeletonVisualizer
)

def run_tests():
    print("[TEST] 1. Testing All 9 Filters...")
    test_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

    assert FilterEngine.filter_rainbow_wave(test_img, time.time()).shape == test_img.shape
    assert FilterEngine.filter_invert(test_img).shape == test_img.shape
    assert FilterEngine.filter_pixelate(test_img, 16).shape == test_img.shape
    assert FilterEngine.filter_cartoon(test_img).shape == test_img.shape
    assert FilterEngine.filter_dual_tone(test_img).shape == test_img.shape
    assert FilterEngine.filter_glitch(test_img, time.time()).shape == test_img.shape
    assert FilterEngine.filter_thermal(test_img).shape == test_img.shape
    assert FilterEngine.filter_edge(test_img).shape == test_img.shape
    assert FilterEngine.filter_blur(test_img, 29).shape == test_img.shape
    print(" -> All 9 Filters: OK")

    print("[TEST] 2. Testing Crop-ROI Compositor...")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    poly = np.array([[300, 200], [700, 200], [700, 500], [300, 500]], dtype=np.int32)
    
    comp = Compositor.composite_portal_roi(
        frame,
        FilterEngine.filter_invert,
        poly,
        feather_ksize=17,
        draw_border=True
    )
    assert comp.shape == frame.shape
    print(" -> Crop-ROI Compositor: OK")

    print("[TEST] 3. Testing Adaptive Smoother & Aggressive Detector...")
    smoother = AdaptiveLandmarkSmoother()
    pts1 = np.ones((21, 3), dtype=np.float32) * 100
    pts2 = np.ones((21, 3), dtype=np.float32) * 150
    s1 = smoother.smooth("Left", pts1)
    s2 = smoother.smooth("Left", pts2)
    assert s2.shape == pts2.shape
    print(" -> Adaptive Smoother: OK")

    detector = GestureDetector(enter_pinch_ratio=0.45, exit_pinch_ratio=0.60)
    # create hand landmarks with pinch
    hand_lms = np.zeros((21, 3), dtype=np.float32)
    hand_lms[0] = [200, 400, 0]  # wrist
    hand_lms[9] = [200, 300, 0]  # middle MCP (palm_scale = 100)
    hand_lms[4] = [220, 250, 0]  # thumb tip
    hand_lms[8] = [225, 250, 0]  # index tip (dist = 5, pinch_ratio = 0.05 < 0.45)
    
    analysis = detector.analyze_hand('left', hand_lms, 1280, 720)
    assert analysis['is_pinch'] == True, "Pinch should be detected"
    assert analysis['just_pinched'] == True, "First pinch frame should trigger just_pinched"
    print(" -> Aggressive Pinch Detector: OK")

    print("\nALL OPTIMIZATION TESTS PASSED! [PASS]")

if __name__ == "__main__":
    run_tests()
