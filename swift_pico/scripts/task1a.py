#!/usr/bin/env python3
"""Task 1A survivor detector, also reused by Task 2C."""

import argparse
import os
import sys

import cv2
import numpy as np


REQUIRED_IDS = {80, 85, 90, 95}


def _detect_aruco(gray):
    """Support both generations of the OpenCV ArUco API."""
    if hasattr(cv2.aruco, 'ArucoDetector'):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        return detector.detectMarkers(gray)[:2]
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return corners, ids


def _arena_corners(corners, ids):
    """Get arena's inner ArUco vertices ordered TL, TR, BR, BL."""
    if ids is None:
        raise ValueError('No marker detected')
    marker_ids = ids.flatten().astype(int)
    if not REQUIRED_IDS.issubset(set(marker_ids)):
        raise ValueError('One or More Ids are missing')

    marker_corners = {marker_id: corners[index][0]
                      for index, marker_id in enumerate(marker_ids)}
    marker_centres = {marker_id: np.mean(points, axis=0)
                      for marker_id, points in marker_corners.items()}
    arena_middle = np.mean(list(marker_centres.values()), axis=0)
    inner_corner = {
        marker_id: points[np.argmin([np.linalg.norm(point - arena_middle) for point in points])]
        for marker_id, points in marker_corners.items()
    }
    by_y = sorted(marker_centres.items(), key=lambda item: item[1][1])
    top = sorted(by_y[:2], key=lambda item: item[1][0])
    bottom = sorted(by_y[2:], key=lambda item: item[1][0])
    tl_id, tr_id = top[0][0], top[1][0]
    bl_id, br_id = bottom[0][0], bottom[1][0]
    return np.float32([inner_corner[tl_id], inner_corner[tr_id],
                       inner_corner[br_id], inner_corner[bl_id]]), sorted(marker_ids.tolist())


def _nearest_grid_label(cx, cy):
    column = int(np.clip(round(cx / 75.0), 1, 11))
    row = int(np.clip(round(cy / 75.0), 1, 11))
    return f"{chr(ord('A') + column - 1)}{row}"


def _contour_centres(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width > 0 and height > 0:
            centres.append((x + width / 2.0, y + height / 2.0))
    return centres


def detect_survivors(image):
    """Run Task 1A on a BGR image; Task 2C uses critical_survivors from this."""
    if image is None or image.size == 0:
        raise ValueError('Could not read image')
    corners, ids = _detect_aruco(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    source, marker_ids = _arena_corners(corners, ids)
    destination = np.float32([[0, 0], [900, 0], [900, 900], [0, 900]])
    warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, destination), (900, 900))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 150, 100]), np.array([179, 255, 255])))
    yellow_mask = cv2.inRange(hsv, np.array([20, 150, 150]), np.array([35, 255, 255]))
    critical = sorted({_nearest_grid_label(x, y) for x, y in _contour_centres(red_mask)})
    stable = sorted({_nearest_grid_label(x, y) for x, y in _contour_centres(yellow_mask)})
    return {
        'marker_ids': marker_ids,
        'critical_survivors': critical,
        'stable_survivors': stable,
        'warped': warped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    image_path = parser.parse_args().image
    try:
        result = detect_survivors(cv2.imread(image_path))
    except ValueError as error:
        print(error)
        return 1
    folder, filename = os.path.dirname(image_path), os.path.basename(image_path)
    output_path = os.path.join(folder, os.path.splitext(filename)[0] + '_results.txt')
    with open(output_path, 'w', encoding='utf-8') as output:
        output.write(f"Detected marker IDs: {result['marker_ids']}\n\n")
        output.write('Critical Survivors: ' + ', '.join(result['critical_survivors']) + '\n')
        output.write('Stable Survivors: ' + ', '.join(result['stable_survivors']) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
