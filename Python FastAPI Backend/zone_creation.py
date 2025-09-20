import cv2
import json
import numpy as np

video_path = "./Parking Videos/Video 1 modif.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read video frame")
    exit()

# Get frame dimensions
frame_height, frame_width = frame.shape[:2]

zones = []
current_zone = []

def mouse_callback(event, x, y, flags, param):
    global current_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        current_zone.append((x, y))

cv2.namedWindow("Define Parking Zones", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Define Parking Zones", frame_width, frame_height)
cv2.setMouseCallback("Define Parking Zones", mouse_callback)

while True:
    temp_frame = frame.copy()

    # Draw current points
    for point in current_zone:
        cv2.circle(temp_frame, point, 5, (0,0,255), -1)
    if len(current_zone) > 1:
        cv2.polylines(temp_frame, [np.array(current_zone)], False, (0,255,0), 2)

    # Draw all saved zones
    for zone in zones:
        cv2.polylines(temp_frame, [np.array(zone)], True, (255,0,0), 2)

    cv2.imshow("Define Parking Zones", temp_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # Enter key → save current polygon
        if len(current_zone) > 2:
            zones.append(current_zone)
        current_zone = []

    elif key == ord('s'):  # Save to JSON
        with open("parking_zones_xx.json", "w") as f:
            json.dump(zones, f)
        print("Zones saved to parking_zones.json")
        break

    elif key == 27:  # ESC → exit without saving
        break

cv2.destroyAllWindows()