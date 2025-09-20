import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import asyncio
import cv2
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time

import torch
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parking_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Parking Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            simplified_message = {}
            if "all_zones" in message:
                for zone_id, zone_data in message["all_zones"].items():
                    simplified_message[zone_id] = zone_data["occupied"]
            
            await websocket.send_json(simplified_message)
        except Exception as e:
            logger.error(f"Error sending message to websocket: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        
        simplified_message = {}
        if "all_zones" in message:
            for zone_id, zone_data in message["all_zones"].items():
                simplified_message[zone_id] = zone_data["occupied"]
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(simplified_message)
            except Exception as e:
                logger.error(f"Error broadcasting to websocket: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

class VideoSource:
    def __init__(self, video_path: str, zones_file: str, threshold: float):
        self.video_path = video_path
        self.zones_file = zones_file
        self.threshold = threshold
        self.zones = []
        self.cap = None
        self.is_running = False
        self.actual_fps = 0
        self.frame_duration = 0  # Duration of each frame in seconds
        self.last_frame_time = 0  # When the last frame was processed
        self.video_start_time = 0  # When video processing started
        self.frame_pos = 0  # Current frame position

    async def start(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error(f"Error: Could not open video file {self.video_path}")
            return False
        
        # Get video properties
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_duration = 1.0 / self.actual_fps
        self.video_start_time = time.time()
        self.frame_pos = 0
        logger.info(f"Video FPS: {self.actual_fps:.2f}, Frame duration: {self.frame_duration:.4f}s")
        
        # Load zones for this video
        try:
            with open(self.zones_file, "r") as f:
                zones_data = json.load(f)
            self.zones = [np.array(zone, dtype=np.int32) for zone in zones_data]
            logger.info(f"Loaded {len(self.zones)} parking zones from {self.zones_file}")
        except Exception as e:
            logger.error(f"Error loading zones: {e}")
            return False
        
        self.is_running = True
        return True

    async def stop(self):
        if self.cap:
            self.cap.release()
        self.is_running = False

    async def get_frame(self):
        if not self.cap or not self.is_running:
            return None
        
        # Calculate when this frame should be displayed based on video time
        current_video_time = time.time() - self.video_start_time
        target_frame_pos = int(current_video_time * self.actual_fps)
        
        # If we're ahead of the video, wait
        if target_frame_pos <= self.frame_pos:
            return None
        
        # If we're behind, seek to the correct frame
        frames_to_skip = target_frame_pos - self.frame_pos - 1
        if frames_to_skip > 0:
            logger.debug(f"Skipping {frames_to_skip} frames to catch up")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_pos)
        
        ret, frame = self.cap.read()
        self.frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        if not ret:
            logger.info(f"End of video reached for {self.video_path}, restarting...")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.video_start_time = time.time()
            self.frame_pos = 0
            ret, frame = self.cap.read()
            if not ret:
                return None
        
        return frame

class ParkingDetector:
    def __init__(self):
        self.model = None
        self.video_sources = []
        self.global_zones = {}  # Maps zone_id to zone data
        self.previous_status = {}
        self.confirmed_status = {}
        self.state_start_time = {}
        self.vehicle_classes = ["car", "truck", "motorbike", "bus"]
        self.min_vehicle_size = 500
        self.is_running = False
        self.confirmation_threshold = timedelta(seconds=2)

        # Define how many zones are in the top rows for each video
        self.top_row_zones = {
            0: 6,  # Video 1 has 6 zones in top row
            1: 6    # Video 2 has 6 zones in top row
        }
        
    def load_model(self):
        try:
            self.model = YOLO("yolov8m.pt")
            logger.info("YOLO model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            return False
    
    def add_video_source(self, video_path: str, zones_file: str, threshold: float):
        """Add a new video source with its own zones and threshold"""
        self.video_sources.append(VideoSource(video_path, zones_file, threshold))
    
    def calculate_overlap(self, zone, car_box):
        try:
            zone_poly = cv2.convexHull(zone.astype(np.float32))
            car_poly = np.array([[car_box[0][0], car_box[0][1]],
                               [car_box[1][0], car_box[0][1]],
                               [car_box[1][0], car_box[1][1]],
                               [car_box[0][0], car_box[1][1]]])
            car_poly = cv2.convexHull(car_poly.astype(np.float32))
            
            intersection_area = cv2.intersectConvexConvex(zone_poly, car_poly)[0]
            zone_area = cv2.contourArea(zone_poly)
            
            if zone_area > 0:
                return intersection_area / zone_area
            return 0
        except Exception as e:
            logger.error(f"Error calculating overlap: {e}")
            return 0

    def get_zone_id(self, video_id: int, zone_idx: int) -> str:
        """Get the zone ID with top rows reversed"""
        top_row_count = self.top_row_zones.get(video_id, 0)
        
        if zone_idx < top_row_count:
            # Reverse numbering for top row zones
            reversed_idx = top_row_count - 1 - zone_idx
            return f"video{video_id + 1}_zone_{reversed_idx + 1}"
        else:
            return f"video{video_id + 1}_zone_{zone_idx + 1}"

    async def process_frame(self, frame, zones, threshold, video_id):
        try:
            car_boxes = []
            changes = []
            now = datetime.now()
            
            results = self.model(frame, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    if class_name in self.vehicle_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        
                        if area > self.min_vehicle_size:
                            car_boxes.append(((x1, y1), (x2, y2), class_name))
            
            for idx, zone in enumerate(zones):
                # Use the reversed zone ID for top rows
                zone_id = self.get_zone_id(video_id, idx)
                
                zone_occupied = False
                total_overlap = 0
                
                for car_box in car_boxes:
                    overlap = self.calculate_overlap(zone, car_box)
                    total_overlap += overlap
                    
                    if overlap > threshold:
                        zone_occupied = True
                        break
                
                current_status = {
                    "occupied": zone_occupied,
                    "overlap_percentage": round(total_overlap * 100, 1),
                    "timestamp": now.isoformat(),
                    "video_id": video_id
                }
                
                if zone_id not in self.previous_status or \
                   self.previous_status[zone_id]["occupied"] != zone_occupied:
                    self.state_start_time[zone_id] = now
                else:
                    state_duration = now - self.state_start_time[zone_id]
                    if state_duration >= self.confirmation_threshold:
                        if self.confirmed_status.get(zone_id, {}).get("occupied") != zone_occupied:
                            old_status = self.confirmed_status.get(zone_id, {}).get("occupied", False)
                            self.confirmed_status[zone_id] = current_status
                            
                            changes.append({
                                "zone_id": zone_id,
                                "previous_status": "occupied" if old_status else "free",
                                "current_status": "occupied" if zone_occupied else "free",
                                "overlap_percentage": round(total_overlap * 100, 1),
                                "timestamp": now.isoformat(),
                                "state_duration_seconds": state_duration.total_seconds(),
                                "video_id": video_id
                            })
                            logger.info(
                                f"Zone {zone_id} confirmed change from {old_status} to {zone_occupied} "
                                f"after {state_duration.total_seconds():.1f} seconds"
                            )
                
                self.previous_status[zone_id] = current_status
            
            return changes
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return []
    
    async def process_video_source(self, video_source: VideoSource, video_id: int):
        if not await video_source.start():
            return
        
        try:
            while self.is_running and video_source.is_running:
                start_time = time.time()
                
                frame = await video_source.get_frame()
                if frame is None:
                    # Calculate how long to wait until next frame is due
                    next_frame_time = video_source.video_start_time + (video_source.frame_pos * video_source.frame_duration)
                    sleep_time = max(0, next_frame_time - time.time())
                    await asyncio.sleep(sleep_time)
                    continue
                
                changes = await self.process_frame(
                    frame, 
                    video_source.zones, 
                    video_source.threshold,
                    video_id
                )
                
                if changes:
                    message = {
                        "type": "status_change",
                        "changes": changes,
                        "all_zones": self.confirmed_status,
                        "timestamp": datetime.now().isoformat()
                    }
                    await manager.broadcast(message)
                
                # Calculate processing time and adjust sleep if needed
                processing_time = time.time() - start_time
                sleep_time = max(0, video_source.frame_duration - processing_time)
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in video processing loop for {video_source.video_path}: {e}")
        finally:
            await video_source.stop()
            logger.info(f"Video processing stopped for {video_source.video_path}")

    async def start_detection(self):
        if not self.model:
            logger.error("Model not loaded")
            return
        
        self.is_running = True
        logger.info("Starting parking detection for all video sources")
        
        try:
            tasks = []
            for idx, source in enumerate(self.video_sources):
                tasks.append(self.process_video_source(source, idx))
            
            await asyncio.gather(*tasks)
                
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
        finally:
            self.is_running = False
            logger.info("Parking detection stopped for all video sources")

detector = ParkingDetector()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Parking Detection API")
    if not detector.load_model():
        logger.error("Failed to load YOLO model")
        return
    
    # Add your video sources with their specific thresholds
    detector.add_video_source(
        video_path="./Parking Videos/Video 1 modif.mp4",
        zones_file="parking_zones_1.json",
        threshold=0.2  # Lower threshold for Video 1
    )
    detector.add_video_source(
        video_path="./Parking Videos/Video 2 modif.mp4",
        zones_file="parking_zones_2.json",
        threshold=0.3  # Higher threshold for Video 2
    )
    
    # Initialize status tracking for all zones
    for video_id, source in enumerate(detector.video_sources):
        for zone_idx in range(len(source.zones)):
            zone_id = f"video{video_id + 1}_zone_{zone_idx + 1}"
            detector.state_start_time[zone_id] = datetime.now()
            detector.confirmed_status[zone_id] = {
                "occupied": False, 
                "timestamp": datetime.now().isoformat(),
                "video_id": video_id
            }
            detector.previous_status[zone_id] = {
                "occupied": False, 
                "timestamp": datetime.now().isoformat(),
                "video_id": video_id
            }
    
    asyncio.create_task(detector.start_detection())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time parking status updates"""
    await manager.connect(websocket)
    try:
        # Send current status immediately upon connection
        if detector.confirmed_status:
            initial_message = {
                "type": "initial_status",
                "all_zones": detector.confirmed_status,
                "timestamp": datetime.now().isoformat()
            }
            await manager.send_personal_message(initial_message, websocket)
        
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)