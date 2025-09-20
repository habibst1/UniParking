# 🚗 UniParking

UniParking is an end-to-en solution that helps drivers quickly find available parking spots in real time.  
The system combines **computer vision**, **real-time communication**, and a **mobile app** to deliver an intuitive parking experience.


## 🎥 Demo


## 📋 Project Overview

Parking in busy areas like universities, malls, or city centers is often time-consuming and frustrating.  
This project addresses the problem by using cameras, AI, and a mobile application to monitor parking occupancy live.

- 🤖 **Computer Vision (YOLOv8 + OpenCV)**  
  Cameras are placed facing the parking lots. A YOLOv8 model detects vehicles in each defined parking zone, determining whether a spot is free or occupied.

- ⚙️ **Backend (FastAPI + WebSockets)**  
  A FastAPI server processes the video streams, applies object detection, and maintains the real-time status of all parking zones.  
  Parking updates are broadcast to connected clients over WebSockets.

- 🛠️ **Zone Creation Tool**  
  A Python utility with OpenCV allows defining custom parking zones on video frames.  
  Each zone is saved in JSON format and used by the backend for detection.

- 📱 **Frontend (Flutter Mobile App)**  
  A cross-platform app built with Flutter displays live parking availability.  
  Users can:  
  - View real-time occupancy of each parking lot  
  - Get visual indicators for free vs. occupied spots  
  - Tap on a spot to see its current status  

## 🚀 How It Works

1. Parking cameras capture live video feeds.  
2. The YOLOv8 model processes frames to detect vehicles.  
3. Detected vehicles are matched with predefined parking zones.  
4. The backend updates spot availability and broadcasts results.  
5. The Flutter app receives updates instantly through WebSockets.  
6. Users can open the app and directly see where free spots are available.

## 🏢 Use Cases

- 🎓 University and school parking areas  
- 🛍️ Shopping malls and business centers  
- 🅿️ Public or private garages  
- 📊 Any large-scale parking facility where availability is critical  

---

This project showcases the integration of **AI, backend services, and mobile development** to solve a real-world problem in smart mobility.
