# Event-Based Video Recorder with YOLO

## Overview
This project demonstrates an event-triggered video recording system using YOLOv8.
When a specific object (default: `person`) is detected, the system saves a short video clip
including frames before and after the event. Metadata is logged in JSON format.

##  Features
- Real-time event detection with YOLOv8
- Buffered video (default: 5s before + 5s after)
- Clips saved in `.mp4` format
- Metadata stored in `metadata.json`:
  - Event type
  - Timestamp
  - Simulated GPS coordinates
  - File path
  - Frame count

## How to Run
pip install -r requirements.txt
python recorder.py
