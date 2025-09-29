import cv2
import time
import os
import json
import random
from collections import deque
from datetime import datetime
from threading import Thread, Lock
from ultralytics import YOLO

# ---------- Utility functions ----------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def now_str():
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

def simulate_gps(base=(28.6139, 77.2090), jitter=0.01):
    lat = base[0] + random.uniform(-jitter, jitter)
    lon = base[1] + random.uniform(-jitter, jitter)
    return [round(lat, 6), round(lon, 6)]

class MetadataStore:
    def __init__(self, path):
        self.path = path
        self.lock = Lock()
        ensure_dir(os.path.dirname(path) or ".")
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump([], f)

    def append(self, item):
        with self.lock:
            with open(self.path, "r") as f:
                data = json.load(f)
            data.append(item)
            with open(self.path, "w") as f:
                json.dump(data, f, indent=2)

def write_clip(frames, fps, out_path):
    if len(frames) == 0:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"[+] Saved clip: {out_path}")

# ---------- Main Recorder ----------
def run_event_recorder_yolo(source=0, pre_seconds=5, post_seconds=5, save_dir="clips", trigger_class="person"):
    ensure_dir(save_dir)
    metadata = MetadataStore(os.path.join(save_dir, "metadata.json"))

    # Load YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 20.0

    pre_frames = int(pre_seconds * fps)
    post_frames = int(post_seconds * fps)

    buffer_deque = deque(maxlen=pre_frames)
    saving_threads = []

    print(f"Running... Event triggers when YOLO detects: {trigger_class}")
    print("Press 'q' to quit.")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        buffer_deque.append(frame.copy())
        frame_count += 1

        # Resize for faster YOLO detection (e.g., 640 width)
        resized = cv2.resize(frame, (640, 360))

        # Run YOLO only every 3rd frame (for speed)
        detected_classes = []
        if frame_count % 3 == 0:
            results = model.predict(resized, imgsz=640, verbose=False)
            detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
            annotated = results[0].plot()
        else:
            annotated = resized

        # Show video in real-time
        cv2.imshow("YOLO Event Recorder", annotated)

        # Trigger event if desired class detected
        if trigger_class in detected_classes:
            print(f"[*] Event triggered! Detected {trigger_class}")

            clip_name = f"event_{now_str()}_{random.randint(1000,9999)}.mp4"
            out_path = os.path.join(save_dir, clip_name)

            pre_list = list(buffer_deque)
            post_list = []

            # Collect post-event frames
            for _ in range(post_frames):
                ret2, f2 = cap.read()
                if not ret2:
                    break
                post_list.append(f2.copy())
                cv2.imshow("YOLO Event Recorder", f2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            clip_frames = pre_list + post_list
            t = Thread(target=write_clip, args=(clip_frames, fps, out_path))
            t.start()
            saving_threads.append(t)

            # Save metadata
            meta = {
                "event_type": f"yolo_{trigger_class}",
                "timestamp": now_str(),
                "gps": simulate_gps(),
                "video_file": out_path,
                "frames": len(clip_frames)
            }
            metadata.append(meta)
            print(f"[+] Metadata saved for {clip_name}")

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    for t in saving_threads:
        t.join()
    print("Stopped. All clips saved.")

if __name__ == "__main__":
    run_event_recorder_yolo(source=0, pre_seconds=5, post_seconds=5, trigger_class="person")
