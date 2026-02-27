"""
Sentinel — Delhi CCTV Simulation Generator

Generates 3 synthetic MP4 clips simulating Delhi traffic cameras:
  1. cam_connaught_place.mp4  — Connaught Place intersection
  2. cam_india_gate.mp4       — India Gate boulevard
  3. cam_chandni_chowk.mp4    — Chandni Chowk market street

Each clip has:
  - Dark CCTV-style background with grid overlay
  - Moving objects (persons, auto-rickshaws, cars, trucks, bikes)
  - Timestamp overlay with camera ID and GPS coordinates
  - 10-second duration at 30fps

Usage:
  python scripts/generate_delhi_cams.py
"""
import cv2
import numpy as np
from pathlib import Path
import math
import random
import sys


DELHI_CAMERAS = [
    {
        "id": "DEL-CP-001",
        "name": "Connaught Place Junction",
        "lat": "28.6315",
        "lon": "77.2167",
        "objects": [
            {"label": "PERSON", "color": (246, 130, 59), "w": 30, "h": 70, "count": 6},
            {"label": "AUTO", "color": (11, 158, 245), "w": 50, "h": 35, "count": 3},
            {"label": "CAR", "color": (129, 185, 16), "w": 80, "h": 45, "count": 2},
        ],
        "filename": "cam_connaught_place.mp4",
    },
    {
        "id": "DEL-IG-002",
        "name": "India Gate Boulevard",
        "lat": "28.6129",
        "lon": "77.2295",
        "objects": [
            {"label": "PERSON", "color": (246, 130, 59), "w": 28, "h": 65, "count": 4},
            {"label": "CAR", "color": (129, 185, 16), "w": 90, "h": 50, "count": 4},
            {"label": "BUS", "color": (212, 182, 6), "w": 120, "h": 55, "count": 1},
            {"label": "BIKE", "color": (246, 92, 139), "w": 25, "h": 30, "count": 3},
        ],
        "filename": "cam_india_gate.mp4",
    },
    {
        "id": "DEL-CC-003",
        "name": "Chandni Chowk Market",
        "lat": "28.6507",
        "lon": "77.2334",
        "objects": [
            {"label": "PERSON", "color": (246, 130, 59), "w": 25, "h": 60, "count": 10},
            {"label": "AUTO", "color": (11, 158, 245), "w": 45, "h": 32, "count": 4},
            {"label": "BIKE", "color": (246, 92, 139), "w": 22, "h": 28, "count": 5},
        ],
        "filename": "cam_chandni_chowk.mp4",
    },
]


def generate_delhi_video(cam: dict, output_dir: str, width=1280, height=720, fps=30.0, duration=10.0):
    """Generate a single Delhi CCTV simulation clip."""
    out_path = str(Path(output_dir) / cam["filename"])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    total_frames = int(fps * duration)

    # Create objects
    random.seed(hash(cam["id"]))
    np.random.seed(abs(hash(cam["id"])) % (2**31))

    objects = []
    obj_id = 0
    for obj_def in cam["objects"]:
        for _ in range(obj_def["count"]):
            obj_id += 1
            objects.append({
                "id": obj_id,
                "label": obj_def["label"],
                "color": obj_def["color"],
                "w": obj_def["w"] + random.randint(-5, 5),
                "h": obj_def["h"] + random.randint(-5, 5),
                "x": float(random.randint(50, width - 150)),
                "y": float(random.randint(50, height - 150)),
                "vx": random.uniform(-4, 4),
                "vy": random.uniform(-2, 2),
            })

    for frame_idx in range(total_frames):
        # Dark CCTV background
        frame = np.full((height, width, 3), (12, 16, 24), dtype=np.uint8)

        # Grid overlay
        for gx in range(0, width, 80):
            cv2.line(frame, (gx, 0), (gx, height), (20, 25, 35), 1)
        for gy in range(0, height, 80):
            cv2.line(frame, (0, gy), (width, gy), (20, 25, 35), 1)

        # Road lines (horizontal)
        road_y = height // 2
        cv2.line(frame, (0, road_y - 2), (width, road_y - 2), (40, 50, 65), 2)
        cv2.line(frame, (0, road_y + 2), (width, road_y + 2), (40, 50, 65), 2)
        # Dashed center
        for dx in range(0, width, 40):
            cv2.line(frame, (dx, road_y), (dx + 20, road_y), (60, 70, 90), 1)

        # Timestamp overlay
        t_sec = frame_idx / fps
        ts = f"2026-02-26 22:{int(t_sec)//60:02d}:{int(t_sec)%60:02d}.{int((t_sec%1)*100):02d}"
        header = f"{cam['id']}  {cam['name']}  {ts}  GPS:{cam['lat']},{cam['lon']}"
        cv2.putText(frame, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 100, 130), 1)
        cv2.putText(frame, f"F:{frame_idx:05d}  SENTINEL SIMULATED FEED", (width - 320, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 75, 100), 1)

        # Bottom info bar
        cv2.rectangle(frame, (0, height - 22), (width, height), (15, 19, 28), -1)
        cv2.putText(frame, f"Objects: {len(objects)}  |  Synthetic Delhi CCTV  |  Not real surveillance",
                    (8, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 85, 110), 1)

        # Update & draw objects
        for obj in objects:
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]

            # Bounce
            if obj["x"] < 0 or obj["x"] + obj["w"] > width:
                obj["vx"] *= -1
            if obj["y"] < 30 or obj["y"] + obj["h"] > height - 25:
                obj["vy"] *= -1
            obj["x"] = max(0, min(obj["x"], width - obj["w"]))
            obj["y"] = max(30, min(obj["y"], height - 25 - obj["h"]))

            x, y, w, h = int(obj["x"]), int(obj["y"]), obj["w"], obj["h"]

            # Filled rect (semi-transparent via overlay)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), obj["color"], -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            # Border
            cv2.rectangle(frame, (x, y), (x + w, y + h), obj["color"], 2)

            # Label + confidence
            conf = 0.70 + 0.25 * math.sin(frame_idx * 0.03 + obj["id"])
            label = f"{obj['label']}#{obj['id']} {conf:.0%}"
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, obj["color"], 1)

        # Vignette effect
        rows, cols = frame.shape[:2]
        X = cv2.getGaussianKernel(cols, cols * 0.6)
        Y = cv2.getGaussianKernel(rows, rows * 0.6)
        M = Y * X.T
        M = M / M.max()
        for i in range(3):
            frame[:, :, i] = (frame[:, :, i] * (0.3 + 0.7 * M)).astype(np.uint8)

        writer.write(frame)

    writer.release()
    return out_path


def main():
    output_dir = Path("data/delhi_cams")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SENTINEL — Delhi CCTV Simulation Generator")
    print("=" * 60)

    for cam in DELHI_CAMERAS:
        print(f"\n  Generating: {cam['id']} — {cam['name']}")
        print(f"  GPS: {cam['lat']}, {cam['lon']}")
        print(f"  Objects: {sum(o['count'] for o in cam['objects'])}")
        path = generate_delhi_video(cam, str(output_dir))
        size_mb = Path(path).stat().st_size / 1024 / 1024
        print(f"  Output: {path} ({size_mb:.1f} MB)")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Generated {len(DELHI_CAMERAS)} Delhi simulation clips")
    print(f"  📁 Location: {output_dir.absolute()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
