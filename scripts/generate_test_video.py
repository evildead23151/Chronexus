"""
Sentinel — Generate Synthetic Test Video

Creates a simple test MP4 with moving rectangles simulating
detected objects (persons, vehicles) for pipeline testing.
No external dependencies beyond OpenCV.
"""
import cv2
import numpy as np
from pathlib import Path
import math


def generate_test_video(
    output_path: str = "data/sample.mp4",
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    duration_s: float = 10.0,
    num_objects: int = 5,
) -> str:
    """Generate a synthetic test video with moving objects."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(fps * duration_s)

    # Define objects with different colors and sizes
    objects = []
    colors = [
        (59, 130, 246),   # Blue (person)
        (245, 158, 11),   # Amber (vehicle)
        (16, 185, 129),   # Green (vehicle)
        (139, 92, 246),   # Violet (person)
        (6, 182, 212),    # Cyan (person)
    ]
    labels = ['PERSON', 'VEHICLE', 'VEHICLE', 'PERSON', 'PERSON']
    sizes = [(40, 80), (120, 60), (100, 50), (35, 75), (38, 72)]

    np.random.seed(42)
    for i in range(num_objects):
        objects.append({
            'x': np.random.randint(100, width - 200),
            'y': np.random.randint(100, height - 200),
            'vx': np.random.uniform(-3, 3),
            'vy': np.random.uniform(-2, 2),
            'w': sizes[i % len(sizes)][0],
            'h': sizes[i % len(sizes)][1],
            'color': colors[i % len(colors)],
            'label': labels[i % len(labels)],
            'id': i + 1,
        })

    for frame_idx in range(total_frames):
        # Dark background with subtle grid
        frame = np.full((height, width, 3), (15, 20, 30), dtype=np.uint8)

        # Draw grid
        for gx in range(0, width, 100):
            cv2.line(frame, (gx, 0), (gx, height), (25, 30, 42), 1)
        for gy in range(0, height, 100):
            cv2.line(frame, (0, gy), (width, gy), (25, 30, 42), 1)

        # Camera overlay
        timestamp = f"CAM-001  {frame_idx / fps:.2f}s  F:{frame_idx:05d}"
        cv2.putText(frame, timestamp, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 116, 139), 1)
        cv2.putText(frame, "SENTINEL TEST FEED", (width - 230, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 116, 139), 1)

        # Update and draw objects
        for obj in objects:
            # Move
            obj['x'] += obj['vx']
            obj['y'] += obj['vy']

            # Bounce off walls
            if obj['x'] < 0 or obj['x'] + obj['w'] > width:
                obj['vx'] *= -1
            if obj['y'] < 0 or obj['y'] + obj['h'] > height:
                obj['vy'] *= -1

            obj['x'] = max(0, min(obj['x'], width - obj['w']))
            obj['y'] = max(0, min(obj['y'], height - obj['h']))

            x, y, w, h = int(obj['x']), int(obj['y']), obj['w'], obj['h']

            # Draw filled rect with slight transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), obj['color'], -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # Draw border
            cv2.rectangle(frame, (x, y), (x + w, y + h), obj['color'], 2)

            # Label
            label = f"{obj['label']} #{obj['id']}"
            cv2.putText(frame, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, obj['color'], 1)

            # Simulated confidence
            conf = 0.75 + 0.2 * math.sin(frame_idx * 0.05 + obj['id'])
            cv2.putText(frame, f"{conf:.0%}", (x + w + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1)

        # Crosshair center
        cx, cy = width // 2, height // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (50, 60, 80), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (50, 60, 80), 1)

        writer.write(frame)

    writer.release()
    print(f"Generated test video: {output_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration:   {duration_s}s @ {fps} fps")
    print(f"  Frames:     {total_frames}")
    print(f"  Objects:    {num_objects}")
    return output_path


if __name__ == "__main__":
    generate_test_video()
