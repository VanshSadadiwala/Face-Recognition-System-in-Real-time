import argparse
import os
import time
from collections import deque

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ================================================================
# Configuration
# ================================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30

LAPTOP_CAMERA_INDEX = 0

REFERENCE_IMAGE_FOLDER = "reference_faces"
PERSON_NAME = "Match"

# ArcFace cosine similarity threshold.
# Increase to make matching stricter.
MATCH_THRESHOLD = 0.4

# Run face detection and recognition once every N frames.
# Increase this to 8 or 10 if the system remains slow.
PROCESS_EVERY_N_FRAMES = 8

# Smaller detector size gives much better CPU performance.
DETECTION_SIZE = (320, 320)

# Temporal voting reduces unstable matching.
MATCH_HISTORY_LEN = 1
MATCH_VOTE_THRESHOLD = 1

# RealSense depth display adds extra processing.
SHOW_DEPTH = False

# Resize the recognition input before processing.
# 1.0 means original 640x480.
# Use 0.75 or 0.5 for additional performance.
PROCESSING_SCALE = 0.75

# Show calculated display FPS.
SHOW_FPS = True


# ================================================================
# InsightFace model setup
# ================================================================

def create_face_analyzer():
    """
    Loads InsightFace using CPU execution.
    Only face detection and recognition modules are loaded.
    """

    print("[MODEL] Loading CPUExecutionProvider...")

    face_app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )

    face_app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE,
        det_thresh=0.55,
    )

    print("[MODEL] InsightFace is running on CPU.")

    return face_app


# ================================================================
# Embedding utilities
# ================================================================

def normalize_embedding(embedding):
    """
    L2-normalizes an embedding.
    """

    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


def cosine_similarity(embedding_a, embedding_b):
    """
    Computes cosine similarity between two normalized embeddings.
    """

    return float(np.dot(embedding_a, embedding_b))


# ================================================================
# Reference face loading
# ================================================================

def load_reference_encodings(folder_path, face_app):
    """
    Loads face embeddings from images stored in the reference folder.
    """

    if not os.path.isdir(folder_path):
        print(f"[WARN] Folder '{folder_path}' does not exist.")
        print("[WARN] Create it and add reference face images.")
        return []

    reference_encodings = []

    supported_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    )

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(supported_extensions):
            continue

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"[WARN] Could not read '{filename}'.")
            continue

        try:
            faces = face_app.get(image)
        except Exception as error:
            print(
                f"[WARN] Face processing failed for "
                f"'{filename}': {error}"
            )
            continue

        if len(faces) == 0:
            print(f"[WARN] No face detected in '{filename}'.")
            continue

        if len(faces) > 1:
            print(
                f"[WARN] Multiple faces found in '{filename}'. "
                "Using the largest face."
            )

            faces = sorted(
                faces,
                key=lambda face: (
                    (face.bbox[2] - face.bbox[0])
                    * (face.bbox[3] - face.bbox[1])
                ),
                reverse=True,
            )

        embedding = faces[0].normed_embedding

        if embedding is None:
            print(
                f"[WARN] No recognition embedding produced "
                f"for '{filename}'."
            )
            continue

        reference_encodings.append(
            normalize_embedding(embedding)
        )

        print(f"[INFO] Loaded reference: {filename}")

    print(
        f"[INFO] Total reference embeddings: "
        f"{len(reference_encodings)}"
    )

    return reference_encodings


# ================================================================
# Face processing
# ================================================================

def get_tracking_key(bbox, original_scale):
    """
    Produces a lightweight tracking key based on face location.
    """

    left, top, right, bottom = bbox

    left = int(left / original_scale)
    top = int(top / original_scale)
    right = int(right / original_scale)
    bottom = int(bottom / original_scale)

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    # Larger cells make the identity history less sensitive
    # to small face movements.
    return center_x // 80, center_y // 80


def recognize_faces(
    frame,
    face_app,
    reference_encodings,
    match_histories,
):
    """
    Runs face detection and ArcFace matching.

    Returns annotations rather than drawing directly, allowing the same
    annotations to be displayed on later camera frames.
    """

    annotations = []

    if not reference_encodings:
        return annotations

    if PROCESSING_SCALE != 1.0:
        processing_frame = cv2.resize(
            frame,
            None,
            fx=PROCESSING_SCALE,
            fy=PROCESSING_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        processing_frame = frame

    try:
        faces = face_app.get(processing_frame)
    except Exception as error:
        print(f"[WARN] Face inference failed: {error}")
        return annotations

    current_keys = set()

    frame_height, frame_width = frame.shape[:2]

    for face in faces:
        if face.normed_embedding is None:
            continue

        scaled_bbox = face.bbox.astype(float)

        tracking_key = get_tracking_key(
            scaled_bbox,
            PROCESSING_SCALE,
        )

        current_keys.add(tracking_key)

        embedding = normalize_embedding(
            face.normed_embedding
        )

        similarities = [
            cosine_similarity(embedding, reference)
            for reference in reference_encodings
        ]

        best_similarity = max(similarities)
        raw_match = best_similarity >= MATCH_THRESHOLD

        if tracking_key not in match_histories:
            match_histories[tracking_key] = deque(
                maxlen=MATCH_HISTORY_LEN
            )

        match_histories[tracking_key].append(raw_match)

        positive_votes = sum(
            match_histories[tracking_key]
        )

        stable_match = (
            positive_votes >= MATCH_VOTE_THRESHOLD
        )

        left = int(scaled_bbox[0] / PROCESSING_SCALE)
        top = int(scaled_bbox[1] / PROCESSING_SCALE)
        right = int(scaled_bbox[2] / PROCESSING_SCALE)
        bottom = int(scaled_bbox[3] / PROCESSING_SCALE)

        left = max(0, min(left, frame_width - 1))
        top = max(0, min(top, frame_height - 1))
        right = max(0, min(right, frame_width - 1))
        bottom = max(0, min(bottom, frame_height - 1))

        # Show active votes out of max history length: e.g., [3/5]
        votes_info = f"[{positive_votes}/{MATCH_HISTORY_LEN}]"

        if stable_match:
            label = f"{PERSON_NAME} ({best_similarity:.2f}) {votes_info}"
            color = (0, 255, 0)
        else:
            label = f"Unknown ({best_similarity:.2f}) {votes_info}"
            color = (0, 0, 255)
        annotations.append(
            {
                "bbox": (left, top, right, bottom),
                "label": label,
                "color": color,
            }
        )

    stale_keys = [
        key
        for key in match_histories
        if key not in current_keys
    ]

    for key in stale_keys:
        del match_histories[key]

    return annotations


def draw_annotations(frame, annotations):
    """
    Draws cached face boxes and labels on the current camera frame.
    """

    result_frame = frame.copy()

    for annotation in annotations:
        left, top, right, bottom = annotation["bbox"]
        label = annotation["label"]
        color = annotation["color"]

        cv2.rectangle(
            result_frame,
            (left, top),
            (right, bottom),
            color,
            2,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        text_size, baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        text_width, text_height = text_size

        label_top = max(
            0,
            top - text_height - baseline - 10,
        )

        label_bottom = max(
            text_height + baseline + 5,
            top,
        )

        label_right = min(
            frame.shape[1] - 1,
            left + text_width + 10,
        )

        cv2.rectangle(
            result_frame,
            (left, label_top),
            (label_right, label_bottom),
            color,
            cv2.FILLED,
        )

        cv2.putText(
            result_frame,
            label,
            (left + 5, label_bottom - baseline - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return result_frame


def draw_system_information(
    frame,
    display_fps,
    inference_fps,
):
    """
    Draws display and inference information.
    """

    if SHOW_FPS:
        cv2.putText(
            frame,
            f"Display FPS: {display_fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Recognition FPS: {inference_fps:.1f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "Press Q to exit",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ================================================================
# Laptop webcam
# ================================================================

def run_laptop_camera(
    face_app,
    reference_encodings,
    camera_index=0,
):
    """
    Runs optimized face recognition using the laptop webcam.
    """

    print(
        f"[CAMERA] Opening laptop camera index "
        f"{camera_index}..."
    )

    if os.name == "nt":
        capture = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW,
        )
    else:
        capture = cv2.VideoCapture(camera_index)

    capture.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )
    capture.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )
    capture.set(
        cv2.CAP_PROP_FPS,
        FRAME_FPS,
    )

    # Reduces internal camera buffering where supported.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        print("[ERROR] Could not open laptop webcam.")
        return

    match_histories = {}
    cached_annotations = []

    frame_number = 0

    display_fps = 0.0
    inference_fps = 0.0

    display_frame_count = 0
    display_timer = time.perf_counter()

    print("[SYSTEM] Laptop camera running.")
    print("[SYSTEM] Press 'q' to exit.")

    try:
        while True:
            success, frame = capture.read()

            if not success or frame is None:
                continue

            frame_number += 1
            display_frame_count += 1

            if (
                frame_number == 1
                or frame_number % PROCESS_EVERY_N_FRAMES == 0
            ):
                inference_start = time.perf_counter()

                cached_annotations = recognize_faces(
                    frame=frame,
                    face_app=face_app,
                    reference_encodings=reference_encodings,
                    match_histories=match_histories,
                )

                inference_time = (
                    time.perf_counter()
                    - inference_start
                )

                if inference_time > 0:
                    inference_fps = 1.0 / inference_time

            current_time = time.perf_counter()
            display_elapsed = current_time - display_timer

            if display_elapsed >= 1.0:
                display_fps = (
                    display_frame_count / display_elapsed
                )

                display_frame_count = 0
                display_timer = current_time

            result_frame = draw_annotations(
                frame,
                cached_annotations,
            )

            if not reference_encodings:
                cv2.putText(
                    result_frame,
                    "No reference images loaded",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            draw_system_information(
                result_frame,
                display_fps,
                inference_fps,
            )

            cv2.imshow(
                "Laptop Camera - Face Recognition",
                result_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[SYSTEM] Keyboard interruption received.")

    finally:
        print("[CAMERA] Closing laptop webcam...")
        capture.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Laptop camera closed.")


# ================================================================
# Intel RealSense D435i
# ================================================================

def run_realsense_camera(
    face_app,
    reference_encodings,
):
    """
    Runs optimized face recognition using RealSense color frames.

    Depth frames are displayed only when SHOW_DEPTH is True.
    """

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[ERROR] pyrealsense2 is not installed.")
        print("[INFO] Install it using:")
        print("       pip install pyrealsense2")
        return

    print("[CAMERA] Starting Intel RealSense D435i...")

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        rs.format.bgr8,
        FRAME_FPS,
    )

    if SHOW_DEPTH:
        config.enable_stream(
            rs.stream.depth,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            rs.format.z16,
            FRAME_FPS,
        )

    pipeline_started = False

    try:
        profile = pipeline.start(config)
        pipeline_started = True

        device = profile.get_device()
        device_name = device.get_info(
            rs.camera_info.name
        )

        print(f"[CAMERA] Connected device: {device_name}")

        # Try to reduce RealSense frame queue delay.
        for sensor in device.query_sensors():
            if sensor.supports(
                rs.option.frames_queue_size
            ):
                try:
                    sensor.set_option(
                        rs.option.frames_queue_size,
                        1,
                    )
                except RuntimeError:
                    pass

    except Exception as error:
        print("[ERROR] Could not start RealSense.")
        print(f"[ERROR] {error}")

        if pipeline_started:
            pipeline.stop()

        return

    align = None
    colorizer = None

    if SHOW_DEPTH:
        align = rs.align(rs.stream.color)
        colorizer = rs.colorizer()

    match_histories = {}
    cached_annotations = []

    frame_number = 0

    display_fps = 0.0
    inference_fps = 0.0

    display_frame_count = 0
    display_timer = time.perf_counter()

    print("[SYSTEM] RealSense camera running.")
    print("[SYSTEM] Press 'q' to exit.")

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(
                    timeout_ms=5000
                )
            except RuntimeError as error:
                print(
                    f"[WARN] RealSense timeout: {error}"
                )
                continue

            depth_image = None

            if SHOW_DEPTH:
                aligned_frames = align.process(frames)

                color_frame = (
                    aligned_frames.get_color_frame()
                )
                depth_frame = (
                    aligned_frames.get_depth_frame()
                )

                if not color_frame or not depth_frame:
                    continue

                depth_color_frame = colorizer.colorize(
                    depth_frame
                )

                depth_image = np.asanyarray(
                    depth_color_frame.get_data()
                )

            else:
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

            color_image = np.asanyarray(
                color_frame.get_data()
            )

            frame_number += 1
            display_frame_count += 1

            if (
                frame_number == 1
                or frame_number % PROCESS_EVERY_N_FRAMES == 0
            ):
                inference_start = time.perf_counter()

                cached_annotations = recognize_faces(
                    frame=color_image,
                    face_app=face_app,
                    reference_encodings=reference_encodings,
                    match_histories=match_histories,
                )

                inference_time = (
                    time.perf_counter()
                    - inference_start
                )

                if inference_time > 0:
                    inference_fps = 1.0 / inference_time

            current_time = time.perf_counter()
            display_elapsed = current_time - display_timer

            if display_elapsed >= 1.0:
                display_fps = (
                    display_frame_count / display_elapsed
                )

                display_frame_count = 0
                display_timer = current_time

            result_frame = draw_annotations(
                color_image,
                cached_annotations,
            )

            if not reference_encodings:
                cv2.putText(
                    result_frame,
                    "No reference images loaded",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            draw_system_information(
                result_frame,
                display_fps,
                inference_fps,
            )

            cv2.imshow(
                "RealSense Color - Face Recognition",
                result_frame,
            )

            if SHOW_DEPTH and depth_image is not None:
                cv2.imshow(
                    "RealSense Depth",
                    depth_image,
                )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[SYSTEM] Keyboard interruption received.")

    finally:
        print("[CAMERA] Stopping RealSense...")

        if pipeline_started:
            pipeline.stop()

        cv2.destroyAllWindows()
        print("[SYSTEM] RealSense camera closed.")


# ================================================================
# Command-line arguments
# ================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Optimized InsightFace recognition using a laptop "
            "webcam or Intel RealSense D435i."
        )
    )

    parser.add_argument(
        "--source",
        choices=["laptop", "realsense"],
        default="laptop",
        help="Select laptop webcam or RealSense camera.",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=LAPTOP_CAMERA_INDEX,
        help="OpenCV laptop camera index.",
    )

    parser.add_argument(
        "--reference-folder",
        type=str,
        default=REFERENCE_IMAGE_FOLDER,
        help="Folder containing reference face images.",
    )

    return parser.parse_args()


# ================================================================
# Main
# ================================================================

def main():
    args = parse_arguments()

    print("=" * 60)
    print("Optimized InsightFace Recognition")
    print("=" * 60)

    try:
        face_app = create_face_analyzer()
    except Exception as error:
        print(
            f"[FATAL] InsightFace initialization failed: "
            f"{error}"
        )
        return

    reference_encodings = load_reference_encodings(
        folder_path=args.reference_folder,
        face_app=face_app,
    )

    if args.source == "laptop":
        run_laptop_camera(
            face_app=face_app,
            reference_encodings=reference_encodings,
            camera_index=args.camera_index,
        )

    elif args.source == "realsense":
        run_realsense_camera(
            face_app=face_app,
            reference_encodings=reference_encodings,
        )


if __name__ == "__main__":
    main()