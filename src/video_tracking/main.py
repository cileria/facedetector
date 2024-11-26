import argparse
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from deepface import DeepFace
from deepface.commons import folder_utils
from deepface.models.facial_recognition import Facenet, VGGFace
from ultralytics import YOLO

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

MODEL_NAME = "VGG-Face"

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
VIDEOS_DIR = PROJECT_ROOT / "videos"
MODEL_DIR = PROJECT_ROOT / "model"
OUTPUT_DIR = PROJECT_ROOT / "output"
INPUT_DIR = PROJECT_ROOT / "input"
TRAINING_DATA_DIR = VIDEOS_DIR / "trainingsdata"

# Set the device to MPS (Metal Performance Shaders) GPU if available, otherwise use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def initialize_environment(
    deepface_home: Path, db_path: Path, seed_images_path: Path
) -> None:
    """Initializes environment, copies seed images, and prepares DeepFace database."""
    os.environ["DEEPFACE_HOME"] = str(deepface_home.absolute())
    folder_utils.initialize_folder()

    # Always recreate the database in the input directory
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

    # Check if seed images path exists
    if not seed_images_path.exists():
        raise FileNotFoundError(
            f"Training data directory not found: {seed_images_path}"
        )

    # Copy all images from seed directory
    for file_path in seed_images_path.rglob("*"):
        if file_path.suffix.lower() in (".jpg", ".png"):
            folder_name = file_path.parent.name
            dst_filename = f"{folder_name}-{file_path.name}"
            dst_path = db_path / dst_filename
            logger.info(f"Copying training image: {file_path.name} to {dst_path}")
            shutil.copy(file_path, dst_path)

    logger.info(f"Seed images have been copied from {seed_images_path} to {db_path}")

    # Load models
    logger.info("Loading VGGFace model...")
    VGGFace.load_model()
    logger.info("Loading Facenet model...")
    Facenet.load_facenet512d_model()

    # Warm up DeepFace database
    try:
        logger.info("Warming up DeepFace database...")
        DeepFace.find(
            img_path=str(PROJECT_ROOT / "dummy.png"),
            db_path=str(db_path),
            model_name=MODEL_NAME,
            enforce_detection=False,
        )
    except Exception as e:
        logger.error(f"Error during DeepFace warm-up: {e}")


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    for directory in [MODEL_DIR, OUTPUT_DIR, INPUT_DIR, TRAINING_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def process_video(
    video_path: str,
    output_dir: str,
    db_path: str,
    face_distance_threshold: float = 0.4,
    person_confidence_score: float = 0.80,
    retry_in_seconds: int = 1,
    frame_skip: int = 10,  # Process every nth frame
    resize_factor: float = 0.5,  # Resize frames to half resolution
) -> None:
    """Processes the video to detect faces and recognize them using DeepFace."""
    model = YOLO("model/yolo11n.pt").to(device)  # Pretrained YOLO11n model
    cap = cv2.VideoCapture(video_path)

    # Get video properties for output video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_factor)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_factor)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(
        "output/tracking_video.mp4v",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    class_names = (
        model.names
    )  # Assuming model.names returns a list or dictionary of class names
    face_id_cache: dict[
        tuple, FaceNotFound | FaceFound
    ] = {}  # Cache to store ID to recognized name mapping

    frame_count = 0

    # Set of classes you consider as animals
    animal_classes = {"dog", "cat"} 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video reached or failed to read the frame.")
            break  # Break the loop if no frame is read (end of video)
        
        # Skip frames
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Resize frame for faster processing
        #frame = cv2.resize(frame, (frame_width, frame_height))

        # Run YOLO model on the frame with tracking enabled
        results = model.track(frame, persist=True, verbose=False)

        for result in results:
            boxes = result.boxes  # Get bounding boxes
            for box in boxes:
                class_id = int(box.cls[0])  # Get class ID as an integer
                class_name = class_names[
                    class_id
                ]  # Get the class name from the class ID
                object_id = box.id  # Unique ID assigned by YOLO for tracking
                object_id_hash = object_id_to_key(object_id)

                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract coordinates
                confidence = box.conf[0]  # Confidence score
                label = f"{class_name}({object_id_hash}) {confidence:.2f}"  # Initial label with class and confidence

                # Check for animal detection separately
                if class_name in animal_classes:
                    logger.info(f"Animal detected: {class_name} with confidence {confidence:.2f}")
                    # You can add any specific logic for animals here if needed
                    label += f" Animal detected: {class_name}"

                color = (0, 0, 255)
                if (
                    is_object("person", person_confidence_score, class_name, confidence)
                    or object_id_hash in face_id_cache
                ):
                    color = (0, 255, 0)

                    run_detection = True
                    if object_id_hash in face_id_cache:
                        run_detection = False
                        logger.debug(f"Cached detection ObjectId: {object_id_hash}")
                        if isinstance(face_id_cache[object_id_hash], FaceNotFound):
                            # Skip until we have reached next checkpoint
                            if (
                                frame_count
                                < face_id_cache[object_id_hash].skip_frames_until
                            ):
                                label += " Match: Unknown"
                            else:
                                # Rerun detection
                                del face_id_cache[object_id_hash]
                                run_detection = True
                        else:
                            # If the person is already recognized, use the cached name
                            label += f" Match: {face_id_cache[object_id_hash].match_name} Distance: {face_id_cache[object_id_hash].distance:.2f}"

                    if run_detection:
                        logger.info(f"Run detection for ObjectId: {object_id_hash}")
                        # Extract the face region from the frame
                        face_region = frame[y1:y2, x1:x2]
                        # Convert face region to a tensor and move it to the GPU
                        face_region_np = np.array(face_region)

                        try:
                            result_df = DeepFace.find(
                                img_path=face_region_np,
                                db_path=db_path,
                                model_name=MODEL_NAME,
                                threshold=face_distance_threshold,
                                enforce_detection=False,
                                silent=True,
                            )

                            if len(result_df) > 0 and result_df[0].size > 0:
                                best_match = result_df[0].iloc[0]
                                logger.info(
                                    f"Match found for ObjectId: {object_id_hash} - Distance: {best_match['distance']:.2f} Threshold: {best_match['threshold']}"
                                )

                                best_match_path = best_match["identity"]
                                match_name = os.path.basename(best_match_path).split(
                                    "."
                                )[0]
                                logger.info(
                                    f"Match found for ObjectId: {object_id_hash} - Match: {match_name}"
                                )
                                face_id_cache[object_id_hash] = FaceFound(
                                    object_id,
                                    class_name,
                                    match_name,
                                    distance=best_match["distance"],
                                )
                                label += f" Match: {face_id_cache[object_id_hash].match_name} Distance: {face_id_cache[object_id_hash].distance:.2f}"
                            else:
                                # Skip frames to improve performance.
                                face_id_cache[object_id_hash] = FaceNotFound(
                                    object_id,
                                    class_name,
                                    skip_frames_until=frame_count
                                    + (fps * retry_in_seconds),
                                )
                                label += " Match: Unknown"

                        except Exception as e:
                            logger.error(
                                f"Error during face recognition for ID {object_id_hash}: {e}"
                            )

                # Draw the bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 3, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                if (
                    class_name == "person"
                ):  # Save images if a person is detected for debugging.
                    save_detected_frame(frame, output_dir, class_name, frame_count)

        cv2.imshow("Object Detection", frame)
        out.write(frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Video processing stopped by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info("Video processing completed and resources released.")


def save_detected_frame(
    frame: np.ndarray, output_dir: str, class_name: str, frame_count: int
) -> None:
    """Saves the detected frame to the output directory."""
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    output_image_path = os.path.join(class_output_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(output_image_path, frame)
    logger.debug(f"Frame {frame_count} saved to {output_image_path}.")


def object_id_to_key(tensor: torch.Tensor) -> tuple:
    return tuple(tensor.flatten().tolist())


def is_object(
    expected_class_name: str,
    expected_confidence: float,
    actual_class_name: str,
    actual_confidence: float,
) -> bool:
    return (
        expected_class_name == actual_class_name
        and expected_confidence <= actual_confidence
    )


@dataclass
class FaceNotFound:
    object_id: torch.Tensor
    class_name: str
    skip_frames_until: int


@dataclass
class FaceFound:
    object_id: torch.Tensor
    class_name: str
    match_name: str
    distance: float


# Main execution
def main(
    video_paths: list[str],
    output_dir: Path,
    input_dir: Path,
    face_distance_threshold: float = 0.4,
    person_confidence_score: float = 0.80,
) -> None:
    """Main function to process videos."""
    ensure_directories()

    # Initialize environment
    deepface_home = MODEL_DIR / "deepface"
    training_data = (
        TRAINING_DATA_DIR / "jan"
    )  # Using 'jan' as the default training data directory

    if not training_data.exists():
        logger.error(f"Training data directory not found: {training_data}")
        logger.info(f"Please ensure your training data is in: {training_data}")
        return

    try:
        initialize_environment(deepface_home, input_dir, training_data)
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        return

    # Process each video
    for video_path in video_paths:
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"Video path does not exist: {video_path}")
            continue

        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path

        logger.info(f"Processing video: {video_path}")
        try:
            process_video(
                video_path=str(video_path),
                output_dir=str(output_dir),
                db_path=str(input_dir),
                face_distance_threshold=face_distance_threshold,
                person_confidence_score=person_confidence_score,
            )
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")


def cli() -> None:
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Video processing with face recognition"
    )
    parser.add_argument(
        "video_paths", nargs="+", type=str, help="Paths to video files to process"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for processed frames",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Input directory for face database",
    )
    parser.add_argument(
        "--face-distance-threshold",
        type=float,
        default=0.4,
        help="Threshold for face recognition distance",
    )
    parser.add_argument(
        "--person-confidence-score",
        type=float,
        default=0.80,
        help="Confidence threshold for person detection",
    )

    args = parser.parse_args()
    main(
        video_paths=args.video_paths,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        face_distance_threshold=args.face_distance_threshold,
        person_confidence_score=args.person_confidence_score,
    )


if __name__ == "__main__":
    cli()
