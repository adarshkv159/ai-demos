import sys
import time
import os
import numpy as np
import cv2
import const
from logging import getLogger
import tflite_runtime.interpreter as tflite
import csv
from datetime import datetime
import serial
from angle_calculations import generate_arm_command

logger = getLogger(__name__)

# ======================
# Utility Functions
# ======================
def find_and_append_util_path():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        potential_util_path = os.path.join(current_dir, 'util')
        if os.path.exists(potential_util_path):
            sys.path.append(potential_util_path)
            return
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Couldn't find 'util' directory. Please ensure it's in the project directory structure.")

find_and_append_util_path()

from utils import file_abs_path, get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models
from image_utils import load_image as load_image_img
from pose_resnet_util import compute, keep_aspect

# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
POSE_THRESHOLD = 0.5  # Minimum confidence threshold for keypoints
SSD_MODEL_NAME = 'ssd_mobilenet_v1_quant'
SSD_MODEL_PATH = file_abs_path(__file__, f'{SSD_MODEL_NAME}.tflite')
SSD_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/ssd_mobilenet_v1/'
POSE_MODEL_NAME = 'pose_resnet_50_256x192_int8'
POSE_MODEL_PATH = file_abs_path(__file__, f'{POSE_MODEL_NAME}.tflite')
POSE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/pose_resnet/'

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Simple Baseline for Pose Estimation', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-th', '--threshold',
    default=0.5, type=float,
    help='Detection confidence threshold (default: 0.5)'
)
parser.add_argument(
    '-d', '--delegate',
    default='',
    help='Delegate path for TFLite'
)
parser.add_argument(
    '--record_interval',
    default=2, type=int,
    help='Interval in seconds between recording keypoints (default: 2)'
)
args = update_parser(parser)

if args.float:
    POSE_MODEL_NAME = 'pose_resnet_50_256x192_float32'
    POSE_MODEL_PATH = file_abs_path(__file__, f'{POSE_MODEL_NAME}.tflite')

# ======================
# CSV and Serial Functions
# ======================
def initialize_csv():
    """Initialize or clear the CSV file with headers for all keypoints"""
    if os.path.exists('keypoints.csv'):
        os.remove('keypoints.csv')
    
    with open('keypoints.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Timestamp',
            'Keypoint',
            'X',
            'Y',
            'Confidence'
        ])

def should_record(current_time, last_record_time):
    """Check if enough time has passed since last recording"""
    return (current_time - last_record_time).total_seconds() >= args.record_interval

def write_keypoints_to_csv(person, timestamp=None, frame_width=640, frame_height=480, ser=None):
    """Write keypoints to CSV if all required points are detected with sufficient confidence"""
    global last_keypoints, last_send_time
    
    required_points = {
        'left_shoulder': const.POSE_KEYPOINT_SHOULDER_LEFT,
        'left_elbow': const.POSE_KEYPOINT_ELBOW_LEFT,
        'left_wrist': const.POSE_KEYPOINT_WRIST_LEFT,
        'left_hip': const.POSE_KEYPOINT_HIP_LEFT
    }
    
    timestamp = timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    all_detected = True
    keypoint_data = []
    current_keypoints = {}
    
    for name, point in required_points.items():
        if person.points[point].score > POSE_THRESHOLD:
            x = int(person.points[point].x * frame_width)
            y = int(person.points[point].y * frame_height)
            confidence = person.points[point].score
            keypoint_data.append([timestamp, name, x, y, confidence])
            current_keypoints[name] = (x, y)
        else:
            all_detected = False
            break
    
    if all_detected:
        with open('keypoints.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(keypoint_data)
        
        # Generate and send arm command
        if ser and ser.is_open:
            try:
                json_command = generate_arm_command(current_keypoints)
                ser.write((json_command + "\n").encode())
                logger.info(f"Sent command: {json_command}")
                last_keypoints = current_keypoints
                last_send_time = datetime.now()
            except Exception as e:
                logger.error(f"Failed to send arm command: {e}")
        else:
            logger.warning("Serial port not initialized or closed. Skipping command send.")
        
        return True
    return False

# ======================
# Display result
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)

def draw_keypoint(input_img, person, point):
    threshold = POSE_THRESHOLD
    if person.points[point].score > threshold:
        color = hsv_to_rgb(255*point/const.POSE_KEYPOINT_CNT, 255, 255)
        x = int(input_img.shape[1] * person.points[point].x)
        y = int(input_img.shape[0] * person.points[point].y)
        cv2.circle(input_img, (x, y), 5, color, -1)

def display_result(input_img, person):
    points_to_draw = [
        const.POSE_KEYPOINT_NOSE,
        const.POSE_KEYPOINT_SHOULDER_CENTER,
        const.POSE_KEYPOINT_SHOULDER_LEFT,
        const.POSE_KEYPOINT_SHOULDER_RIGHT,
        const.POSE_KEYPOINT_EYE_LEFT,
        const.POSE_KEYPOINT_EYE_RIGHT,
        const.POSE_KEYPOINT_EAR_LEFT,
        const.POSE_KEYPOINT_EAR_RIGHT,
        const.POSE_KEYPOINT_ELBOW_LEFT,
        const.POSE_KEYPOINT_ELBOW_RIGHT,
        const.POSE_KEYPOINT_WRIST_LEFT,
        const.POSE_KEYPOINT_WRIST_RIGHT,
        const.POSE_KEYPOINT_BODY_CENTER,
        const.POSE_KEYPOINT_HIP_LEFT,
        const.POSE_KEYPOINT_HIP_RIGHT,
        const.POSE_KEYPOINT_KNEE_LEFT,
        const.POSE_KEYPOINT_ANKLE_LEFT,
        const.POSE_KEYPOINT_KNEE_RIGHT,
        const.POSE_KEYPOINT_ANKLE_RIGHT
    ]
    
    for point in points_to_draw:
        draw_keypoint(input_img, person, point)

# ======================
# Pose Estimation
# ======================
def pose_estimation(boxes, interpreter_pose, img):
    dtype = np.int8
    if args.float:
        dtype = np.float32
    
    pose_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[0], img.shape[1]
    count = len(boxes)

    pose_detections = []
    for idx in range(count):
        top_left = (int(boxes[idx][0]), int(boxes[idx][1]))
        bottom_right = (int(boxes[idx][2]), int(boxes[idx][3]))
        px1, py1, px2, py2 = keep_aspect(
            top_left, bottom_right, pose_img
        )
        crop_img = pose_img[py1:py2, px1:px2, :]

        if crop_img.size == 0:
            logger.warning(f"Empty crop image for box {idx}. Skipping pose estimation.")
            pose_detections.append(None)
            continue

        offset_x = px1/img.shape[1]
        offset_y = py1/img.shape[0]
        scale_x = crop_img.shape[1]/img.shape[1]
        scale_y = crop_img.shape[0]/img.shape[0]
        detections = compute(
            interpreter_pose, crop_img, offset_x, offset_y, scale_x, scale_y, dtype
        )
        if detections is None:
            logger.warning(f"Pose estimation failed for box {idx}.")
            pose_detections.append(None)
        else:
            pose_detections.append(detections)
    return pose_detections

# ======================
# Main functions
# ======================
def recognize_from_image(interpreter_pose, interpreter_detect, ser):
    initialize_csv()
    
    detect_input_details = interpreter_detect.get_input_details()
    detect_output_details = interpreter_detect.get_output_details()
    detect_height = detect_input_details[0]['shape'][1]
    detect_width = detect_input_details[0]['shape'][2]

    for image_path in args.input:
        logger.info(image_path)
        
        img = load_image_img(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            continue
        orig_h, orig_w = img.shape[:2]
        resized_img = cv2.resize(img, (detect_width, detect_height))
        input_data = np.expand_dims(resized_img, axis=0).astype(np.uint8)

        logger.info('Start detection inference...')
        interpreter_detect.set_tensor(detect_input_details[0]['index'], input_data)
        interpreter_detect.invoke()

        boxes = interpreter_detect.get_tensor(detect_output_details[0]['index'])[0]
        labels = interpreter_detect.get_tensor(detect_output_details[1]['index'])[0]
        scores = interpreter_detect.get_tensor(detect_output_details[2]['index'])[0]
        num_detections = int(interpreter_detect.get_tensor(detect_output_details[3]['index'])[0])

        person_boxes = []
        for i in range(num_detections):
            if scores[i] > args.threshold and int(labels[i]) == 0:
                ymin, xmin, ymax, xmax = boxes[i]
                x0 = int(xmin * orig_w)
                y0 = int(ymin * orig_h)
                x1 = int(xmax * orig_w)
                y1 = int(ymax * orig_h)
                person_boxes.append([x0, y0, x1, y1])
        
        logger.info(f'Found {len(person_boxes)} persons')
        
        pose_detections = pose_estimation(person_boxes, interpreter_pose, img)
        
        for i, box in enumerate(person_boxes):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            
            if i < len(pose_detections) and pose_detections[i] is not None:
                display_result(img, pose_detections[i])
                write_keypoints_to_csv(pose_detections[i], frame_width=orig_w, frame_height=orig_h, ser=ser)
            else:
                logger.warning(f"No valid pose detection for person {i}.")
        
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img)
    
    logger.info('Script finished successfully.')

def recognize_from_video(interpreter_pose, interpreter_detect, ser):
    initialize_csv()
    
    detect_input_details = interpreter_detect.get_input_details()
    detect_output_details = interpreter_detect.get_output_details()
    detect_height = detect_input_details[0]['shape'][1]
    detect_width = detect_input_details[0]['shape'][2]

    if args.video.isdigit():
        cap_input = int(args.video)
    else:
        cap_input = args.video
    
    vid = cv2.VideoCapture(cap_input)
    if not vid.isOpened():
        logger.error(f"Could not open video source {args.video}")
        exit(1)
    
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    
    writer = None
    if args.savepath != SAVE_IMAGE_PATH:
        writer = cv2.VideoWriter(
            args.savepath,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
    
    frame_count = 0
    total_processing_time = 0
    last_record_time = datetime.min
    
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
        current_time = datetime.now()
        
        resized_frame = cv2.resize(frame, (detect_width, detect_height))
        input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)
        
        interpreter_detect.set_tensor(detect_input_details[0]['index'], input_data)
        interpreter_detect.invoke()
        
        boxes = interpreter_detect.get_tensor(detect_output_details[0]['index'])[0]
        labels = interpreter_detect.get_tensor(detect_output_details[1]['index'])[0]
        scores = interpreter_detect.get_tensor(detect_output_details[2]['index'])[0]
        num_detections = int(interpreter_detect.get_tensor(detect_output_details[3]['index'])[0])
        
        person_boxes = []
        for i in range(num_detections):
            if scores[i] > args.threshold and int(labels[i]) == 0:
                ymin, xmin, ymax, xmax = boxes[i]
                x0 = int(xmin * frame_width)
                y0 = int(ymin * frame_height)
                x1 = int(xmax * frame_width)
                y1 = int(ymax * frame_height)
                person_boxes.append([x0, y0, x1, y1])
        
        pose_detections = pose_estimation(person_boxes, interpreter_pose, frame)
        
        if should_record(current_time, last_record_time):
            for i, box in enumerate(person_boxes):
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                
                if i < len(pose_detections) and pose_detections[i] is not None:
                    display_result(frame, pose_detections[i])
                    timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
                    if write_keypoints_to_csv(pose_detections[i], timestamp, frame_width, frame_height, ser=ser):
                        last_record_time = current_time
                else:
                    logger.warning(f"No valid pose detection for person {i}.")
        else:
            for i, box in enumerate(person_boxes):
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                if i < len(pose_detections) and pose_detections[i] is not None:
                    display_result(frame, pose_detections[i])
                else:
                    logger.warning(f"No valid pose detection for person {i}.")
        
        processing_time = time.time() - start_time
        total_processing_time += processing_time
        current_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
        
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        next_record_in = max(0, args.record_interval - (current_time - last_record_time).total_seconds())
        cv2.putText(frame, f"Next record in: {next_record_in:.1f}s", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Pose Estimation", frame)
        
        if writer is not None:
            writer.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vid.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')

def main():
    global last_keypoints, last_send_time
    last_keypoints = None
    last_send_time = datetime.min

    # Initialize serial port
    ser = None
    try:
        ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
        time.sleep(2)
        print("Serial port initialized.")
    except Exception as e:
        print(f"Failed to initialize serial port: {e}")
        ser = None

    check_and_download_models(SSD_MODEL_PATH, SSD_REMOTE_PATH)
    check_and_download_models(POSE_MODEL_PATH, POSE_REMOTE_PATH)
    
    if args.delegate:
        interpreter_detect = tflite.Interpreter(
            model_path=SSD_MODEL_PATH,
            experimental_delegates=[tflite.load_delegate(args.delegate)]
        )
        interpreter_pose = tflite.Interpreter(
            model_path=POSE_MODEL_PATH,
            experimental_delegates=[tflite.load_delegate(args.delegate)]
        )
        print("Loaded on NPU")
    else:
        interpreter_detect = tflite.Interpreter(model_path=SSD_MODEL_PATH)
        interpreter_pose = tflite.Interpreter(model_path=POSE_MODEL_PATH)
    
    interpreter_detect.allocate_tensors()
    interpreter_pose.allocate_tensors()
    
    try:
        if args.video is not None:
            recognize_from_video(interpreter_pose, interpreter_detect, ser)
        else:
            recognize_from_image(interpreter_pose, interpreter_detect, ser)
    finally:
        if ser is not None and ser.is_open:
            ser.close()
            print("Serial port closed.")

if __name__ == '__main__':
    main()
