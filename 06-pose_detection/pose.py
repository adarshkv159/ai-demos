import sys
import time
import os
import numpy as np
import cv2
import const
from logging import getLogger
import tflite_runtime.interpreter as tflite  # Changed from TensorFlow to TFLite

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

from utils import file_abs_path, get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image as load_image_img  # noqa: E402
from pose_resnet_util import compute, keep_aspect  # noqa: E402

# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'
POSE_THRESHOLD = 0.4
SSD_MODEL_NAME = 'ssd_mobilenet_v1_quant'
SSD_MODEL_PATH = file_abs_path(__file__, f'{SSD_MODEL_NAME}.tflite')
SSD_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/ssd_mobilenet_v1/'

# ======================
# Arguemnt Parser Config
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
args = update_parser(parser)

if args.float:
    POSE_MODEL_NAME = 'pose_resnet_50_256x192_float32'
else:
    POSE_MODEL_NAME = 'pose_resnet_50_256x192_int8'
POSE_MODEL_PATH = file_abs_path(__file__, f'{POSE_MODEL_NAME}.tflite')
POSE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/pose_resnet/'

# ======================
# COCO Keypoints Skeleton Definitions
# ======================
# Skeleton connections for COCO (18 keypoints) format
# (see: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]
FACE_KEYPOINTS = {0, 1, 2, 3, 4}  # Usually nose, eyes, ears in COCO

# ======================
# Display result (Draw Keypoints & Skeleton except face)
# ======================
def draw_keypoints_and_skeleton(input_img, person):
    threshold = POSE_THRESHOLD
    points = []
    # Draw keypoints
    for i, point in enumerate(person.points):
        if point.score > threshold:
            x = int(input_img.shape[1] * point.x)
            y = int(input_img.shape[0] * point.y)
            points.append((x, y))
            # Draw circle for keypoint
            cv2.circle(input_img, (x, y), 6, (0, 255, 0, 255), -1)
        else:
            points.append(None)
    # Draw skeleton (except face keypoints)
    for pair in COCO_SKELETON:
        kp1, kp2 = pair
        if kp1 in FACE_KEYPOINTS or kp2 in FACE_KEYPOINTS:
            continue  # skip face connections
        if points[kp1] is not None and points[kp2] is not None:
            cv2.line(input_img, points[kp1], points[kp2], (0, 0, 255, 255), 2)

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

        offset_x = px1/img.shape[1]
        offset_y = py1/img.shape[0]
        scale_x = crop_img.shape[1]/img.shape[1]
        scale_y = crop_img.shape[0]/img.shape[0]
        detections = compute(
            interpreter_pose, crop_img, offset_x, offset_y, scale_x, scale_y, dtype
        )
        pose_detections.append(detections)
    return pose_detections

# ======================
# Main functions
# ======================
def recognize_from_image(interpreter_pose, interpreter_detect):
    # Get model details
    detect_input_details = interpreter_detect.get_input_details()
    detect_output_details = interpreter_detect.get_output_details()
    detect_height = detect_input_details[0]['shape'][1]
    detect_width = detect_input_details[0]['shape'][2]

    # input image loop
    for image_path in args.input:
        logger.info(image_path)
        
        # Load and preprocess image
        img = load_image_img(image_path)
        orig_h, orig_w = img.shape[:2]
        resized_img = cv2.resize(img, (detect_width, detect_height))
        input_data = np.expand_dims(resized_img, axis=0).astype(np.uint8)

        # Run person detection
        logger.info('Start detection inference...')
        interpreter_detect.set_tensor(detect_input_details[0]['index'], input_data)
        interpreter_detect.invoke()

        # Get detection results
        boxes = interpreter_detect.get_tensor(detect_output_details[0]['index'])[0]
        labels = interpreter_detect.get_tensor(detect_output_details[1]['index'])[0]
        scores = interpreter_detect.get_tensor(detect_output_details[2]['index'])[0]
        num_detections = int(interpreter_detect.get_tensor(detect_output_details[3]['index'])[0])

        # Process detections
        person_boxes = []
        for i in range(num_detections):
            if scores[i] > args.threshold and int(labels[i]) == 0:  # Person class
                ymin, xmin, ymax, xmax = boxes[i]
                x0 = int(xmin * orig_w)
                y0 = int(ymin * orig_h)
                x1 = int(xmax * orig_w)
                y1 = int(ymax * orig_h)
                person_boxes.append([x0, y0, x1, y1])
        
        logger.info(f'Found {len(person_boxes)} persons')
        
        # Run pose estimation
        pose_detections = pose_estimation(person_boxes, interpreter_pose, img)
        
        # Draw results
        for i, box in enumerate(person_boxes):
            # Draw bounding box
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            
            # Draw keypoints and skeleton except face
            if pose_detections[i]:
                draw_keypoints_and_skeleton(img, pose_detections[i])
        
        # Save output
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, img)
    
    logger.info('Script finished successfully.')

def recognize_from_video(interpreter_pose, interpreter_detect):
    # Get model details
    detect_input_details = interpreter_detect.get_input_details()
    detect_output_details = interpreter_detect.get_output_details()
    detect_height = detect_input_details[0]['shape'][1]
    detect_width = detect_input_details[0]['shape'][2]

    # Initialize video capture
    if args.video.isdigit():
        cap_input = int(args.video)
    else:
        cap_input = args.video
    
    vid = cv2.VideoCapture(cap_input)
    if not vid.isOpened():
        logger.error(f"Could not open video source {args.input}")
        exit(1)
    
    # Get video properties
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if needed
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
    
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Preprocess frame
        resized_frame = cv2.resize(frame, (detect_width, detect_height))
        input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)
        
        # Run person detection
        interpreter_detect.set_tensor(detect_input_details[0]['index'], input_data)
        interpreter_detect.invoke()
        
        # Get detection results
        boxes = interpreter_detect.get_tensor(detect_output_details[0]['index'])[0]
        labels = interpreter_detect.get_tensor(detect_output_details[1]['index'])[0]
        scores = interpreter_detect.get_tensor(detect_output_details[2]['index'])[0]
        num_detections = int(interpreter_detect.get_tensor(detect_output_details[3]['index'])[0])
        
        # Process detections
        person_boxes = []
        for i in range(num_detections):
            if scores[i] > args.threshold and int(labels[i]) == 0:  # Person class
                ymin, xmin, ymax, xmax = boxes[i]
                x0 = int(xmin * frame_width)
                y0 = int(ymin * frame_height)
                x1 = int(xmax * frame_width)
                y1 = int(ymax * frame_height)
                person_boxes.append([x0, y0, x1, y1])
        
        # Run pose estimation
        pose_detections = pose_estimation(person_boxes, interpreter_pose, frame)
        
        # Draw results
        for i, box in enumerate(person_boxes):
            # Draw bounding box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            
            # Draw keypoints and skeleton except face
            if pose_detections[i]:
                draw_keypoints_and_skeleton(frame, pose_detections[i])
        
        # Calculate FPS
        processing_time = time.time() - start_time
        total_processing_time += processing_time
        frame_count += 1
        fps = frame_count / total_processing_time
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Pose Estimation", frame)
        
        # Save frame if needed
        if writer is not None:
            writer.write(frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    vid.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')

def main():
    # Download models if needed
    check_and_download_models(SSD_MODEL_PATH, SSD_REMOTE_PATH)
    check_and_download_models(POSE_MODEL_PATH, POSE_REMOTE_PATH)
    # Initialize SSD model (for object detection)
    if args.delegate:
        interpreter_detect = tflite.Interpreter(
            model_path=SSD_MODEL_PATH,
            experimental_delegates=[tflite.load_delegate(args.delegate)]
        )
        print("Loaded on NPU")
    else:
        interpreter_detect = tflite.Interpreter(model_path=SSD_MODEL_PATH)
    interpreter_detect.allocate_tensors()
    # Initialize Pose model
    if args.delegate:
        interpreter_pose = tflite.Interpreter(
            model_path=POSE_MODEL_PATH,
            experimental_delegates=[tflite.load_delegate(args.delegate)]
        )
        print("Loaded on NPU")
    else:
        interpreter_pose = tflite.Interpreter(model_path=POSE_MODEL_PATH)
    interpreter_pose.allocate_tensors()
    # Run the appropriate mode
    if args.video is not None:
        recognize_from_video(interpreter_pose, interpreter_detect)
    else:
        recognize_from_image(interpreter_pose, interpreter_detect)

if __name__ == '__main__':
    main()
