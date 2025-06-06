import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


model_path = "posenet_mobilenet_v1_075_353_481_quant.tflite"
NPU_DELEGATE_PATH = "/usr/lib/libvx_delegate.so"

try:
    delegate = tflite.load_delegate(NPU_DELEGATE_PATH)
    interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
    print("Running on NPU")
except Exception as e:
    print(f"Failed to load NPU delegate: {e}")
    print("Falling back to CPU execution.")
    interpreter = tflite.Interpreter(model_path=model_path)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (255, 0, 255), (0, 2): (0, 255, 255),
    (1, 3): (255, 0, 255), (2, 4): (0, 255, 255),
    (0, 5): (255, 0, 255), (0, 6): (0, 255, 255),
    (5, 7): (255, 0, 255), (7, 9): (255, 0, 255),
    (6, 8): (0, 255, 255), (8, 10): (0, 255, 255),
    (5, 6): (255, 255, 0), (5, 11): (255, 0, 255),
    (6, 12): (0, 255, 255), (11, 12): (255, 255, 0),
    (11, 13): (255, 0, 255), (13, 15): (255, 0, 255),
    (12, 14): (0, 255, 255), (14, 16): (0, 255, 255)
}

def dequantize(tensor, scale, zero_point):
    return scale * (tensor.astype(np.float32) - zero_point)

def decode_pose(heatmaps, offsets, output_stride=16):
    heatmaps = heatmaps.squeeze()
    offsets = offsets.squeeze()
    num_keypoints = heatmaps.shape[-1]

    keypoints = np.zeros((num_keypoints, 2))
    scores = np.zeros(num_keypoints)

    for i in range(num_keypoints):
        hmap = heatmaps[:, :, i]
        y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
        
        # Normalize score between 0 and 1
        raw_score = hmap[y, x]
        max_score = np.max(hmap)
        scores[i] = raw_score / max_score if max_score > 0 else 0.0

        offset_y = offsets[y, x, i]
        offset_x = offsets[y, x, i + num_keypoints]

        keypoints[i, 0] = x * output_stride + offset_x
        keypoints[i, 1] = y * output_stride + offset_y

    return keypoints, scores


def select_main_person(keypoints, scores, min_score=0.4):
    # Step 1: Filter keypoints by confidence
    valid_mask = scores > min_score
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 3:
        return np.zeros_like(scores)

    valid_kps = keypoints[valid_indices]

    # Step 2: Normalize coordinates (improves DBSCAN clustering)
    scaler = StandardScaler()
    scaled_kps = scaler.fit_transform(valid_kps)

    # Step 3: Apply DBSCAN clustering
    # eps = 0.8 is a good starting point for standardized data
    clustering = DBSCAN(eps=1, min_samples=3).fit(scaled_kps)
    labels = clustering.labels_

    # Step 4: Aggregate scores per cluster
    cluster_scores = {}
    for i, label in enumerate(labels):
        if label == -1:  # Ignore noise
            continue
        cluster_scores[label] = cluster_scores.get(label, 0) + scores[valid_indices[i]]

    if not cluster_scores:
        return np.zeros_like(scores)

    # Step 5: Find cluster with the highest total score
    best_cluster = max(cluster_scores.items(), key=lambda x: x[1])[0]

    # Step 6: Create filtered scores (only for best cluster)
    filtered_scores = np.zeros_like(scores)
    for idx, label in zip(valid_indices, labels):
        if label == best_cluster:
            filtered_scores[idx] = scores[idx]

    return filtered_scores


def draw_prediction_on_image(image, keypoints, scores, threshold=0.3):
    for i, (x, y) in enumerate(keypoints):
        if scores[i] > threshold:
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    for (i1, i2), color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if scores[i1] > threshold and scores[i2] > threshold:
            pt1 = tuple(keypoints[i1].astype(int))
            pt2 = tuple(keypoints[i2].astype(int))
            cv2.line(image, pt1, pt2, color, 2)

    return image

def preprocess_image(image, input_size):
    image = cv2.resize(image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image[np.newaxis, :].astype(np.uint8)

def main():
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]
    input_size = (input_width, input_height)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_image = preprocess_image(frame, input_size)
        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()

        heatmaps_raw = interpreter.get_tensor(output_details[0]['index'])
        heatmaps = dequantize(heatmaps_raw,
                             output_details[0]['quantization'][0],
                             output_details[0]['quantization'][1])

        offsets_raw = interpreter.get_tensor(output_details[1]['index'])
        offsets = dequantize(offsets_raw,
                             output_details[1]['quantization'][0],
                             output_details[1]['quantization'][1])

        keypoints, scores = decode_pose(heatmaps, offsets)
        
        # Select only the main person
        filtered_scores = select_main_person(keypoints, scores)
        
        # Rescale to original frame
        scale_x = frame.shape[1] / input_width
        scale_y = frame.shape[0] / input_height
        keypoints *= [scale_x, scale_y]

        output_image = draw_prediction_on_image(frame.copy(), keypoints, filtered_scores)
        cv2.imshow("PoseNet (Single Person)", output_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
