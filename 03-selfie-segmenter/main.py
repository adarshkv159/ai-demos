import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def normalize_input(input_data, input_shape):
    """Fit the image size for model and change colorspace"""
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    resized_data = cv2.resize(input_data, input_shape)
    normalized_data = np.ascontiguousarray(resized_data / 255.0)
    normalized_data = normalized_data.astype("float32")
    normalized_data = normalized_data[None, ...]
    return normalized_data

# Load model with NPU delegate
model_path = "./selfie_segmenter_int8.tflite"
interpreter = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates=[tflite.load_delegate("/usr/lib/libvx_delegate.so")]
)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
input_shape = interpreter.get_input_details()[0]["shape"]
output_index = interpreter.get_output_details()[0]["index"]

# Open default camera (e.g., /dev/video0)
vid = cv2.VideoCapture(0)

if not vid.isOpened():
    print("Failed to open video capture.")
    exit()

while True:
    ret, frame = vid.read()
    if not ret:
        print("Failed to capture image.")
        break

    input_frame = normalize_input(frame, (input_shape[2], input_shape[1]))

    interpreter.set_tensor(input_index, input_frame)
    interpreter.invoke()
    mask = interpreter.get_tensor(output_index)[0]

    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), cv2.INTER_CUBIC)
    condition = np.stack((mask,) * 3, axis=-1) > 0.1

    foreground = np.full(shape=frame.shape, fill_value=255, dtype=np.uint8)
    background = np.full(shape=frame.shape, fill_value=0, dtype=np.uint8)

    segmentation = np.where(condition, foreground, background)

    cv2.imshow("Segmentation Output", segmentation)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
