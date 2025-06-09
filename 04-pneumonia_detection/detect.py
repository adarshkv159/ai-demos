import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

# Path to the NPU delegate library
npu_delegate_path = "/usr/lib/libvx_delegate.so"

# Initialize the TFLite interpreter with the NPU delegate
interpreter = tflite.Interpreter(
    model_path="trained.tflite",
    experimental_delegates=[tflite.load_delegate(npu_delegate_path)]
)

interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# Function to preprocess the image
def preprocess_frame(frame, input_shape):
    frame = cv2.resize(frame, (input_shape[2], input_shape[1]))  # Resize to model input size
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    frame = frame.astype(input_details[0]['dtype'])  # Cast to required dtype
    return frame

# Function to dequantize the output
def dequantize_output(quantized_values, scale, zero_point):
    return [(val - zero_point) * scale for val in quantized_values]

# Function to interpret the prediction
def interpret_prediction(output_data, labels, scale, zero_point):
    dequantized_values = dequantize_output(output_data[0], scale, zero_point)
    max_index = np.argmax(dequantized_values)
    predicted_class = labels[max_index]
    confidence = dequantized_values[max_index]
    return predicted_class, confidence, dequantized_values

# Define the labels
labels = ["bacteria", "normal", "virus"]

# Quantization parameters
scale = output_details[0]['quantization'][0]
zero_point = output_details[0]['quantization'][1]

# Open default camera (e.g., /dev/video0)
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

print("Camera feed started. Press 'q' to quit.")

# Process the camera feed
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    input_data = preprocess_frame(frame, input_details[0]['shape'])

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class, confidence, dequantized_values = interpret_prediction(output_data, labels, scale, zero_point)

    result_text = f"Prediction: {predicted_class} ({confidence:.2f})"
    cv2.putText(frame, result_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

