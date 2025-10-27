import cv2
import time
import numpy as np
import argparse
import pyttsx3
import os
import subprocess
import threading
import whisper
import tempfile
import uuid
import string

from face_detection import YoloFace
from face_recognition import Facenet
from face_database import FaceDatabase

# Configuration
WHISPER_MODEL = "tiny"
NAME_PROMPT = "Adarsh, Aarav, Nayan, Riya, Lakshmi, Arjun, Priya, Deepa, Neha, Rohan, Anjali, Vijay, Vinay, Vikram, Sanjay, Suraj"
COMMAND_PROMPT = "new, add, remove, delete, quit, cancel"
CONFIRM_PROMPT = "yes, no"
AUDIO_DEVICE = "default"
TTS_DEVICE = "default"
COMMAND_DURATION = 3
NAME_DURATION = 4
CONFIRM_DURATION = 3
TTS_DEBOUNCE = 3.0
COMMAND_DEBOUNCE = 2.0
NO_FACE_TIMEOUT = 10.0
FRAME_SKIP = 5
SIDE_PANEL_WIDTH = 350
SIDE_PANEL_HEIGHT = 600
TEXT_TIMEOUT = 5.0
RECORDING_TIMEOUT = 6.0
MIN_MESSAGE_DISPLAY = 1.0
FACE_CAPTURE_TIMEOUT = 3.0
NO_FACE_CHECK_INTERVAL = 1.0

# Window configuration for resizing
DEFAULT_WINDOW_WIDTH = 990  # 640 + 350
DEFAULT_WINDOW_HEIGHT = 600
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 400

class AppState:
    def __init__(self):
        self.last_spoken_name = None
        self.current_command = None
        self.running = True
        self.processing_command = False
        self.last_command_time = 0
        self.no_face_time = None
        self.frame_count = 0
        self.command_thread = None
        self.add_state = None
        self.add_embeddings = None
        self.add_name = None
        self.face_capture_start = None
        self.remove_names = None
        self.recording_message = None
        self.recording_message_time = 0
        self.is_recording = False
        self.waiting_for_number = False
        self.last_audio_check = 0
        # New state variables for face capture feedback
        self.last_face_check_time = 0
        self.face_capture_attempts = 0
        self.max_face_capture_attempts = 5
        self.face_detected_for_capture = False
        # New state variables for enhanced functionality
        self.whisper_processing = False
        self.whisper_start_time = 0
        self.recording_start_time = 0
        self.recording_duration = 0
        self.current_operation = "idle"  # idle, recording, processing, listening
        self.window_resized = False
        self.panel_scale = 1.0

    def reset_to_idle(self):
        """Reset application state to idle mode"""
        print("Resetting to idle state...")
        self.current_command = None
        self.processing_command = False
        self.add_state = None
        self.add_embeddings = None
        self.add_name = None
        self.face_capture_start = None
        self.remove_names = None
        self.waiting_for_number = False
        self.face_capture_attempts = 0
        self.face_detected_for_capture = False
        self.last_face_check_time = 0
        self.current_operation = "idle"
        self.recording_message = "Ready for commands"
        self.recording_message_time = time.time()
        # Stop any ongoing TTS
        if hasattr(tts, 'engine') and tts.engine:
            try:
                tts.engine.stop()
            except:
                pass

    def process_command(self):
        """Process the current voice command"""
        if self.command_thread and self.command_thread.is_alive():
            print("Command processing skipped (thread busy)")
            return

        def run_command():
            current_time = time.time()
            if self.processing_command or (current_time - self.last_command_time < COMMAND_DEBOUNCE):
                print("Command processing skipped (debounce or busy)")
                return

            self.processing_command = True
            self.last_command_time = current_time
            self.current_operation = "processing"

            try:
                print(f"Processing command: {self.current_command}")
                if self.current_command == "add":
                    self.add_state = "capture_face"
                    self.face_capture_start = None
                    self.remove_names = None
                    self.face_capture_attempts = 0
                    self.face_detected_for_capture = False
                    self.last_face_check_time = 0
                    self.current_operation = "adding"
                    tts.say("Please show your face to the camera")
                elif self.current_command == "remove":
                    self.remove_names = database.get_names()
                    self.waiting_for_number = True
                    self.current_operation = "removing"
                    handle_remove_command()
                elif self.current_command == "quit" or self.current_command == "cancel":
                    self.reset_to_idle()
                    tts.say("Operation cancelled, returning to idle mode")
                elif self.current_command == "exit":
                    self.running = False
            finally:
                if self.current_command not in ["quit", "cancel"]:
                    self.current_command = None
                self.processing_command = False

        self.command_thread = threading.Thread(target=run_command, daemon=True)
        self.command_thread.start()

    def process_add_state(self, frame):
        """Handle the face addition process with improved feedback"""
        current_time = time.time()
        
        if self.add_state == "capture_face":
            if self.face_capture_start is None:
                if tts.tts_thread and tts.tts_thread.is_alive():
                    return
                self.face_capture_start = current_time
                self.last_face_check_time = current_time

            # Check for face detection
            boxes = detector.detect(frame)
            face_detected = boxes is not None and len(boxes) > 0
            
            if face_detected:
                if not self.face_detected_for_capture:
                    self.face_detected_for_capture = True
                    print("Face detected for capture")
                
                # Proceed with face capture after a short delay to ensure stable detection
                if current_time - self.face_capture_start >= 2.0:
                    box = boxes[0]
                    box[[0, 2]] *= frame.shape[1]
                    box[[1, 3]] *= frame.shape[0]
                    x1, y1, x2, y2 = box.astype(np.int32)
                    x1, y1 = max(x1 - 10, 0), max(y1 - 10, 0)
                    x2, y2 = min(x2 + 10, frame.shape[1]), min(y2 + 10, frame.shape[0])
                    face = frame[y1:y2, x1:x2]
                    
                    # Get embeddings and proceed to name input
                    self.add_embeddings = recognizer.get_embeddings(face)
                    if self.add_embeddings is not None:
                        self.add_state = "say_name"
                        tts.say("Face captured. Please say the name")
                        return
                    else:
                        tts.say("Face capture failed. Please try again")
                        self.add_state = None
                        self.face_capture_start = None
                        self.current_operation = "idle"
                        return
            else:
                self.face_detected_for_capture = False
                # Provide feedback if no face is detected for too long
                if current_time - self.last_face_check_time >= NO_FACE_CHECK_INTERVAL:
                    self.last_face_check_time = current_time
                    self.face_capture_attempts += 1
                    
                    if self.face_capture_attempts <= self.max_face_capture_attempts:
                        # Don't interrupt ongoing TTS
                        if not (tts.tts_thread and tts.tts_thread.is_alive()):
                            tts.say("Please show your face to the camera")
                    else:
                        tts.say("Face capture timeout. Please try again")
                        self.add_state = None
                        self.face_capture_start = None
                        self.current_operation = "idle"
                        return

            # Timeout check
            if current_time - self.face_capture_start > 15.0:  # 15 second timeout
                tts.say("Face capture timeout. Please try again")
                self.add_state = None
                self.face_capture_start = None
                self.current_operation = "idle"
                return

        elif self.add_state == "say_name":
            if tts.tts_thread and tts.tts_thread.is_alive():
                return

            name = recognize_name()
            if not name:
                tts.say("Name not recognized")
                self.add_state = "say_name"
                tts.say("Please say the name")
                return

            self.add_state = "confirm_name"
            tts.say(f"Did you say {name}? Please say yes or no")
            self.add_name = name

        elif self.add_state == "confirm_name":
            if tts.tts_thread and tts.tts_thread.is_alive():
                return

            confirmation = recognize_confirmation()
            if confirmation == "yes":
                database.add_name(self.add_name, self.add_embeddings)
                tts.say(f"{self.add_name} added successfully")
                self.add_state = None
                self.add_embeddings = None
                self.add_name = None
                self.face_capture_start = None
                self.current_operation = "idle"
            elif confirmation == "no":
                self.add_state = "say_name"
                tts.say("Please say the name")
            else:
                tts.say("Please say yes or no")

def check_audio_device():
    """Verify audio device is available and working"""
    try:
        test_file = f"test_{uuid.uuid4()}.wav"
        result = subprocess.run(
            ["arecord", "-D", AUDIO_DEVICE, "-d", "1", test_file],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            timeout=2
        )

        if os.path.exists(test_file):
            os.remove(test_file)

        return True

        print(f"Audio check failed: {result.stderr.decode()}")
    except Exception as e:
        print(f"Audio device error: {e}")
        return False

# Initialize critical components
print("Initializing system...")
parser = argparse.ArgumentParser(description="Face recognition with voice command support")
parser.add_argument('-i', '--input', default='0', help='input device index (e.g., 0 for /dev/video0)')
parser.add_argument('-d', '--delegate', default='', help='delegate path')
args = parser.parse_args()

if not check_audio_device():
    print(f"ERROR: Cannot access audio device {AUDIO_DEVICE}. Try:")
    print("1. Run 'arecord -l' to list available devices")
    print("2. Check if audio device is available")
    exit(1)

print("Loading Whisper model...")
whisper_model = whisper.load_model(WHISPER_MODEL)
print("Whisper model loaded.")

vid = cv2.VideoCapture(int(args.input) if args.input.isdigit() else 0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = YoloFace("yoloface_int8.tflite", args.delegate)
recognizer = Facenet("facenet_512_int_quantized.tflite", args.delegate)
database = FaceDatabase()

tts_lock = threading.Lock()
app_state = AppState()

class TTSEngine:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 150)
        except Exception as e:
            print(f"TTS init error: {e}")
            self.engine = None

        self.tts_thread = None
        self.last_tts_time = 0
        self.current_text = ""
        self.text_start_time = 0
        self.is_speaking = False

    def say(self, text):
        current_time = time.time()
        
        # Handle quit/cancel commands immediately
        if text.lower() in ["operation cancelled, returning to idle mode", "quit", "cancel"]:
            if self.tts_thread and self.tts_thread.is_alive():
                try:
                    if self.engine:
                        self.engine.stop()
                except:
                    pass
        
        if self.tts_thread and self.tts_thread.is_alive():
            print(f"TTS skipped: {text} (busy)")
            return

        if current_time - self.last_tts_time < TTS_DEBOUNCE:
            print(f"TTS skipped: {text} (debounce)")
            return

        self.current_text = text
        self.text_start_time = current_time
        self.is_speaking = True

        def run_tts():
            with tts_lock:
                if not self.engine:
                    self.is_speaking = False
                    return

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    tmp_file = tmpfile.name

                try:
                    self.engine.save_to_file(text, tmp_file)
                    self.engine.runAndWait()
                    subprocess.run(f"aplay -D {TTS_DEVICE} {tmp_file}", shell=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"TTS error: {e}")
                finally:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                    self.is_speaking = False

        self.tts_thread = threading.Thread(target=run_tts, daemon=True)
        self.tts_thread.start()
        self.last_tts_time = current_time
        print(f"TTS: {text}")

tts = TTSEngine()

def record_audio(filename, duration=4):
    """Record audio with comprehensive error handling and status updates"""
    try:
        # Set recording state
        app_state.is_recording = True
        app_state.recording_start_time = time.time()
        app_state.recording_duration = duration
        app_state.current_operation = "recording"
        app_state.recording_message = f"Recording audio... ({duration}s)"
        app_state.recording_message_time = time.time()
        print(f"[record_audio] Recording started for {duration} seconds...")

        # Build arecord command
        cmd = [
            'arecord',
            '-D', AUDIO_DEVICE,
            '-f', 'S16_LE',
            '-r', '16000',
            '-c', '1',
            '-d', str(duration),
            filename
        ]

        with subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        ) as proc:
            try:
                return_code = proc.wait(timeout=duration + 2)
                if return_code != 0:
                    error_output = proc.stderr.read() if proc.stderr else "Unknown error"
                    raise RuntimeError(f"arecord failed (code {return_code}): {error_output}")

                if not os.path.exists(filename):
                    raise RuntimeError("No recording file created")

                if os.path.getsize(filename) < 1024:
                    raise RuntimeError("Recording file too small (silent recording?)")

                print("[record_audio] Recording completed successfully")
                app_state.recording_message = "Recording completed - Processing..."
                app_state.recording_message_time = time.time()
                return True

            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise RuntimeError("Recording timed out")

    except Exception as e:
        error_msg = str(e)
        print(f"[record_audio] Error: {error_msg}")
        app_state.recording_message = f"Recording failed: {error_msg}"
        app_state.recording_message_time = time.time()
        return False

    finally:
        app_state.is_recording = False
        # Clean up failed temp files
        if 'filename' in locals() and os.path.exists(filename) and not app_state.recording_message.startswith("Recording completed"):
            try:
                os.remove(filename)
            except:
                pass

def whisper_transcribe(audio_file, prompt_context=""):
    """Enhanced Whisper transcription with processing indicators"""
    try:
        app_state.whisper_processing = True
        app_state.whisper_start_time = time.time()
        app_state.current_operation = "processing"
        app_state.recording_message = "Processing speech with Whisper..."
        app_state.recording_message_time = time.time()
        
        print("Starting Whisper transcription...")
        audio_input = whisper.load_audio(audio_file)
        audio_input = whisper.pad_or_trim(audio_input)
        mel = whisper.log_mel_spectrogram(audio_input).to(whisper_model.device)
        options = whisper.DecodingOptions(language="en", fp16=False, prompt=prompt_context, temperature=0.5)
        result = whisper.decode(whisper_model, mel, options)
        text = result.text.strip().lower()
        
        processing_time = time.time() - app_state.whisper_start_time
        print(f"Whisper transcription completed in {processing_time:.2f}s: {text}")
        app_state.recording_message = f"Speech processed: '{text}'"
        app_state.recording_message_time = time.time()
        
        return text
    except Exception as e:
        print(f"Whisper error: {e}")
        app_state.recording_message = f"Speech processing failed: {str(e)}"
        app_state.recording_message_time = time.time()
        return ""
    finally:
        app_state.whisper_processing = False
        if os.path.exists(audio_file):
            os.remove(audio_file)

def recognize_command():
    tmp_file = f"cmd_{uuid.uuid4()}.wav"
    app_state.recording_message = "Listening for command..."
    app_state.recording_message_time = time.time()
    
    if not record_audio(tmp_file, COMMAND_DURATION):
        print("Failed to record command audio")
        return ""

    cmd_text = whisper_transcribe(tmp_file, f"Commands: {COMMAND_PROMPT}")
    if "new" in cmd_text or "add" in cmd_text:
        return "add"
    elif "remove" in cmd_text or "delete" in cmd_text:
        return "remove"
    elif "quit" in cmd_text or "cancel" in cmd_text:
        return "quit"
    elif "exit" in cmd_text:
        return "exit"

    print(f"Command not recognized: {cmd_text}")
    app_state.recording_message = f"Command not recognized: {cmd_text}"
    app_state.recording_message_time = time.time()
    return ""

def recognize_name():
    tmp_file = f"name_{uuid.uuid4()}.wav"
    app_state.recording_message = "Listening for name..."
    app_state.recording_message_time = time.time()
    
    if not record_audio(tmp_file, NAME_DURATION):
        print("Failed to record name audio")
        return None

    name_text = whisper_transcribe(tmp_file, f"Example names: {NAME_PROMPT}")
    if not name_text:
        print("No name transcribed")
        return None

    # Improved name cleaning and validation
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = name_text.translate(translator).strip()

    # Split into words and clean each one
    words = [word.strip().capitalize() for word in cleaned_text.split() if word.strip()]

    # More flexible validation (1–3 parts)
    if not words or len(words) > 3:
        print(f"Invalid name format: {words}")
        return None

    invalid_substrings = ['example', 'name', 'names', 'call', 'my', 'quit', 'cancel']
    if any(sub in name_text.lower() for sub in invalid_substrings):
        print(f"Contains invalid substring: {name_text}")
        return None

    allowed_suffixes = ['jr', 'sr', 'ii', 'iii', 'iv']
    filtered_words = []
    for word in words:
        if word.lower() in allowed_suffixes and word == words[-1]:
            continue
        filtered_words.append(word)

    if not filtered_words:
        print("No valid name parts after filtering")
        return None

    name = ' '.join(filtered_words)
    if len(name) < 2 or len(name) > 30:
        print(f"Name length invalid: {name}")
        return None

    print(f"Recognized name: {name}")
    return name

def recognize_confirmation():
    tmp_file = f"confirm_{uuid.uuid4()}.wav"
    app_state.recording_message = "Listening for confirmation..."
    app_state.recording_message_time = time.time()
    
    if not record_audio(tmp_file, CONFIRM_DURATION):
        print("Failed to record confirmation audio")
        return None

    confirm_text = whisper_transcribe(tmp_file, f"Responses: {CONFIRM_PROMPT}")
    if "yes" in confirm_text:
        return "yes"
    elif "no" in confirm_text:
        return "no"
    elif "quit" in confirm_text or "cancel" in confirm_text:
        app_state.reset_to_idle()
        tts.say("Operation cancelled")
        return None

    print(f"Confirmation not recognized: {confirm_text}")
    return None

def recognize_number():
    tmp_file = f"number_{uuid.uuid4()}.wav"
    app_state.recording_message = "Listening for number..."
    app_state.recording_message_time = time.time()
    
    if not record_audio(tmp_file, CONFIRM_DURATION):
        print("Failed to record number audio")
        return None

    number_text = whisper_transcribe(tmp_file, "Numbers: one, two, three, four, five, six, seven, eight, nine, ten, quit, cancel")
    
    if "quit" in number_text or "cancel" in number_text:
        app_state.reset_to_idle()
        tts.say("Operation cancelled")
        return None
    
    number_map = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }

    number = number_map.get(number_text.lower(), None)
    print(f"Recognized number: {number}")
    return number

def process_faces(frame):
    app_state.frame_count += 1
    if app_state.frame_count % FRAME_SKIP != 0:
        return None

    boxes = detector.detect(frame)
    current_time = time.time()

    if boxes is None or len(boxes) == 0:
        if app_state.last_spoken_name is not None:
            app_state.last_spoken_name = None
            app_state.no_face_time = current_time
            print("No face detected: Cleared last spoken name")
        elif app_state.no_face_time and (current_time - app_state.no_face_time) > NO_FACE_TIMEOUT:
            app_state.no_face_time = None
            print("No face timeout: Ready for new greeting")
        return None

    app_state.no_face_time = None
    box = boxes[0]
    box[[0, 2]] *= frame.shape[1]
    box[[1, 3]] *= frame.shape[0]
    x1, y1, x2, y2 = box.astype(np.int32)
    x1, y1 = max(x1 - 10, 0), max(y1 - 10, 0)
    x2, y2 = min(x2 + 10, frame.shape[1]), min(y2 + 10, frame.shape[0])

    face = frame[y1:y2, x1:x2]
    embeddings = recognizer.get_embeddings(face)
    name, confidence = database.find_name(embeddings)

    label = f"{name} ({int(confidence*100)}%)" if name else "Unknown"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if name and name.lower() != "unknown" and name != app_state.last_spoken_name:
        print(f"TTS say hello to {name}")
        tts.say(f"Hello {name}")
        app_state.last_spoken_name = name
        print(f"Set last spoken name to {name}")

    return embeddings

def handle_remove_command():
    names = database.get_names()
    if not names:
        tts.say("No names stored")
        app_state.remove_names = None
        app_state.waiting_for_number = False
        app_state.current_operation = "idle"
        return

    app_state.remove_names = names
    print("Known names:")
    for i, name in enumerate(names, 1):
        print(f"{i}. {name}")

    tts.say("Please say the number of the name to remove, or say quit to cancel")

def render_side_panel():
    """Create the side panel image with enhanced recording status display"""
    # Calculate panel size based on window scaling
    panel_height = int(SIDE_PANEL_HEIGHT * app_state.panel_scale)
    panel_width = int(SIDE_PANEL_WIDTH * app_state.panel_scale)
    
    side_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    y_offset = int(20 * app_state.panel_scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * app_state.panel_scale
    color = (255, 255, 255)
    thickness = max(1, int(app_state.panel_scale))

    # System status header with more detailed status
    current_time = time.time()
    
    # Enhanced status display
    if app_state.is_recording:
        elapsed = current_time - app_state.recording_start_time
        remaining = max(0, app_state.recording_duration - elapsed)
        status = f"Recording... ({remaining:.1f}s)"
        status_color = (0, 255, 255)  # Yellow for recording
    elif app_state.whisper_processing:
        elapsed = current_time - app_state.whisper_start_time
        status = f"AI Processing... ({elapsed:.1f}s)"
        status_color = (255, 0, 255)  # Magenta for AI processing
    elif tts.is_speaking:
        status = "Speaking..."
        status_color = (0, 255, 0)  # Green for speaking
    elif app_state.add_state == "capture_face":
        if app_state.face_detected_for_capture:
            status = "Face Detected"
            status_color = (0, 255, 0)  # Green
        else:
            status = "Looking for Face"
            status_color = (0, 165, 255)  # Orange
    elif app_state.processing_command:
        status = "Processing Command..."
        status_color = (255, 255, 0)  # Cyan
    elif app_state.current_operation == "adding":
        status = "Adding Person"
        status_color = (255, 165, 0)  # Orange
    elif app_state.current_operation == "removing":
        status = "Removing Person"
        status_color = (255, 165, 0)  # Orange
    else:
        status = "Ready"
        status_color = (255, 255, 255)  # White
    
    cv2.putText(side_panel, f"Status: {status}", (int(10*app_state.panel_scale), y_offset), 
                font, font_scale, status_color, thickness)
    y_offset += int(30*app_state.panel_scale)

    # Operation indicator
    if app_state.current_operation != "idle":
        op_color = (200, 200, 255)
        cv2.putText(side_panel, f"Operation: {app_state.current_operation.title()}", 
                   (int(10*app_state.panel_scale), y_offset), font, font_scale*0.9, op_color, thickness)
        y_offset += int(25*app_state.panel_scale)

    # Recording progress bar
    if app_state.is_recording and app_state.recording_duration > 0:
        elapsed = current_time - app_state.recording_start_time
        progress = min(1.0, elapsed / app_state.recording_duration)
        bar_width = int(250 * app_state.panel_scale)
        bar_height = int(10 * app_state.panel_scale)
        bar_x = int(20 * app_state.panel_scale)
        bar_y = y_offset
        
        # Draw progress bar background
        cv2.rectangle(side_panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Draw progress
        progress_width = int(bar_width * progress)
        cv2.rectangle(side_panel, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 255), -1)
        y_offset += bar_height + int(15*app_state.panel_scale)

    # Recording messages with enhanced display
    if app_state.recording_message:
        message_time = current_time - app_state.recording_message_time
        if message_time < RECORDING_TIMEOUT:
            cv2.putText(side_panel, "Audio Status:", (int(10*app_state.panel_scale), y_offset), 
                       font, font_scale, (200, 200, 255), thickness)
            y_offset += int(25*app_state.panel_scale)
            
            # Split long messages into multiple lines
            lines = []
            max_chars = int(30 / app_state.panel_scale)
            words = app_state.recording_message.split()
            current_line = ""
            
            for word in words:
                if len(current_line + " " + word) <= max_chars:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            for line in lines:
                # Color code based on message type
                if "started" in line.lower() or "recording" in line.lower():
                    msg_color = (0, 255, 255)  # Yellow for started/recording
                elif "completed" in line.lower() or "success" in line.lower():
                    msg_color = (0, 255, 0)    # Green for success
                elif "failed" in line.lower() or "error" in line.lower():
                    msg_color = (0, 0, 255)    # Red for errors
                elif "processing" in line.lower():
                    msg_color = (255, 0, 255)  # Magenta for processing
                else:
                    msg_color = color
                
                cv2.putText(side_panel, line, (int(20*app_state.panel_scale), y_offset), 
                           font, font_scale*0.9, msg_color, thickness)
                y_offset += int(25*app_state.panel_scale)

    # Face capture status during add operation
    if app_state.add_state == "capture_face":
        y_offset += int(15*app_state.panel_scale)
        cv2.putText(side_panel, "Face Capture:", (int(10*app_state.panel_scale), y_offset), 
                   font, font_scale, (0, 255, 255), thickness)
        y_offset += int(25*app_state.panel_scale)
        
        if app_state.face_detected_for_capture:
            cv2.putText(side_panel, "Face found - capturing...", (int(20*app_state.panel_scale), y_offset), 
                       font, font_scale*0.9, (0, 255, 0), thickness)
        else:
            cv2.putText(side_panel, "Show face to camera", (int(20*app_state.panel_scale), y_offset), 
                       font, font_scale*0.9, (0, 165, 255), thickness)
        y_offset += int(25*app_state.panel_scale)
        
        # Show attempt counter
        if app_state.face_capture_attempts > 0:
            cv2.putText(side_panel, f"Attempts: {app_state.face_capture_attempts}/{app_state.max_face_capture_attempts}", 
                       (int(20*app_state.panel_scale), y_offset), font, font_scale*0.8, (255, 255, 0), thickness)
            y_offset += int(25*app_state.panel_scale)

    # TTS responses
    if tts.current_text and (current_time - tts.text_start_time < TEXT_TIMEOUT):
        y_offset += int(15*app_state.panel_scale)
        cv2.putText(side_panel, "System Response:", (int(10*app_state.panel_scale), y_offset), 
                   font, font_scale, (0, 255, 0), thickness)
        y_offset += int(25*app_state.panel_scale)
        
        # Split TTS text into lines
        max_chars = int(30 / app_state.panel_scale)
        words = tts.current_text.split()
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            if len(test_line) <= max_chars:
                line = test_line
            else:
                cv2.putText(side_panel, line, (int(20*app_state.panel_scale), y_offset), 
                           font, font_scale*0.9, color, thickness)
                y_offset += int(25*app_state.panel_scale)
                line = word
        if line:
            cv2.putText(side_panel, line, (int(20*app_state.panel_scale), y_offset), 
                       font, font_scale*0.9, color, thickness)
            y_offset += int(25*app_state.panel_scale)

    # Names list for removal
    if app_state.remove_names:
        y_offset += int(15*app_state.panel_scale)
        cv2.putText(side_panel, "Select name to remove:", (int(10*app_state.panel_scale), y_offset), 
                   font, font_scale, (0, 255, 255), thickness)
        y_offset += int(25*app_state.panel_scale)
        for i, name in enumerate(app_state.remove_names, 1):
            cv2.putText(side_panel, f"{i}. {name}", (int(30*app_state.panel_scale), y_offset), 
                       font, font_scale*0.9, color, thickness)
            y_offset += int(22*app_state.panel_scale)

    # Commands help
    y_offset += int(20*app_state.panel_scale)
    cv2.putText(side_panel, "Voice Commands:", (int(10*app_state.panel_scale), y_offset), 
               font, font_scale*0.8, (150, 150, 150), thickness)
    y_offset += int(20*app_state.panel_scale)
    
    commands = ["'v' + 'add' - Add person", "'v' + 'remove' - Remove", "'v' + 'quit' - Cancel", "'q' - Exit app"]
    for cmd in commands:
        cv2.putText(side_panel, cmd, (int(15*app_state.panel_scale), y_offset), 
                   font, font_scale*0.7, (100, 100, 100), thickness)
        y_offset += int(18*app_state.panel_scale)

    return side_panel

def compose_frame_with_panel(frame):
    """Compose a single canvas that contains the webcam frame and the side panel with resizing support."""
    h, w = frame.shape[:2]
    panel = render_side_panel()

    # Calculate scaling for better fit if window was resized
    panel_h, panel_w = panel.shape[:2]
    canvas_h = max(h, panel_h)
    canvas_w = w + panel_w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place the frame on the left
    canvas[0:h, 0:w] = frame

    # Place the panel on the right, top-aligned
    canvas[0:panel_h, w:w+panel_w] = panel

    return canvas

def on_window_resize(event, x, y, flags, param):
    """Handle window resize events"""
    if event == cv2.EVENT_MOUSEWHEEL:
        # Simple scaling with mouse wheel (optional feature)
        if flags > 0:  # Scroll up
            app_state.panel_scale = min(2.0, app_state.panel_scale + 0.1)
        else:  # Scroll down
            app_state.panel_scale = max(0.5, app_state.panel_scale - 0.1)
        print(f"Panel scale: {app_state.panel_scale:.1f}")

def main():
    last_key_time = 0
    
    # Create resizable window
    cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Recognition', DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    cv2.setMouseCallback('Face Recognition', on_window_resize)

    # Initial system message
    app_state.recording_message = "System initialized - Ready for voice commands"
    app_state.recording_message_time = time.time()

    print("Face Recognition System Started")
    print("Controls:")
    print("  'v' + voice command - Voice control")
    print("  'q' - Quit application")
    print("  Mouse wheel - Scale side panel")
    print("Voice Commands: 'add', 'remove', 'quit/cancel'")

    while app_state.running:
        ret, frame = vid.read()
        if not ret:
            print("Camera read error")
            break

        current_time = time.time()

        # Process faces and commands
        process_faces(frame)
        if app_state.add_state:
            app_state.process_add_state(frame)

        # Display recording indicator over the frame
        if app_state.is_recording:
            elapsed = current_time - app_state.recording_start_time
            remaining = max(0, app_state.recording_duration - elapsed)
            cv2.putText(frame, f"Recording... {remaining:.1f}s", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display Whisper processing indicator
        if app_state.whisper_processing:
            elapsed = current_time - app_state.whisper_start_time
            cv2.putText(frame, f"AI Processing... {elapsed:.1f}s", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Display face capture status during add operation
        if app_state.add_state == "capture_face":
            if app_state.face_detected_for_capture:
                cv2.putText(frame, "Face detected - Hold still", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Show face to camera", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # UI controls
        cv2.putText(frame, "Press 'v' to speak command, 'q' to quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Compose single window content
        composed = compose_frame_with_panel(frame)

        # Show single resizable window
        cv2.imshow('Face Recognition', composed)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('v') and not app_state.processing_command and (current_time - last_key_time > COMMAND_DEBOUNCE):
            if app_state.is_recording or app_state.whisper_processing:
                print("Cannot start new command - already recording/processing")
                continue
                
            last_key_time = current_time
            print("Key 'v' pressed: starting command recognition")
            app_state.current_command = recognize_command()
            if app_state.current_command:
                app_state.process_command()

        elif app_state.waiting_for_number and ord('0') <= key <= ord('9'):
            number = int(chr(key))
            names = app_state.remove_names
            if names and 1 <= number <= len(names):
                name = names[number - 1]
                if database.del_name(name):
                    tts.say(f"Removed {name}")
                else:
                    tts.say(f"{name} not found")
                app_state.remove_names = None
                app_state.waiting_for_number = False
                app_state.current_operation = "idle"
            else:
                tts.say("Invalid number")
                app_state.remove_names = None
                app_state.waiting_for_number = False
                app_state.current_operation = "idle"

        elif key == ord('c'):  # Cancel current operation
            app_state.reset_to_idle()
            tts.say("Operation cancelled")

        elif key == ord('q'):
            print("Key 'q' pressed: exiting")
            app_state.running = False

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

