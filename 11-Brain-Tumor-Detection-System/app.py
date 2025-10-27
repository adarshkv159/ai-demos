import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import argparse
import sys
import os
from datetime import datetime

# Use tflite_runtime instead of tensorflow
try:
    import tflite_runtime.interpreter as tflite
    print("✅ Using tflite_runtime")
except ImportError:
    try:
        import tensorflow.lite as tflite
        print("⚠️  tflite_runtime not found, falling back to tensorflow.lite")
    except ImportError:
        print("❌ Neither tflite_runtime nor tensorflow.lite found!")
        sys.exit(1)

class BrainTumorDetectionApp:
    def __init__(self, root, delegate_path=None):
        self.root = root
        self.root.title("Brain Tumor Detection System - Quantized Model (TFLite Runtime)")
        self.root.geometry("1100x700")
        self.root.configure(bg='#f0f0f0')
        
        self.delegate_path = delegate_path
        
        # Load the quantized TFLite model with delegate support
        try:
            self.model_path = 'brain_tumor_fully_quantized_int8.tflite'
            self.load_model_with_delegate()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Quantization parameters from model info
            self.input_scale = 0.003921568859368563  # 1/255
            self.input_zero_point = 0
            self.output_scale = 0.00390625
            self.output_zero_point = -128
            
            print("✅ Quantized TFLite model loaded successfully!")
            print(f"📐 Input: {self.input_details[0]['shape']}, dtype: {self.input_details[0]['dtype']}")
            print(f"📊 Output: {self.output_details[0]['shape']}, dtype: {self.output_details[0]['dtype']}")
            print(f"🔢 Input scale: {self.input_scale}, zero_point: {self.input_zero_point}")
            print(f"🔢 Output scale: {self.output_scale}, zero_point: {self.output_zero_point}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load quantized model: {str(e)}")
            print(f"Model loading error: {e}")
            return
        
        # Class labels - Same order as your training
        self.class_labels = ['meningioma', 'glioma', 'notumor', 'pituitary']
        self.image_size = 224
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        self.prediction_result = None
        
        self.setup_ui()
    
    def load_model_with_delegate(self):
        """Load TFLite model with NPU delegate support and CPU fallback"""
        if self.delegate_path and os.path.exists(self.delegate_path):
            try:
                print(f"🚀 Attempting to load NPU delegate: {self.delegate_path}")
                ext_delegate = [tflite.load_delegate(self.delegate_path)]
                self.interpreter = tflite.Interpreter(
                    model_path=self.model_path, 
                    experimental_delegates=ext_delegate
                )
                self.interpreter.allocate_tensors()
                self.acceleration_mode = "NPU (Hardware Accelerated)"
                print("✅ NPU delegate loaded successfully!")
                return
            except Exception as e:
                print(f"⚠️  NPU delegate failed: {e}")
                print("🔄 Falling back to CPU...")
        
        # Fallback to CPU-only execution
        try:
            print("🔄 Loading model with CPU execution...")
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.acceleration_mode = "CPU Only"
            print("✅ CPU fallback successful!")
        except Exception as e:
            print(f"❌ CPU fallback failed: {e}")
            raise e
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(self.root, text="Brain Tumor Detection System (TFLite Runtime)", 
                              font=('Arial', 18, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=10)
        
        # Main frame (holds three columns)
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Left frame for controls
        control_frame = tk.Frame(main_frame, bg='#f0f0f0', width=220)
        control_frame.pack(side='left', fill='y', padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Center frame for image
        center_frame = tk.Frame(main_frame, bg='#f0f0f0')
        center_frame.pack(side='left', expand=True, fill='both')
        
        # Right frame for results (moved from bottom to right)
        results_container = tk.Frame(main_frame, bg='#f0f0f0', width=360)
        results_container.pack(side='right', fill='y', padx=(10, 0))
        results_container.pack_propagate(False)
        
        # Upload button
        self.upload_btn = tk.Button(control_frame, text="📁 Upload Image", 
                                   command=self.upload_image,
                                   font=('Arial', 12, 'bold'),
                                   bg='#3498db', fg='white',
                                   width=15, height=2,
                                   relief='raised', bd=3)
        self.upload_btn.pack(pady=10)
        
        # Predict button
        self.predict_btn = tk.Button(control_frame, text="🔍 Detect Tumor", 
                                    command=self.predict_tumor,
                                    font=('Arial', 12, 'bold'),
                                    bg='#e74c3c', fg='white',
                                    width=15, height=2,
                                    relief='raised', bd=3,
                                    state='disabled')
        self.predict_btn.pack(pady=10)
        
        # Clear button
        self.clear_btn = tk.Button(control_frame, text="🗑️ Clear", 
                                  command=self.clear_all,
                                  font=('Arial', 12, 'bold'),
                                  bg='#95a5a6', fg='white',
                                  width=15, height=2,
                                  relief='raised', bd=3)
        self.clear_btn.pack(pady=10)
        
        # Save result button
        self.save_btn = tk.Button(control_frame, text="💾 Save Result", 
                                 command=self.save_result,
                                 font=('Arial', 12, 'bold'),
                                 bg='#27ae60', fg='white',
                                 width=15, height=2,
                                 relief='raised', bd=3,
                                 state='disabled')
        self.save_btn.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(pady=20, fill='x')
        
        # Model info frame
        info_frame = tk.LabelFrame(control_frame, text="Model Info", 
                                  font=('Arial', 10, 'bold'), bg='#f0f0f0')
        info_frame.pack(pady=20, fill='x')
        
        acceleration_info = getattr(self, 'acceleration_mode', 'Loading...')
        model_info = f"• INT8 Quantized\n• TFLite Runtime\n• {acceleration_info}\n• Size: ~75% smaller"
        tk.Label(info_frame, text=model_info, font=('Arial', 9), 
                bg='#f0f0f0', justify='left').pack(pady=5)
        
        # Delegate info frame (if delegate is used)
        if self.delegate_path:
            delegate_frame = tk.LabelFrame(control_frame, text="NPU Delegate", 
                                          font=('Arial', 10, 'bold'), bg='#f0f0f0')
            delegate_frame.pack(pady=10, fill='x')
            delegate_name = os.path.basename(self.delegate_path) if self.delegate_path else "None"
            delegate_info = f"• Delegate: {delegate_name}\n• Hardware Acceleration"
            tk.Label(delegate_frame, text=delegate_info, font=('Arial', 9), 
                    bg='#f0f0f0', justify='left').pack(pady=5)
        
        # Center: Image display frame
        self.image_frame = tk.LabelFrame(center_frame, text="Input Image", 
                                       font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.image_frame.pack(fill='both', expand=True)
        
        # Image label
        self.image_label = tk.Label(self.image_frame, text="No image selected\n\nClick 'Upload Image' to begin", 
                                   font=('Arial', 14), bg='white', fg='#7f8c8d',
                                   width=40, height=15, relief='sunken', bd=2)
        self.image_label.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Right: Results frame moved to the right side
        self.results_frame = tk.LabelFrame(results_container, text="Detection Results", 
                                         font=('Arial', 12, 'bold'), bg='#f0f0f0')
        self.results_frame.pack(fill='both', expand=True)
        
        # Results display
        self.results_text = tk.Text(self.results_frame, height=8, font=('Arial', 11),
                                   bg='white', fg='#2c3e50', relief='sunken', bd=2,
                                   wrap='word', state='disabled')
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        status_message = f"Ready - Model loaded with {getattr(self, 'acceleration_mode', 'CPU')}"
        self.status_var.set(status_message)
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                            relief='sunken', anchor='w', font=('Arial', 10),
                            bg='#ecf0f1', fg='#2c3e50')
        status_bar.pack(side='bottom', fill='x')
    
    def upload_image(self):
        """Handle image upload"""
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('BMP files', '*.bmp'),
            ('TIFF files', '*.tiff *.tif'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Brain MRI Image",
            filetypes=file_types
        )
        
        if file_path:
            try:
                self.load_and_display_image(file_path)
                self.current_image_path = file_path
                self.predict_btn.config(state='normal')
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                
                # Clear previous results
                self.clear_results()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_var.set("Error loading image")
    
    def load_and_display_image(self, file_path):
        """Load and display image in the GUI"""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Could not load image. Please check the file format.")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_image = image_rgb
        
        display_image = self.resize_image_for_display(image_rgb, max_size=(600, 500))
        
        pil_image = Image.fromarray(display_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep a reference
    
    def resize_image_for_display(self, image, max_size=(600, 500)):
        """Resize image for display while maintaining aspect ratio"""
        height, width = image.shape[:2]
        max_width, max_height = max_size
        scale = min(max_width/width, max_height/height)
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized
        return image
    
    def preprocess_image_for_quantized_model(self, image):
        """Preprocess image for quantized model inference"""
        resized = cv2.resize(image, (self.image_size, self.image_size))
        normalized = resized.astype(np.float32) / 255.0
        quantized = np.round(normalized / self.input_scale + self.input_zero_point)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        quantized = np.expand_dims(quantized, axis=0)
        return quantized
    
    def dequantize_output(self, quantized_output):
        """Convert quantized INT8 output back to float32 probabilities"""
        float_output = self.output_scale * (quantized_output.astype(np.float32) - self.output_zero_point)
        return float_output
    
    def format_class_name(self, class_name):
        """Format class name for display"""
        if class_name.lower() == 'notumor':
            return 'No Tumor'
        elif class_name.lower() == 'meningioma':
            return 'Meningioma'
        elif class_name.lower() == 'glioma':
            return 'Glioma'
        elif class_name.lower() == 'pituitary':
            return 'Pituitary'
        else:
            return class_name.title()
    
    def predict_tumor(self):
        """Perform tumor prediction using quantized model with delegate support"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        try:
            self.progress.start()
            self.predict_btn.config(state='disabled')
            accel_mode = getattr(self, 'acceleration_mode', 'Unknown')
            self.status_var.set(f"Processing with {accel_mode}...")
            self.root.update()
            
            quantized_input = self.preprocess_image_for_quantized_model(self.current_image)
            self.interpreter.set_tensor(self.input_details[0]['index'], quantized_input)
            
            import time
            start_time = time.perf_counter()
            self.interpreter.invoke()
            inference_time = (time.perf_counter() - start_time) * 1000  # ms
            
            quantized_output = self.interpreter.get_tensor(self.output_details[0]['index'])
            float_output = self.dequantize_output(quantized_output)
            
            logits = float_output[0]
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            predicted_class_index = np.argmax(probabilities)
            confidence_score = probabilities[predicted_class_index]
            predicted_class = self.class_labels[predicted_class_index]
            
            if predicted_class.lower() == 'notumor':
                result_text = "No Tumor Detected"
                result_color = "green"
            else:
                formatted_class = self.format_class_name(predicted_class)
                result_text = f"Tumor Detected: {formatted_class}"
                result_color = "red"
            
            self.prediction_result = {
                'result': result_text,
                'predicted_class': predicted_class,
                'confidence': confidence_score,
                'probabilities': probabilities,
                'color': result_color,
                'inference_time': inference_time,
                'acceleration_mode': accel_mode,
                'quantized_output': quantized_output.tolist(),
                'float_output': float_output.tolist()
            }
            
            self.display_results()
            self.save_btn.config(state='normal')
            self.status_var.set(f"Inference completed ({inference_time:.1f}ms) - {accel_mode}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_var.set("Prediction failed")
            import traceback
            print("Full error traceback:")
            print(traceback.format_exc())
        
        finally:
            self.progress.stop()
            self.predict_btn.config(state='normal')
    
    def display_results(self):
        """Display prediction results with enhanced color highlighting"""
        if not self.prediction_result:
            return
        
        result = self.prediction_result
        
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        
        results_content = f"""BRAIN TUMOR DETECTION RESULTS (TFLite Runtime)
{'='*60}

Primary Result: {result['result']}
Confidence Score: {result['confidence']*100:.2f}%

DETAILED CLASSIFICATION PROBABILITIES:
{'='*60}
"""
        
        # Insert the basic content first
        self.results_text.insert(1.0, results_content)
        
        # Add classification probabilities with color highlighting
        line_num = 8  # Starting line for probabilities
        for i, (class_name, prob) in enumerate(zip(self.class_labels, result['probabilities'])):
            is_predicted = i == np.argmax(result['probabilities'])
            status = "✓ DETECTED" if is_predicted else ""
            class_display = self.format_class_name(class_name)
            
            prob_line = f"{class_display:12}: {prob*100:6.2f}% {status}\n"
            self.results_text.insert(f"{line_num}.0", prob_line)
            
            # Color highlighting for detected classes
            if is_predicted:
                if class_name.lower() == 'notumor':
                    # Green for no tumor
                    self.results_text.tag_add("no_tumor_detected", f"{line_num}.0", f"{line_num}.end")
                    self.results_text.tag_config("no_tumor_detected", 
                                                foreground="#27ae60", 
                                                font=('Arial', 11, 'bold'),
                                                background="#e8f5e8")
                else:
                    # Red for tumor detected
                    self.results_text.tag_add("tumor_detected", f"{line_num}.0", f"{line_num}.end")
                    self.results_text.tag_config("tumor_detected", 
                                                foreground="#e74c3c", 
                                                font=('Arial', 11, 'bold'),
                                                background="#fdf2f2")
            line_num += 1
        
        # Add remaining info
        remaining_content = f"\n{'='*60}\n"
        remaining_content += f"Model Type: INT8 Quantized TensorFlow Lite\n"
        remaining_content += f"Runtime: TFLite Runtime\n"
        remaining_content += f"Acceleration: {result.get('acceleration_mode', 'Unknown')}\n"
        remaining_content += f"Inference Time: {result.get('inference_time', 0):.1f} ms\n"
        remaining_content += f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        remaining_content += f"Input image: {os.path.basename(self.current_image_path) if self.current_image_path else 'Unknown'}\n"
        
        if 'quantized_output' in result:
            qo = result['quantized_output'][0]
            remaining_content += f"\nDEBUG INFO:\n"
            remaining_content += f"Quantized output range: [{min(qo)}, {max(qo)}]\n"
        
        self.results_text.insert(tk.END, remaining_content)
        
        # Enhanced color highlighting for the primary result line
        primary_result_line = "3.0"
        if result['color'] == 'red':
            # Highlight the entire primary result line in red for tumor detection
            self.results_text.tag_add("primary_tumor", "4.0", "4.end")
            self.results_text.tag_config("primary_tumor", 
                                       foreground="#e74c3c", 
                                       font=('Arial', 11, 'bold'),
                                       background="#fdf2f2")
        else:
            # Highlight the entire primary result line in green for no tumor
            self.results_text.tag_add("primary_no_tumor", "4.0", "4.end")
            self.results_text.tag_config("primary_no_tumor", 
                                       foreground="#27ae60", 
                                       font=('Arial', 11, 'bold'),
                                       background="#e8f5e8")
        
        self.results_text.config(state='disabled')
    
    def clear_results(self):
        """Clear results display"""
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)  
        self.results_text.config(state='disabled')
        self.save_btn.config(state='disabled')
    
    def clear_all(self):
        """Clear all data and reset interface"""
        self.current_image = None
        self.current_image_path = None
        self.prediction_result = None
        
        self.image_label.config(image="", text="No image selected\n\nClick 'Upload Image' to begin")
        self.image_label.image = None
        
        self.clear_results()
        self.predict_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        
        accel_mode = getattr(self, 'acceleration_mode', 'CPU')
        self.status_var.set(f"Ready - Model loaded with {accel_mode}")
    
    def save_result(self):
        """Save results to file"""
        if not self.prediction_result:
            messagebox.showwarning("Warning", "No results to save!")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Results",
                defaultextension=".txt",
                filetypes=[
                    ('Text files', '*.txt'),
                    ('All files', '*.*')
                ]
            )
            
            if file_path:
                results_content = self.results_text.get(1.0, tk.END)
                with open(file_path, 'w') as f:
                    f.write(results_content)
                
                messagebox.showinfo("Success", f"Results saved to:\n{file_path}")
                self.status_var.set(f"Results saved: {os.path.basename(file_path)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Brain Tumor Detection with TFLite Runtime')
    parser.add_argument(
        '-d', '--delegate',
        default='',
        help='Path to NPU delegate library (e.g., libvx_delegate.so for IMX93 NPU)'
    )
    parser.add_argument(
        '--model',
        default='brain_tumor_fully_quantized_int8.tflite',
        help='Path to the quantized TFLite model file'
    )
    return parser.parse_args()

def main():
    """Main function to run the application"""
    args = parse_arguments()
    
    model_file = args.model
    if not os.path.exists(model_file):
        print(f"❌ Model file '{model_file}' not found!")
        if len(sys.argv) == 1:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error", 
                               f"Quantized model file '{model_file}' not found!\n\n"
                               "Please ensure the quantized TFLite model file is in the correct location.")
            return
        else:
            sys.exit(1)
    
    print("🚀 Starting Brain Tumor Detection App with TFLite Runtime")
    print(f"📄 Model: {model_file}")
    if args.delegate:
        print(f"🔧 NPU Delegate: {args.delegate}")
        if not os.path.exists(args.delegate):
            print(f"⚠️  Delegate file not found, will fallback to CPU")
    else:
        print("💻 Running CPU-only mode")
    
    root = tk.Tk()
    app = BrainTumorDetectionApp(root, delegate_path=args.delegate)
    
    if args.model != 'brain_tumor_fully_quantized_int8.tflite':
        app.model_path = model_file
    
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    print("✅ GUI initialized successfully")
    root.mainloop()

if __name__ == "__main__":
    main()

