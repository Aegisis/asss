import pandas as pd
import cv2
from deepface import DeepFace
import threading
import queue
from PIL import Image
from customtkinter import (
    CTk, CTkFrame, CTkLabel, CTkButton, CTkEntry, 
    CTkImage, CTkScrollableFrame
)
from CTkMessagebox import CTkMessagebox
import numpy as np
from mainbackendmk1 import Backend
from tkinter import StringVar
from customtkinter import CTkToplevel
from datetime import datetime
import base64
import json
import os
import time

class CameraThread(threading.Thread):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.running = True
        self.last_frame = None
        self.face_detected_frames = 0
        self.frame_count = 0
        
    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_count += 1
                self.last_frame = frame
                
                # Process every 4th frame
                if self.frame_count % 4 != 0:
                    # Just update display
                    if hasattr(self.parent, 'camera_label'):
                        try:
                            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            photo = CTkImage(light_image=image, size=(640, 480))
                            self.parent.camera_label.configure(image=photo)
                        except Exception as e:
                            print(f"Error updating camera display: {e}")
                    continue
                
                try:
                    # Use SSD detector which is faster than RetinaFace
                    face_objs = DeepFace.extract_faces(
                        frame,
                        detector_backend='ssd',
                        enforce_detection=False,
                        align=True
                    )
                    
                    if face_objs and len(face_objs) > 0:
                        self.face_detected_frames += 1
                        
                        # Draw rectangle around detected face
                        for face_obj in face_objs:
                            facial_area = face_obj['facial_area']
                            x = facial_area['x']
                            y = facial_area['y']
                            w = facial_area['w']
                            h = facial_area['h']
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        if self.face_detected_frames >= 2:  # Reduced from 3 to 2
                            # Get the face region and resize
                            face = face_objs[0]['face']
                            face = cv2.resize(face, (224, 224))
                            
                            # Use Facenet512 model which is faster than VGG-Face
                            embedding_obj = DeepFace.represent(
                                face,
                                detector_backend='skip',
                                enforce_detection=False,
                                align=True,
                                model_name='Facenet512'
                            )
                            
                            if embedding_obj:
                                face_embedding = np.array(embedding_obj[0]['embedding'])
                                print(f"Embedding shape: {face_embedding.shape}")  # Debugging line

                                # Ensure the shape matches the expected input for verification
                                if face_embedding.shape[0] != 128:  # Adjust this based on your verification model
                                    print("Warning: Embedding shape does not match expected size for verification.")
                                    # Handle the shape mismatch accordingly

                                self.process_face(frame, face_embedding)
                                self.face_detected_frames = 0  # Reset counter
                                break
                
                except Exception as e:
                    print(f"Face detection error: {e}")
                    self.face_detected_frames = 0
                    pass
                
                if hasattr(self.parent, 'camera_label'):
                    try:
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        photo = CTkImage(light_image=image, size=(640, 480))
                        self.parent.camera_label.configure(image=photo)
                    except Exception as e:
                        print(f"Error updating camera display: {e}")
                    
        cap.release()

    def process_face(self, frame, face_embedding):
        """Process detected face and find best match"""
        try:
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.parent.USER_DETECTED_PATH), exist_ok=True)
            
            # Save the image
            cv2.imwrite(self.parent.USER_DETECTED_PATH, frame)
            
            # Convert and save to JSON
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Save face data
            face_data = {
                "timestamp": timestamp,
                "image": img_base64,
                "capture_date": datetime.now().isoformat(),
                "encoding": face_embedding.tolist()
            }
            
            self.parent.save_face_data(face_data)
            
            # Try to find matching user
            user_id, name = self.parent.backend.verify_face(face_embedding)
            
            # Emotion recognition
            emotion_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            detected_emotion = emotion_result[0]['dominant_emotion'] if emotion_result else None
            
            def show_result():
                try:
                    # Stop camera and animation
                    self.running = False
                    if hasattr(self.parent, 'animation'):
                        self.parent.animation.release()
                    
                    if user_id:
                        # If face recognized, go directly to menu
                        self.parent.after(100, lambda: self.parent.show_menu_for_user(user_id, name, detected_emotion))
                    else:
                        # If face not recognized, show registration choice
                        self.parent.after(100, lambda: self.parent.show_registration_choice(face_embedding))
                except Exception as e:
                    print(f"Error in show_result: {e}")
                    self.parent.show_error("An error occurred. Please try again.")
            
            # Schedule the result display
            self.parent.after(0, show_result)
            
        except Exception as e:
            print(f"Error processing face: {e}")
            self.parent.show_error("Failed to process face. Please try again.")

class SmartFoodSystem(CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize file paths with absolute paths
        self.MENU_DATA_PATH = (r"C:\Users\omila\OneDrive\Desktop\gpro\New folder\indian_food.csv")  # Put the CSV file in the same directory as your script
        self.USER_PHOTOS_PATH = (r"C:\Users\omila\OneDrive\Desktop\gpro\ii\user_photos.json")
        self.USER_DETECTED_PATH = (r"C:\Users\omila\OneDrive\Desktop\gpro\ii\user_detected.jpg")
        self.VIDEO_PATH = (r"C:\Users\omila\OneDrive\Desktop\gpro\mk.2test\New folder\Animation - 1733991071989.mp4")
        
        # Initialize backend first
        self.backend = Backend()
        
        # Window configuration
        self.title("GP Robo - Your AI Waiter")
        self.geometry("1200x800")
        self.configure(fg_color="#FFFFFF")  # Light background
        
        # Initialize variables
        self.scan_window = None
        self.camera_thread = None
        self.cart_items = []
        self.cart_window = None
        
        # Load menu data
        self.load_menu_data()
        
        # Setup styles
        self.setup_styles()
        
        # Show landing page first
        self.show_landing_page()
        
        # Add this to your existing initialization
        self.current_order_items = []  # Track current order items

    def setup_styles(self):
        """Setup custom styles with light fonts on light theme"""
        self.styles = {
            'heading': {
                'font': ('Helvetica', 24, 'bold'),
                'text_color': '#000000'  # Black text
            },
            'subheading': {
                'font': ('Helvetica', 18, 'bold'),
                'text_color': '#333333'  # Dark gray text
            },
            'body': {
                'font': ('Helvetica', 12),
                'text_color': '#000000'  # Black text
            }
        }
        
    def show_landing_page(self):
        """Show initial landing page"""
        # Clear any existing widgets
        for widget in self.winfo_children():
            widget.destroy()
        
        # Create new main container with white background
        self.main_container = CTkFrame(self, fg_color="#FFFFFF")  # Set white background
        self.main_container.pack(expand=True, fill="both")
        
        # Welcome text
        CTkLabel(
            self.main_container,
            text="Welcome to GP Robo",
            font=("Helvetica", 32, "bold"),
            text_color="#000000"  # Black text
        ).pack(pady=(200, 10))
        
        CTkLabel(
            self.main_container,
            text="Your AI Waiter",
            font=("Helvetica", 18),
            text_color="#333333"  # Dark gray text
        ).pack(pady=(0, 50))
        
        # Robot icon
        CTkLabel(
            self.main_container,
            text="🤖",
            font=("Helvetica", 48)
        ).pack(pady=(0, 50))
        
        # Touch to continue button
        CTkButton(
            self.main_container,
            text="Touch to Continue",
            command=self.show_options_page,
            font=("Helvetica", 20),
            fg_color="#4CAF50",
            hover_color="#45a049",
            width=300,
            height=50,
            corner_radius=25
        ).pack(pady=50)

    def show_options_page(self):
        """Show face detection directly after touch to continue"""
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Start face detection immediately
        self.start_face_recognition()

    def show_registration_choice(self, face_encoding):
        """Show stylish registration choice with modern UI"""
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Main content frame with new color
        content_frame = CTkFrame(
            self.main_container,
            fg_color="#d9d9d9",  # Changed from (#f0f2f5, "#2b2b2b")
            corner_radius=20
        )
        content_frame.pack(expand=True, fill="both", padx=40, pady=40)
        
        # Welcome message with adjusted text color
        CTkLabel(
            content_frame,
            text="Welcome to GP Robo! 🤖",
            font=("Helvetica", 32, "bold"),
            text_color="#333333"  # Dark gray for main heading
        ).pack(pady=(40, 20))
        
        # Benefits frame with new color
        benefits_frame = CTkFrame(
            content_frame,
            fg_color="#d9d9d9",  # Changed from (#ffffff, "#333333")
            corner_radius=15,
            border_width=1,
            border_color=("#e0e0e0", "#404040")
        )
        benefits_frame.pack(padx=60, pady=(0, 30), fill="x")
        
        # Benefits content with icons
        benefits = [
            ("🎁", "Exclusive Rewards", "Special birthday treats and member discounts"),
            ("🎯", "Personalized Menu", "AI-powered food recommendations"),
            ("⚡", "Quick Access", "Instant recognition for faster ordering"),
            ("🍽️", "Previous Order History", "Track your previous orders and preferences")
        ]
        
        for icon, title, desc in benefits:
            benefit_item = CTkFrame(
                benefits_frame,
                fg_color="transparent"
            )
            benefit_item.pack(padx=20, pady=10, fill="x")
            
            # Icon
            CTkLabel(
                benefit_item,
                text=icon,
                font=("Helvetica", 24),
                text_color="#333333"  # Dark gray for icons
            ).pack(side="left", padx=(10, 15))
            
            # Benefit text frame
            text_frame = CTkFrame(
                benefit_item,
                fg_color="transparent"
            )
            text_frame.pack(side="left", fill="x", expand=True)
            
            CTkLabel(
                text_frame,
                text=title,
                font=("Helvetica", 14, "bold"),
                text_color="#333333",  # Dark gray for benefit titles
                anchor="w"
            ).pack(fill="x")
            
            CTkLabel(
                text_frame,
                text=desc,
                font=("Helvetica", 12),
                text_color="#666666",  # Medium gray for descriptions
                anchor="w"
            ).pack(fill="x")
        
        # Buttons frame with modern design
        buttons_frame = CTkFrame(
            content_frame,
            fg_color="#d9d9d9"  # Match parent background
        )
        buttons_frame.pack(pady=30)
        
        # Register button
        register_btn = CTkButton(
            buttons_frame,
            text="Register Now",
            font=("Helvetica", 16, "bold"),
            fg_color="#4a4a4a",  # Darker gray
            hover_color="#3a3a3a",  # Even darker on hover
            text_color="white",
            corner_radius=25,
            width=200,
            height=50,
            command=lambda: self.show_registration_form(face_encoding)  # Added back command
        )
        register_btn.pack(side="left", padx=10)
        
        # Guest button
        guest_btn = CTkButton(
            buttons_frame,
            text="Continue as Guest",
            font=("Helvetica", 16, "bold"),
            fg_color="#8c8c8c",  # Medium gray
            hover_color="#7a7a7a",  # Darker on hover
            text_color="white",
            corner_radius=25,
            width=200,
            height=50,
            command=self.show_menu_without_registration  # Added back command
        )
        guest_btn.pack(side="left", padx=10)
        
        # Privacy note
        CTkLabel(
            content_frame,
            text="🔒 Your data is protected and never shared",
            font=("Helvetica", 12),
            text_color="#666666"  # Medium gray for privacy note
        ).pack(pady=(20, 40))

    def start_face_recognition(self, require_smile=False):
        """Start face recognition with automatic capture"""
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Create animation frame
        animation_frame = CTkFrame(self.main_container)
        animation_frame.pack(expand=True)
        
        # Initialize camera thread first
        self.camera_thread = CameraThread(self)
        self.camera_thread.daemon = True
        
        # Create animation label as instance variable
        self.animation_label = CTkLabel(animation_frame, text="")
        self.animation_label.pack(expand=True)
        
        try:
            # Load animation with absolute path
            self.animation = cv2.VideoCapture(self.VIDEO_PATH)
            if not self.animation.isOpened():
                raise Exception("Could not open animation file")
            
            # Get video dimensions
            width = int(self.animation.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.animation.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            def update_animation():
                if not hasattr(self, 'animation') or not self.animation.isOpened():
                    return
                    
                try:
                    ret, frame = self.animation.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame_rgb)
                        photo = CTkImage(light_image=image, size=(width, height))
                        if hasattr(self, 'animation_label') and self.animation_label.winfo_exists():
                            self.animation_label.configure(image=photo)
                            self.animation_label.image = photo  # Keep reference
                    else:
                        self.animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                    if hasattr(self, 'animation'):
                        self.after(33, update_animation)
                except Exception as e:
                    print(f"Animation error: {e}")
            
            # Start animation
            update_animation()
            
            # Start camera thread
            self.camera_thread.start()
            
        except Exception as e:
            print(f"Error loading animation: {e}")
            CTkLabel(
                animation_frame,
                text="🤖 Scanning...",
                font=("Helvetica", 24)
            ).pack(pady=20)
        
        # Status messages
        self.status_label = CTkLabel(
            self.main_container,
            text="Detecting face...",
            font=self.styles['body']['font']
        )
        self.status_label.pack(pady=10)
        
        # Cancel button
        CTkButton(
            self.main_container,
            text="Cancel",
            font=self.styles['body']['font'],
            command=self.stop_camera_and_return,
            fg_color="#f44336",
            hover_color="#da190b",
            width=150
        ).pack(pady=10)

    def capture_photo(self):
        """Capture photo and process face recognition"""
        if hasattr(self, 'camera_thread') and self.camera_thread.last_frame is not None:
            try:
                frame = self.camera_thread.last_frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = DeepFace.extract_faces(rgb_frame, detector_backend='retinaface', enforce_detection=False)
                
                if not face_locations:
                    CTkMessagebox(title="Error", message="No face detected! Please try again.")
                    return
                
                # Get face encodings first
                face_encodings = DeepFace.represent(rgb_frame, detector_backend='retinaface', enforce_detection=False, align=True)
                if not face_encodings:
                    CTkMessagebox(title="Error", message="Could not process face. Please try again.")
                    return
                
                # Create photos directory if it doesn't exist
                photos_dir = os.path.join(os.getcwd(), "user_data", "photos")
                os.makedirs(photos_dir, exist_ok=True)
                
                # Save image with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = os.path.join(photos_dir, f"user_{timestamp}.jpg")
                
                # Save physical image
                cv2.imwrite(image_path, frame)
                
                # Convert image to base64 for JSON storage
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Save to JSON
                self.save_image_to_json(img_base64, timestamp)
                
                # Clear camera display
                self.camera_label.configure(image=None)
                
                # Check if user exists
                user_id, name = self.backend.verify_face(face_encodings[0])
                
                if user_id:
                    CTkMessagebox(title="Success", message=f"Welcome back, {name}!")
                    self.show_menu_for_user(user_id, name)
                else:
                    CTkMessagebox(title="New User", message="Face not recognized. Please register.")
                    self.show_registration_form(face_encodings[0])
            except Exception as e:
                print(f"Error capturing photo: {e}")
                CTkMessagebox(title="Error", message="Failed to capture photo. Please try again.")

    def save_image_to_json(self, img_base64, timestamp):
        """Save captured image to JSON database"""
        try:
            # Create base directory if it doesn't exist
            base_dir = os.path.join(os.getcwd(), "user_data")
            os.makedirs(base_dir, exist_ok=True)
            
            # Set paths relative to base directory
            json_file = os.path.join(base_dir, "user_photos.json")
            
            # Load existing data or create new
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {"images": []}
            
            # Add new image
            image_data = {
                "timestamp": timestamp,
                "image": img_base64,
                "capture_date": datetime.now().isoformat()
            }
            
            data["images"].append(image_data)
            
            # Save updated data
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)
                
        except Exception as e:
            print(f"Error saving image to JSON: {e}")

    def stop_camera_and_show_menu(self, user_id, name):
        """Stop camera and show user menu"""
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.running = False
            self.camera_thread.join()
        self.show_menu_for_user(user_id, name)

    def stop_camera_and_register(self, face_encoding):
        """Stop camera and show registration form"""
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.running = False
            self.camera_thread.join()
        self.show_registration_form(face_encoding)

    def stop_camera_and_return(self):
        """Stop camera and return to landing page"""
        if hasattr(self, 'animation_running'):
            self.animation_running = False
            
        if hasattr(self, 'camera_thread') and self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.running = False
            self.camera_thread.join()
            
        self.show_landing_page()

    def load_menu_data(self):
        """Load menu data from CSV file"""
        try:
            self.menu_data = pd.read_csv(self.MENU_DATA_PATH)
            # Clean up course names to match our categories
            self.menu_data['Course'] = self.menu_data['Course'].map({
                'dessert': 'Dessert',
                'main course': 'Main Course',
                'snack': 'Snacks',
                'snacks': 'Snacks'
            }).fillna('Other')
            print(f"Loaded {len(self.menu_data)} menu items")
        except Exception as e:
            print(f"Error loading menu data: {e}")
            self.menu_data = pd.DataFrame()

    def show_error(self, message):
        """Display error message to user."""
        CTkMessagebox(
            title="Error",
            message=message,
            icon="cancel"
        )

    def show_registration_form(self, face_encoding):
        """Show registration form with validation"""
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Registration form container with fixed width
        form_frame = CTkFrame(
            self.main_container,
            fg_color="#d9d9d9",
            corner_radius=15,
            width=500,  # Fixed width
            height=600  # Fixed height
        )
        form_frame.pack(expand=True, padx=20, pady=40)
        form_frame.pack_propagate(False)  # Maintain fixed size
        
        # Title with more padding
        CTkLabel(
            form_frame,
            text="New User Registration",
            font=self.styles['heading']['font'],
            text_color="#333333"
        ).pack(pady=(40, 30))
        
        # Form fields container with fixed width
        fields_frame = CTkFrame(
            form_frame, 
            fg_color="#d9d9d9",
            width=400  # Fixed width for fields container
        )
        fields_frame.pack(pady=20, padx=50)
        
        # Form fields with validation
        fields = {
            'name': {
                'label': "Full Name",
                'validate': lambda x: len(x.strip()) >= 3,
                'error': "Name must be at least 3 characters"
            },
            'phone': {
                'label': "Phone Number",
                'validate': lambda x: x.isdigit() and len(x) == 10,
                'error': "Phone number must be 10 digits"
            },
            'dob': {
                'label': "Date of Birth (YYYY-MM-DD)",
                'validate': lambda x: self.validate_date(x),
                'error': "Invalid date format. Use YYYY-MM-DD"
            }
        }
        
        # Create entry fields and error labels
        entries = {}
        error_labels = {}
        
        for field, config in fields.items():
            # Label
            CTkLabel(
                fields_frame,
                text=config['label'],
                font=("Helvetica", 14),
                text_color="#333333"
            ).pack(anchor="w", pady=(10, 0))
            
            # Entry
            entry = CTkEntry(
                fields_frame,
                width=300,
                height=35,
                font=("Helvetica", 12)
            )
            entry.pack(pady=(5, 0))
            entries[field] = entry
            
            # Error label
            error_label = CTkLabel(
                fields_frame,
                text="",
                font=("Helvetica", 12),
                text_color="#FF4757"
            )
            error_label.pack(anchor="w")
            error_labels[field] = error_label
        
        def validate_and_submit():
            # Clear previous errors
            for error_label in error_labels.values():
                error_label.configure(text="")
            
            # Validate all fields
            valid = True
            values = {}
            
            for field, config in fields.items():
                value = entries[field].get().strip()
                values[field] = value
                
                if not config['validate'](value):
                    error_labels[field].configure(text=config['error'])
                    valid = False
            
            if valid:
                # Register user
                user_id = self.backend.register_user(
                    values['name'],
                    values['phone'],
                    values['dob'],
                    face_encoding
                )
                
                if user_id:
                    CTkMessagebox(
                        title="Success",
                        message="Registration successful!",
                        icon="check"
                    )
                    self.show_menu_for_user(user_id, values['name'])
                else:
                    CTkMessagebox(
                        title="Error",
                        message="Registration failed. Please try again.",
                        icon="cancel"
                    )
        
        # Buttons frame
        buttons_frame = CTkFrame(form_frame, fg_color="#d9d9d9")
        buttons_frame.pack(pady=30)
        
        # Register button
        CTkButton(
            buttons_frame,
            text="Register",
            command=validate_and_submit,
            font=("Helvetica", 14),
            fg_color="#4CAF50",
            hover_color="#45a049",
            width=150,
            height=40
        ).pack(side="left", padx=10)
        
        # Cancel button
        CTkButton(
            buttons_frame,
            text="Cancel",
            command=self.show_landing_page,
            font=("Helvetica", 14),
            fg_color="#FF4757",
            hover_color="#FF6B81",
            width=150,
            height=40
        ).pack(side="left", padx=10)

    def show_menu_for_user(self, user_id, name, detected_emotion=None):
        """Show menu for registered user with categories and cart"""
        try:
            # Clear main container
            for widget in self.main_container.winfo_children():
                widget.destroy()
            
            # Create header with light theme
            header_frame = CTkFrame(self.main_container, fg_color="#d9d9d9")  # Light gray background
            header_frame.pack(fill="x", padx=20, pady=10)
            
            CTkLabel(
                header_frame,
                text=f"Welcome, {name}!",
                font=self.styles['subheading']['font'],
                text_color="#333333"  # Dark gray text
            ).pack(side="left", padx=20)
            
            # Header buttons (right side)
            buttons_frame = CTkFrame(header_frame, fg_color="transparent")
            buttons_frame.pack(side="right", padx=10)
            
            if user_id:
                # Past Orders button
                CTkButton(
                    buttons_frame,
                    text="📋 Past Orders",
                    command=lambda: self.show_past_orders_window(user_id),
                    font=("Helvetica", 12),
                    fg_color="#4a4a4a",  # Dark gray
                    hover_color="#3a3a3a",
                    width=100,
                    height=30,
                    corner_radius=15
                ).pack(side="left", padx=5)
            
            # Cart button
            CTkButton(
                buttons_frame,
                text=f"🛒 Cart ({len(self.cart_items)})",
                command=lambda: self.show_cart(user_id),
                font=("Helvetica", 12),
                fg_color="#8c8c8c",  # Medium gray
                hover_color="#7a7a7a",
                text_color="white",
                width=100,
                height=30,
                corner_radius=15
            ).pack(side="left", padx=5)
            
            # Profile button
            CTkButton(
                buttons_frame,
                text="👤 Profile",
                command=lambda: self.show_user_profile(user_id),
                font=("Helvetica", 12),
                fg_color="#4a4a4a",  # Dark gray
                hover_color="#3a3a3a",
                text_color="white",
                width=100,
                height=30,
                corner_radius=15
            ).pack(side="left", padx=5)
            
            # Main content area with light theme
            content_frame = CTkFrame(self.main_container, fg_color="#ffffff")  # White background
            content_frame.pack(fill="both", expand=True, padx=20, pady=10)
            
            # Left sidebar with light theme
            left_sidebar = CTkFrame(content_frame, fg_color="#d9d9d9")  # Light gray
            left_sidebar.pack(side="left", fill="y", padx=(0, 20), pady=10)
            
            # Cart section with light theme
            cart_frame = CTkFrame(left_sidebar, fg_color="#e6e6e6")  # Lighter gray
            cart_frame.pack(fill="x", padx=10, pady=10)
            
            CTkLabel(
                cart_frame,
                text="Your Order",
                font=self.styles['subheading']['font'],
                text_color="#333333"  # Dark gray text
            ).pack(pady=10)
            
            # Scrollable cart items
            cart_items_frame = CTkScrollableFrame(cart_frame, height=200, fg_color="#f0f0f0")  # Very light gray
            cart_items_frame.pack(fill="x", padx=10, pady=5)
            
            total_amount = 0
            for item in self.cart_items:
                item_frame = CTkFrame(cart_items_frame, fg_color="#ffffff")  # White background
                item_frame.pack(fill="x", pady=2)
                
                CTkLabel(
                    item_frame,
                    text=item['Name'],
                    font=self.styles['body']['font'],
                    text_color="#333333"  # Dark gray text
                ).pack(side="left", padx=5)
                
                CTkLabel(
                    item_frame,
                    text=f"₹{item.get('Price', 100)}",
                    font=self.styles['body']['font'],
                    text_color="#333333"  # Dark gray text
                ).pack(side="right", padx=5)
                
                total_amount += item.get('Price', 100)
            
            # Total and Order button
            CTkLabel(
                cart_frame,
                text=f"Total: ₹{total_amount}",
                font=self.styles['subheading']['font'],
                text_color="#333333"  # Dark gray text
            ).pack(pady=10)
            
            CTkButton(
                cart_frame,
                text="Order Now",
                command=lambda: self.show_cart(user_id),
                fg_color="#4a4a4a",  # Dark gray
                hover_color="#3a3a3a",
                text_color="white"
            ).pack(pady=10)
            
            # Recommendations section
            CTkLabel(
                left_sidebar,
                text="Recommended for You",
                font=self.styles['subheading']['font'],
                text_color="#333333"  # Dark gray text
            ).pack(pady=10)
            
            rec_frame = CTkScrollableFrame(left_sidebar, height=300, fg_color="#f0f0f0")  # Very light gray
            rec_frame.pack(fill="both", expand=True, padx=10, pady=5)
            
            # Menu section
            menu_frame = CTkFrame(content_frame, fg_color="#ffffff")  # White background
            menu_frame.pack(side="right", fill="both", expand=True)
            
            # Category buttons
            categories_frame = CTkFrame(menu_frame, fg_color="transparent")
            categories_frame.pack(fill="x", padx=20, pady=10)
            
            categories = {
                "Main Course": "🍛",
                "Dessert": "🍨",
                "Snacks": "🥪"
            }
            
            for category, emoji in categories.items():
                CTkButton(
                    categories_frame,
                    text=f"{emoji} {category}",
                    command=lambda c=category: self.show_category(c, user_id),
                    fg_color="#4a4a4a",  # Dark gray
                    hover_color="#3a3a3a",
                    text_color="white",
                    width=120,
                    height=30,
                    corner_radius=15
                ).pack(side="left", padx=5)
            
            # Menu items frame
            self.menu_frame = CTkScrollableFrame(menu_frame, fg_color="#f0f0f0")  # Very light gray
            self.menu_frame.pack(fill="both", expand=True, padx=20, pady=10)
            
            # Show first category by default
            self.show_category("Main Course", user_id)
            
            # Get recommendations based on past orders and detected emotion
            recommendations = self.backend.get_recommendations(user_id, detected_emotion)
            print(f"Recommendations based on emotion '{detected_emotion}': {recommendations}")  # Debugging line
            
            # Display recommendations in the UI
            # Add your code to display these recommendations in the menu
            
        except Exception as e:
            print(f"Error showing menu: {e}")
            CTkMessagebox(
                title="Error",
                message="An error occurred while loading the menu. Please try again.",
                icon="cancel"
            )
            self.show_landing_page()

    def show_user_profile(self, user_id):
        """Show user profile in a separate window"""
        try:
            # Get user details from database
            user_details = self.backend.get_user_details(user_id)
            
            # Create new window with dark theme
            profile_window = CTkToplevel()
            profile_window.grab_set()  # Make window modal
            profile_window.title("User Profile")
            profile_window.geometry("400x600")
            profile_window.configure(fg_color="#1A1A1A")
            
            # Center the window
            profile_window.update_idletasks()
            x = (profile_window.winfo_screenwidth() - profile_window.winfo_width()) // 2
            y = (profile_window.winfo_screenheight() - profile_window.winfo_height()) // 2
            profile_window.geometry(f"+{x}+{y}")
            
            # Main profile frame
            profile_frame = CTkFrame(
                profile_window,
                fg_color="#2D2D2D",  # Darker gray
                corner_radius=15,
            )
            profile_frame.pack(expand=True, fill="both", padx=20, pady=20)
            
            # Title
            CTkLabel(
                profile_frame,
                text="User Profile",
                font=self.styles['heading']['font']
            ).pack(pady=20)
            
            try:
                # Load and display user image
                user_image = Image.open("D:/ai project gp robo/aimlproject-main/user_detected.jpg")
                user_image = user_image.resize((150, 150), Image.LANCZOS)
                photo = CTkImage(light_image=user_image, size=(150, 150))
                
                image_label = CTkLabel(
                    profile_frame,
                    image=photo,
                    text=""
                )
                image_label.pack(pady=10)
            except Exception as e:
                print(f"Error loading user image: {e}")
            
            # User details
            details_frame = CTkFrame(profile_frame, fg_color="transparent")
            details_frame.pack(pady=20, padx=30, fill="x")
            
            # Name
            CTkLabel(
                details_frame,
                text="Name:",
                font=self.styles['body']['font']
            ).pack(anchor="w")
            CTkLabel(
                details_frame,
                text=user_details['name'],
                font=('Helvetica', 14, 'bold')
            ).pack(anchor="w", pady=(0, 10))
            
            # Phone
            CTkLabel(
                details_frame,
                text="Phone:",
                font=self.styles['body']['font']
            ).pack(anchor="w")
            CTkLabel(
                details_frame,
                text=user_details['phone'],
                font=('Helvetica', 14, 'bold')
            ).pack(anchor="w", pady=(0, 10))
            
            # DOB
            CTkLabel(
                details_frame,
                text="Date of Birth:",
                font=self.styles['body']['font']
            ).pack(anchor="w")
            CTkLabel(
                details_frame,
                text=user_details['dob'],
                font=('Helvetica', 14, 'bold')
            ).pack(anchor="w", pady=(0, 10))
            
            # Buttons frame
            buttons_frame = CTkFrame(profile_frame, fg_color="transparent")
            buttons_frame.pack(pady=20)
            
            def close_profile():
                profile_window.grab_release()
                profile_window.destroy()
            
            # Update button commands
            CTkButton(
                buttons_frame,
                text="Close",
                command=close_profile,
                fg_color="#FF3C5A",
                hover_color="#FF1A1A",
                width=120
            ).pack(side="left", padx=10)
            
            CTkButton(
                buttons_frame,
                text="Cancel",
                command=close_profile,
                fg_color="#666666",
                hover_color="#555555",
                width=120
            ).pack(side="left", padx=10)
            
        except Exception as e:
            print(f"Error showing profile: {e}")
            CTkMessagebox(title="Error", message="Could not load profile")

    def add_to_cart(self, item, user_id=None):
        """Add item to cart"""
        try:
            # Add price to item dictionary
            item_with_price = item.copy()
            item_with_price['Price'] = 100  # Replace with actual price
            
            self.cart_items.append(item_with_price)
            print(f"Added to cart: {item['Name']}")
            print(f"Cart now has {len(self.cart_items)} items")
            
            CTkMessagebox(
                title="Added to Cart",
                message=f"{item['Name']} added to cart!\nPrice: ₹{item_with_price['Price']}",
                icon="check"
            )
            
            # Refresh the current view - use the correct method based on user_id
            if user_id:
                self.show_menu_for_user(user_id, self.backend.get_user_name(user_id))
            else:
                self.show_menu_without_registration()
                
        except Exception as e:
            print(f"Error adding to cart: {e}")
            CTkMessagebox(
                title="Error",
                message="Could not add item to cart",
                icon="cancel"
            )

    def show_cart(self, user_id):
        """Show cart with order confirmation"""
        if not self.cart_items:
            CTkMessagebox(title="Cart Empty", message="Your cart is empty!")
            return
        
        try:
            # Create and configure cart window
            cart_window = CTkToplevel(self)
            cart_window.title("Your Cart")
            cart_window.geometry("500x600")
            cart_window.configure(fg_color="#1A1A1A")
            
            # Make window modal and wait for it to be ready
            cart_window.transient(self)
            cart_window.focus_force()
            cart_window.grab_set()
            
            # Cart items container
            cart_frame = CTkScrollableFrame(
                cart_window,
                fg_color="#262626"
            )
            cart_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            total_amount = 0
            
            # Display cart items
            for item in self.cart_items:
                item_frame = CTkFrame(cart_frame, fg_color="#333333")
                item_frame.pack(fill="x", padx=10, pady=5)
                
                CTkLabel(
                    item_frame,
                    text=item['Name'],
                    font=self.styles['body']['font'],
                    text_color='#FFFFFF'
                ).pack(side="left", padx=10, pady=10)
                
                price = 100
                total_amount += price
                
                CTkLabel(
                    item_frame,
                    text=f"₹{price}",
                    font=self.styles['body']['font'],
                    text_color='#FFFFFF'
                ).pack(side="right", padx=10)
                
                CTkButton(
                    item_frame,
                    text="Remove",
                    command=lambda i=item: self.remove_from_cart(i, cart_window, user_id),
                    fg_color="#dc3545",
                    hover_color="#c82333",
                    width=80
                ).pack(side="right", padx=5)
            
            # Total amount
            total_frame = CTkFrame(cart_window, fg_color="#333333")
            total_frame.pack(fill="x", padx=20, pady=10)
            
            CTkLabel(
                total_frame,
                text=f"Total: ₹{total_amount}",
                font=self.styles['subheading']['font'],
                text_color='#FFFFFF'
            ).pack(pady=10)
            
            # Confirm order button
            CTkButton(
                cart_window,
                text="Confirm Order",
                command=lambda: self.place_order(user_id, cart_window),
                fg_color="#28a745",
                hover_color="#218838",
                width=200
            ).pack(pady=20)
            
        except Exception as e:
            print(f"Error showing cart: {e}")
            CTkMessagebox(title="Error", message="Could not load cart")

    def place_order(self, user_id, cart_window):
        """Place order for registered user"""
        try:
            # Get order items
            order_items = [item['Name'] for item in self.cart_items]
            total_amount = sum(item.get('Price', 100) for item in self.cart_items)
            
            # Save order to database
            if self.backend.save_order(user_id, order_items, total_amount):
                # Clear cart
                self.cart_items = []
                cart_window.destroy()
                
                # Show success message and wait for user to acknowledge
                msg = CTkMessagebox(
                    title="Success",
                    message="Order placed successfully!\nThank you for your order!",
                    icon="check",
                    option_1="OK"
                )
                msg.get()  # Wait for user response
                
                # Reset window state
                self.configure(fg_color="#FFFFFF")  # Reset background color
                self.main_container.destroy()  # Remove current container
                
                # Create new main container
                self.main_container = CTkFrame(self)
                self.main_container.pack(expand=True, fill="both")
                
                # Show initial landing page
                self.show_landing_page()
            else:
                raise Exception("Failed to save order")
                
        except Exception as e:
            print(f"Error placing order: {e}")
            CTkMessagebox(title="Error", message="Failed to place order!")

    def place_guest_order(self, cart_window):
        """Place order for guest user"""
        try:
            # Get order items
            order_items = [item['Name'] for item in self.cart_items]
            total_amount = sum(item.get('Price', 100) for item in self.cart_items)
            
            # Save guest order
            if self.backend.save_guest_order(self.guest_session_id, order_items, total_amount):
                # Clear cart
                self.cart_items = []
                cart_window.destroy()
                
                # Show success message and wait for user to acknowledge
                msg = CTkMessagebox(
                    title="Success",
                    message="Order placed successfully!\nThank you for your order!",
                    icon="check",
                    option_1="OK"
                )
                msg.get()  # Wait for user response
                
                # Reset window state
                self.configure(fg_color="#FFFFFF")  # Reset background color
                self.main_container.destroy()  # Remove current container
                
                # Create new main container
                self.main_container = CTkFrame(self)
                self.main_container.pack(expand=True, fill="both")
                
                # Show initial landing page
                self.show_landing_page()
            else:
                raise Exception("Failed to save order")
                
        except Exception as e:
            print(f"Error placing guest order: {e}")
            CTkMessagebox(title="Error", message="Failed to place order!")

    def remove_from_cart(self, item, cart_window, user_id):
        """Remove item from cart"""
        self.cart_items.remove(item)
        cart_window.destroy()
        self.show_cart(user_id)  # Refresh cart window

    def show_menu_without_registration(self):
        """Show menu for guest users without registration"""
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Create header
        header_frame = CTkFrame(self.main_container, fg_color="#333333")
        header_frame.pack(fill="x", padx=20, pady=10)
        
        CTkLabel(
            header_frame,
            text="Welcome, Guest!",
            font=self.styles['heading']['font']
        ).pack(side="left", padx=20)
        
        # Cart button for guest
        CTkButton(
            header_frame,
            text=f"🛒 Cart ({len(self.cart_items)})",
            command=lambda: self.show_guest_cart(),
            fg_color="#FF3C5A"
        ).pack(side="right", padx=20)
        
        # Register button
        CTkButton(
            header_frame,
            text="Register for Benefits",
            command=lambda: self.show_consent_form(),
            fg_color="#28a745"
        ).pack(side="right", padx=20)
        
        # Menu container
        menu_frame = CTkScrollableFrame(
            self.main_container,
            fg_color="#262626"
        )
        menu_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Menu items
        for index, row in self.menu_data.iterrows():
            item_frame = CTkFrame(menu_frame, fg_color="#333333")
            item_frame.pack(fill="x", padx=10, pady=5)
            
            # Food details
            details_frame = CTkFrame(item_frame, fg_color="transparent")
            details_frame.pack(side="left", fill="x", expand=True, padx=10, pady=5)
            
            CTkLabel(
                details_frame,
                text=row['Name'],
                font=self.styles['subheading']['font']
            ).pack(anchor="w")
            
            CTkLabel(
                details_frame,
                text=f"Type: {row['Veg_Non']} | Course: {row['Course']}",
                font=self.styles['body']['font']
            ).pack(anchor="w")
            
            # Price and Order button
            button_frame = CTkFrame(item_frame, fg_color="transparent")
            button_frame.pack(side="right", padx=10)
            
            CTkLabel(
                button_frame,
                text="₹100",  # Replace with actual price
                font=self.styles['body']['font']
            ).pack(side="left", padx=10)
            
            CTkButton(
                button_frame,
                text="Add to Cart",
                command=lambda item=row: self.add_to_guest_cart(item),
                fg_color="#28a745"
            ).pack(side="left", padx=5)

    def add_to_guest_cart(self, item):
        """Add item to guest cart"""
        self.cart_items.append(item)
        CTkMessagebox(title="Success", message=f"{item['Name']} added to cart!")
        self.show_menu_without_registration()

    def show_guest_cart(self):
        """Show cart contents for guest"""
        if not self.cart_items:
            CTkMessagebox(title="Cart Empty", message="Your cart is empty!")
            return
        
        cart_window = CTkToplevel(self)
        cart_window.title("Shopping Cart")
        cart_window.geometry("500x600")
        
        # Cart items container
        cart_frame = CTkScrollableFrame(cart_window)
        cart_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        total_amount = 0
        
        # Display cart items
        for item in self.cart_items:
            item_frame = CTkFrame(cart_frame, fg_color="#333333")
            item_frame.pack(fill="x", pady=5)
            
            CTkLabel(
                item_frame,
                text=item['Name'],
                font=self.styles['body']['font']
            ).pack(side="left", padx=10, pady=10)
            
            price = 100  # Replace with actual price
            total_amount += price
            
            CTkLabel(
                item_frame,
                text=f"₹{price}",
                font=self.styles['body']['font']
            ).pack(side="right", padx=10)
            
            CTkButton(
                item_frame,
                text="Remove",
                command=lambda i=item: self.remove_from_guest_cart(i),
                fg_color="#dc3545",
                width=80
            ).pack(side="right", padx=5)
        
        # Total amount
        total_frame = CTkFrame(cart_window, fg_color="#333333")
        total_frame.pack(fill="x", padx=20, pady=10)
        
        CTkLabel(
            total_frame,
            text=f"Total: ₹{total_amount}",
            font=self.styles['subheading']['font']
        ).pack(pady=10)
        
        # Order button
        CTkButton(
            cart_window,
            text="Place Order",
            command=lambda: self.place_guest_order(cart_window),
            fg_color="#28a745"
        ).pack(pady=20)

    def remove_from_guest_cart(self, item):
        """Remove item from guest cart"""
        self.cart_items.remove(item)
        self.show_guest_cart()

    def save_face_data(self, face_data):
        """Save face data to JSON database"""
        try:
            # Create base directory if it doesn't exist
            base_dir = os.path.join(os.getcwd(), "user_data")
            os.makedirs(base_dir, exist_ok=True)
            
            # Set paths relative to base directory
            json_file = os.path.join(base_dir, "user_photos.json")
            
            # Load existing data or create new
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {"faces": []}
            
            # Add new face data
            data["faces"].append(face_data)
            
            # Save updated data
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)
            
        except Exception as e:
            print(f"Error saving face data: {e}")

    def show_category(self, category, user_id=None):
        """Display menu items for selected category"""
        # Clear current items
        for widget in self.menu_frame.winfo_children():
            widget.destroy()
        
        # Category title
        CTkLabel(
            self.menu_frame,
            text=f"{category} Menu",
            font=self.styles['subheading']['font']
        ).pack(pady=(20,10))
        
        # Filter menu items by category
        category_items = self.menu_data[self.menu_data['Course'] == category]
        
        if category_items.empty:
            CTkLabel(
                self.menu_frame,
                text=f"No items found in {category}",
                font=self.styles['body']['font']
            ).pack(pady=20)
            return
        
        # Create scrollable frame for items
        items_frame = CTkScrollableFrame(self.menu_frame, fg_color="transparent")
        items_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Display items
        for _, item in category_items.iterrows():
            item_frame = CTkFrame(items_frame, fg_color="#333333")
            item_frame.pack(fill="x", padx=10, pady=5)
            
            # Left side - Item details
            details_frame = CTkFrame(item_frame, fg_color="transparent")
            details_frame.pack(side="left", fill="x", expand=True, padx=10, pady=5)
            
            # Item name
            CTkLabel(
                details_frame,
                text=item['Name'],
                font=self.styles['body']['font'],
                text_color="#FFFFFF"
            ).pack(anchor="w")
            
            # Item details
            CTkLabel(
                details_frame,
                text=f"Type: {item['Veg_Non']} | Flavor: {item['Flavour']}",
                font=('Helvetica', 10),
                text_color="#CCCCCC"
            ).pack(anchor="w")
            
            # Right side - Price and Add to Cart
            action_frame = CTkFrame(item_frame, fg_color="transparent")
            action_frame.pack(side="right", padx=10)
            
            # Price
            price = 100  # Replace with actual price from your data if available
            CTkLabel(
                action_frame,
                text=f"₹{price}",
                font=("Helvetica", 18, "bold"),
                text_color="#4CAF50"  # Green color for price
            ).pack(side="left", padx=(0,10))
            
            # Add to cart button
            CTkButton(
                action_frame,
                text="🛒 Add",
                command=lambda i=item: self.add_to_cart(i, user_id) if user_id else self.add_to_guest_cart(i),
                fg_color="#28a745",
                hover_color="#218838",
                width=80,
                height=32,
                corner_radius=16
            ).pack(side="left")
            
            # Optional: Add quantity selector
            # quantity_var = StringVar(value="1")
            # CTkEntry(
            #     action_frame,
            #     textvariable=quantity_var,
            #     width=40,
            #     height=32
            # ).pack(side="left", padx=5)

    def show_recommendations(self, user_id):
        """Show recommendations based on past orders and popular items"""
        # Create recommendations frame
        rec_frame = CTkFrame(self.menu_frame, fg_color="#333333")
        rec_frame.pack(fill="x", padx=10, pady=10)
        
        # Title
        CTkLabel(
            rec_frame,
            text="✨ Recommended for You",
            font=self.styles['subheading']['font']
        ).pack(pady=10)
        
        # Get past orders based recommendations
        past_orders = self.backend.get_user_orders(user_id)
        recommended_items = []
        
        if past_orders:
            # Extract past ordered items
            past_items = []
            for _, items, _ in past_orders:
                past_items.extend(items.split(','))
            
            # Get similar items based on past orders
            for item in past_items:
                item = item.strip()
                # Check if item exists in menu data
                matching_items = self.menu_data[self.menu_data['Name'] == item]
                if not matching_items.empty:
                    item_category = matching_items['Course'].iloc[0]
                    # Get items from same category
                    similar_items = self.menu_data[
                        (self.menu_data['Course'] == item_category) &
                        (self.menu_data['Name'] != item)
                    ]  # Added closing bracket here
                    if not similar_items.empty:
                        recommended_items.extend(similar_items.sample(min(2, len(similar_items)))['Name'].tolist())

    def show_recommendations_page(self, user_id):
        """Show full recommendations page"""
        # Clear current items
        for widget in self.menu_frame.winfo_children():
            widget.destroy()
        
        # Title
        CTkLabel(
            self.menu_frame,
            text="Personalized Recommendations",
            font=self.styles['heading']['font'],
            text_color="#000000"  # Black text
        ).pack(pady=20)
        
        # Get past orders based recommendations
        past_orders = self.backend.get_user_orders(user_id)
        recommended_items = []
        
        if past_orders:
            # Based on past orders
            past_items = []
            for _, items, _ in past_orders:
                past_items.extend(items.split(','))
            
            # Get similar items by category
            for item in past_items:
                item = item.strip()
                if item in self.menu_data['Name'].values:
                    item_category = self.menu_data[self.menu_data['Name'] == item]['Course'].iloc[0]
                    similar_items = self.menu_data[
                        (self.menu_data['Course'] == item_category) &
                        (self.menu_data['Name'] != item)
                    ]
                    if not similar_items.empty:
                        recommended_items.extend(similar_items.sample(min(3, len(similar_items)))['Name'].tolist())
        
        # Add general recommendations if needed
        if len(recommended_items) < 10:
            general_recs = self.backend.get_recommendations(user_id)
            recommended_items.extend(general_recs)
        
        # Remove duplicates and limit
        recommended_items = list(dict.fromkeys(recommended_items))[:10]
        
        if recommended_items:
            for item_name in recommended_items:
                item_data = self.menu_data[self.menu_data['Name'] == item_name].iloc[0]
                item_frame = CTkFrame(self.menu_frame, fg_color="#F5F5F5")
                item_frame.pack(fill="x", padx=10, pady=5)
                
                # Item details
                details_frame = CTkFrame(item_frame, fg_color="transparent")
                details_frame.pack(side="left", fill="x", expand=True, padx=10, pady=5)
                
                CTkLabel(
                    details_frame,
                    text=item_name,
                    font=self.styles['body']['font'],
                    text_color="#000000"  # Black text
                ).pack(anchor="w")
                
                CTkLabel(
                    details_frame,
                    text=f"Type: {item_data['Veg_Non']} | Course: {item_data['Course']}",
                    font=('Helvetica', 10),
                    text_color="#333333"  # Dark gray text
                ).pack(anchor="w")
                
                # Add to cart button
                CTkButton(
                    item_frame,
                    text="Add to Cart",
                    command=lambda i=item_data: self.add_to_cart(i, user_id),
                    fg_color="#4CAF50",
                    hover_color="#45a049",
                    width=100
                ).pack(side="right", padx=10)
        else:
            CTkLabel(
                self.menu_frame,
                text="No recommendations available yet.\nTry ordering some items!",
                font=self.styles['body']['font'],
                text_color="#333333"  # Dark gray text
            ).pack(pady=20)

    def show_menu_layout(self, user_id=None, is_guest=False):
        """Show improved menu layout with past orders at bottom right"""
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Create main container with dark theme
        main_frame = CTkFrame(self.main_container, fg_color="#1A1A1A")
        main_frame.pack(expand=True, fill="both")
        
        # Header section
        header_frame = CTkFrame(main_frame, fg_color="#2D2D2D", height=60)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        # Welcome message
        welcome_text = f"Welcome, {self.backend.get_user_name(user_id)}!" if user_id else "Welcome, Guest!"
        CTkLabel(
            header_frame,
            text=welcome_text,
            font=("Helvetica", 18, "bold"),
            text_color="#FFFFFF"
        ).pack(side="left", padx=10)
        
        # Header buttons (right side)
        buttons_frame = CTkFrame(header_frame, fg_color="transparent")
        buttons_frame.pack(side="right", padx=10)
        
        if user_id:
            # Past Orders button
            CTkButton(
                buttons_frame,
                text="📋 Past Orders",
                command=lambda: self.show_past_orders_window(user_id),
                font=("Helvetica", 12),
                fg_color="#5352ED",
                hover_color="#4241BC",
                width=100,
                height=30,
                corner_radius=15
            ).pack(side="left", padx=5)
        
        # Cart button
        CTkButton(
            buttons_frame,
            text=f"🛒 Cart ({len(self.cart_items)})",
            command=lambda: self.show_cart(user_id) if user_id else self.show_guest_cart(),
            font=("Helvetica", 12),
            fg_color="#FF4757",
            hover_color="#FF6B81",
            width=100,
            height=30,
            corner_radius=15
        ).pack(side="left", padx=5)
        
        # Profile button
        CTkButton(
            buttons_frame,
            text="👤 Profile",
            command=lambda: self.show_user_profile(user_id),
            font=("Helvetica", 12),
            fg_color="#4CAF50",
            hover_color="#45a049",
            width=100,
            height=30,
            corner_radius=15
        ).pack(side="left", padx=5)
        
        # Create main content frame
        content_frame = CTkFrame(main_frame, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Menu section (left side)
        menu_section = CTkFrame(content_frame, fg_color="transparent")
        menu_section.pack(side="left", fill="both", expand=True)
        
        # Category buttons
        categories_frame = CTkFrame(menu_section, fg_color="#2D2D2D")
        categories_frame.pack(fill="x", pady=5)
        
        courses = ["Main Course", "Dessert", "Snacks", "Beverages"]
        course_colors = {
            "Main Course": "#FF4757",
            "Dessert": "#2ED573",
            "Snacks": "#FFA502",
            "Beverages": "#5352ED"
        }
        
        for course in courses:
            CTkButton(
                categories_frame,
                text=course,
                font=("Helvetica", 12),
                fg_color=course_colors.get(course, "#747D8C"),
                hover_color=course_colors.get(course, "#747D8C"),
                width=120,
                height=30,
                corner_radius=15
            ).pack(side="left", padx=5, pady=5)
        
        # Menu items container
        menu_container = CTkScrollableFrame(
            menu_section,
            fg_color="transparent",
            height=600
        )
        menu_container.pack(fill="both", expand=True, pady=5)
        
        # Past Orders section (bottom right)
        if user_id:
            past_orders_frame = CTkFrame(content_frame, fg_color="#2D2D2D", width=300)
            past_orders_frame.pack(side="right", fill="y", padx=(10, 0))
            past_orders_frame.pack_propagate(False)  # Maintain fixed width
            
            # Past Orders Header
            CTkLabel(
                past_orders_frame,
                text="Recent Orders",
                font=("Helvetica", 16, "bold"),
                text_color="#FFFFFF"
            ).pack(pady=10)
            
            # Past orders list
            past_orders_list = CTkScrollableFrame(
                past_orders_frame,
                width=280,
                height=200,  # Reduced height
                fg_color="#333333"
            )
            past_orders_list.pack(fill="both", expand=True, padx=5, pady=5)
            
            try:
                past_orders = self.backend.get_user_orders(user_id)
                if past_orders:
                    for order_id, items, timestamp in past_orders[:5]:  # Show only last 5 orders
                        # Order header
                        order_frame = CTkFrame(past_orders_list, fg_color="#404040")
                        order_frame.pack(fill="x", padx=5, pady=2)
                        
                        # Order ID and Date
                        header_frame = CTkFrame(order_frame, fg_color="transparent")
                        header_frame.pack(fill="x", padx=5, pady=2)
                        
                        CTkLabel(
                            header_frame,
                            text=f"#{order_id}",
                            font=("Helvetica", 12, "bold"),
                            text_color="#FFFFFF"
                        ).pack(side="left")
                        
                        # Safely handle timestamp display
                        try:
                            if isinstance(timestamp, str) and timestamp != "No date":
                                display_date = timestamp.split()[0]  # Get just the date part
                            else:
                                display_date = "No date"
                        except Exception:
                            display_date = "No date"
                        
                        CTkLabel(
                            header_frame,
                            text=display_date, 
                            font=("Helvetica", 10),
                            text_color="#AAAAAA"
                        ).pack(side="right")
                        
                        # Items list frame
                        items_frame = CTkFrame(past_orders_list, fg_color="#404040")
                        items_frame.pack(fill="x", padx=5, pady=1)
                        
                        # Items
                        items_list = [item.strip() for item in items.split(',')]
                        for item in items_list:
                            item_frame = CTkFrame(items_frame, fg_color="transparent")  # Changed from past_orders_list to items_frame
                            item_frame.pack(fill="x", padx=5, pady=1)
                            
                            CTkLabel(
                                item_frame,
                                text=item,
                                font=("Helvetica", 11),
                                text_color="#CCCCCC"
                            ).pack(side="left")
                            
                            CTkButton(
                                item_frame,
                                text="+ Add",
                                command=lambda item=item.strip(): self.add_past_order_item(item, user_id),  # Fixed lambda
                                font=("Helvetica", 10),
                                fg_color="#4CAF50",
                                hover_color="#45a049",
                                width=50,
                                height=24,
                                corner_radius=12
                            ).pack(side="right", padx=5)
                
                    # Reorder entire order button
                    CTkButton(
                        order_frame,
                        text="Reorder All Items",
                        command=lambda items=items_list: self.reorder_all_items(items, user_id, self.winfo_toplevel()),
                        font=("Helvetica", 12),
                        fg_color="#FF4757",
                        hover_color="#FF6B81",
                        width=150,
                        height=32,
                        corner_radius=16
                    ).pack(pady=10)
                
            except Exception as e:
                print(f"Error displaying past orders: {e}")
                CTkLabel(
                    past_orders_list,
                    text="⚠️ Error loading past orders",
                    font=("Helvetica", 16, "bold"),
                    text_color="#FF4757"
                ).pack(pady=10)

        # Add this near where you create the order container
        orders_container = CTkFrame(
            past_orders_frame,
            fg_color="#d9d9d9"
        )
        orders_container.pack(side="left", fill="both", expand=True)
        
        # Add current order display on the right
        current_order_frame = CTkFrame(
            past_orders_frame,
            fg_color="#d9d9d9",
            width=300
        )
        current_order_frame.pack(side="right", fill="y", padx=20, pady=20)
        
        CTkLabel(
            current_order_frame,
            text="Your Current Order",
            font=("Helvetica", 16, "bold"),
            text_color="#333333"
        ).pack(pady=10)
        
        self.order_items_frame = CTkScrollableFrame(
            current_order_frame,
            fg_color="#d9d9d9",
            width=250
        )
        self.order_items_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def toggle_past_orders(self, past_orders_frame):
        """Toggle visibility of past orders dropdown"""
        if past_orders_frame.winfo_viewable():
            past_orders_frame.pack_forget()
        else:
            past_orders_frame.pack(fill="x", padx=10, pady=0)

    def add_past_order_item(self, item_name, user_id):
        """Add item from past order to cart and update current order display"""
        try:
            # Clean up item name and find closest match
            item_name = item_name.strip()
            
            # Try exact match first
            menu_item = self.menu_data[self.menu_data['Name'].str.lower() == item_name.lower()]
            
            if menu_item.empty:
                # If no exact match, try partial match
                menu_item = self.menu_data[self.menu_data['Name'].str.lower().str.contains(item_name.lower())]
            
            if not menu_item.empty:
                item_data = menu_item.iloc[0]
                cart_item = {
                    'Name': item_data['Name'],
                    'Veg_Non': item_data['Veg_Non'],
                    'Course': item_data['Course'],
                    'Price': 100
                }
                
                # Add to cart
                self.cart_items.append(cart_item)
                
                # Update current order display
                self.update_current_order_display(cart_item)
                
                # Refresh the menu to show updated cart
                self.show_menu_for_user(user_id, None)
                
                # Show success message
                CTkMessagebox(
                    title="Success",
                    message=f"Added {item_data['Name']} to cart!",
                    icon="check"
                )
                return True
            
        except Exception as e:
            print(f"Error adding past order item to cart: {e}")
            return False

    def update_current_order_display(self, item):
        """Update the current order display"""
        if not hasattr(self, 'current_order_frame'):
            # Create current order frame if it doesn't exist
            self.current_order_frame = CTkFrame(
                self.main_container,
                fg_color="#d9d9d9",
                corner_radius=15
            )
            self.current_order_frame.pack(side="right", fill="y", padx=20, pady=20)
            
            # Title
            CTkLabel(
                self.current_order_frame,
                text="Your Current Order",
                font=("Helvetica", 16, "bold"),
                text_color="#333333"
            ).pack(pady=10)
            
            # Items container
            self.order_items_frame = CTkScrollableFrame(
                self.current_order_frame,
                fg_color="#d9d9d9",
                width=250
            )
            self.order_items_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Add new item to display
        item_frame = CTkFrame(
            self.order_items_frame,
            fg_color="#ffffff",
            corner_radius=10
        )
        item_frame.pack(fill="x", padx=5, pady=5)
        
        # Item name and price
        CTkLabel(
            item_frame,
            text=item['Name'],
            font=("Helvetica", 12),
            text_color="#333333"
        ).pack(side="left", padx=10, pady=5)
        
        CTkLabel(
            item_frame,
            text=f"₹{item['Price']}",
            font=("Helvetica", 12, "bold"),
            text_color="#4a4a4a"
        ).pack(side="right", padx=10)

    def show_past_orders_window(self, user_id):
        """Show past orders in a separate window"""
        # Create new window
        past_orders_window = CTkToplevel(self)
        past_orders_window.title("Past Orders")
        past_orders_window.geometry("800x700")  # Made wider
        past_orders_window.configure(fg_color="#1A1A1A")
        
        # Make window modal
        past_orders_window.transient(self)
        past_orders_window.grab_set()
        
        # Title with icon
        title_frame = CTkFrame(past_orders_window, fg_color="transparent")
        title_frame.pack(pady=20)
        
        CTkLabel(
            title_frame,
            text="📋 Your Order History",
            font=("Helvetica", 24, "bold"),
            text_color="#FFFFFF"
        ).pack()
        
        # Create scrollable frame for orders
        orders_frame = CTkScrollableFrame(
            past_orders_window,
            fg_color="#2D2D2D",
            width=750,
            height=600
        )
        orders_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        try:
            past_orders = self.backend.get_user_orders(user_id)
            if past_orders:
                for order_id, items, timestamp in past_orders:
                    # Order container with better spacing
                    order_container = CTkFrame(orders_frame, fg_color="#333333", corner_radius=15)
                    order_container.pack(fill="x", padx=10, pady=10)
                    
                    # Order header with better styling
                    header_frame = CTkFrame(order_container, fg_color="#404040", corner_radius=10)
                    header_frame.pack(fill="x", padx=10, pady=10)
                    
                    order_info_frame = CTkFrame(header_frame, fg_color="transparent")
                    order_info_frame.pack(side="left", padx=10)
                    
                    CTkLabel(
                        order_info_frame,
                        text=f"Order #{order_id}",
                        font=("Helvetica", 16, "bold"),
                        text_color="#FFFFFF"
                    ).pack(side="left", padx=(0, 10))
                    
                    # Date display
                    try:
                        display_date = timestamp.split()[0] if isinstance(timestamp, str) else "No date"
                        CTkLabel(
                            order_info_frame,
                            text=f"📅 {display_date}",
                            font=("Helvetica", 12),
                            text_color="#AAAAAA"
                        ).pack(side="left")
                    except:
                        pass
                    
                    # Reorder all button in header
                    CTkButton(
                        header_frame,
                        text="🔄 Reorder All",
                        command=lambda items=items.split(','): self.reorder_all_items(items, user_id, self.winfo_toplevel()),
                        font=("Helvetica", 12, "bold"),
                        fg_color="#FF4757",
                        hover_color="#FF6B81",
                        width=120,
                        height=32,
                        corner_radius=16
                    ).pack(side="right", padx=10)
                    
                    # Items section
                    items_frame = CTkFrame(order_container, fg_color="transparent")
                    items_frame.pack(fill="x", padx=15, pady=10)
                    
                    # Grid layout for items (2 columns)
                    items_list = [item.strip() for item in items.split(',')]
                    for i, item in enumerate(items_list):
                        # Item card
                        item_card = CTkFrame(items_frame, fg_color="#2D2D2D", corner_radius=10)
                        item_card.pack(fill="x", pady=5)
                        
                        # Item info
                        item_info = CTkFrame(item_card, fg_color="transparent")
                        item_info.pack(side="left", fill="x", expand=True, padx=10, pady=5)
                        
                        # Item name with icon based on type
                        item_type = "🥬" if "veg" in item.lower() else "🍖"
                        CTkLabel(
                            item_info,
                            text=f"{item_type} {item}",
                            font=("Helvetica", 14),
                            text_color="#FFFFFF"
                        ).pack(anchor="w")
                        
                        # Price (if available in menu data)
                        try:
                            price = "₹100"  # Replace with actual price lookup
                            CTkLabel(
                                item_info,
                                text=price,
                                font=("Helvetica", 12, "bold"),
                                text_color="#4CAF50"
                            ).pack(side="left", padx=10)
                        except:
                            pass
                        
                        # Add to cart button
                        button_frame = CTkFrame(item_card, fg_color="transparent")
                        button_frame.pack(side="right", padx=10)
                        
                        CTkButton(
                            button_frame,
                            text="🛒 Add to Cart",
                            command=lambda item=item.strip(): self.add_past_order_item_and_notify(item, user_id, past_orders_window),  # Fixed lambda
                            font=("Helvetica", 12),
                            fg_color="#4CAF50",
                            hover_color="#45a049",
                            width=110,
                            height=30,
                            corner_radius=15
                        ).pack(pady=5)
                    
                    # Divider
                    if order_id != past_orders[-1][0]:  # Don't add divider after last order
                        CTkFrame(
                            orders_frame, 
                            fg_color="#404040", 
                            height=2
                        ).pack(fill="x", padx=20, pady=10)
                
            else:
                # Empty state with icon
                CTkLabel(
                    orders_frame,
                    text="🛍️ No orders yet",
                    font=("Helvetica", 18, "bold"),
                    text_color="#AAAAAA"
                ).pack(pady=20)
                
                CTkLabel(
                    orders_frame,
                    text="Your order history will appear here",
                    font=("Helvetica", 14),
                    text_color="#888888"
                ).pack()
                
        except Exception as e:
            print(f"Error showing past orders: {e}")
            CTkLabel(
                orders_frame,
                text="⚠️ Error loading past orders",
                font=("Helvetica", 16, "bold"),
                text_color="#FF4757"
            ).pack(pady=20)

    def add_past_order_item_and_notify(self, item_name, user_id, window):
        """Add item to cart and show notification"""
        self.add_past_order_item(item_name, user_id)
        
        # Check if the window is still open
        if window.winfo_exists():
            # Show success message
            success_label = CTkLabel(
                window,
                text=f"Added {item_name} to cart!",
                font=("Helvetica", 12),
                text_color="#4CAF50"
            )
            success_label.pack(pady=5)
            
            # Remove message after 2 seconds
            window.after(2000, lambda: success_label.destroy())
        else:
            print("The past orders window has been closed.")

    def reorder_all_items(self, items, user_id, window):
        """Add all items from an order to cart"""
        try:
            for item in items:
                self.add_past_order_item(item, user_id)
            
            # Refresh the menu to show all added items
            self.show_menu_for_user(user_id, None)
            
            # Show success message
            CTkLabel(
                window,
                text="All items added to cart!",
                font=("Helvetica", 12),
                text_color="#4CAF50"
            ).pack(pady=5)
            
            # Remove message after 2 seconds
            window.after(2000, lambda label=window.winfo_children()[-1]: label.destroy())
            
        except Exception as e:
            print(f"Error reordering items: {e}")
            CTkMessagebox(
                title="Error",
                message="Could not add all items to cart",
                icon="cancel"
            )

if __name__ == "__main__":
    app = SmartFoodSystem()
    app.mainloop()