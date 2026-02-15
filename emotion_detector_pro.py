"""
Enhanced Facial Analysis System with Person Detection
Real-time Emotion + Age + Gender + Person Verification
Using YOLOv8-Nano for intelligent person detection
"""

import cv2
from deepface import DeepFace
import numpy as np
from collections import deque
import time
from ultralytics import YOLO

class EnhancedFacialAnalysisSystem:
    def __init__(self):
        """Initialize the enhanced system with person detection"""
        print("="*70)
        print("🎭 ENHANCED FACIAL ANALYSIS WITH PERSON DETECTION")
        print("="*70)
        print("\n📥 Loading AI models...")
        print("⏳ First-time setup: Downloading YOLOv8-Nano (6 MB) + DeepFace models...")
        print("   This may take 2-3 minutes on first run...\n")
        
        # Load YOLO person detector
        print("🔍 Loading YOLOv8-Nano for person detection...")
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Nano version (6 MB)
            print("✅ YOLOv8-Nano loaded successfully!")
        except Exception as e:
            print(f"⚠️  Warning: Could not load YOLO: {e}")
            print("   Falling back to face detection only...")
            self.yolo_model = None
        
        # Emotion labels and colors
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        self.emotion_colors = {
            'angry': (0, 0, 255),
            'disgust': (0, 128, 0),
            'fear': (128, 0, 128),
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'surprise': (0, 255, 255),
            'neutral': (128, 128, 128)
        }
        
        self.emotion_emojis = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'sad': '😢',
            'surprise': '😮',
            'neutral': '😐'
        }
        
        self.gender_icons = {
            'Man': '♂️',
            'Woman': '♀️'
        }
        
        # For smoothing predictions
        self.emotion_history = deque(maxlen=10)
        self.age_history = deque(maxlen=10)
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Analysis cache
        self.last_analysis = None
        self.frames_since_analysis = 0
        
        # Person detection cache
        self.last_person_detected = False
        self.frames_since_person_check = 0
        
        print("✅ Enhanced facial analysis system initialized!")
        print("   Features: Person Detection + Emotion + Age + Gender")
        print("="*70 + "\n")
    
    def detect_person(self, frame):
        """Detect if there's a real person in the frame using YOLO"""
        if self.yolo_model is None:
            return True  # Fallback: assume person present
        
        try:
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False, conf=0.5)
            
            # Check if any person detected (class 0 = person in COCO dataset)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id == 0:  # Person class
                        return True
            
            return False
            
        except Exception as e:
            print(f"⚠️  Person detection error: {e}")
            return True  # Fallback: assume person present
    
    def get_detected_objects(self, frame):
        """Get all detected objects in frame for display"""
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(frame, verbose=False, conf=0.5)
            
            detected_objects = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    
                    detected_objects.append({
                        'name': class_name,
                        'confidence': confidence
                    })
            
            return detected_objects
            
        except Exception as e:
            return []
    
    def analyze_face(self, frame):
        """Analyze face for emotion, age, gender"""
        try:
            result = DeepFace.analyze(
                frame, 
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            analysis = {
                'emotion': result['dominant_emotion'],
                'emotion_scores': result['emotion'],
                'age': int(result['age']),
                'gender': result['dominant_gender'],
                'gender_confidence': result['gender'][result['dominant_gender']]
            }
            
            self.emotion_history.append(analysis['emotion'])
            self.age_history.append(analysis['age'])
            
            return analysis
            
        except Exception as e:
            return {
                'emotion': 'neutral',
                'emotion_scores': {emotion: 0 for emotion in self.emotions},
                'age': 0,
                'gender': 'Unknown',
                'gender_confidence': 0
            }
    
    def get_smoothed_values(self):
        """Get smoothed emotion and age"""
        if not self.emotion_history:
            return 'neutral', 0
        
        emotion = max(set(self.emotion_history), key=self.emotion_history.count)
        age = int(np.mean(list(self.age_history))) if self.age_history else 0
        
        return emotion, age
    
    def draw_enhanced_overlay(self, frame, analysis, face_location, person_detected, detected_objects):
        """Draw comprehensive analysis with person detection info"""
        height, width = frame.shape[:2]
        
        emotion = analysis['emotion']
        age = analysis['age']
        gender = analysis['gender']
        gender_conf = analysis['gender_confidence']
        
        smooth_emotion, smooth_age = self.get_smoothed_values()
        
        # Draw face box if detected and person confirmed
        if face_location is not None and len(face_location) > 0 and person_detected:
            x, y, w, h = face_location
            color = self.emotion_colors.get(smooth_emotion, (255, 255, 255))
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Prepare label
            emoji = self.emotion_emojis.get(smooth_emotion, '😐')
            gender_icon = self.gender_icons.get(gender, '👤')
            
            label_lines = [
                f"{emoji} {smooth_emotion.upper()}",
                f"{gender_icon} {gender} | Age: {smooth_age}",
            ]
            
            label_height = 80
            
            # Background for text
            cv2.rectangle(frame, (x, y-label_height), (x+w, y), color, -1)
            
            # Draw text lines
            y_offset = y - 55
            for line in label_lines:
                cv2.putText(frame, line, (x+5, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
        
        # Enhanced info panel
        panel_width = 350
        panel_x = width - panel_width - 20
        panel_y = 50
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x-15, panel_y-15), 
                     (width-10, height-60),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "ENHANCED ANALYSIS", (panel_x, panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panel_y += 35
        
        # Person Detection Status
        person_color = (0, 255, 0) if person_detected else (0, 0, 255)
        person_status = "PERSON DETECTED" if person_detected else "NO PERSON"
        person_icon = "✅" if person_detected else "❌"
        
        cv2.putText(frame, f"{person_icon} {person_status}", (panel_x, panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, person_color, 2)
        panel_y += 35
        
        # Detected Objects
        if detected_objects:
            cv2.putText(frame, "SCENE OBJECTS:", (panel_x, panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
            panel_y += 25
            
            # Show top 3 objects
            for obj in detected_objects[:3]:
                obj_text = f"  {obj['name'].capitalize()}"
                cv2.putText(frame, obj_text, (panel_x, panel_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                panel_y += 20
            
            panel_y += 10
        
        # Only show facial analysis if person detected
        if person_detected and age > 0:
            # Demographics Section
            cv2.putText(frame, "DEMOGRAPHICS:", (panel_x, panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            panel_y += 30
            
            # Age
            age_text = f"Age: {smooth_age} years"
            cv2.putText(frame, age_text, (panel_x, panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            panel_y += 25
            
            # Gender
            gender_icon = self.gender_icons.get(gender, '👤')
            gender_text = f"{gender_icon} Gender: {gender}"
            cv2.putText(frame, gender_text, (panel_x, panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            panel_y += 25
            
            # Gender confidence bar
            conf_width = int((gender_conf / 100) * 150)
            cv2.rectangle(frame, (panel_x, panel_y), 
                         (panel_x+150, panel_y+12), (50, 50, 50), -1)
            cv2.rectangle(frame, (panel_x, panel_y), 
                         (panel_x+conf_width, panel_y+12), (100, 200, 255), -1)
            cv2.putText(frame, f"{gender_conf:.1f}%", (panel_x+160, panel_y+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            panel_y += 30
            
            # Emotion Section
            cv2.putText(frame, "EMOTIONS:", (panel_x, panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            panel_y += 30
            
            # Top 3 emotions
            emotion_scores = analysis['emotion_scores']
            sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for emo, score in sorted_emotions:
                emoji = self.emotion_emojis.get(emo, '😐')
                text = f"{emoji} {emo.capitalize()}"
                cv2.putText(frame, text, (panel_x, panel_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                bar_length = int((score / 100) * 150)
                color = self.emotion_colors.get(emo, (255, 255, 255))
                
                cv2.rectangle(frame, (panel_x, panel_y+5), 
                             (panel_x+150, panel_y+18), (50, 50, 50), -1)
                cv2.rectangle(frame, (panel_x, panel_y+5), 
                             (panel_x+bar_length, panel_y+18), color, -1)
                
                cv2.putText(frame, f"{score:.0f}%", (panel_x+160, panel_y+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                panel_y += 35
        else:
            # No person detected message
            cv2.putText(frame, "Waiting for person...", (panel_x, panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        
        return frame
    
    def run_webcam(self):
        """Run enhanced real-time analysis with person detection"""
        print("🎥 Starting enhanced webcam system...")
        print("📸 Press 'q' to quit")
        print("📸 Press 's' to take screenshot\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✅ Webcam started successfully!")
        print("🎭 Enhanced analysis with person detection active...\n")
        
        fps_counter = 0
        fps = 0
        start_time = time.time()
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error reading frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Person detection every 15 frames (performance optimization)
            self.frames_since_person_check += 1
            if self.frames_since_person_check >= 15:
                self.last_person_detected = self.detect_person(frame)
                self.frames_since_person_check = 0
            
            person_detected = self.last_person_detected
            
            # Get detected objects (less frequently for performance)
            detected_objects = []
            if fps_counter % 30 == 0:  # Every 30 frames
                detected_objects = self.get_detected_objects(frame)
            
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )
            
            # Facial analysis only if person detected
            self.frames_since_analysis += 1
            if self.frames_since_analysis >= 10 and person_detected:
                analysis = self.analyze_face(frame)
                self.last_analysis = analysis
                self.frames_since_analysis = 0
            elif self.last_analysis is None:
                analysis = self.analyze_face(frame)
                self.last_analysis = analysis
            else:
                analysis = self.last_analysis
            
            # Draw enhanced overlay
            face_location = faces[0] if len(faces) > 0 else None
            frame = self.draw_enhanced_overlay(
                frame, analysis, face_location, person_detected, detected_objects
            )
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - start_time)
                start_time = time.time()
            
            # Title bar
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (20, 20, 20), -1)
            cv2.putText(frame, "Enhanced Facial Analysis + Person Detection", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # FPS counter
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-150, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Instructions
            cv2.putText(frame, "Press 'q' to quit | 's' for screenshot", 
                       (20, frame.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display
            cv2.imshow('Enhanced Facial Analysis - Professional Edition', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"enhanced_analysis_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Webcam stopped")
        print("👋 Goodbye!")

if __name__ == "__main__":
    print("\n🚀 Starting Enhanced Facial Analysis System...")
    print("   With YOLOv8 Person Detection\n")
    
    analyzer = EnhancedFacialAnalysisSystem()
    analyzer.run_webcam()