import cv2  
import mediapipe as mp
import numpy as np
import time
import webbrowser
import winsound  # Beep sound for Windows
import pyttsx3  # AI Voice Alert

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

def speak_alert():
    engine.say("Drowsiness Alert! Wake up!")
    engine.runAndWait()

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indices
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(eye_landmarks, frame_width, frame_height):
    p1 = np.array([eye_landmarks[0].x * frame_width, eye_landmarks[0].y * frame_height])
    p2 = np.array([eye_landmarks[1].x * frame_width, eye_landmarks[1].y * frame_height])
    p3 = np.array([eye_landmarks[2].x * frame_width, eye_landmarks[2].y * frame_height])
    p4 = np.array([eye_landmarks[3].x * frame_width, eye_landmarks[3].y * frame_height])
    p5 = np.array([eye_landmarks[4].x * frame_width, eye_landmarks[4].y * frame_height])
    p6 = np.array([eye_landmarks[5].x * frame_width, eye_landmarks[5].y * frame_height])

    # EAR Formula
    ear = (np.linalg.norm(p2 - p3) + np.linalg.norm(p5 - p6)) / (2.0 * np.linalg.norm(p1 - p4))
    return ear

# Initialize webcam
cap = cv2.VideoCapture(0)
drowsy_time = None  

# Spotify song link (Replace with your preferred song)
most_played_song_url = "https://open.spotify.com/track/6YcBBZVKYYRirfeAt6BeGA?si=3872de83487247fb"  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE_LANDMARKS]
            right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE_LANDMARKS]

            left_ear = calculate_ear(left_eye, frame_width, frame_height)
            right_ear = calculate_ear(right_eye, frame_width, frame_height)
            avg_ear = (left_ear + right_ear) / 2.0

            # **DRAW GREEN DOTS ON EYES**
            for landmark in left_eye + right_eye:
                x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Green dots on eye landmarks

            # Drowsiness detection (Threshold = 0.38)
            if avg_ear < 0.38:  
                if drowsy_time is None:
                    drowsy_time = time.time()  # Start timing
                elif time.time() - drowsy_time > 2:  # If closed for > 2 seconds
                    print("\n🚨 DROWSINESS ALERT! WAKE UP! 🚨\n sound beep on\n playing music \n")  # Alert in terminal
                    
                    # Beep sound (1000 Hz, 700 ms)
                    winsound.Beep(1000, 700)
                    
                    # AI Voice Alert
                    speak_alert()
                    
                    # Open and play the Spotify song
                    webbrowser.open(most_played_song_url)
                    time.sleep(2)  # Small delay to prevent multiple alerts
                    drowsy_time = None  # Reset immediately so it can detect again
            else:
                drowsy_time = None  # Reset timer if eyes open

    # Show video feed
    cv2.imshow("Drowsiness Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
