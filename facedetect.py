import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture frame. Check your webcam.")
            continue

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            try:
                # Resize ROI for faster model inference (DeepFace models typically expect ~224x224)
                if face_roi.size == 0:
                    continue
                small_roi = cv2.resize(face_roi, (224, 224))

                result = DeepFace.analyze(small_roi, actions=['emotion'], enforce_detection=False)

                # DeepFace.analyze may return a dict or a list of dicts depending on version
                if isinstance(result, list) and len(result) > 0:
                    data = result[0]
                elif isinstance(result, dict):
                    data = result
                else:
                    data = {}

                emotion = data.get('dominant_emotion', 'Unknown')

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                # Log and continue (don't crash the whole loop because of one face)
                print("DeepFace error:", e)

        cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    # Allow graceful exit on Ctrl+C
    print('\nInterrupted by user')
finally:
    cap.release()
    cv2.destroyAllWindows()