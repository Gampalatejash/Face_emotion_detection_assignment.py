
import cv2
from deepface import DeepFace

# Path to the input video file
video_path = "input_video.mp4"  # replace your video path

video_capture = cv2.VideoCapture(video_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while video_capture.isOpened():
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break  # Exit the loop

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around faces and analyze emotions
    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop face for emotion detection
        face = frame[y:y+h, x:x+w]

        try:
            # Use DeepFace for anlysis of emotions
            analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = analysis['dominant_emotion']

            # Display the detected emotion
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        except:
            cv2.putText(frame, "Emotion Not Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # num of people are shown or dectected
    num_people = len(faces)
    cv2.putText(frame, f"People Count: {num_people}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Video Feed", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
