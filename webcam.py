import cv2

# Open the default camera (usually the webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream")
else:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
