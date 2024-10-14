import cv2
import numpy as np
import datetime

# Function to start video recording
def start_recording(output_filename, width, height, codec, fps):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Function to stop video recording
def stop_recording(video_writer):
    video_writer.release()

# Function to draw rectangle around the moving object and add timestamp
def draw_rectangle(frame, bbox, timestamp):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add timestamp to the frame
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Function to authenticate the user
def authenticate():
    # Hardcoded username and password (replace with secure methods in a real-world scenario)
    correct_username = "user"
    correct_password = "password"

    # Get user input
    entered_username = input("Enter username: ")
    entered_password = input("Enter password: ")

    # Check if the entered credentials are correct
    return entered_username == correct_username and entered_password == correct_password

# Main function
def main(input_source, output_filename, record_no_movement, detection_threshold=25, min_contour_area=500):
    # Perform authentication
    if not authenticate():
        print("Authentication failed. Exiting.")
        return

    # Continue with video processing
    cap = cv2.VideoCapture(input_source)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    codec = 'mp4v'  # You can change the codec as needed

    recording = False
    video_writer = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if not recording or record_no_movement:
            if not recording and record_no_movement:
                # Add timestamp to the output filename
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename_with_time = f"output_{current_time}.mov"
                video_writer = start_recording(output_filename_with_time, width, height, codec, fps)
                recording = True

            # Initialize the background model
            if 'bg_model' not in locals():
                bg_model = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=False)

            fg_mask = bg_model.apply(gray)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    draw_rectangle(frame, (x, y, w, h), timestamp)

                    if recording:
                        video_writer.write(frame)

                    break  # Break on the first detected moving object

        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            draw_rectangle(frame, (0, 0, 0, 0), timestamp)  # Draw an empty rectangle with timestamp
            video_writer.write(frame)

        cv2.imshow('Video', frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # Press 'Esc' to exit
            break

    cap.release()

    if recording:
        stop_recording(video_writer)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_source = 0  # Use 0 for webcam, or provide the path to a video file
    output_filename = "output.mov"
    record_no_movement = True  # Set to False if you don't want to record when there is no movement
    main(input_source, output_filename, record_no_movement)
