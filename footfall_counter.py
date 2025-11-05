import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# CONFIGURATION 
VIDEO_PATH = 'test_video.mp4'
MODEL_NAME = 'yolov8n.pt'
LINE_Y_POSITION = 360          # Y-coordinate of the ROI line
CONFIDENCE_THRESHOLD = 0.3     # Detection confidence
PERSON_CLASS_ID = 0            # COCO class 0 is 'person'

def main():
    # 1. Initialize YOLO
    print(f"Loading model {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    # 2. Initialize DeepSort tracker
    print("Initializing DeepSort...")
    # max_age: Max frames to keep a track alive without new detections
    tracker = DeepSort(max_age = 30)

    # 3. Open video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    output_path = 'footfall_output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 4. Initialize counting variables
    entry_count = 0
    exit_count = 0
    # Dictionary to store the last known y-coordinate of each track
    track_history = {}

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 5. Perform detection
        results = model(frame, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD, verbose=False)

        # 6. Format detections for DeepSort
        # DeepSort : [(bbox), confidence, class]
        detections_list = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = box.conf[0]
                cls = int(box.cls[0])
                
                # Bbox : [x1, y1, w, h]
                w = x2 - x1
                h = y2 - y1
                bbox = [x1, y1, w, h]
                detections_list.append((bbox, conf, cls))
        
        # 7. Update tracker
        # tracks : [x1, y1, x2, y2, track_id, class_id, conf]
        tracks = tracker.update_tracks(detections_list, frame=frame)

        # 8. Draw line and process tracks
        cv2.line(frame, (0, LINE_Y_POSITION), (frame_width, LINE_Y_POSITION), (0, 255, 0), 2)

        for track in tracks:
            # Check if track is confirmed
            if not track.is_confirmed():
                continue
                
            x1, y1, x2, y2 = track.to_tlbr() # Get [x1, y1, x2, y2]
            track_id = track.track_id
            
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 9. Counting Logic
            centroid_y = int((y1 + y2) / 2)
            
            if track_id not in track_history:
                track_history[track_id] = centroid_y
            else:
                prev_y = track_history[track_id]
                
                # Crossing from UP to DOWN (Exit)
                if prev_y < LINE_Y_POSITION and centroid_y >= LINE_Y_POSITION:
                    exit_count += 1
                    print(f"ID {track_id}: Exiting (Total Exits: {exit_count})")
                
                # Crossing from DOWN to UP (Entry)
                elif prev_y > LINE_Y_POSITION and centroid_y <= LINE_Y_POSITION:
                    entry_count += 1
                    print(f"ID {track_id}: Entering (Total Entries: {entry_count})")

                track_history[track_id] = centroid_y

        # 10. Display counts on frame
        cv2.putText(frame, f'Entries: {entry_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Exits: {exit_count}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 11. Write frame to output video
        out.write(frame)

    # 12. Cleanup
    print("\nProcessing finished.")
    print(f"Total Entries: {entry_count}")
    print(f"Total Exits: {exit_count}")
    print(f"Output video saved to: {output_path}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()