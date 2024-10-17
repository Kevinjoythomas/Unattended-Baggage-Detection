import cv2
import math
from ultralytics import YOLO
import random
from tracker import Tracker
import numpy as np
import configparser
import time

model = YOLO('./models/yolov8s.pt') 

parser = configparser.ConfigParser()
parser.read('./config/config.ini')

show_gui = parser['DEFAULT'].getboolean('show_gui')
source = parser['DEFAULT']['source']
size_str = parser['DEFAULT'].get('image_size')
img_sz = tuple(map(int, size_str.split(',')))
distance_threshold = parser['DEFAULT'].getint('distance_from_bag')

print(f"Video source from config: {source}")
print(f"Show gui: {show_gui}")
print(f"Image size: {img_sz}")
print(f"Distance from bag: {distance_threshold}")

# Open the video source
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Tracker
person_tracker = Tracker()
bag_tracker = Tracker()

# Bag indices 
bag_indexes = [24, 26, 28]
distance_threshold = 150  # Define the distance threshold
colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for j in range(30)]
frame_count = 0


# Mapping of bags to their assigned persons
bag_to_person = {}

def calculate_distance(box1, box2):
    # Calculate the center of each box
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    
    distance = math.sqrt((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2)
    return distance

yolo_times = []
deep_sort_times = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video or failed to read frame.")
            break
        frame_count +=1
        if(frame_count%6!=0):
                
                continue
                if show_gui:
                    print("SHOWING WITHOUT INFERENCE")
                    cv2.imshow("YOLO", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        start_time = time.time()
        # print("SHOWED WITH INFERENCE")
    
        # Perform inference on the frame
        yolo_start_time = time.time()
        print("frame size is ",frame.shape[1]," and ",frame.shape[0])
        # frame = cv2.resize(frame, img_sz)
        results = model(frame)
        yolo_end_time = time.time()
        yolo_time = yolo_end_time - yolo_start_time
        yolo_times.append(yolo_time)

        # Separate detections for baggage and persons
        detections_bag = []
        detections_people = []
        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = box
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                class_id = int(class_id)
                if class_id in bag_indexes:
                    detections_bag.append([x1, y1, x2, y2, score])
                elif class_id == 0:
                    detections_people.append([x1, y1, x2, y2, score])

        detections_people = np.array(detections_people)

        deep_sort_start_time = time.time()
        person_tracker.update(frame, detections_people)
        bag_tracker.update(frame,detections_bag)
        deep_sort_end_time = time.time()
        deep_sort_time = deep_sort_end_time - deep_sort_start_time
        deep_sort_times.append(deep_sort_time)

        # for track in person_tracker.tracks:
        #     bbox = track.bbox
        #     track_id = track.track_id
        #     x1, y1, x2, y2 = bbox
        #     x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), colors[int(int(track_id) % len(colors))], 3)
        #     cv2.putText(frame, str(track_id)+": Person", (max(x1, 0), max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[int(int(track_id) % len(colors))], 2)

        # Loop through detected baggage objects
        for bag_box in bag_tracker.tracks:
            track_id = bag_box.track_id
            bag_bbox = bag_box.bbox
            x1, y1, x2, y2 = bag_bbox
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            is_unattended = True
            min_distance = float('inf')
            assigned_person = None

            # Check if the bag is already assigned to a person
            if track_id in bag_to_person:
                assigned_person = bag_to_person[track_id]
                for person_box in person_tracker.tracks:
                    person_id = person_box.track_id
                    person_bbox = person_box.bbox
                    if person_id == bag_to_person[track_id]:
                        distance = calculate_distance(bag_bbox, person_bbox)
                        if distance > distance_threshold:
                                is_unattended = True
                                print("BAG IS UNATTENDED\n\n\n")

                        else:
                                is_unattended = False
                        break
    
            else:
                # Assign the nearest person to the bag
                for person_box in person_tracker.tracks:
                    person_id = person_box.track_id
                    person_bbox = person_box.bbox
                    distance = calculate_distance(bag_bbox, person_bbox)
                    if distance < min_distance:
                        min_distance = distance
                        if min_distance < distance_threshold:
                            assigned_person = person_id
                            is_unattended = False

                if assigned_person:
                    bag_to_person[track_id] = assigned_person

            left, top, right, bottom = map(int, bag_bbox)
            color = (0, 255, 0) if not is_unattended else (0, 0, 255)
            label = 'Baggage' if not is_unattended else 'Unattended'
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (max(left, 0), max(top - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        inference_end_time = time.time()
        print(f"YOLO Iteration time: {yolo_time:.4f} seconds")
        print(f"Deep Sort Iteration time: {deep_sort_time:.4f} seconds")
        print(f"Total Iteration time: {inference_end_time-start_time:.4f} seconds")

        # Resize the frame before displaying

        # Display the frame
        if show_gui:
            frame = cv2.resize(frame, (720,720))
            cv2.imshow("YOLO", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Process interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()

    if yolo_times and deep_sort_times:
        average_yolo_time = sum(yolo_times) / len(yolo_times)
        average_deep_sort_time = sum(deep_sort_times) / len(deep_sort_times)
        print(f"Average YOLO inference time per frame: {average_yolo_time:.4f} seconds")
        print(f"Average Deep SORT processing time per frame: {average_deep_sort_time:.4f} seconds")
