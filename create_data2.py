import cv2
import numpy as np
import os
import pickle
import time

# Settings
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
model_file = 'trained_model.yml'
width, height = 130, 100
MAX_ATTEMPTS = 3
RECOGNITION_TIMEOUT = 30  # Overall timeout for recognition in seconds
CONFIDENCE_THRESHOLD = 30  # Recognition threshold (recognize if confidence >= this value)
WAIT_TIME_AFTER_RECOGNITION = 10  # Time to wait for another person in seconds

def capture_images(num_people, images_per_person):
    print('Capturing images for training. Press ESC to stop capturing for each person.')
    (images, labels, names, id) = ([], [], {}, 0)

    for person_num in range(num_people):
        person_name = input(f"\nEnter the name for person {person_num + 1}: ")
        path = os.path.join(datasets, person_name)
        if not os.path.isdir(path):
            os.makedirs(path)

        count = 0
        print(f"Starting capture for {person_name}. Press 'ESC' to stop capturing.")

        while count < images_per_person:
            ret, im = webcam.read()
            if not ret:
                print("Failed to grab frame")
                break

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face = gray[y:y + h, x:x + w]
                    face_resize = cv2.resize(face, (width, height))
                    cv2.imwrite(f'{path}/{count + 1}.png', face_resize)
                    labels.append(id)
                    images.append(face_resize)
                    count += 1
                    print(f"Captured image {count}/{images_per_person} for {person_name}")
                    if count >= images_per_person:
                        break

            cv2.putText(im, f"Capturing: {count}/{images_per_person}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Capture', im)
            
            key = cv2.waitKey(100)
            if key == 27:  # Press 'ESC' to exit
                break

        names[id] = person_name
        id += 1

    print("\nFinished capturing images for all people.")
    return images, labels, names

def train_model(images, labels):
    print('Training model...')
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)
    model.save(model_file)

    # Save names and IDs mapping using pickle
    with open('names.pkl', 'wb') as file:
        pickle.dump(names, file)

    print(f"Model trained and saved to {model_file}")
    print("names.pkl file created.")

def recognize_faces():
    print("\nStarting continuous face recognition. Press ESC to exit.")
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_file)

    with open('names.pkl', 'rb') as file:
        names = pickle.load(file)

    # Dictionary to track recognition status for each face
    face_trackers = {}

    while True:
        ret, im = webcam.read()
        if not ret:
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        current_time = time.time()

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))

            # Create a unique key for this face based on its position
            face_key = f"{x}_{y}_{w}_{h}"
            
            # Initialize tracking for new faces
            if face_key not in face_trackers:
                face_trackers[face_key] = {
                    'start_time': current_time,
                    'attempts': 0,
                    'status': 'attempting',  
                    'name': None,
                    'displayed_message_recognized': False,
                    'displayed_message_unrecognized': False  
                }

            tracker = face_trackers[face_key]
            time_elapsed = current_time - tracker['start_time']

            # Check if already recognized or unrecognized and wait accordingly
            if tracker['status'] == 'recognized':
                if not tracker['displayed_message_recognized']:
                    print(f"{tracker['name']} is recognized! Another person please.")
                    tracker['displayed_message_recognized'] = True
                
                # Wait for the specified time before continuing recognition
                if time_elapsed < WAIT_TIME_AFTER_RECOGNITION:
                    continue
            
            elif tracker['status'] == 'unrecognized':
                if not tracker['displayed_message_unrecognized']:
                    print("Face unrecognized due to timeout. Another person please.")
                    tracker['displayed_message_unrecognized'] = True
                
                # Wait for the specified time before continuing recognition
                if time_elapsed < WAIT_TIME_AFTER_RECOGNITION:
                    continue

            # Attempt recognition if within timeout and under max attempts
            if time_elapsed <= RECOGNITION_TIMEOUT and tracker['attempts'] < MAX_ATTEMPTS:
                try:
                    tracker['attempts'] += 1
                    label, confidence = recognizer.predict(face_resize)
                    
                    cv2.putText(im, f"Confidence: {confidence:.2f}", (x,y-30),
                               cv2.FONT_HERSHEY_SIMPLEX ,0.5,(255 ,255 ,0),2)

                    # Recognize if confidence is greater than or equal to threshold
                    if confidence >= CONFIDENCE_THRESHOLD:  
                        person_name = names.get(label,"Unknown")
                        tracker['status'] = 'recognized'
                        tracker['name'] = person_name

                        # Print recognition message once per recognized event
                        if not tracker['displayed_message_recognized']:
                            print(f"{person_name} is recognized! Another person please.")
                            tracker['displayed_message_recognized'] = True  

                        cv2.putText(im,f"{person_name}",(x,y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX ,0.7,(0 ,255 ,0),2)

                        return True

                    else:
                        # Mark as unrecognized when confidence is below threshold
                        tracker['status'] = 'unrecognized'
                        print("Unknown face detected.")

                except Exception as e:
                    print(f"Error during recognition: {e}")
                    cv2.putText(im,"Error",(x,y-10),
                               cv2.FONT_HERSHEY_SIMPLEX ,0.7,(0 ,0 ,255),2)

            # Mark as unrecognized if we've exceeded the time limit
            elif time_elapsed > RECOGNITION_TIMEOUT:
                tracker['status'] = 'unrecognized'
                print("Face unrecognized due to timeout. Another person please.")
                
                # Display unsuccessful recognition message directly on the frame.
                cv2.putText(im,"Unsuccessful Recognition", (50, im.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX ,1,(0 ,0 ,255),3)
                
                # Show the frame with the message for a brief period.
                cv2.imshow("Recognition", im) 
                cv2.waitKey(3000)  # Show the message for 3 seconds

        cv2.imshow('Recognition', im)
        
        # Clean up old face trackers after WAIT_TIME_AFTER_RECOGNITION has passed.
        current_faces = set(f"{x}_{y}_{w}_{h}" for (x,y,w,h) in faces)
        face_trackers_to_remove = [k for k,v in face_trackers.items() 
                                    if k not in current_faces and 
                                    v['status'] in ['recognized', 'unrecognized']]
        
        for k in face_trackers_to_remove:
            del face_trackers[k]

        key = cv2.waitKey(10)
        if key == 27:  
            break
    
    return False

# Main loop to control detection and recognition flow
webcam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(haar_file)

while True:
    num_people = int(input("Enter the number of people to detect: "))
    images_per_person = int(input("Enter the number of images to capture per person: "))
    
    images, labels, names = capture_images(num_people, images_per_person)

    # Convert lists to numpy arrays for training
    if len(images) > 0:
        (images, labels) = [np.array(lis) for lis in [images, labels]]
        
        train_model(images, labels)

    while True:
        recognized = recognize_faces()

        # Ask user whether they want to detect another person or continue recognizing faces.
        choice = input("Do you want to detect another person? (yes/no): ").strip().lower()
        
        if choice == "yes":
            break   # Break out of inner loop to start detection again.
        
        elif choice == "no":
            continue   # Continue recognizing faces.

print("Exiting program.")
webcam.release()
cv2.destroyAllWindows()
