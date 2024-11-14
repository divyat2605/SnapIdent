**SnapIdent: Face Detection and Image Capture Project
**SnapIdent is a simple yet effective project that demonstrates real-time face detection and image capturing using Python and OpenCV. The system uses a webcam to detect faces in real-time and captures images, which are then saved into a dedicated folder for further processing or training in face recognition systems.

Project Overview
SnapIdent captures and stores images of faces detected through a webcam feed. The project creates a database of face images stored in a specific directory for future use in face recognition systems or other AI applications. The images are resized for consistency, and the program automatically organizes them into a dataset folder under a user-specific subfolder.

Features
Real-time Face Detection: Detects faces in real-time using the webcam feed.
Automatic Image Capture: Captures and saves images of detected faces.
Organized Data Storage: Saves images in a structured folder under datasets/{subfolder_name}.
Customizable Subfolder Name: The name of the subfolder can be customized (default: 'divi').
Simple and Lightweight: Uses OpenCV for efficient face detection and webcam interaction.
Requirements
Python 3.x
OpenCV
NumPy
Install the required libraries using:
pip install numpy
pip install opencv
How to Run the Project
Clone the repository:
git clone https://github.com/your-username/SnapIdent.git
cd SnapIdent
Ensure that the haarcascade_frontalface_default.xml file is in the same directory as the script, or update the path accordingly.
Run the script
python face_detection.py
The program will activate your webcam and start detecting faces. It will capture 30 images of detected faces and save them in project/datasets/divi/ by default.
Press ESC to stop the program and close the webcam feed.
Future Enhancements
Data Augmentation: Incorporate techniques like rotation, flipping, and scaling to diversify the dataset and improve recognition accuracy.
Face Recognition: Implement a machine learning model to identify and recognize faces from the captured dataset.
Real-time Face Recognition: Enhance the project by adding real-time face recognition functionality to identify users on the fly.
User Interface: Develop a simple GUI for controlling the application and viewing captured images.
Database Integration: Store captured images and associated data (e.g., names, timestamps) in a database for better organization and management.
