# Error Checking
try:
    from queue import Queue
    import PIL
    import io
    import os
    import pickle
    import face_recognition
    import numpy as np
    import firebase_admin
    from firebase_admin import credentials
    import cv2
    from firebase_admin import db
    from firebase_admin import storage
    from datetime import datetime, time
    import time as timer
    from tkinter import *
    from tkinter import ttk
    from PIL import Image, ImageTk
    import threading
    from ultralytics import YOLO
    import pyttsx3
except ImportError as e:
    print("Import error: Module", str(e))
except Exception as e:
    print("An error occurred with Module:", str(e))

# Firebase Error Check
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://your_google_firebase_url.com",
        'storageBucket': "yourbucket.com"
    })
    print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"An error occurred during Firebase Admin SDK initialization: {e}")

bucket = storage.bucket()

# Error check + loading encoding file
try:
    # Load the encoding file
    print("Loading Encoding File ...")
    with open('EncodeFile.p', 'rb') as file:
        encodeListKnownWithIds = pickle.load(file)
    encodeListKnownWithIds, employeeIds = encodeListKnownWithIds
    print("Encode File Loaded")
except FileNotFoundError:
    print("The encoding file 'EncodeFile.p' was not found.")
except EOFError:
    print("Error occurred while reading the encoding file. It might be corrupted or empty.")
except Exception as e:
    print(f"An error occurred while loading the encoding file: {e}")

# SETUP #

# set reset time
reset_time_of_day = time(18, 5)  # Reset at 2:00 AM
last_reset_date = None  # Track the last reset date

# Define Classes/Lists
peoplelist = []
detected_classes = []
positive_classes = ["boots", "gloves", "helmet on", "vest"]
negative_classes = ["no boots", "no glove", "no helmet", "no vest"]

# Get base directory of project (to allow for relative file pathing changes instead of from content root)
base_dir = os.path.dirname(__file__)

try:
    # Define Object Detection Model
    model_path = os.path.join(base_dir, "yolo-Weights/best.pt")
    model = YOLO(model_path)
    print("Object Detection Model defined successfully.")
except FileNotFoundError:
    print(f"Model file '{model_path}' not found.")
except Exception as e:
    print(f"An error occurred while defining the Object Detection Model: {e}")

try:
    # Setup of web camera
    vid = cv2.VideoCapture(0)
    vid.set(3, 640)
    vid.set(4, 480)
    print("Web camera setup completed successfully.")
except Exception as e:
    print(f"An error occurred during web camera setup: {e}")
# footnote: if your computer does not have an inbuilt camera, a USB webcam will do the trick

# Initialization of TTS #
engine = pyttsx3.init()


# Initialize a Queue for threading and processing
frameQueue = Queue(maxsize=1)  # maxsize controls the maximum number of items in the queue

#  variables here for speech
helmet_active = False
vest_active = False
gloves_active = False
boots_active = False
voice_active = False



def say_message(message):
    # Convert text to speech
    engine.say(message)
    engine.runAndWait()


def toggle_helmet():
    global active_features
    global helmet_active
    global voice_active
    helmet_active = not helmet_active
    update_button_image(check_button1, helmet_active)
    active_features.append("helmet on") if helmet_active else active_features.remove("helmet on")
    if voice_active is True:
        threading.Thread(target=say_message, args=(f"Helmet Detection has been {'Deactivated' if not helmet_active else 'Activated'}",)).start()


def toggle_vest():
    global active_features
    global vest_active
    global voice_active
    vest_active = not vest_active
    update_button_image(check_button3, vest_active)
    active_features.append("vest") if vest_active else active_features.remove("vest")

    if voice_active is True:
        threading.Thread(target=say_message, args=(f"Vest Detection has been {'Deactivated' if not vest_active else 'Activated'}",)).start()


def toggle_gloves():
    global active_features
    global gloves_active
    global voice_active
    gloves_active = not gloves_active
    update_button_image(check_button2, gloves_active)
    active_features.append("gloves") if gloves_active else active_features.remove("gloves")
    if voice_active is True:
        threading.Thread(target=say_message, args=(f"Gloves Detection has been {'Deactivated' if not gloves_active else 'Activated'}",)).start()


def toggle_boots():
    global active_features
    global boots_active
    global voice_active
    boots_active = not boots_active
    update_button_image(check_button4, boots_active)
    active_features.append("boots") if boots_active else active_features.remove("boots")
    if voice_active is True:
        threading.Thread(target=say_message, args=(f"Boots Detection has been {'Deactivated' if not boots_active else 'Activated'}",)).start()


def toggle_voice():
    global voice_active
    voice_active = not voice_active
    update_button_image_voice(check_button5, voice_active)
    threading.Thread(target=say_message, args=(f"Voice assistant has been {'Deactivated' if not voice_active else 'Activated'}",)).start()


def update_button_image(button, active):
    button.config(image=(static_photo_notrequired if not active else static_photo_required))


def update_button_image_voice(button, active):
    button.config(image=(static_photo_nobutton if not active else static_photo_yesbutton))


def check_all_value():
    root.after(100, check_all_value)


# End of TTS #

# Start of GUI #
root = Tk()
root.title("SMARTGUARD")
root.geometry("1580x720")
root.resizable(False, False)
frm = ttk.Frame(root)  # print the webcam video
frm.grid()

# checking if images can load
try:
    # Define images
    static_image_scanning_path = os.path.join(base_dir, "Processes/scanning.jpeg")
    static_image_background_path = os.path.join(base_dir, "Processes/background.png")
    static_image_gui_path = os.path.join(base_dir, "Processes/faced.jpeg")
    static_image_granted_path = os.path.join(base_dir, "Processes/accessgranted.jpeg")
    static_image_face_path = os.path.join(base_dir, "Processes/noface.png")
    static_image_denied_path = os.path.join(base_dir, "Processes/accessdenied.png")
    static_image_notrequired_path = os.path.join(base_dir, "Processes/notrequired.png")
    static_image_required_path = os.path.join(base_dir, "Processes/required.png")
    static_image_yesbutton_path = os.path.join(base_dir, "Processes/yesbutton.png")
    static_image_nobutton_path = os.path.join(base_dir, "Processes/nobutton.png")
except FileNotFoundError as file_not_found_error:
    print(f"File not found error: {file_not_found_error}")
except Exception as e:
    print(f"An error occurred: {e}")

# Load frame image + position
static_image_background = Image.open(static_image_background_path)
static_photo_background = ImageTk.PhotoImage(static_image_background)
static_image_label = Label(root, image=static_photo_background)
static_image_label.place(x=0, y=0)

# load scanning image + position
static_image = Image.open(static_image_scanning_path)
static_photo = ImageTk.PhotoImage(static_image)
static_image_label = Label(root, image=static_photo)
static_image_label.place(x=1108, y=44)  # Adjust column and row as needed

# Loading GUI image to replace
static_image_gui = Image.open(static_image_gui_path)
static_photo_gui = ImageTk.PhotoImage(static_image_gui)

# Loading access granted image to replace
static_image_granted = Image.open(static_image_granted_path)
static_photo_granted = ImageTk.PhotoImage(static_image_granted)

# Loading Access Denied Image to replace
static_image_denied = Image.open(static_image_denied_path)
static_photo_denied = ImageTk.PhotoImage(static_image_denied)

# Initialize Face image
static_image_face = Image.open(static_image_face_path)
static_photo_face = ImageTk.PhotoImage(static_image_face)
static_image_label_face = Label(root, image=static_photo_face)

# Load Not required button image
static_image_notrequired = Image.open(static_image_notrequired_path)
static_photo_notrequired = ImageTk.PhotoImage(static_image_notrequired)
static_image_notrequired_label = Label(root, image=static_photo_notrequired)

# Load required button image
static_image_required = Image.open(static_image_required_path)
static_photo_required = ImageTk.PhotoImage(static_image_required)
static_image_required_label = Label(root, image=static_photo_required)

# Load voice activated button image
static_image_yesbutton = Image.open(static_image_yesbutton_path)
static_photo_yesbutton = ImageTk.PhotoImage(static_image_yesbutton)
static_image_yesbutton_label = Label(root, image=static_photo_yesbutton)

# Load voice deactivated button image
static_image_nobutton = Image.open(static_image_nobutton_path)
static_photo_nobutton = ImageTk.PhotoImage(static_image_nobutton)
static_image_nobutton_label = Label(root, image=static_photo_nobutton)

# Initialize Employee Label
employeeIdLabel = ttk.Label(root, text="", font=('Helvetica', 12, 'bold'))
employeePositionLabel = ttk.Label(root, text="", font=('Helvetica', 12, 'bold'))

# PPE Option
active_features = []
check_button1 = Button(root, image=static_photo_notrequired, command=toggle_helmet)
check_button1.place(x=125, y=80)
check_button1.image = static_photo_notrequired

check_button2 = Button(root, image=static_photo_notrequired, command=toggle_gloves)
check_button2.place(x=125, y=180)
check_button2.image = static_photo_notrequired

check_button3 = Button(root, image=static_photo_notrequired, command=toggle_vest)
check_button3.place(x=125, y=280)
check_button3.image = static_photo_notrequired

check_button4 = Button(root, image=static_photo_notrequired, command=toggle_boots)
check_button4.place(x=125, y=380)
check_button4.image = static_photo_notrequired

check_button5 = Button(root, image=static_photo_nobutton, command=toggle_voice)
check_button5.place(x=125, y=490)
check_button5.image = static_photo_nobutton

# People Counter Initialization
Counter_Label = ttk.Label(root, text="", font=('Helvetica', 24, 'bold'))
Counter_Label.place(x=150, y=650)  # to check later
Counter_Label.config(text=f"{len(peoplelist)}")

# Create a Frame for the camera feed
camera_frame = ttk.Frame(root)
camera_frame.place(x=355, y=162)

# Create a label widget for the camera feed
label_widget = Label(camera_frame)
label_widget.pack()

# End of GUI #

# Start of User Defined Functions #
def open_camera():
    global active_features
    global detected_classes  # Access the global variable
    _, frame = vid.read()
    if _:
        if not frameQueue.full():
            frameQueue.put(frame)  # Put the frame into the queue

    # Object detection with YOLO
    results = model(frame, stream=True)

    # Coordinates and annotations
    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            # Detected Classes
            detected_classes.append(model.names[cls])  # Append the detected class to the list
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # Custom bbox for each class
            if model.names[cls] in positive_classes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif model.names[cls] in negative_classes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 0, 0)
            thickness = 2

            cv2.putText(frame, model.names[cls], org, font, fontScale, color, thickness)

    # Convert frame to display in the Tkinter window
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)
    label_widget.after(10, open_camera)

    # Face recognition running in the thread background

    def face_recognition_thread():
        while True:
            if not frameQueue.empty() and detected_classes.count("person") == 1:
                frame = frameQueue.get()  # Get the frame from the queue
                # Processing for facial recognition
                imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                faceCurFrame = face_recognition.face_locations(imgS)
                encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

                # Face recognition logic
                if faceCurFrame:
                    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                        matches = face_recognition.compare_faces(encodeListKnownWithIds, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnownWithIds, encodeFace)

                        matchIndex = np.argmin(faceDis)

                        if matches[matchIndex] and employeeIds[matchIndex] not in peoplelist:
                            peoplelist.append(employeeIds[matchIndex])

                            # Retrieve Employee Info
                            employee_IDNo = db.reference(f'Employee/{employeeIds[matchIndex]}/Employee ID').get()
                            employee_Name = db.reference(f'Employee/{employeeIds[matchIndex]}/Name').get()
                            employee_pos = db.reference(f'Employee/{employeeIds[matchIndex]}/Position').get()
                            # Detected known face, print to check, to remove when complete
                            print("Known Face Detected: ", employee_Name)
                            print("employee id", employee_IDNo)
                            blob = bucket.blob(f'Images/{employee_IDNo}.png')  # Ensure correct path
                            image_data = blob.download_as_bytes()  # Download the image data as bytes

                            # Convert the byte data to an image
                            image = PIL.Image.open(io.BytesIO(image_data))
                            photo = ImageTk.PhotoImage(image)

                            # Display the image in the Tkinter label
                            static_image_label_face.config(image=photo)
                            static_image_label_face.place(x=1209, y=175)
                            static_image_label.config(image=static_photo_gui)
                            employeeIdLabel.config(text=f"{employee_IDNo}")
                            employeeIdLabel.place(x=1303, y=480)
                            employeePositionLabel.config(text=f"{employee_pos}")
                            employeePositionLabel.place(x=1303, y=535)
                            engine.say(text=f"Employee Name {employee_Name} Detected")
                            engine.runAndWait()
                            timer.sleep(2)

                            if not active_features or all(item in detected_classes for item in active_features):
                                employeeIdLabel.place_forget()
                                static_image_label_face.place_forget()
                                employeeIdLabel.place_forget()
                                employeePositionLabel.place_forget()
                                static_image_label.config(image=static_photo_granted)
                                engine.say("Access granted")
                                engine.runAndWait()
                                timer.sleep(2)
                                static_image_label.config(image=static_photo)  # re-load scanning
                                Counter_Label.config(text=f"{len(peoplelist)}")
                                frameQueue.task_done()
                            else:
                                employeeIdLabel.place_forget()
                                static_image_label_face.place_forget()
                                employeeIdLabel.place_forget()
                                employeePositionLabel.place_forget()
                                missing_features = [item for item in active_features if item not in detected_classes]
                                static_image_label.config(image=static_photo_denied)
                                timer.sleep(2)
                                engine.say("Access Denied")
                                engine.runAndWait()
                                engine.say(f"PPE {' '.join(map(str, missing_features))} Missing")
                                engine.runAndWait()
                                timer.sleep(2)
                                static_image_label.config(image=static_photo)  # re-load scanning
                                peoplelist.remove(employeeIds[matchIndex])
                                frameQueue.task_done()
                        elif not matches[matchIndex]:
                            employeeIdLabel.place_forget()
                            static_image_label_face.place_forget()
                            employeeIdLabel.place_forget()
                            employeePositionLabel.place_forget()
                            static_image_label.config(image=static_photo_denied)
                            timer.sleep(2)
                            engine.say("Access Denied")
                            engine.runAndWait()
                            engine.say("Unauthorized Personnel Detected")
                            engine.runAndWait()
                            timer.sleep(2)
                            static_image_label.config(image=static_photo)  # re-load scanning
                            frameQueue.task_done()
                frameQueue.task_done()
            if detected_classes.count("person") > 1:
                engine.say("More than 1 person detected, face recognition failed")
                engine.runAndWait()
                frameQueue.task_done()
    face_recognition_thread = threading.Thread(target=face_recognition_thread, daemon=True)
    face_recognition_thread.start()

    detected_classes = []

    # to reset at the end of a selected timing
    def check_and_reset_counter():
        global last_reset_date, peoplelist, encodeListKnownWithIds, employeeIds
        current_time = datetime.now()
        current_date = current_time.date()
        reset_datetime = datetime.combine(current_date, reset_time_of_day)

        # Check if it's time to reset
        if (last_reset_date is None or last_reset_date < current_date) and current_time >= reset_datetime:
            # Reset peoplelist
            peoplelist = []
            Counter_Label.config(text=f"{len(peoplelist)}")

            # Reload the encoding file if it might have been updated
            try:
                print("Reloading Encoding File ...")
                with open('EncodeFile.p', 'rb') as file:
                    encodeListKnownWithIds, employeeIds = pickle.load(file)
                print("Encode File Reloaded")
            except FileNotFoundError:
                print("The encoding file 'EncodeFile.p' was not found.")
            except EOFError:
                print("Error occurred while reading the encoding file. It might be corrupted or empty.")
            except Exception as e:
                print(f"An error occurred while reloading the encoding file: {e}")

            last_reset_date = current_date  # Update the last reset date

    check_and_reset_counter()
# End of User-Defined Functions #


# Main Functions
root.after(100, open_camera)  # Call the open_camera function to open the camera feed by default
root.after(500, check_all_value)  # Schedule the check_all_value function
root.mainloop()  # Tkinter main loop
vid.release()  # Release the VideoCapture object





