import cv2
import face_recognition
import os
import pickle
import firebase_admin
from firebase_admin import credentials, storage

# this file uploads all saved images in the "Images" file to the firebase automatically when loaded
# after running delete all images file

def initialize_firebase():
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://your_google_firebase.com/",
        'storageBucket': "your_bucket.com"
    })


def upload_images_to_firebase(folder_path):
    bucket = storage.bucket()
    image_paths = os.listdir(folder_path)
    image_list = []
    employee_ids = []

    for path in image_paths:
        try:
            full_path = os.path.join(folder_path, path)
            img = cv2.imread(full_path)
            if img is not None:
                image_list.append(img)
                employee_ids.append(os.path.splitext(path)[0])

                blob = bucket.blob(full_path)
                blob.upload_from_filename(full_path)
            else:
                print(f"Error reading image {path}")
        except Exception as e:
            print(f"Error processing {path}: {e}")

    return image_list, employee_ids


def find_encodings(images):
    encode_list = []
    for img in images:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)
            if encode:
                encode_list.append(encode[0])
            else:
                print("No faces found in the image.")
        except Exception as e:
            print(f"Error encoding image: {e}")

    return encode_list


def save_encodings_with_ids(encode_list, ids):
    with open("EncodeFile.p", 'wb') as file:
        pickle.dump([encode_list, ids], file)


if __name__ == "__main__":
    initialize_firebase()
    folder_path = 'Images/'
    img_list, employee_ids = upload_images_to_firebase(folder_path)
    print(employee_ids)

    print("Encoding Started ...")
    encode_list_known = find_encodings(img_list)
    print("Encoding Complete")

    save_encodings_with_ids(encode_list_known, employee_ids)
    print("File Saved")

