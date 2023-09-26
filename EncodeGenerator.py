import os
import cv2
import pickle
import face_recognition
import firebase_admin
from firebase_admin import credentials, db, storage

def initialize_firebase():
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)


def load_images_from_folder(folder_path):
    path_list = os.listdir(folder_path)
    img_list = [cv2.imread(os.path.join(folder_path, path)) for path in path_list]
    student_ids = [os.path.splitext(path)[0] for path in path_list]
    return img_list, student_ids

def upload_images_to_firebase(folder_path, path_list):
    bucket = storage.bucket()
    for path in path_list:
        file_name = f'{folder_path}/{path}'
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)

def find_encodings(images_list):
    return [face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0] for img in images_list]

def save_encodings_to_file(encodings, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(encodings, file)

def main():
    initialize_firebase()

    folder_path = 'Images'
    img_list, student_ids = load_images_from_folder(folder_path)

    print(student_ids)

    path_list = os.listdir(folder_path)
    upload_images_to_firebase(folder_path, path_list)

    print("Encoding Started ...")
    encode_list_known = find_encodings(img_list)
    encode_list_known_with_ids = [encode_list_known, student_ids]
    print("Encoding Complete")

    save_encodings_to_file(encode_list_known_with_ids, "EncodeFile.p")
    print("File Saved")

if __name__ == "__main__":
    main()
