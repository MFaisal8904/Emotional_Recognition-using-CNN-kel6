import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load model, face cascade, dan definisikan kategori
model = load_model(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'models', 'model.h5')))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_labels = ['MARAH', 'JIJIK', 'TAKUT', 'SENANG', 'NETRAL', 'SEDIH', 'TERKEJUT']

# Fungsi untuk Preprocess input dan melakukan prediksi
def classify_image(image):
    # Cek apakah gambar merupakan non-grayscale, convert jika bukan
    if len(image.shape) == 3 and image.shape[2] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)  # Tambah dimensi untuk mendefinsikan gambar grayscale
    image = np.expand_dims(image, axis=0)   # Tambah dimensi agar bisa diterima model

    # Predict
    predictions = model.predict(image)
    class_id = np.argmax(predictions[0]) # Bulatkan skor confidence
    predicted_label = emotion_labels[class_id]
    confidence = predictions[0][class_id]

    return f"Class: {class_id}, Confidence: {confidence:.2f}, Label: {predicted_label}"
        
# GUI menggunakan Tkinter
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry('1280x720')
        
        # Webcam frame
        self.video_frame = ttk.Frame(self.window)
        self.video_frame.pack(padx=10, pady=10)

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()

        # Prediction label
        self.prediction_label = ttk.Label(self.window, text="Prediction: ")
        self.prediction_label.pack(pady=10)

        # Tombol upload gambar
        self.upload_btn = ttk.Button(self.window, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=5)

        # Tombol toggle webcam
        self.webcam_running = False # Default webcam mati
        self.toggle_webcam_btn = ttk.Button(self.window, text="Start Webcam", command=self.toggle_webcam)
        self.toggle_webcam_btn.pack(pady=5)

    def start_webcam(self):
        if not self.webcam_running:
            self.webcam_running = True
            self.cap = cv2.VideoCapture(0)  # Nyalakan webcam
            self.update_video()
            self.toggle_webcam_btn.config(text="Stop Webcam")
            print("Webcam started.")
            
    def toggle_webcam(self):
        if self.webcam_running:
            self.stop_webcam()
        else:
            self.start_webcam()
    
    def stop_webcam(self):
        if self.webcam_running:
            self.webcam_running = False
            self.cap.release()
            self.toggle_webcam_btn.config(text="Start Webcam")
            self.video_label.config(image='')
            self.video_label.config(text='Webcam stopped.')
            self.video_label.image = None
            print("Webcam stopped.")
    
    def update_video(self):
        if self.webcam_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Ubah jadi grayscale untuk memudahkan deteksi wajah
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                # Deteksi wajah
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                # Proses wajah yang dideteksi
                for (x, y, w, h) in faces:
                    # Gambar kotak disekitar wajah
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Crop daerah wajah
                    face_area = frame[y:y+h, x:x+w]
                    # Preprocess daerah yang dicrop
                    processed_face = cv2.cvtColor(face_area, cv2.COLOR_RGB2GRAY)
                    processed_face = cv2.resize(processed_face, (48, 48))
                    processed_face = processed_face.astype("float32") / 255.0
                    processed_face = np.expand_dims(processed_face, axis=-1)
                    processed_face = np.expand_dims(processed_face, axis=0)
                    # Predict daerah yang dicrop
                    prediction = model.predict(processed_face)
                    class_id = np.argmax(prediction)
                    predicted_label = emotion_labels[class_id]
                    confidence = prediction[0][class_id]
                    # Tampilkan hasil prediksi disekitar daerah wajah
                    cv2.putText(frame, f"Class: {class_id}, Conf: {confidence:.2f}, Label: {predicted_label}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                self.display_image(frame, self.video_label)

            self.window.after(10, self.update_video)
        else:
            print("Webcam is not running.")

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = ImageOps.exif_transpose(image)  # Penanganan rotasi EXIF untuk gambar JPEG
            image = np.array(image)

            # Tampilkan gambar
            self.display_image(image, self.video_label)

            # Lakukan prediksi dan update label hasil prediksi
            prediction = classify_image(image)
            self.prediction_label.config(text=prediction)

    def display_image(self, img, label):
        # Cek apakah gambar merupakan grayscale
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            # Berikan channel tambahan ke gambar grayscale agar diterima fungsi
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Batasi ukuran maksimum display gambar input
        max_width, max_height = 800, 600
        img_height, img_width = img.shape[:2]
        scaling_factor = min(max_width/img_width, max_height/img_height)
        if scaling_factor < 1:
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk  # Untuk menghindari garbage collection
        label.config(image=imgtk)
        label.pack()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Tampilkan jendela UI
root = tk.Tk()
app = App(root, "Emotion Recognition")
root.mainloop()
