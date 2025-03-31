import numpy as np
import cv2
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30

class TrafficSignRecognizer:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Traffic Sign Recognition")
        self.model = tf.keras.models.load_model(model_path)
        
        Label(root, text="Upload a Traffic Sign Image", font=("Arial", 14)).pack(pady=10)
        Button(root, text="Select Image", command=self.load_image, font=("Arial", 12), bg="blue", fg="white").pack(pady=10)
        
        self.image_label = Label(root)
        self.image_label.pack(pady=10)
        
        self.prediction_label = Label(root, text="", font=("Arial", 12))
        self.prediction_label.pack(pady=10)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.ppm")])
        if not file_path:
            return
        
        image = cv2.imread(file_path)
        if image is None:
            self.prediction_label.config(text="Error: Could not read the image file.")
            return
        
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        self.predict(image, file_path)
    
    def predict(self, image, file_path):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        predicted_category = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        
        self.image_label.config(image=img)
        self.image_label.image = img
        
        self.prediction_label.config(text=f"Predicted Category: {predicted_category}\nConfidence: {confidence:.2f}%")

if __name__ == "__main__":
    model_file = "best_model.h5"  # Change this if needed
    root = tk.Tk()
    app = TrafficSignRecognizer(root, model_file)
    root.mainloop()
