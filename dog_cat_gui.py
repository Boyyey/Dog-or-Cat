import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

IMG_HEIGHT = 150
IMG_WIDTH = 150

# Load the trained model
model = tf.keras.models.load_model('dog_cat_classifier.h5')

class_names = ['Dog', 'Cat']  # 0: Dog, 1: Cat

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)[0][0]
    # Debug: print raw prediction value
    print(f"Raw prediction value: {prediction}")
    label = class_names[1] if prediction > 0.5 else class_names[0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"Predicted label: {label}, Confidence: {confidence}")
    return label, confidence

class DogCatApp:
    def __init__(self, master):
        self.master = master
        master.title('Dog or Cat Classifier')
        self.label = Label(master, text='Select an image of a dog or a cat')
        self.label.pack()
        self.image_label = Label(master)
        self.image_label.pack()
        self.result_label = Label(master, text='', font=('Arial', 16))
        self.result_label.pack()
        self.select_button = Button(master, text='Select Image', command=self.load_image)
        self.select_button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[('Image Files', '*.jpg *.jpeg *.png')])
        if file_path:
            img = Image.open(file_path)
            img.thumbnail((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk
            label, confidence = predict_image(file_path)
            self.result_label.config(text=f'Prediction: {label} (Confidence: {confidence:.2f})')

if __name__ == '__main__':
    root = tk.Tk()
    app = DogCatApp(root)
    root.mainloop()
