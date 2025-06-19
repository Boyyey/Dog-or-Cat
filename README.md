# 🐶🐱 Dog or Cat Classifier with Deep Learning & GUI

A beautiful and functional deep learning project that classifies images of dogs and cats using a trained Convolutional Neural Network (CNN) — complete with an elegant **Tkinter GUI** for real-time image predictions.

---

## 🚀 Features

- 🧠 Trained deep CNN using TensorFlow and Keras
- 🖼️ Image preprocessing and augmentation
- 📈 Training/validation accuracy & loss visualization
- 💾 Model saving and loading
- 🖱️ Simple GUI built with Tkinter for image selection
- ✅ Real-time prediction with confidence score

---

## 📸 Demo

https://github.com/your-username/dog-or-cat-classifier/assets/demo.gif *(optional if you have a screen recording or GIF)*

---

## 🛠️ Tech Stack

- Python 3
- TensorFlow / Keras
- NumPy
- PIL (Python Imaging Library)
- Matplotlib
- Tkinter (GUI)
- ImageDataGenerator for preprocessing

---

## 📈 Model Training Overview

- **Data Source**: [Cats and Dogs Dataset](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)
- **Image Size**: 150x150
- **Augmentations**: Rotation, Zoom, Flip, Shift, Shear
- **Architecture**:
  - 3 Convolution + MaxPooling layers
  - Fully connected dense layer
  - Sigmoid activation for binary classification
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Accuracy Achieved**: ~90% on validation

---

## 🧪 How It Works

1. Launch the GUI using:

   ```bash
   python dog_cat_gui.py
     ```
2.Select any image of a dog or cat from your local files.

3.The model instantly predicts and displays the result with a confidence score.

---

**🎯 Future Improvements**

Add drag & drop support for images

Deploy as a web app using Streamlit or Flask

Add more animal classes for multiclass classification

---
Improve model architecture (e.g., with Transfer Learning)
