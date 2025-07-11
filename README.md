

# 🚦 Traffic Sign Recognition with Convolutional Neural Networks (CNN)

As the development of self-driving cars continues to advance, one of the key technological challenges is enabling vehicles to understand their environment through computer vision. A critical component of this is **traffic sign recognition**—accurately identifying and interpreting road signs such as stop signs, speed limits, and yield signs. This functionality is essential for ensuring that autonomous vehicles can make safe and lawful driving decisions.

## 🧠 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify road signs from images. Using a labeled dataset of traffic sign images, the model learns to recognize visual patterns associated with different types of signs. Once trained, the CNN can accurately classify previously unseen road sign images, contributing to safer autonomous driving systems.

## 📁 Dataset Structure

The dataset consists of images stored in directories structured as follows:

data_dir/
├── 0/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── 1/
│   └── ...
├── ...
└── num_categories - 1/
    └── ...
```

Each directory (0 to `num_categories` - 1) represents a unique traffic sign category and contains images labeled accordingly.

## 🔧 load\_data() Function

The `load_data(data_dir)` function:

* **Loads images** using `cv2` (OpenCV) as NumPy arrays.
* **Resizes** each image to fixed dimensions (`IMG_WIDTH`, `IMG_HEIGHT`) for neural network compatibility.
* **Labels** each image based on its parent folder's category.
* Ensures **cross-platform compatibility** using `os.path.join` for path construction.
* Returns a tuple: `(images, labels)`

```python
(images, labels)
```

* `images`: List of all processed image arrays.
* `labels`: Corresponding list of category integers.

## 🧱 Neural Network Architecture

* **Input shape**: `(IMG_WIDTH, IMG_HEIGHT, 3)` — RGB color images.
* **Output layer**: `num_categories` units with a **softmax** activation function to generate a probability distribution over all classes.
* The internal architecture is customizable, with options to experiment with:

  * Number and size of **Convolutional layers**
  * Number and size of **Pooling layers**
  * Different **hidden layer** configurations
  * Use of **Dropout** for regularization


## 📦 Dependencies

* Python 3.x
* OpenCV (`opencv-python`)
* NumPy
* TensorFlow / Keras
* OS module (built-in)

