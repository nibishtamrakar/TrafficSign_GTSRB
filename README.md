# TrafficSign_GTSRB
Traffic Sign Recognition project using Kaggle's GTSRB dataset and a Convolutional Neural Network (CNN) implemented with TensorFlow. This repository includes data preprocessing, model architecture, training, evaluation, and visualization of results.

# Training Set
This project utilizes the German Traffic Sign Recognition Benchmark (GTSRB) dataset, sourced from Kaggle. The GTSRB dataset contains traffic sign images categorized into 43 classes, representing various types of road signs.

Dataset Details:
Source: [GTSRB on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) <br>
Number of Classes: 43<br>
Image Format: RGB<br>
Resolution: Varies (preprocessing includes resizing to a uniform size)<br>
Training Samples: Over 39,000 images<br>
Validation Samples: Custom split during preprocessing (e.g., 80/20 split)

The dataset was preprocessed to normalize image pixel values and resize images for input compatibility with the Convolutional Neural Network (CNN). Data augmentation techniques were also applied to enhance model generalization, including rotation, flipping, and scaling.

### LeNet Convolution Architecture

The Convolutional Neural Network (CNN) used in this project is based on a modified LeNet architecture. The network is designed for efficient traffic sign recognition, leveraging convolutional and pooling layers followed by fully connected layers for classification. 

#### Model Summary:
| Layer (Type)                  | Output Shape       | Parameters   |
|-------------------------------|--------------------|--------------|
| `conv2d` (Conv2D)             | (None, 28, 28, 60)| 1,560        |
| `conv2d_1` (Conv2D)           | (None, 24, 24, 60)| 90,060       |
| `max_pooling2d` (MaxPooling2D)| (None, 12, 12, 60)| 0            |
| `conv2d_2` (Conv2D)           | (None, 10, 10, 30)| 16,230       |
| `conv2d_3` (Conv2D)           | (None, 8, 8, 30)  | 8,130        |
| `max_pooling2d_1` (MaxPooling2D)| (None, 4, 4, 30)| 0            |
| `dropout` (Dropout)           | (None, 4, 4, 30)  | 0            |
| `flatten` (Flatten)           | (None, 480)       | 0            |
| `dense` (Dense)               | (None, 50)        | 24,050       |
| `dropout_1` (Dropout)         | (None, 50)        | 0            |
| `dense_1` (Dense)             | (None, 43)        | 2,193        |

#### Total Parameters:
- **Trainable Parameters**: 142,223 (555.56 KB)
- **Non-trainable Parameters**: 0

#### Key Features:
1. **Convolutional Layers**: Extract spatial features from input images.
2. **Max Pooling Layers**: Downsample feature maps to reduce spatial dimensions.
3. **Dropout**: Regularization to prevent overfitting.
4. **Flatten Layer**: Converts 2D feature maps into a 1D vector for dense layers.
5. **Fully Connected Layers**: Perform the final classification into 43 traffic sign classes.

This architecture is lightweight and well-suited for recognizing traffic signs in the GTSRB dataset.
