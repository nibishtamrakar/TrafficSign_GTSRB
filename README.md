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

#### Distribution of the Data
![Bar Graph](/images/dist.png)

# LeNet Convolution Architecture

This project's first Convolutional Neural Network (CNN) is based on a modified LeNet architecture. The network is designed for efficient traffic sign recognition, leveraging convolutional and pooling layers followed by fully connected layers for classification. 

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

# Convolutional Neural Network Architecture

This project employs another Convolutional Neural Network (CNN) designed specifically for the classification of traffic signs in the GTSRB dataset. The architecture is structured to balance complexity and efficiency, ensuring robust feature extraction and classification.

#### Model Summary:
| Layer (Type)                  | Output Shape       | Parameters   |
|-------------------------------|--------------------|--------------|
| `conv2d_4` (Conv2D)           | (None, 28, 28, 32)| 832          |
| `conv2d_5` (Conv2D)           | (None, 24, 24, 32)| 25,632       |
| `max_pooling2d_2` (MaxPooling2D)| (None, 12, 12, 32)| 0            |
| `dropout_2` (Dropout)         | (None, 12, 12, 32)| 0            |
| `conv2d_6` (Conv2D)           | (None, 10, 10, 64)| 18,496       |
| `max_pooling2d_3` (MaxPooling2D)| (None, 5, 5, 64) | 0            |
| `dropout_3` (Dropout)         | (None, 5, 5, 64)  | 0            |
| `flatten_1` (Flatten)         | (None, 1600)      | 0            |
| `dense_2` (Dense)             | (None, 256)       | 409,856      |
| `dropout_4` (Dropout)         | (None, 256)       | 0            |
| `dense_3` (Dense)             | (None, 43)        | 11,051       |

#### Total Parameters:
- **Trainable Parameters**: 465,867 (1.78 MB)
- **Non-trainable Parameters**: 0

#### Key Features:
1. **Convolutional Layers**:
   - The network begins with two Conv2D layers (32 filters), followed by MaxPooling and Dropout layers to extract low-level spatial features.
   - Deeper Conv2D layers (64 filters) further refine the features.

2. **Pooling and Dropout**:
   - MaxPooling layers reduce spatial dimensions to lower computational cost.
   - Dropout layers are applied to regularize the model and prevent overfitting.

3. **Fully Connected Layers**:
   - After flattening, a dense layer with 256 units learns high-level representations.
   - The final dense layer has 43 units corresponding to the number of traffic sign classes.

This architecture is well-suited for traffic sign recognition, balancing performance and computational efficiency.

# Testing on video feed
After the images were tested and predicted, the next step is to test it in video data. In order to do this the model is saved into a “.h5” file using TensorFlow’s model.save() function. Then the model was imported into pyhton file where the code for the software to test the video is written. Using OpenCV’s “cv2.VideoCapture(0)” the live feed from the camera is taken and then is preprocessed. The preprocessing for the video is done the same way as for preprocessing the images for training the model. This is done so that the frames of the video are of the same format as the pre-trained model that is imported into the program. Then the code is run through a loop while it predicts the images and displays the prediction and the probability of the prediction. Anything that is printed into the screen is limited to being above the threshold of 70% probability.

![TEST](/images/Picture1.jpg)
