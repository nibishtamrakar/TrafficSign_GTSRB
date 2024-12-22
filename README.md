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
Validation Samples: Custom split during preprocessing (e.g., 80/20 split)<br>
The dataset was preprocessed to normalize image pixel values and resize images for input compatibility with the Convolutional Neural Network (CNN). Data augmentation techniques were also applied to enhance model generalization, including rotation, flipping, and scaling.<br>
