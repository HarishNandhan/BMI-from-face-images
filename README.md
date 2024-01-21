# BMI Estimation from Facial Images 📷💪

## Overview
This project, developed during the sixth semester's 'Mini Project' course, tackles the challenge of estimating Body Mass Index (BMI) from facial images. Leveraging the Polk County Prison dataset, public details of prisoners were used for training, focusing on height, weight, gender, etc.

## Face Detection and Model Training
For face detection, the Multi-Task Cascaded Convolution Neural Network (MTCNN) with a five-point technique was employed. Distances between facial features were calculated and stored for model training. Utilizing transfer learning with pretrained models (ResNet, VGG16) in TensorFlow, the VGG16 model excelled after hyperparameter tuning. The trained model, saved as a '.h5' file, accurately detects faces and outputs age, gender, and BMI.

## 🏗️Project Structure

  - ![Architecture Diagram 1](https://github.com/HarishNandhan/BMI-from-face-images/blob/main/img/images_Github_readme/architecture1.jpg)
  - ![Architecture Diagram 2](https://github.com/HarishNandhan/BMI-from-face-images/blob/main/img/images_Github_readme/architecture2.jpg)

## 📊Exploratory Data Analysis

  - ![EDA Image 1](https://github.com/HarishNandhan/BMI-from-face-images/blob/main/img/images_Github_readme/eda1.jpg)
  - ![EDA Image 2](https://github.com/HarishNandhan/BMI-from-face-images/blob/main/img/images_Github_readme/eda2.jpg)

## 🤖Face Detection and Model Evaluation

  - ![5-Point Face Detection Image](https://github.com/HarishNandhan/BMI-from-face-images/blob/main/img/images_Github_readme/facedetection.jpg)
  - ![Tensorboard Evaluation of Models](https://github.com/HarishNandhan/BMI-from-face-images/blob/main/img/images_Github_readme/tensorboard_results.jpg)

## 📷Final Output

  - ![Screenshot 1](https://github.com/HarishNandhan/BMI-from-face-images/blob/main/img/images_Github_readme/output1.jpg)
  - ![Screenshot 2](https://github.com/HarishNandhan/BMI-from-face-images/blob/main/img/images_Github_readme/output2.jpg)

## Project Report
For a detailed understanding, refer to the [Project Report](https://github.com/HarishNandhan/BMI-from-face-images/blob/main/FACIAL%20BMI.docx.pdf) attached to the GitHub repository.

Feel free to explore the project and contribute! If you have any questions or suggestions, please open an [issue](https://github.com/HarishNandhan/BMI-from-face-images/issues).
