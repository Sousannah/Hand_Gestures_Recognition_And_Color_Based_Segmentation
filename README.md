# Hand_Gestures_Recognition_And_Color_Based_Segmentation

This GitHub repository contains code for a Convolutional Neural Network (CNN) model to detect five different hand gestures from color-based segmented hand photos. Additionally, it includes a Real-time Hand Gesture Recognition module using OpenCV. In this updated version, color-based segmentation using blue colored gloves is incorporated for better segmentation accuracy.

![Hand Gesture Detection and Segmentation](https://github.com/Sousannah/Hand_Gestures_Recognition_And_Color_Based_Segmentation/blob/main/color-based-screenshot.png)

## Project Structure

The repository is organized as follows:

1. **Data Preparation:**
   - The dataset is stored in the "data" directory, with subdirectories for each class representing different hand gestures. You can download the dataset from [this Kaggle link](https://www.kaggle.com/datasets/sarjit07/hand-gesture-recog-dataset/data) and extract it into the "data" directory.
  
2. **CNN Model:**
   - The initial model architecture is defined using Keras Sequential API in the main file.
   - The first attempt may show signs of overfitting, so a second model is created with dropout and regularization to address this issue.
   - The third model utilizes a pre-trained ResNet50 model for feature extraction, followed by additional layers for classification.

3. **Training and Evaluation:**
   - The models are trained using the training set and evaluated on the validation and test sets.
   - Training history, loss, and accuracy plots are visualized for each model.
   - Confusion matrices and classification reports provide detailed performance metrics.

4. **Real-time Hand Gesture Recognition:**
   - The "OpenCV_test" file demonstrates real-time hand gesture recognition using OpenCV.
   - It captures video frames from the camera, preprocesses them, and utilizes the trained model to predict gestures.

5. **Dataset Testing:**
   - The "CNN02_Model_Test" file allows testing the trained model on any provided segmented hand photos.
   - Provide the directory path containing the segmented hand photos, and the model will predict the gestures.

## Instructions:

1. **Training:**
   - Run the "CNN_Train" file to train and evaluate the CNN models.
   - Run the "segmentation_ResNet(one real data)" file to Train and evaluate the ResNet model on the data without doing data augmentation
   - Run the "segmentation_ResNet(one augmented data)" file to Train and evaluate the ResNet model on the data after doing data augmentation
   - Experiment with model architectures and hyperparameters to achieve optimal performance.

2. **Real-time Testing:**
   - Execute the "OpenCV_test" file to see real-time hand gesture recognition using OpenCV.

3. **Model Saving:**
   - The trained models are saved in the repository for later use.

## Requirements:

- Python 3.x
- Libraries: TensorFlow, Keras, OpenCV, scikit-learn, matplotlib, seaborn

Feel free to customize and extend the code according to your requirements. For any issues or suggestions, please create an issue in the [repository](https://github.com/Sousannah/hand-gestures-recognition-and-segmentation-using-CNN-and-OpenCV).

**Note: The trained ResNet models perform better than the CNN model**

Happy coding! 🚀
