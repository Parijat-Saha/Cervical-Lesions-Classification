# Cervical-Lesions-Classification
The main goal of the project is to classify the type of cancer in the cervix region of females
from the 3 types of Cervix lesions taken in our medical image data from Kaggle. Here
Machine Learning technique along with Deep Learning CNN models have been
incorporated for identifying the 3 types of cervical disease mentioned in the scope.Both
Machine Learning model Logistic Regression and CNN has been considered for solving
the problem and was found optimal for the problem in hand as CNN shows promising
result in image and video recognition and is in use to solve many real-world medical
problems involving medical image analysis and Logistic Regression can provide
generalized results for complex medical images which are easily interpretable.

Requirement Elaboration
1. Collection of datasets 
Cervical images are collected from Intel & MobileODT Cervical Cancer Screening dataset
available on Kaggle. Kaggle is the largest data science community that offers powerful
tools and resources for data science projects and researchers’ databases [30]. Also, this
database consists of a total of 6734 labeled images of three stages of CC images, including
type 1 cervical intraepithelial neoplasia 1 (CIN1), CIN2 (type 2), and CIN3 (type 3). In this
project, 1191 type_1, 3567 type_2, and 1976 type_3 images are collected from Additional
images, 250 type_1, 781 type_2, and 450 type_3 images are collected from train images,
and 4018 test images are without any label in Intel & MobileODT Cervical Cancer
Screening.
2. Importing dataset 
The Train and Test Data has been directly imported from the system directory so
that all the data can be easily handled and in addition to that additional data for
train and test will also be included for validation training.
3. Preprocessing Dataset 
Prepossessing of dataset includes data cleaning to remove any missing or
corrupted images, ensuring accurate labeling of each image with the
corresponding cervical cancer class, resizing the images to a consistent resolution,
normalizing pixel values for uniformity, and splitting the data into training,
validation, and testing sets. Additionally, data augmentation techniques can be
applied to enhance the diversity of the training data, and label encoding may be
necessary to represent class labels numerically. These preprocessing steps
collectively ensure that the dataset is well-structured, balanced, and ready for
training a deep learning model to accurately classify cervical cancer across
multiple classes.
4.  Data Visualization 
The processed data statistics are necessary to be visualized in graphical format to
decide upon the necessary parameters which are to be set for out deep learning
models. The python-matplotlib library provides us various.
5. Object Detection
Object detection in images locates and categorizes objects. It involves feature
extraction, region proposal, and deep learning with CNNs. The output includes
bounding boxes, class labels, and confidence scores, with applications in
autonomous driving, surveillance, and more.
Object detection using YOLO (You Only Look Once) on the dataset for cervical
cancer multi-class classification involves several steps. YOLO divides each image
into a grid of cells and predicts bounding boxes and class probabilities within these
cells. First, the YOLO model will be trained on the annotated dataset to learn to
identify cervical cancer regions. During inference, the trained model will scan the
test images, identify the cervical cancer regions, and predict the corresponding
class labels (e.g., Type-1, Type-2, Type-3) along with bounding box coordinates.
This enables precise localization and classification of cervical cancer lesions within
the images, facilitating accurate multi-class classification.
6. Data Augmentation
Image augmentation artificially creates training images through different ways of
processing or combination of multiple processing, such as random rotation, shifts, shear
and flips, etc.
7. Creation of Algorithm
The necessary steps and approaches that would be necessary for building our model
are being studied. Once a good understanding of how the models and algorithms
need to work our objective of researching and building a successful algorithm is
realized.
8. Model Training
You'll train your neural network models, like Logistic Regression, ResNet, or
VGG19, using the preprocessed dataset to make accurate predictions regarding
different stages of cervical diseases. This training process helps your models learn
the patterns and features within the data, enabling them to classify cervical
diseases effectively during the testing phase, which is essential for early detection
and diagnosis.
9. Error Calculation
Training of neural networks proceeds by calculating the derivative of the output error with
respect to the input to a layer and the weights of that layer and then propagating the
derivatives back through the preceding layers.
10. Model Comparison
Model comparison in this project involves evaluating the performance of different neural
network models, such as Logistic Regression model, ResNet, and VGG19, to determine
which one achieves the highest accuracy in classifying cervical diseases. This step helps
you select the most effective model for early disease detection, ensuring that your system
provides the most reliable results.
11. Classification of Cervical Disease
Our classification of cervical disease entails using a trained neural network model
to categorize colposcopy images into different classes representing various stages
and types of cervical diseases.
