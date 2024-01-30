#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import cv2
import shutil
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
sns.set_style('darkgrid')


# In[2]:


def process_data(sdir, img_size):
    working_dir = r'./'
    output_dir = os.path.join(working_dir, 'resized_test_data')
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    filepaths = []
    labels = []
    plant_categories = sorted(os.listdir(sdir))
    for plant_category in plant_categories:
        plant_path = os.path.join(sdir, plant_category)
        dst_plant_path = os.path.join(output_dir, plant_category)
        os.mkdir(dst_plant_path)
        class_list = sorted(os.listdir(plant_path))
        for plant_class in class_list:
            class_path = os.path.join(plant_path, plant_class)
            dst_class_path = os.path.join(dst_plant_path, plant_class)
            os.mkdir(dst_class_path)
            file_list = sorted(os.listdir(class_path))
            description = f'{plant_category:25s}-{plant_class:9s}'
            for file in tqdm(file_list, ncols=130, desc=description, unit='files', colour='blue'):
                file_path = os.path.join(class_path, file)
                dst_file_path = os.path.join(dst_class_path, file)
                filepaths.append(dst_file_path)
                img = cv2.imread(file_path)
                img = cv2.resize(img, img_size)
                cv2.imwrite(dst_file_path, img)
                labels.append(plant_category + '-' + plant_class)
    file_series = pd.Series(filepaths, name='filepaths')
    label_series = pd.Series(labels, name='labels')
    dataframe = pd.concat([file_series, label_series], axis=1)
    return dataframe


source_directory = r'test_data'
image_size = (200, 300)
test_data = process_data(source_directory, image_size)


# In[3]:


def create_data_generators(batch_size, testing_df, image_size):
    test_and_valid_data_generator = ImageDataGenerator()

    test_data_length = len(testing_df)
    test_data_batch_size = sorted([int(test_data_length / n) for n in range(1, test_data_length + 1) if test_data_length % n == 0 and test_data_length / n <= 80], reverse=True)[0]
    test_data_steps = int(test_data_length / test_data_batch_size)

    test_generator_message = '{0:70s} for test data generator'.format(' ')
    print(test_generator_message, '\r', end='')
    test_data_generator = test_and_valid_data_generator.flow_from_dataframe(testing_df, x_col='filepaths', y_col='labels', target_size=image_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_data_batch_size)
    return  test_data_generator

data_batch_size = 5
test_data_generator = create_data_generators(data_batch_size, test_data,  image_size)


# In[4]:


model = tf.keras.models.load_model('plants-99.59.h5')


# In[7]:


def display_sample_images(data_generator):
    class_indices_dict = data_generator.class_indices
    class_names = list(class_indices_dict.keys())
    sample_images, sample_labels = next(data_generator)
    plt.figure(figsize=(25, 25))
    num_samples = len(sample_labels)
    if num_samples < 25:
        num_rows = num_samples
    else:
        num_rows = 25
    for i in range(num_rows):
        plt.subplot(5, 5, i + 1)
        sample_image = sample_images[i] / 255
        plt.imshow(sample_image)
        label_index = np.argmax(sample_labels[i])
        class_name = class_names[label_index]
        plt.title(class_name, color='red', fontsize=18)
        plt.axis('off')
    plt.show()

display_sample_images(test_data_generator)


# In[5]:


def evaluate_model_performance(test_data_generator):
    predicted_labels = []
    error_samples = []
    error_predictions = []
    true_labels = test_data_generator.labels
    class_names = list(test_data_generator.class_indices.keys())
    class_count = len(class_names)
    error_count = 0
    predictions = model.predict(test_data_generator, verbose=1)

    print(predictions)
    total_test_samples = len(predictions)

    for i, prediction in enumerate(predictions):
        file_path = test_data_generator.filenames[i]
        predicted_index = np.argmax(prediction)
        true_index = test_data_generator.labels[i]

        if predicted_index != true_index:
            error_count += 1
            file_path = test_data_generator.filenames[i]
            error_class = class_names[predicted_index]
            error_tuple = (file_path, error_class)
            error_samples.append(error_tuple)

        predicted_labels.append(predicted_index)

    accuracy = (1 - error_count / total_test_samples) * 100
    accuracy_message = f'There were {error_count} errors in {total_test_samples} tests for an accuracy of {accuracy:6.2f}'
    print(accuracy_message)

    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    weighted_f1_score = f1_score(true_labels, predicted_labels, average='weighted') * 100

    if class_count <= 30:
        confusion_matrix_data = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(12, 8))
        sns.heatmap(confusion_matrix_data, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(class_count) + 0.5, class_names, rotation=90)
        plt.yticks(np.arange(class_count) + 0.5, class_names, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    classification_report_data = classification_report(true_labels, predicted_labels, target_names=class_names, digits=4)
    print("Classification Report:\n----------------------\n", classification_report_data)

    return error_count, total_test_samples, error_samples, weighted_f1_score

error_count, total_test_samples, error_samples, weighted_f1_score = evaluate_model_performance(test_data_generator)


# In[6]:


model.summary()


# In[ ]:




