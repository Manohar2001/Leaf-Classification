#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import time
from tqdm import tqdm
from sklearn.metrics
ort f1_score
from IPython.display import YouTubeVideo
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
print('loaded')


# In[2]:


def process_data(sdir, img_size):
    working_dir = r'./'
    output_dir = os.path.join(working_dir, 'resized_training_data')
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
    train_data, temp_data = train_test_split(dataframe, train_size=.7, shuffle=True, random_state=123, stratify=dataframe['labels'])
    validation_data, test_data = train_test_split(temp_data, train_size=.5, shuffle=True, random_state=123, stratify=temp_data['labels'])
    plant_classes = sorted(train_data['labels'].unique())
    class_count = len(plant_classes)
    print('Number of classes in the processed dataset: ', class_count)
    class_counts = list(train_data['labels'].value_counts())
    print('The maximum number of files in any class in the train data: ', max(class_counts))
    print('The minimum number of files in any class in the train data: ', min(class_counts))
    print('Train data length: ', len(train_data))
    print('Test data length: ', len(test_data))
    print('Validation data length: ', len(validation_data))
    return train_data, test_data, validation_data, plant_classes, class_count

source_directory = r'Data'
image_size = (200, 300)
train_data, test_data, validation_data, plant_classes, class_count = process_data(source_directory, image_size)


# In[3]:


def trim_dataframe(input_df, max_samples, min_samples, label_column):
    df = input_df.copy()
    unique_labels = df[label_column].unique()
    num_classes = len(unique_labels)
    data_length = len(df)
    print('Original dataframe has a length of', data_length, 'with', num_classes, 'classes')
    grouped_data = df.groupby(label_column)
    trimmed_df = pd.DataFrame(columns=df.columns)
    for label in df[label_column].unique():
        group = grouped_data.get_group(label)
        count = len(group)
        if count > max_samples:
            sampled_group = group.sample(n=max_samples, random_state=123, axis=0)
            trimmed_df = pd.concat([trimmed_df, sampled_group], axis=0)
        else:
            if min_samples <= count < max_samples:
                sampled_group = group
                trimmed_df = pd.concat([trimmed_df, sampled_group], axis=0)
    print('After trimming, the maximum samples in any class are now', max_samples, 'and the minimum samples in any class are', min_samples)
    updated_classes = trimmed_df[label_column].unique()
    updated_class_count = len(updated_classes)
    updated_length = len(trimmed_df)
    print('The trimmed dataframe now has a length of', updated_length, 'with', updated_class_count, 'classes')
    return trimmed_df, updated_classes, updated_class_count

max_samples_limit = 100
min_samples_limit = 54
label_column_name = 'labels'
trimmed_train_df, updated_classes, updated_class_count = trim_dataframe(train_data, max_samples_limit, min_samples_limit, label_column_name)


# In[4]:


def balance_dataframe(input_df, target_sample_count, working_directory, image_size):
    df = input_df.copy()
    print('Initial length of the dataframe is', len(df))
    augmentation_dir = os.path.join(working_directory, 'augmentation')
    if os.path.isdir(augmentation_dir):
        shutil.rmtree(augmentation_dir)
    os.mkdir(augmentation_dir)

    for label in df['labels'].unique():
        class_directory = os.path.join(augmentation_dir, label)
        os.mkdir(class_directory)

    total_augmented_images = 0
    image_generator = ImageDataGenerator(
        horizontal_flip=True, rotation_range=20, width_shift_range=.2,
        height_shift_range=.2, zoom_range=.2)
    label_groups = df.groupby('labels')

    for label in df['labels'].unique():
        group = label_groups.get_group(label)
        sample_count = len(group)

        if sample_count < target_sample_count:
            augmented_image_count = 0
            images_to_generate = target_sample_count - sample_count
            target_directory = os.path.join(augmentation_dir, label)
            message = '{0:40s} for class {1:^30s} creating {2:^5s} augmented images'.format(' ', label, str(images_to_generate))
            print(message, '\r', end='')
            augmentation_generator = image_generator.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=image_size,
                                            class_mode=None, batch_size=1, shuffle=False,
                                            save_to_dir=target_directory, save_prefix='aug-', color_mode='rgb',
                                            save_format='jpg')
            while augmented_image_count < images_to_generate:
                images = next(augmentation_generator)
                augmented_image_count += len(images)
            total_augmented_images += augmented_image_count

    print('Total augmented images created:', total_augmented_images)

    augmented_filepaths = []
    augmented_labels = []
    class_list = os.listdir(augmentation_dir)

    for klass in class_list:
        class_path = os.path.join(augmentation_dir, klass)
        file_list = os.listdir(class_path)

        for file in file_list:
            file_path = os.path.join(class_path, file)
            augmented_filepaths.append(file_path)
            augmented_labels.append(klass)

    filepaths_series = pd.Series(augmented_filepaths, name='filepaths')
    labels_series = pd.Series(augmented_labels, name='labels')
    augmented_df = pd.concat([filepaths_series, labels_series], axis=1)
    df = pd.concat([df, augmented_df], axis=0).reset_index(drop=True)

    print('Length of the augmented dataframe is now', len(df))

    return df

target_sample_count = 100
working_directory = r'./'
image_size = (200, 300)
train_data = balance_dataframe(train_data, target_sample_count, working_directory, image_size)


# In[5]:


def create_data_generators(batch_size, training_df, testing_df, validation_df, image_size):
    train_data_generator = ImageDataGenerator()
    test_and_valid_data_generator = ImageDataGenerator()

    training_generator_message = '{0:70s} for training data generator'.format(' ')
    print(training_generator_message, '\r', end='')
    training_data_generator = train_data_generator.flow_from_dataframe(training_df, x_col='filepaths', y_col='labels', target_size=image_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

    validation_generator_message = '{0:70s} for validation data generator'.format(' ')
    print(validation_generator_message, '\r', end='')
    validation_data_generator = test_and_valid_data_generator.flow_from_dataframe(validation_df, x_col='filepaths', y_col='labels', target_size=image_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

    test_data_length = len(testing_df)
    test_data_batch_size = sorted([int(test_data_length / n) for n in range(1, test_data_length + 1) if test_data_length % n == 0 and test_data_length / n <= 80], reverse=True)[0]
    test_data_steps = int(test_data_length / test_data_batch_size)

    test_generator_message = '{0:70s} for test data generator'.format(' ')
    print(test_generator_message, '\r', end='')
    test_data_generator = test_and_valid_data_generator.flow_from_dataframe(testing_df, x_col='filepaths', y_col='labels', target_size=image_size,
                                       class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_data_batch_size)

    class_labels = list(training_data_generator.class_indices.keys())
    class_indices = list(training_data_generator.class_indices.values())
    class_count = len(class_labels)
    test_labels = test_data_generator.labels

    print('Test data batch size:', test_data_batch_size, '  Test data steps:', test_data_steps, ' Number of classes:', class_count)

    return training_data_generator, test_data_generator, validation_data_generator, test_data_batch_size, test_data_steps, class_labels

data_batch_size = 5
train_data_generator, test_data_generator, validation_data_generator, test_data_batch_size, test_data_steps, class_labels = create_data_generators(data_batch_size, train_data, test_data, validation_data, image_size)


# In[6]:


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

display_sample_images(train_data_generator)


# In[7]:


def create_custom_model(image_size, learning_rate, model_number=3):
    image_shape = (image_size[0], image_size[1], 3)

    if model_number == 0:
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max')
        model_message = 'Created EfficientNet B0 model'
    elif model_number == 3:
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max')
        model_message = 'Created EfficientNet B3 model'
    elif model_number == 5:
        base_model = tf.keras.applications.efficientnet.EfficientNetB5(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max')
        model_message = 'Created EfficientNet B5 model'
    else:
        base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, weights="imagenet", input_shape=image_shape, pooling='max')
        model_message = 'Created EfficientNet B7 model'

    base_model.trainable = True
    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
    x = Dropout(rate=0.4, seed=123)(x)
    output_layer = Dense(class_count, activation='softmax')(x)
    custom_model = Model(inputs=base_model.input, outputs=output_layer)
    custom_model.compile(Adamax(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model_message = model_message + f' with initial learning rate set to {learning_rate}'
    print(model_message)
    return custom_model

learning_rate = 0.001
model = create_custom_model(image_size, learning_rate,model_number=3)


# In[8]:


training_history = model.fit(x=train_data_generator, epochs=14, verbose=1,  validation_data=validation_data_generator, validation_steps=None, shuffle=False, initial_epoch=0)


# In[9]:


def plot_training_results(training_data, start_epoch):
    training_accuracy = training_data.history['accuracy']
    training_loss = training_data.history['loss']
    validation_accuracy = training_data.history['val_accuracy']
    validation_loss = training_data.history['val_loss']
    total_epochs = len(training_accuracy) + start_epoch
    epochs_list = []
    
    for i in range(start_epoch, total_epochs):
        epochs_list.append(i + 1)
    
    loss_index = np.argmin(validation_loss)
    lowest_validation_loss = validation_loss[loss_index]
    acc_index = np.argmax(validation_accuracy)
    highest_validation_accuracy = validation_accuracy[acc_index]
    
    plt.style.use('fivethirtyeight')
    scatter_label_loss = 'best epoch = ' + str(loss_index + 1 + start_epoch)
    scatter_label_acc = 'best epoch = ' + str(acc_index + 1 + start_epoch)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
    
    axes[0].plot(epochs_list, training_loss, 'r', label='Training loss')
    axes[0].plot(epochs_list, validation_loss, 'g', label='Validation loss')
    axes[0].scatter(loss_index + 1 + start_epoch, lowest_validation_loss, s=150, c='blue', label=scatter_label_loss)
    axes[0].scatter(epochs_list, training_loss, s=100, c='red')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs', fontsize=18)
    axes[0].set_ylabel('Loss', fontsize=18)
    axes[0].legend()
    
    axes[1].plot(epochs_list, training_accuracy, 'r', label='Training Accuracy')
    axes[1].scatter(epochs_list, training_accuracy, s=100, c='red')
    axes[1].plot(epochs_list, validation_accuracy, 'g', label='Validation Accuracy')
    axes[1].scatter(acc_index + 1 + start_epoch, highest_validation_accuracy, s=150, c='blue', label=scatter_label_acc)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs', fontsize=18)
    axes[1].set_ylabel('Accuracy', fontsize=18)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return loss_index

best_loss_epoch = plot_training_results(training_history, 0)


# In[10]:


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


# In[11]:


task_name = 'plants'
model_filename = f'{task_name}-{weighted_f1_score:5.2f}.h5'
model.save(model_filename)
save_message = f'Model saved as {model_filename}'
print(save_message)


# In[12]:


print(plant_classes)


# In[ ]:




