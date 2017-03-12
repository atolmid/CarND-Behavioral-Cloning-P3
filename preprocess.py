from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import csv
#import pickle
from zodbpickle import pickle

# There is 1 output class
nb_classes = 1 

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# print thenumber of samples
print()
print("Number of training examples: {}".format(len(train_samples)))
print()

print()
print("Number of validation examples {}".format(len(validation_samples)))
print()
# print the dataset shape
print()
print("Training Data Shape: {}".format(np.array(train_samples).shape))
print()

print()
print("Validation Data Shape: {}".format(np.array(validation_samples).shape))
print()

batch_size=100


    
def load_data(samples):
    images = []
    angles = []
    i = 0
    for batch_sample in samples:
        try:
            i+=1
            name = './data/IMG/'+batch_sample[0].split('/')[-1]
            #print("name : ", name)
            center_image = cv2.imread(name)
            center_angle = float(batch_sample[3])
            images.append(center_image)
            angles.append(center_angle)
            name1 = './data/IMG/'+batch_sample[1].split('/')[-1]
            #print("name1 : ", name1)
            left_image = cv2.imread(name1)
            images.append(left_image)
            angles.append(center_angle+0.1)
            name2 = './data/IMG/'+batch_sample[2].split('/')[-1]
            #print("name1 : ", name2)
            right_image = cv2.imread(name2)
            images.append(right_image)
            angles.append(center_angle-0.1)
        except ValueError: 
            print('Line {} is corrupt!'.format(i))
    print("images number : ", len(images))
    print("angles number : ", len(angles))
    return images, angles
    

#==============================================================================
# def load_data(samples):
#     images = []
#     angles = []
#     print("samples length : ", len(samples))
#     i = 0
#     for offset in range(0, len(samples), batch_size):
#         batch_samples = samples[offset:offset+batch_size]
#         print("offset : ", offset)
#         try:
#             i+=1
#             for batch_sample in batch_samples:
#                 name = './data/IMG/'+batch_sample[0].split('/')[-1]
#                 center_image = cv2.imread(name)
#                 center_angle = float(batch_sample[3])
#                 images.append(center_image)
#                 angles.append(center_angle)
#         except ValueError: 
#             print('Line {} is corrupt!'.format(offset+i))
#     for batch_sample in samples[offset: len(samples)]:
#             name = './data/IMG/'+batch_sample[0].split('/')[-1]
#             center_image = cv2.imread(name)
#             center_angle = float(batch_sample[3])
#             images.append(center_image)
#             angles.append(center_angle)
#     print("images number : ", len(images))
#     print("angles number : ", len(angles))
#     return images, angles
#==============================================================================

def flip_data(images, angles):
    augmented_images, augmented_angles  = [], []
    for image, angle in zip(images, angles):
        augmented_images.append(image)
        augmented_angles.append(angle)
        augmented_images.append(cv2.flip(image,1))
        augmented_angles.append(angle*-1.0)
    print("augmented_images number : ", len(augmented_images))
    print("augmented_angles number : ", len(augmented_angles))
    return augmented_images, augmented_angles


train_images, train_angles = load_data(train_samples)
train_images, train_angles = flip_data(train_images, train_angles)
validation_images, validation_angles = load_data(validation_samples)
validation_images, validation_angles = flip_data(validation_images, validation_angles)

train = {'train_dataset': train_images, 'train_labels': train_angles}
validation  = {'valid_dataset': validation_images, 'valid_labels': validation_angles}


## Save the data for later access
train_file = 'train.pickle'
validation_file ='validation.pickle'
stop = False

while not stop:
    if not os.path.isfile(train_file):
        print('Saving data to pickle file...')
        try:
            with open(train_file, 'wb') as f:
              pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
#==============================================================================
#             with open(pickle_file, 'wb') as pfile:
#                 pickle.dump(
#                     {
#                         'train_dataset': train_images,
#                         'train_labels': train_angles,
#                         'valid_dataset': validation_images,
#                         'valid_labels': validation_angles
#                     },
#                     pfile, pickle.HIGHEST_PROTOCOL)
#==============================================================================
        except Exception as e:
            print('Unable to save data to', train_file, ':', e)
            raise

        print('Data cached in pickle file.')
        stop = True
    else:
        print("Please use a different file name")
        pickle_file = input("Enter: ")
        
    if not os.path.isfile(validation_file):
        print('Saving data to pickle file...')
        try:
            with open(validation_file, 'wb') as f:
              pickle.dump(validation, f, pickle.HIGHEST_PROTOCOL)
#==============================================================================
#             with open(pickle_file, 'wb') as pfile:
#                 pickle.dump(
#                     {
#                         'train_dataset': train_images,
#                         'train_labels': train_angles,
#                         'valid_dataset': validation_images,
#                         'valid_labels': validation_angles
#                     },
#                     pfile, pickle.HIGHEST_PROTOCOL)
#==============================================================================
        except Exception as e:
            print('Unable to save data to', validation_file, ':', e)
            raise

        print('Data cached in pickle file.')
        stop = True
    else:
        print("Please use a different file name")
        pickle_file = input("Enter: ")
