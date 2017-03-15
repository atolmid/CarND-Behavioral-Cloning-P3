from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import csv
#import pickle
# using zodbpickle, since normal pickle has an issue with large files
from zodbpickle import pickle

# There is 1 output class
nb_classes = 1 

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# print the dataset shape
print()
print("Training Data Shape: {}".format(np.array(train_samples).shape))
print()

print()
print("Validation Data Shape: {}".format(np.array(validation_samples).shape))
print()

batch_size=100
correction = 0.25


# using functions random_shear and random_rotate of github user hangyao, 
# which I found googling additional transforms I could include
def random_shear(image, angle, shear_dist=50):
    rows, cols, _ = image.shape
    d = np.random.randint(-shear_dist, shear_dist+1)
    pt_1 = np.float32([[0, rows], [cols, rows], [cols/2, rows/2]])
    pt_2 = np.float32([[0, rows], [cols, rows], [cols/2+d, rows/2]])
    dsteer = d / (rows/2) * .4875
    M = cv2.getAffineTransform(pt_1, pt_2)
    image = cv2.warpAffine(image, M, (cols,rows), borderMode=1)
    angle += dsteer
    return image, angle


def random_rotate(image, angle, angle_range=5):
    dangle = np.random.uniform(-angle_range, angle_range)
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),dangle,1)
    image = cv2.warpAffine(image, M, (cols,rows), borderMode=1)
    dsteer = - dangle * np.pi / 180 * .4875
    angle += dsteer
    return image, angle

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


def load_data(samples):
    images = []
    angles = []
    i = 0
    for batch_sample in samples:
        try:
            i+=1
            name = './data/IMG/'+batch_sample[0].split('/')[-1]
            center_image = cv2.imread(name)
            center_angle = float(batch_sample[3])
            images.append(center_image)
            angles.append(center_angle)
            name1 = './data/IMG/'+batch_sample[1].split('/')[-1]
            left_image = cv2.imread(name1)
            images.append(left_image)
            angles.append(center_angle+correction)
            name2 = './data/IMG/'+batch_sample[2].split('/')[-1]
            right_image = cv2.imread(name2)
            images.append(right_image)
            angles.append(center_angle-correction)
        except ValueError: 
            print('Line {} is corrupt!'.format(i))
    return images, angles
    


def flip_data(images, angles):
    augmented_images, augmented_angles  = [], []
    for image, angle in zip(images, angles):
        augmented_images.append(image)
        augmented_angles.append(angle)
        augmented_images.append(cv2.flip(image,1))
        augmented_angles.append(angle*-1.0)
    return augmented_images, augmented_angles

def augmentation(images, angles):
    augmented_images, augmented_angles  = [], []
    for image, angle in zip(images, angles):
        augmented_images.append(image)
        augmented_angles.append(angle)
        augmented_images.append(add_random_shadow(image))
        augmented_angles.append(angle)
        augmented_images.append(augment_brightness_camera_images(image))
        augmented_angles.append(angle)
        image_tr, angle_tr = random_rotate(image, angle, angle_range=5)
        augmented_images.append(image_tr)
        augmented_angles.append(angle_tr)
        image_tr, angle_tr = random_shear(image, angle, shear_dist=50)
        augmented_images.append(image_tr)
        augmented_angles.append(angle_tr)
    return augmented_images, augmented_angles



train_images, train_angles = load_data(train_samples)
# print the dataset shape
print()
print("Training Data Shape: {}".format(np.array(train_images).shape))
print()

train_images, train_angles = flip_data(train_images, train_angles)
# print the dataset shape
print()
print("Training Data Shape: {}".format(np.array(train_images).shape))
print()


train_images, train_angles = augmentation(train_images, train_angles)
# print the dataset shape
print()
print("Training Data Shape: {}".format(np.array(train_images).shape))
print()

train = {'train_dataset': train_images, 'train_labels': train_angles}

del train_images
del train_angles

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
        except Exception as e:
            print('Unable to save data to', train_file, ':', e)
            raise

        print('Data cached in pickle file.')
        stop = True
    else:
        print("Please use a different file name")
        pickle_file = input("Enter: ")
        

del train

validation_images, validation_angles = load_data(validation_samples)

validation  = {'valid_dataset': validation_images, 'valid_labels': validation_angles}

del validation_images
del validation_angles

## Save the data for later access
train_file = 'train.pickle'
validation_file ='validation.pickle'
stop = False

while not stop:      
    if not os.path.isfile(validation_file):
        print('Saving data to pickle file...')
        try:
            with open(validation_file, 'wb') as f:
              pickle.dump(validation, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', validation_file, ':', e)
            raise

        print('Data cached in pickle file.')
        stop = True
    else:
        print("Please use a different file name")
        pickle_file = input("Enter: ")
        

del validation
