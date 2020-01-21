import os

import cv2
import csv
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

INPUT_DIM = 64
KERNEL_SIZE = (3, 3)

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(400, input_dim=400, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_cnn():
    model = Sequential()
    model.add(Conv2D(INPUT_DIM, kernel_size=KERNEL_SIZE, activation='relu',\
              input_shape=(INPUT_DIM, INPUT_DIM, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    
    model.add(Dropout(0.5))
    # 50 EPOCHS
    # 87.92% (0.30%)
    '''
    model.add(Conv2D(96, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    # 82.73 % (3.5%)
    # after dropout 79% (5%) 
    
    model.add(Dropout(0.5))
    # 50 epochs 
    model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    '''
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    dirs = [d for d in os.listdir() if (os.path.isdir(d)) and (d[0] != '.')]
    dirs.sort()

    all_images = []
    all_labels = []

    for d in dirs:
        print("working in %s"%d)
        images_dir = [d+"/"+i for i in os.listdir(d) if os.path.isfile(d+"/"+i) and i.endswith(".png")]
        images_dir.sort()
        images = [cv2.imread(imgdir, cv2.IMREAD_GRAYSCALE) for imgdir in images_dir]
        #images = [img.flatten() for img in images]
        images = [cv2.resize(img, (INPUT_DIM, INPUT_DIM), interpolation = cv2.INTER_AREA)\
                  for img in images]
        all_images += images

        print(len(images))
        # first left, then right
        labels = []
        with open(d+"/CollectedData_PrismData_new.csv", 'r') as csv_labels:
            csv_reader = csv.reader(csv_labels, delimiter=',')
            i = 0
            for row in csv_reader:
                if i >= 3:
                    row = list(map(lambda r: '0' if r == '' else r, row))
                    labels.append(row[1:])
                i += 1
        all_labels += labels
    
    all_labels = np.vstack(all_labels)
    all_labels = all_labels.astype(np.float)
    labels_class = (np.sum(all_labels, axis=1) > 0).astype(int)

    images = np.zeros((len(all_images), INPUT_DIM, INPUT_DIM))
    for idx, img in enumerate(all_images):
        images[idx] = img
    print(images.shape)
    print(labels_class.shape)
    
    images = images.reshape(images.shape[0], INPUT_DIM, INPUT_DIM, 1)

    cnn = create_cnn() 
    estimator = KerasClassifier(build_fn=create_cnn, epochs=50, batch_size=10, verbose=1)

    kfold = StratifiedKFold(n_splits=3, shuffle=True)

    results = cross_val_score(estimator, images, labels_class, cv=kfold)

    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    #cnn.fit(images, labels_class, batch_size=10, epochs=50, verbose=1)

    #cnn.predict()

