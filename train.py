import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Input, Lambda, Flatten
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import numpy as np

train_data_path = 'dataset/train/'
validation_data_path = 'dataset/test/'

folders = glob('dataset/train/*')

image_size = [299, 299]
inc = InceptionV3(input_shape = image_size+[3], weights='imagenet', include_top = False)

for layer in inc.layers:
    layer.trainable=False

x = Flatten()(inc.output)
prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=inc.input, outputs=prediction)
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(train_data_path,
                                              target_size=(299, 299),
                                              batch_size=64,
                                              class_mode='categorical')

test_set = test_datagen.flow_from_directory(validation_data_path,
                                              target_size=(299, 299),
                                              batch_size=64,
                                              class_mode='categorical')

run_train = model.fit_generator(train_set,
                                validation_data=test_set,
                                epochs=8,
                                steps_per_epoch=len(train_set),
                                validation_steps=len(test_set)
                                )

model.save('Face_Recognition.h5')
