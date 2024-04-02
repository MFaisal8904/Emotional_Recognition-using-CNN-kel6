# Import package yang dibutuhkan
import os
import scipy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

# Jalur tempat training data
train_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', 'training'))
val_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', 'validation'))

# Gunakan ImageDataGenerator untuk data augmentation dan preprocessing
train_datagen = ImageDataGenerator(
    rescale=.1/255,
    brightness_range=[0.8, 1.2],
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=.1/255,
    brightness_range=[0.8, 1.2],
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

# Load training data menggunakan metode flow_from_diretory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical'
)

# Load valiadation data menggunakan metode flow_from_diretory
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical'
)

# Layer untuk CNN
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Parameter Kompilasi
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

# Train model
model.fit(train_generator, steps_per_epoch=num_train // batch_size, epochs=num_epoch, validation_data=validation_generator, validation_steps=num_val // batch_size)

# Simpan model secara keseluruhan
model.save('model.h5')