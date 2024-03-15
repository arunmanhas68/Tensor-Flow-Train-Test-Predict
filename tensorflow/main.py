import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create a Sequential model
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dropout(0.2))

classifier.add(Dense(units=256, activation='relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units=3, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define data augmentation generators
train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
evaluate_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess the training, test, and evaluation datasets
batch_size = 25

train_generator = train_datagen.flow_from_directory(
    'D:/Tensor/train', target_size=(32, 32), batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
    'D:/Tensor/test', target_size=(32, 32), batch_size=batch_size, class_mode='categorical')
evaluate_generator = evaluate_datagen.flow_from_directory(
    'D:/Tensor/evaluate', target_size=(32, 32), batch_size=batch_size, class_mode='categorical')

# Calculate steps_per_epoch and validation_steps based on the number of samples and batch size
steps_per_epoch = len(train_generator)  # Number of batches in the training set
validation_steps = len(test_generator)  # Number of batches in the test set

# Fit the model
epochs = 25

score_fit = classifier.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=validation_steps
)

import joblib

# Save the model as a .pkl file using joblib
joblib.dump(classifier, 'onemodel.pkl')

# Evaluate the model on the evaluation dataset
score = classifier.evaluate(evaluate_generator, verbose=0)

print("Cover 1 : ===============================================================")
print("Cover 2 : Loss:", score[0])
print("Cover 3 : Accuracy:", score[1])
