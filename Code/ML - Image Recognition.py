# Convolution Neural Network
# Part 1 - Building CNN Architecture and Import Data

# Importing the Keras libraries and packages
# 'Sequential' library used to Initialize NN as sequence of layers (Alternative to Graph initialization)
from keras.models import Sequential
# 'Conv2D' for 1st step of adding convolution layers to images ('Conv3D' for videos with time as 3rd dimension)
from keras.layers import Conv2D
# 'MaxPooling2D' step 2 for pooling of max values from Convolution Layers
from keras.layers import MaxPooling2D
# 'Flatten' Pooled Layers for step 3
from keras.layers import Flatten
# 'Dense' for fully connected layers that feed into classic ANN
from keras.layers import Dense

# Initializing the CNN
# Calling this object a 'classifier' because that's its job
classifier = Sequential()

# Step 1 - Convolution
# Apply a method 'add' on the object 'classifier'
# Filter = Feature Detector = Feature Kernel
# 'Conv2D' (Number of Filters, (Filter Row, Filter Column), input shape of inputs = (3 color channels, 64x64 -> 256x256 dimension of 2D array in each channel))
# Start with 32 filters, work your way up to 64 -> 128 -> 256
# 'input_shape' needs all picture inputs to be the same shape and format (2D array for B&W, 3D for Color images with each 2D array channel being Blue/Green/Red)
# 'input_shape' parameter shape matters (3,64,64) vs (64,64,3)
# 'Relu' Rectifier Activation Function used to get rid of -ve pixel values and increase non-linearity
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# Reduces the size of the Feature Map by half (eg. 5x5 turns into 3x3 or 8x8 turns into 4x4)
# Preserves Spatial Structure and performance of model while reducing computation time
# 'pool_size' at least needs to be 2x2 to preserve Spatial Structure information (context around individual pixels)
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolution layer to improve performance
# Only need 'input_shape' for Input Layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# Take all the Pooled Feature Maps and put them into one huge single Vector that will input into a classic NN
classifier.add(Flatten())

# Step 4 - Full connection
# Add some fully connected hidden layers (start with a number of Node between input and output layers)
# [Input Nodes(huge) - Output Nodes (2: Cat or Dog)] / 2 = ~128?...
# 'Activation' function makes sure relevant Nodes get a stronger vote or no vote
classifier.add(Dense(units = 128, activation = 'relu'))
# Add final Output Layer with binary options
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
# 'adam' Stochastic Gradient Descent optimizer
# 'loss' function. Logarithmic loss for 2 categories use 'binary_crossentropy' and 'categorical_crossentropy' for more objects
# 'metric' is the a performance metric
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

# Create random transformation from Data to increase Dataset and prevent overfitting
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# 'batch_size' is the number of images that go through the CNN every weight update cycle
# Increase 'target_size' to improve model accuracy 

training_set = train_datagen.flow_from_directory('../Data/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


test_set = test_datagen.flow_from_directory('../Data/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')



# Train the model
# Increase 'epochs' to boost model performance (takes longer)
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 2000)



# Save model to file
# Architecture of the model, allowing to reuse trained models
# Weights of the model
# Training configuration (loss, optimizer)
# State of the optimizer, allowing to resume training exactly where you left off
classifier.save('../Data/saved_model/CNN_Cat_Dog_Model.h5')

# Examine model
classifier.summary()

# Examine Weights
classifier.weights

# Examine Optimizer
classifier.optimizer



# Load saved Model
from keras.models import load_model

model = load_model('../Data/saved_model/CNN_Cat_Dog_Model.h5')




# Part 3 - Making new predictions

# Place a new picture of a cat or dog in 'single_prediction' folder and see if your model works
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('../Data/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# Add a 3rd Color dimension to match Model expectation
test_image = image.img_to_array(test_image)
# Add one more dimension to beginning of image array so 'Predict' function can receive it (corresponds to Batch, even if only one batch)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
# We now need to pull up the mapping between 0/1 and cat/dog
training_set.class_indices
# Map is 2D so check the first row, first column value
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
# Print result

print("The model class indices are:", training_set.class_indices)

print("\nPrediction: " + prediction)


