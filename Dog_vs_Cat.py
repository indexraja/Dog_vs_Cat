# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D #first step of making cnn, in which we add concvlation layer. since we are working on images.
from keras.layers import MaxPooling2D #for max pooling, pooling layers
from keras.layers import Flatten #for flattening in which we convert all the pooled features map that we created through convolution of maxpooling into the large feature vector.
from keras.layers import Dense  #this iis the package we used to add the fully connected layers in a classic ann.
from keras.layers import Dropout #for regularization



import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# Initialising the CNN
classifier = Sequential()
#step 1 - Convolution
classifier.add(Conv2D(64, 3, 3, input_shape = (64,64, 3), activation = 'relu')) #(64,3,3),means 64 feature detectors 3 rows, 3 columns, input_shape is input shape of images. all images are not in same format so we have to make them in one format, our image is colored so it will convert into rgb 3d array

#step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2))) #2*2 matrix 
classifier.add(Dropout( 0.2 ))

#Adding second convolutional layer
classifier.add(Conv2D(64, 3, 3, activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2))) #2*2 matrix 

#Adding third convolutional layer
classifier.add(Conv2D(64, 3, 3, activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2))) #2*2 matrix '''
classifier.add(Dropout( 0.2 ))

#Adding final layer
classifier.add(Conv2D(64, 3, 3, activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2))) #2*2 matrix '''


#step 3 - Flattening
classifier.add(Flatten())

#step 4 -full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))#output_dim parameter = no. of nodes in hidden layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #we used binary_crossentropy loss_function

#part 2 - Fitting the CNN to the images
#applying image augmentation on  our images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/rajak/Desktop/CNN/dataset/training_set', target_size = (64, 64), batch_size = 25, class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:/Users/rajak/Desktop/CNN/dataset/test_set', target_size = (64, 64), batch_size = 25, class_mode = 'binary')

classifier.fit_generator(training_set, steps_per_epoch = 320, epochs = 50, validation_data = test_set, validation_steps = 80 )
   



#part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Users/rajak/Desktop/1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0]== 1:
    prediction = 'Dog'
else:
    prediction = 'Cat'