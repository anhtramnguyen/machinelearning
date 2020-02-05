# Visualizing intermediate activations in Convolutional Neural Networks with Keras
import glob
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio as im
import sys, getopt
from tensorflow.keras import models
from tensorflow.keras import optimizers
#from keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

#path to data sets
train_path = 'images/train'
validation_path = 'images/validation'
test_path = 'images/test'

def display_sample_training_images():
    #Display some of our training images:
    columns = 5
    rows = 5
    sample_size = 10
    images = []
    for img_path in glob.glob(train_path + '/HIGH/*.jpg'):
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=(20,10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.title('Sample training images')
    columns = 5
    rows = 5
    for i, image in enumerate(images[:sample_size]):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image)
    plt.show()
    #display more images
    images = []
    for img_path in glob.glob(train_path + '/LOW/*.jpg'):
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=(20,10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.title('Sample validation images')
    for i, image in enumerate(images[:sample_size]):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image)
    plt.show()

def read_image(path):
    img = kimage.load_img(path, target_size=(128, 128))
    tmp = kimage.img_to_array(img)
    tmp = np.expand_dims(tmp, axis=0)
    return tmp

def read_and_stack_image(path):
    img = kimage.load_img(path, target_size=(128, 128))
    tmp = kimage.img_to_array(img)
    tmp = np.expand_dims(tmp, axis=0)
    images = np.vstack([tmp])
    return images

def read_image_fortensor(path):
    img = kimage.load_img(path, target_size=(128, 128))
    tmp = kimage.img_to_array(img)
    tmp = np.expand_dims(tmp, axis=0)
    tmp /= 255.
    return tmp

def predict_an_image(file, count, classifier):
    class_labels = {0:'HIGH', 1: 'LOW'}
    image_tensor = read_image_fortensor(file)
    image_stack = read_and_stack_image(file)
    classes = classifier.predict_classes(image_stack, batch_size=10)

    print("Display file:", file)
    print("Predicted class is:", classes, ' which is ', class_labels.get(classes[0]))

    plt.imshow(image_tensor[0])
    plt.title(file + ":" + class_labels.get(classes[0]))
    plt.show()

    #Display activation layers of the first image of the test batch
    if count==0:
        display_activation_layers(classifier, image_tensor)

def predict_images(classifier):
    #predit images in a directory
    count = 0
    for file in glob.glob(test_path + "/*.jpg"):
        predict_an_image(file, count, classifier)
        count =+ 1

def display_activation_layers(classifier, img_tensor):
    #Instantiate a model from an input tensor and a list of output tensors
    layer_outputs = [layer.output for layer in classifier.layers[:12]] # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    activations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activation

    #Plot all the activations of this same image across each layer
    #Visualizing every channel in every intermediate activation
    layer_names = []
    for layer in classifier.layers[:12]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                if (channel_image.std() != 0):
                    channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()


def display_performance_graphs(history):
    # Displays curves of loss and accuracy during training
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def predict():
    #Initiate the CNN model with Sequential():
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), padding='same', input_shape = (128, 128, 3), activation = 'relu'))
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5)) # antes era 0.25

    # Adding a second convolutional layer
    classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5)) # antes era 0.25

    # Adding a third convolutional layer
    classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5)) # antes era 0.25

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units = 512, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 2, activation = 'softmax'))

    ## Load our classifier with the weights of the best model
    #Now we can load those weights as our final model:
    classifier.load_weights('bestmodel_weights.hdf5')
    #predit images test directory
    predict_images(classifier)

def train():
    display_sample_training_images()
    '''
    Images shapes are of 128 pixels by 128 pixels in RGB scale.
    Initiate the CNN model with Sequential():
    '''
    classifier = Sequential()

    '''
    We specify our convolution layers and add MaxPooling to downsample and Dropout to prevent overfitting. We use Flatten and end with a Dense layer of 2 units, one for each class (HIGH [0], LOW [1]). We specify softmax as our last activation function, which is suggested for multiclass classification.
    '''

    # Step 1 - Convolution
    train_classes = 2 #(high, low)

    classifier.add(Conv2D(32, (3, 3), padding='same', input_shape = (128, 128, 3), activation = 'relu'))
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5)) # antes era 0.25

    # Adding a second convolutional layer
    classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5)) # antes era 0.25

    # Adding a third convolutional layer
    classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5)) # antes era 0.25

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units = 512, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = train_classes, activation = 'softmax'))
    '''
    Display classifer's layer summary
    '''
    print("Model summary:")
    classifier.summary()

    # Compiling the CNN
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.02, amsgrad=False)
    classifier.compile(optimizer = adam, #'rmsprop',
                       loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
    #decay originally 0.01
    '''

    ## Using ImageDataGenerator to read images from directories
    At this point we need to convert our pictures to a shape that the model will accept. For that we use the ImageDataGenerator. We initiate it and feed our images with .flow_from_directory. There are two main folders inside the working directory, called training_set and validation_set. Each of those have 2 subfolders called high and low. I have sent 80% of total images of each shape to the training_set and 20% to the validation_set.

    '''

    train_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size = (128, 128),
                                                     batch_size = 16,
                                                     class_mode = 'categorical')

    validation_set = test_datagen.flow_from_directory(validation_path,
                                                      target_size = (128, 128),
                                                      batch_size = 16,
                                                      class_mode = 'categorical')
    '''
    ## Utilize callback to store the weights of the best model
    The model will train for 20 epochs but we will use ModelCheckpoint to store the weights of the best performing epoch. We will specify val_acc as the metric to use to define the best model. This means we will keep the weights of the epoch that scores highest in terms of accuracy on the test set.
    '''
    checkpointer = ModelCheckpoint(filepath="bestmodel_weights.hdf5",
                                   monitor = 'val_acc',
                                   verbose=1,
                                   save_best_only=True)
    '''
    Now it's time to train the model, here we include the callback to our checkpointer
    '''
    number_steps_per_epoch = 100
    number_epoch = 20
    history = classifier.fit_generator(training_set,
                                       steps_per_epoch = number_steps_per_epoch,
                                       epochs = number_epoch,
                                       callbacks=[checkpointer],
                                       validation_data = validation_set,
                                       validation_steps = 50)
    '''
    The model trained for 20 epochs but reached it's best performance at epoch 10, saving model to bestmodel_weights.hdf5

    That means we have now an hdf5 file which stores the weights of that specific epoch, where the accuracy over the test set was of 95,6%

    ## Load our classifier with the weights of the best model
    Now we can load those weights as our final model:
    '''
    classifier.load_weights('bestmodel_weights.hdf5')
    '''
    ## Saving the complete model
    '''
    #classifier.save('shapes_cnn.h5')
    '''

    ## Displaying curves of loss and accuracy during training
    Let's now inspect how our model performed:
    '''
    #display_performance_graphs(history)
    '''
    ## Classes
    Let's clarify now the class number assigned to each of our figures set, since that is how the model will produce it's predictions:
    high: 0
    low: 1

    ## Predicting new images
    With our model trained and stored, we can load a simple unseen image from our test set and see how it is classified:
    '''
    #predit images test directory
    predict_images(classifier)


def usage():
    print("Usage: python cnnVisualize ")
    print("       to predict images with saved weight:python cnnVisualize --predict ")


######################################
##main program starts
######################################
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h",["predict", "help"])

        if opts:
            for opt, arg in opts:
                if opt in ('-h', '--help'):
                    usage()
                    sys.exit(2)
                elif opt in ('-p','--predict'):
                    predict()
        else:
            train()
    except getopt.GetoptError:
        print("GetoptError")
        sys.exit(2)

if __name__ == "__main__":
   main(sys.argv[1:])
