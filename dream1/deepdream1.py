from keras.applications import VGG16
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from keras import backend as K
import numpy as np
from PIL import Image
import scipy

'''
Deep dream application based on VGG16 classifier, pretrained on imagenet dataset
Written by Grant Holtes, December 2018
www.grantholtes.com
'''
Weights_File = 'model/vgg16_weights.h5'
Model_File = 'model/vgg16_model.h5'


def preprocessImage(image):
    '''preprocess for vgg16'''
    return preprocess_input(np.expand_dims(image, axis=0))

def postProcessArray(x):
    '''transform back to RGB to then be exported'''
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x

def saveImage(NPimage, pathForOutput):
    dream = Image.fromarray(postProcessArray(NPimage[0]))
    dream.save(pathForOutput)

def resizeImage(size, NPimage):
    '''side = max side length of the output'''
    shape = NPimage.shape
    scaleH = size[0] / shape[1]
    scaleW = size[1] / shape[2]
    NPimage = scipy.ndimage.zoom(NPimage,
                                (1, scaleH, scaleW, 1),
                                order=1)
    return NPimage

def getFeatureReps(layer_names, Model):
    '''Get the activations of all layers in the list 'layer_names'''
    featureMatrices = []
    for layer in layer_names:
        selectedLayer = Model.get_layer(layer)
        featureMatrices.append(selectedLayer.output)
    return featureMatrices

def loss(layer_names, Model):
    '''Define how the loss is evaluated'''
    return - K.sum(getFeatureReps(layer_names, Model))

def gradient(NPimage, layer_names, model):
    '''Get the gradient of the loss with respect to each input pixel'''
    gradFunc = K.function([model.input], K.gradients(loss(layer_names, model), [model.input]))
    return gradFunc([NPimage])

def gradientAccent(NPimage, layer_names, iterations = 5, learningRate = 8):
    for iter in range(iterations):
        print("Iteration: {0}".format(iter+1))
        gradIter = gradient(NPimage, layer_names, Model)
        NPimage = np.add(NPimage, -np.multiply(gradIter, learningRate))[0]
    return NPimage

def Main(Model = VGG16(include_top=False, weights='imagenet'),
         pathToImage = "data.jpg",
         pathForOutput = "dream.jpg",
         learningRate = 8,
         maxSize = "Native",
         minSize = 100,
         sizeSteps = 3,
         iterationsPerSize = 10,
         layer_names = ['block5_pool']):

    #Load data
    original = Image.open(pathToImage)
    originalSize = original.size
    original = np.array(original)
    NPimage = preprocessImage(original)
    #Save a copy of original:
    NPoriginal = NPimage.copy()
    originalUpscaledShrunkImage = NPimage.copy()
    #Check size of the image
    if maxSize > max(originalSize):
        maxSize = max(originalSize)
    if maxSize < minSize:
        print("ERROR: maxSize < minSize, quitting")
        exit()
    #make sizes
    sizes = []
    for step in range(sizeSteps):
        Max = round(minSize + step*(maxSize-minSize)/(sizeSteps-1))
        if originalSize[0] > originalSize[1]:
            w = Max
            h = round(Max*originalSize[1] / originalSize[0])
        else:
            w = round(Max*originalSize[0] / originalSize[1])
            h = Max
        sizes.append([h, w])
    previousSize = sizes[0] #Set first size
    #Main
    for size in sizes:
        print("Processing image at size: {0}".format(size))
        NPimage = resizeImage(size, NPimage)
        #shrink original image, compare to unscaled original image to recover lost detail / infomation
        originalUpscaledShrunkImage = resizeImage(size, originalUpscaledShrunkImage)
        originalAtSize = resizeImage(size, NPoriginal)
        #Define lost information
        lostInformation = originalAtSize - originalUpscaledShrunkImage
        #Add lost infomation back to both the copy and dream (NPImage) images
        NPimage += lostInformation
        originalUpscaledShrunkImage += lostInformation
        #Perform Gradient Accent
        NPimage = gradientAccent(NPimage, layer_names, iterationsPerSize, learningRate)

    #final resize:
    if maxSize < max(originalSize):
        #Add remaining lost infomation
        size = [originalSize[1], originalSize[0]]
        NPimage = resizeImage(size, NPimage)
        originalUpscaledShrunkImage = resizeImage(size, originalUpscaledShrunkImage)
        originalAtSize = resizeImage(size, NPoriginal)
        lostInformation = originalAtSize - originalUpscaledShrunkImage
        NPimage += lostInformation

    #Save the image
    saveImage(NPimage, pathForOutput)


#Load Model
tf_session = K.get_session()
Model = VGG16(include_top=False, weights='imagenet')

# save own copy
print('Save inception_v3 Model')
Model.save(Model_File)

# #save weights
print('Save inception_v3 Weights')
Model.save_weights(Weights_File)

print("Model Loaded")

#Run Deep Dream - Example of parameters.
#Run 'Model.summary()' to view other layer_names
#Max size = max side length of the image to be processed in pixels
Main(Model = Model,
     pathToImage = "test.jpg",
     pathForOutput = "dream.jpg",
     learningRate = 8,
     maxSize = 2000,
     minSize = 100,
     sizeSteps = 5,
     iterationsPerSize = 7,
     layer_names = ['block5_pool'])
