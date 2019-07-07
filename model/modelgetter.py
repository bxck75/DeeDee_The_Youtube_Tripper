from keras.applications import *
import keras.applications
Weights_File = 'NASNetMobile_weights.h5'
Model_File = 'NASNetMobile_model.h5'

print(dir(keras.applications))
'DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'NASNetLarge', 'NASNetMobile', 'ResNet50', 'VGG16', 'VGG19', 'Xception'

#Load Model
Model = NASNetMobile(include_top=False, weights='imagenet')

# save own copy
Model.save(Model_File)
print('Saved Model')

# #save weights
Model.save_weights(Weights_File)
print('Saved Weights')

# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in Model.layers])

for aap in layer_dict:
    print(aap)
print("Model Loaded")
