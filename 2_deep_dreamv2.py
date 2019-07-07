###########################################################################################################
#
# Deep Dreaming Heaven
#   by Bxck Kooy
#  made 7 jul 2019
#
# Run the script with:
# 
# python deep_dreamv2.py <input image path> <output image path> <repeat> <iteration> <layer presets>
# 
# img='frames/00000001.jpg'; python 2_deep_dreamv2.py $img 'processed_'$img 1 4 0
# img='frames/ghostmane/0000000148.jpg';name=`echo $img |awk -F '/' '{print $2"/"$3}'`; python 2_deep_dreamv2.py $img 'processed_frames/'$name 4 10 0
# to do a folder of images 
# 
# for file in `ls frames/`; do python deep_dreamv2.py 'frames/'$file 'processed_frames/'$file 4 10; done
# 
############################################################################################################333
from __future__ import print_function
import numpy as np
import scipy
import argparse
import warnings
import imageio,collections
import datetime,os,sys,glob
import tensorflow as tf
import keras.applications
# print(dir(keras.applications))
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import *
from tensorflow.python.client import device_lib
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.nasnet import NASNetMobile
import keras.applications
def get_available_gpus():
    from tensorflow.python.client import device_lib
    return device_lib.list_local_devices()
# print(get_available_gpus())

#warnings settings
os.environ['KMP_WARNINGS'] = 'off'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# set warning level
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    # os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses
    # print(dir(warnings.simplefilter))

#input parsing
parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('out_image_path', metavar='out', type=str,
                    help='Path to for the transformed image')
parser.add_argument('octaves', metavar='oct', type=int,
                    help='octave count',default=4)
parser.add_argument('itterations', metavar='iter', type=int,
                    help='itterations per octave',default=10)
parser.add_argument('preset', metavar='pre', type=int,
                    help='settings preset select',default=0)
args = parser.parse_args()

# You can tweak these setting to obtain other visual effects.
step = 0.01 # Gradient ascent step size
num_octave = args.octaves  # Number of scales at which to run gradient ascent
octave_scale = 1.6  # Size ratio between scales
iterations = args.itterations  # Number of ascent steps per scale
max_loss = 20 # Max loss limit
#path settings
preset_selector = int(args.preset)
base_image_path = args.base_image_path
out_image_path = args.out_image_path

set={}
set['InceptionV3'] =  {'features': {'input_1': 0.3,'concatenate_2': 0.5,'mixed3': 0.3,'activation_60': 1.8},}
set['MobileNet'] =  {'features': {'input_1': 0.5,'conv_pw_13':2,'conv_dw_2_relu':0.4,'conv_pad_4':1.5},}
set['Xception'] =  {'features': {'input_1': 0.3,'block5_sepconv3_act': 1.3,'block13_pool': 2.0,'add_11': 0.3,'block8_sepconv3_act': 0.4},}
set['InceptionResNetV2'] =  {'features': {'input_1': 0.1,'block3_sepconv2': 0.5,'block3_sepconv1': 0.3},}
set['VGG16'] =  {'features': {'input_1': 0.5,'conv_pad_5': 0.5,'block3_sepconv1': 0.3},}
set['NasNetMobile'] =  {'features': {'input_1': 0.1,'block3_sepconv2': 0.5,'block3_sepconv1': 0.3},}
set['ResNet50'] =  {'features': {'input_1': 0.1,'bn_conv1': 0.5 ,'res3b_branch2b': 0.5,'conv1_pad': 0.3},}
set['VGG19'] =  {'features': {'input_1': 0.1,'block3_sepconv2': 0.5,'block3_sepconv1': 0.3},}
set['DenseNet121'] =  {'features': {'input_1': 0.1,'block3_sepconv2': 0.5,'block3_sepconv1': 0.3},}
set['DenseNet169'] =  {'features': {'input_1': 0.1,'block3_sepconv2': 0.5,'block3_sepconv1': 0.3},}
set['DenseNet201'] =  {'features': {'input_1': 0.1,'block3_sepconv2': 0.5,'block3_sepconv1': 0.3},}

ActiveModel= 'InceptionV3' #active model

# proc/cpu limiter
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 16
config.inter_op_parallelism_threads = 16
sess= tf.Session(config=config)

Weights_File = 'model/'+ActiveModel+'_weights.h5'
Model_File = 'model/'+ActiveModel+'_model.h5'

debug = True

def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate tensors.
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

def resize_img(img, size):
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        from timeit import default_timer as timer
        # Timer starting...
        start = timer()
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x = x + step * grad_values
        end = timer()
        precision = 2
        print( "Time spend :"+"{:.{}f}".format((end - start), precision ) +" seconds." )

    return x

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    imageio.imwrite(fname, pil_img)

def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x = x + 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def load_saved_model(model_path,weights_path):
    model = load_model(model_path)
    model.load_weights(weights_path)
    return model




class ModelSwitcher(object):
    def activate_model(self, argument):
        print('[ Loading ] : ' + ActiveModel + ' Model')
        """Dispatch method"""
        method_name = 'Load_' + str(argument)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid month")
        # Call the method as we return it
        return method()
 
    print(set[ActiveModel]['features'])
    def Load_Xception(self):
        return xception.Xception(weights='imagenet',include_top=False),set[ActiveModel]
    def Load_VGG16(self):
        return VGG16(weights='imagenet',include_top=False),set[ActiveModel]
    def Load_VGG19(self):
        return VGG19(weights='imagenet',include_top=False),set[ActiveModel]
    def Load_MobileNet(self):
        return mobilenet.MobileNet(weights='imagenet',include_top=False),set[ActiveModel]
    def Load_NasNetMobile(self):
        return NASNetMobile(weights='imagenet',include_top=False),set[ActiveModel]
    def Load_InceptionResNetV2(self):
        return inceptionresnetv2.InceptionResNetV2(weights='imagenet',include_top=False),set[ActiveModel]
    def Load_ResNet50(self):
        return resnet50.ResNet50(weights='imagenet',include_top=False),set[ActiveModel]
    def Load_InceptionV3(self):
        return inception_v3.InceptionV3(weights='imagenet',include_top=False),set[ActiveModel]
    def Load_DenseNet121(self):
        return densenet121.DenseNet121(weights='imagenet',include_top=False),set[ActiveModel]
    def Load_DenseNet169(self):
        return densenet169.DenseNet169(weights='imagenet',include_top=False),set[ActiveModel]
    def Load_DenseNet201(self):
        return densenet201.DenseNet201(weights='imagenet',include_top=False),set[ActiveModel]
    def get_settings(self):
        return set[ActiveModel]

K.set_learning_phase(0)
# Auto choose Saved or Loaded Model/Weights
if (os.path.exists(Model_File) and os.path.exists(Weights_File)) :
    print('Loading saved model and weights')
    model = load_saved_model(Model_File,Weights_File)
    a=ModelSwitcher()
    selected_model_settings = a.get_settings()
else:
    print('No saved model and weights found!')
    # Switch to the selected model.
    # The model will be loaded with pre-trained ImageNet weights.
    a=ModelSwitcher()
    model,selected_model_settings = a.activate_model(ActiveModel)                
    # save own copy
    print('[ Saving Model] : ' + ActiveModel)
    model.save(Model_File)

dream = model.input
print('[ Model loaded ]')

# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Tensor overview when debug=True
if debug:
    print(dir(keras.applications))
    layer_dict_sorted = collections.OrderedDict(layer_dict)
    for tensor in layer_dict_sorted:
        print(tensor)

print(selected_model_settings['features'])
# Define the loss.
loss = K.variable(0.)
for layer_name in selected_model_settings['features']:
    # Add the L2 norm of the features of a layer to the loss.
    assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
    coeff = selected_model_settings['features'][layer_name]
    x = layer_dict[layer_name].output
    # We avoid border artifacts by only involving non-border pixels in the loss.
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    if K.image_data_format() == 'channels_first':
        loss = loss + coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
    else:
        loss = loss + coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

# Compute the gradients of the dream wrt the loss.
grads = K.gradients(loss, dream)[0]
# Normalize gradients.
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

# Set up function to retrieve the value
# of the loss and gradients given an input image.
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


# Dream!!!!!!!!!!!!!

print(base_image_path)
img = preprocess_image(base_image_path)
if K.image_data_format() == 'channels_first':
    original_shape = img.shape[2:]
else:
    original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

# for all shapes we do 
timer=1
for shape in successive_shapes:
    print('Octave :' + str(timer) + ' Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,iterations=iterations,step=step,max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img
    img = img + lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    timer=timer + 1

# The End we save weights
print('[ Saving  : ' + ActiveModel + ' Weights ]')
model.save_weights(Weights_File)
print('Saving Final Image : ' + out_image_path)
save_img(img, fname=out_image_path)
# Cleans up session
print('closing....')
sess.close()
# Vgg16 LAYERS
# input_1
# block1_conv1
# block1_conv2
# block1_pool
# block2_conv1
# block2_conv2
# block2_pool
# block3_conv1
# block3_conv2
# block3_conv3
# block3_pool
# block4_conv1
# block4_conv2
# block4_conv3
# block4_pool
# block5_conv1
# block5_conv2
# block5_conv3
# block5_pool

# Inceptionv3 LAYERS
# input_1
# conv2d_1
# batch_normalization_1
# activation_1
# conv2d_2
# batch_normalization_2
# activation_2
# conv2d_3
# batch_normalization_3
# activation_3
# max_pooling2d_1
# conv2d_4
# batch_normalization_4
# activation_4
# conv2d_5
# batch_normalization_5
# activation_5
# max_pooling2d_2
# conv2d_9
# batch_normalization_9
# activation_9
# conv2d_7
# conv2d_10
# batch_normalization_7
# batch_normalization_10
# activation_7
# activation_10
# average_pooling2d_1
# conv2d_6
# conv2d_8
# conv2d_11
# conv2d_12
# batch_normalization_6
# batch_normalization_8
# batch_normalization_11
# batch_normalization_12
# activation_6
# activation_8
# activation_11
# activation_12
# mixed0
# conv2d_16
# batch_normalization_16
# activation_16
# conv2d_14
# conv2d_17
# batch_normalization_14
# batch_normalization_17
# activation_14
# activation_17
# average_pooling2d_2
# conv2d_13
# conv2d_15
# conv2d_18
# conv2d_19
# batch_normalization_13
# batch_normalization_15
# batch_normalization_18
# batch_normalization_19
# activation_13
# activation_15
# activation_18
# activation_19
# mixed1
# conv2d_23
# batch_normalization_23
# activation_23
# conv2d_21
# conv2d_24
# batch_normalization_21
# batch_normalization_24
# activation_21
# activation_24
# average_pooling2d_3
# conv2d_20
# conv2d_22
# conv2d_25
# conv2d_26
# batch_normalization_20
# batch_normalization_22
# batch_normalization_25
# batch_normalization_26
# activation_20
# activation_22
# activation_25
# activation_26
# mixed2
# conv2d_28
# batch_normalization_28
# activation_28
# conv2d_29
# batch_normalization_29
# activation_29
# conv2d_27
# conv2d_30
# batch_normalization_27
# batch_normalization_30
# activation_27
# activation_30
# max_pooling2d_3
# mixed3
# conv2d_35
# batch_normalization_35
# activation_35
# conv2d_36
# batch_normalization_36
# activation_36
# conv2d_32
# conv2d_37
# batch_normalization_32
# batch_normalization_37
# activation_32
# activation_37
# conv2d_33
# conv2d_38
# batch_normalization_33
# batch_normalization_38
# activation_33
# activation_38
# average_pooling2d_4
# conv2d_31
# conv2d_34
# conv2d_39
# conv2d_40
# batch_normalization_31
# batch_normalization_34
# batch_normalization_39
# batch_normalization_40
# activation_31
# activation_34
# activation_39
# activation_40
# mixed4
# conv2d_45
# batch_normalization_45
# activation_45
# conv2d_46
# batch_normalization_46
# activation_46
# conv2d_42
# conv2d_47
# batch_normalization_42
# batch_normalization_47
# activation_42
# activation_47
# conv2d_43
# conv2d_48
# batch_normalization_43
# batch_normalization_48
# activation_43
# activation_48
# average_pooling2d_5
# conv2d_41
# conv2d_44
# conv2d_49
# conv2d_50
# batch_normalization_41
# batch_normalization_44
# batch_normalization_49
# batch_normalization_50
# activation_41
# activation_44
# activation_49
# activation_50
# mixed5
# conv2d_55
# batch_normalization_55
# activation_55
# conv2d_56
# batch_normalization_56
# activation_56
# conv2d_52
# conv2d_57
# batch_normalization_52
# batch_normalization_57
# activation_52
# activation_57
# conv2d_53
# conv2d_58
# batch_normalization_53
# batch_normalization_58
# activation_53
# activation_58
# average_pooling2d_6
# conv2d_51
# conv2d_54
# conv2d_59
# conv2d_60
# batch_normalization_51
# batch_normalization_54
# batch_normalization_59
# batch_normalization_60
# activation_51
# activation_54
# activation_59
# activation_60
# mixed6
# conv2d_65
# batch_normalization_65
# activation_65
# conv2d_66
# batch_normalization_66
# activation_66
# conv2d_62
# conv2d_67
# batch_normalization_62
# batch_normalization_67
# activation_62
# activation_67
# conv2d_63
# conv2d_68
# batch_normalization_63
# batch_normalization_68
# activation_63
# activation_68
# average_pooling2d_7
# conv2d_61
# conv2d_64
# conv2d_69
# conv2d_70
# batch_normalization_61
# batch_normalization_64
# batch_normalization_69
# batch_normalization_70
# activation_61
# activation_64
# activation_69
# activation_70
# mixed7
# conv2d_73
# batch_normalization_73
# activation_73
# conv2d_74
# batch_normalization_74
# activation_74
# conv2d_71
# conv2d_75
# batch_normalization_71
# batch_normalization_75
# activation_71
# activation_75
# conv2d_72
# conv2d_76
# batch_normalization_72
# batch_normalization_76
# activation_72
# activation_76
# max_pooling2d_4
# mixed8
# conv2d_81
# batch_normalization_81
# activation_81
# conv2d_78
# conv2d_82
# batch_normalization_78
# batch_normalization_82
# activation_78
# activation_82
# conv2d_79
# conv2d_80
# conv2d_83
# conv2d_84
# average_pooling2d_8
# conv2d_77
# batch_normalization_79
# batch_normalization_80
# batch_normalization_83
# batch_normalization_84
# conv2d_85
# batch_normalization_77
# activation_79
# activation_80
# activation_83
# activation_84
# batch_normalization_85
# activation_77
# mixed9_0
# concatenate_1
# activation_85
# mixed9
# conv2d_90
# batch_normalization_90
# activation_90
# conv2d_87
# conv2d_91
# batch_normalization_87
# batch_normalization_91
# activation_87
# activation_91
# conv2d_88
# conv2d_89
# conv2d_92
# conv2d_93
# average_pooling2d_9
# conv2d_86
# batch_normalization_88
# batch_normalization_89
# batch_normalization_92
# batch_normalization_93
# conv2d_94
# batch_normalization_86
# activation_88
# activation_89
# activation_92
# activation_93
# batch_normalization_94
# activation_86
# mixed9_1
# concatenate_2
# activation_94
# mixed10


#resnetincp LAYERS

# input_1
# conv2d_1
# batch_normalization_1
# activation_1
# conv2d_2
# batch_normalization_2
# activation_2
# conv2d_3
# batch_normalization_3
# activation_3
# max_pooling2d_1
# conv2d_4
# batch_normalization_4
# activation_4
# conv2d_5
# batch_normalization_5
# activation_5
# max_pooling2d_2
# conv2d_9
# batch_normalization_9
# activation_9
# conv2d_7
# conv2d_10
# batch_normalization_7
# batch_normalization_10
# activation_7
# activation_10
# average_pooling2d_1
# conv2d_6
# conv2d_8
# conv2d_11
# conv2d_12
# batch_normalization_6
# batch_normalization_8
# batch_normalization_11
# batch_normalization_12
# activation_6
# activation_8
# activation_11
# activation_12
# mixed_5b
# conv2d_16
# batch_normalization_16
# activation_16
# conv2d_14
# conv2d_17
# batch_normalization_14
# batch_normalization_17
# activation_14
# activation_17
# conv2d_13
# conv2d_15
# conv2d_18
# batch_normalization_13
# batch_normalization_15
# batch_normalization_18
# activation_13
# activation_15
# activation_18
# block35_1_mixed
# block35_1_conv
# block35_1
# block35_1_ac
# conv2d_22
# batch_normalization_22
# activation_22
# conv2d_20
# conv2d_23
# batch_normalization_20
# batch_normalization_23
# activation_20
# activation_23
# conv2d_19
# conv2d_21
# conv2d_24
# batch_normalization_19
# batch_normalization_21
# batch_normalization_24
# activation_19
# activation_21
# activation_24
# block35_2_mixed
# block35_2_conv
# block35_2
# block35_2_ac
# conv2d_28
# batch_normalization_28
# activation_28
# conv2d_26
# conv2d_29
# batch_normalization_26
# batch_normalization_29
# activation_26
# activation_29
# conv2d_25
# conv2d_27
# conv2d_30
# batch_normalization_25
# batch_normalization_27
# batch_normalization_30
# activation_25
# activation_27
# activation_30
# block35_3_mixed
# block35_3_conv
# block35_3
# block35_3_ac
# conv2d_34
# batch_normalization_34
# activation_34
# conv2d_32
# conv2d_35
# batch_normalization_32
# batch_normalization_35
# activation_32
# activation_35
# conv2d_31
# conv2d_33
# conv2d_36
# batch_normalization_31
# batch_normalization_33
# batch_normalization_36
# activation_31
# activation_33
# activation_36
# block35_4_mixed
# block35_4_conv
# block35_4
# block35_4_ac
# conv2d_40
# batch_normalization_40
# activation_40
# conv2d_38
# conv2d_41
# batch_normalization_38
# batch_normalization_41
# activation_38
# activation_41
# conv2d_37
# conv2d_39
# conv2d_42
# batch_normalization_37
# batch_normalization_39
# batch_normalization_42
# activation_37
# activation_39
# activation_42
# block35_5_mixed
# block35_5_conv
# block35_5
# block35_5_ac
# conv2d_46
# batch_normalization_46
# activation_46
# conv2d_44
# conv2d_47
# batch_normalization_44
# batch_normalization_47
# activation_44
# activation_47
# conv2d_43
# conv2d_45
# conv2d_48
# batch_normalization_43
# batch_normalization_45
# batch_normalization_48
# activation_43
# activation_45
# activation_48
# block35_6_mixed
# block35_6_conv
# block35_6
# block35_6_ac
# conv2d_52
# batch_normalization_52
# activation_52
# conv2d_50
# conv2d_53
# batch_normalization_50
# batch_normalization_53
# activation_50
# activation_53
# conv2d_49
# conv2d_51
# conv2d_54
# batch_normalization_49
# batch_normalization_51
# batch_normalization_54
# activation_49
# activation_51
# activation_54
# block35_7_mixed
# block35_7_conv
# block35_7
# block35_7_ac
# conv2d_58
# batch_normalization_58
# activation_58
# conv2d_56
# conv2d_59
# batch_normalization_56
# batch_normalization_59
# activation_56
# activation_59
# conv2d_55
# conv2d_57
# conv2d_60
# batch_normalization_55
# batch_normalization_57
# batch_normalization_60
# activation_55
# activation_57
# activation_60
# block35_8_mixed
# block35_8_conv
# block35_8
# block35_8_ac
# conv2d_64
# batch_normalization_64
# activation_64
# conv2d_62
# conv2d_65
# batch_normalization_62
# batch_normalization_65
# activation_62
# activation_65
# conv2d_61
# conv2d_63
# conv2d_66
# batch_normalization_61
# batch_normalization_63
# batch_normalization_66
# activation_61
# activation_63
# activation_66
# block35_9_mixed
# block35_9_conv
# block35_9
# block35_9_ac
# conv2d_70
# batch_normalization_70
# activation_70
# conv2d_68
# conv2d_71
# batch_normalization_68
# batch_normalization_71
# activation_68
# activation_71
# conv2d_67
# conv2d_69
# conv2d_72
# batch_normalization_67
# batch_normalization_69
# batch_normalization_72
# activation_67
# activation_69
# activation_72
# block35_10_mixed
# block35_10_conv
# block35_10
# block35_10_ac
# conv2d_74
# batch_normalization_74
# activation_74
# conv2d_75
# batch_normalization_75
# activation_75
# conv2d_73
# conv2d_76
# batch_normalization_73
# batch_normalization_76
# activation_73
# activation_76
# max_pooling2d_3
# mixed_6a
# conv2d_78
# batch_normalization_78
# activation_78
# conv2d_79
# batch_normalization_79
# activation_79
# conv2d_77
# conv2d_80
# batch_normalization_77
# batch_normalization_80
# activation_77
# activation_80
# block17_1_mixed
# block17_1_conv
# block17_1
# block17_1_ac
# conv2d_82
# batch_normalization_82
# activation_82
# conv2d_83
# batch_normalization_83
# activation_83
# conv2d_81
# conv2d_84
# batch_normalization_81
# batch_normalization_84
# activation_81
# activation_84
# block17_2_mixed
# block17_2_conv
# block17_2
# block17_2_ac
# conv2d_86
# batch_normalization_86
# activation_86
# conv2d_87
# batch_normalization_87
# activation_87
# conv2d_85
# conv2d_88
# batch_normalization_85
# batch_normalization_88
# activation_85
# activation_88
# block17_3_mixed
# block17_3_conv
# block17_3
# block17_3_ac
# conv2d_90
# batch_normalization_90
# activation_90
# conv2d_91
# batch_normalization_91
# activation_91
# conv2d_89
# conv2d_92
# batch_normalization_89
# batch_normalization_92
# activation_89
# activation_92
# block17_4_mixed
# block17_4_conv
# block17_4
# block17_4_ac
# conv2d_94
# batch_normalization_94
# activation_94
# conv2d_95
# batch_normalization_95
# activation_95
# conv2d_93
# conv2d_96
# batch_normalization_93
# batch_normalization_96
# activation_93
# activation_96
# block17_5_mixed
# block17_5_conv
# block17_5
# block17_5_ac
# conv2d_98
# batch_normalization_98
# activation_98
# conv2d_99
# batch_normalization_99
# activation_99
# conv2d_97
# conv2d_100
# batch_normalization_97
# batch_normalization_100
# activation_97
# activation_100
# block17_6_mixed
# block17_6_conv
# block17_6
# block17_6_ac
# conv2d_102
# batch_normalization_102
# activation_102
# conv2d_103
# batch_normalization_103
# activation_103
# conv2d_101
# conv2d_104
# batch_normalization_101
# batch_normalization_104
# activation_101
# activation_104
# block17_7_mixed
# block17_7_conv
# block17_7
# block17_7_ac
# conv2d_106
# batch_normalization_106
# activation_106
# conv2d_107
# batch_normalization_107
# activation_107
# conv2d_105
# conv2d_108
# batch_normalization_105
# batch_normalization_108
# activation_105
# activation_108
# block17_8_mixed
# block17_8_conv
# block17_8
# block17_8_ac
# conv2d_110
# batch_normalization_110
# activation_110
# conv2d_111
# batch_normalization_111
# activation_111
# conv2d_109
# conv2d_112
# batch_normalization_109
# batch_normalization_112
# activation_109
# activation_112
# block17_9_mixed
# block17_9_conv
# block17_9
# block17_9_ac
# conv2d_114
# batch_normalization_114
# activation_114
# conv2d_115
# batch_normalization_115
# activation_115
# conv2d_113
# conv2d_116
# batch_normalization_113
# batch_normalization_116
# activation_113
# activation_116
# block17_10_mixed
# block17_10_conv
# block17_10
# block17_10_ac
# conv2d_118
# batch_normalization_118
# activation_118
# conv2d_119
# batch_normalization_119
# activation_119
# conv2d_117
# conv2d_120
# batch_normalization_117
# batch_normalization_120
# activation_117
# activation_120
# block17_11_mixed
# block17_11_conv
# block17_11
# block17_11_ac
# conv2d_122
# batch_normalization_122
# activation_122
# conv2d_123
# batch_normalization_123
# activation_123
# conv2d_121
# conv2d_124
# batch_normalization_121
# batch_normalization_124
# activation_121
# activation_124
# block17_12_mixed
# block17_12_conv
# block17_12
# block17_12_ac
# conv2d_126
# batch_normalization_126
# activation_126
# conv2d_127
# batch_normalization_127
# activation_127
# conv2d_125
# conv2d_128
# batch_normalization_125
# batch_normalization_128
# activation_125
# activation_128
# block17_13_mixed
# block17_13_conv
# block17_13
# block17_13_ac
# conv2d_130
# batch_normalization_130
# activation_130
# conv2d_131
# batch_normalization_131
# activation_131
# conv2d_129
# conv2d_132
# batch_normalization_129
# batch_normalization_132
# activation_129
# activation_132
# block17_14_mixed
# block17_14_conv
# block17_14
# block17_14_ac
# conv2d_134
# batch_normalization_134
# activation_134
# conv2d_135
# batch_normalization_135
# activation_135
# conv2d_133
# conv2d_136
# batch_normalization_133
# batch_normalization_136
# activation_133
# activation_136
# block17_15_mixed
# block17_15_conv
# block17_15
# block17_15_ac
# conv2d_138
# batch_normalization_138
# activation_138
# conv2d_139
# batch_normalization_139
# activation_139
# conv2d_137
# conv2d_140
# batch_normalization_137
# batch_normalization_140
# activation_137
# activation_140
# block17_16_mixed
# block17_16_conv
# block17_16
# block17_16_ac
# conv2d_142
# batch_normalization_142
# activation_142
# conv2d_143
# batch_normalization_143
# activation_143
# conv2d_141
# conv2d_144
# batch_normalization_141
# batch_normalization_144
# activation_141
# activation_144
# block17_17_mixed
# block17_17_conv
# block17_17
# block17_17_ac
# conv2d_146
# batch_normalization_146
# activation_146
# conv2d_147
# batch_normalization_147
# activation_147
# conv2d_145
# conv2d_148
# batch_normalization_145
# batch_normalization_148
# activation_145
# activation_148
# block17_18_mixed
# block17_18_conv
# block17_18
# block17_18_ac
# conv2d_150
# batch_normalization_150
# activation_150
# conv2d_151
# batch_normalization_151
# activation_151
# conv2d_149
# conv2d_152
# batch_normalization_149
# batch_normalization_152
# activation_149
# activation_152
# block17_19_mixed
# block17_19_conv
# block17_19
# block17_19_ac
# conv2d_154
# batch_normalization_154
# activation_154
# conv2d_155
# batch_normalization_155
# activation_155
# conv2d_153
# conv2d_156
# batch_normalization_153
# batch_normalization_156
# activation_153
# activation_156
# block17_20_mixed
# block17_20_conv
# block17_20
# block17_20_ac
# conv2d_161
# batch_normalization_161
# activation_161
# conv2d_157
# conv2d_159
# conv2d_162
# batch_normalization_157
# batch_normalization_159
# batch_normalization_162
# activation_157
# activation_159
# activation_162
# conv2d_158
# conv2d_160
# conv2d_163
# batch_normalization_158
# batch_normalization_160
# batch_normalization_163
# activation_158
# activation_160
# activation_163
# max_pooling2d_4
# mixed_7a
# conv2d_165
# batch_normalization_165
# activation_165
# conv2d_166
# batch_normalization_166
# activation_166
# conv2d_164
# conv2d_167
# batch_normalization_164
# batch_normalization_167
# activation_164
# activation_167
# block8_1_mixed
# block8_1_conv
# block8_1
# block8_1_ac
# conv2d_169
# batch_normalization_169
# activation_169
# conv2d_170
# batch_normalization_170
# activation_170
# conv2d_168
# conv2d_171
# batch_normalization_168
# batch_normalization_171
# activation_168
# activation_171
# block8_2_mixed
# block8_2_conv
# block8_2
# block8_2_ac
# conv2d_173
# batch_normalization_173
# activation_173
# conv2d_174
# batch_normalization_174
# activation_174
# conv2d_172
# conv2d_175
# batch_normalization_172
# batch_normalization_175
# activation_172
# activation_175
# block8_3_mixed
# block8_3_conv
# block8_3
# block8_3_ac
# conv2d_177
# batch_normalization_177
# activation_177
# conv2d_178
# batch_normalization_178
# activation_178
# conv2d_176
# conv2d_179
# batch_normalization_176
# batch_normalization_179
# activation_176
# activation_179
# block8_4_mixed
# block8_4_conv
# block8_4
# block8_4_ac
# conv2d_181
# batch_normalization_181
# activation_181
# conv2d_182
# batch_normalization_182
# activation_182
# conv2d_180
# conv2d_183
# batch_normalization_180
# batch_normalization_183
# activation_180
# activation_183
# block8_5_mixed
# block8_5_conv
# block8_5
# block8_5_ac
# conv2d_185
# batch_normalization_185
# activation_185
# conv2d_186
# batch_normalization_186
# activation_186
# conv2d_184
# conv2d_187
# batch_normalization_184
# batch_normalization_187
# activation_184
# activation_187
# block8_6_mixed
# block8_6_conv
# block8_6
# block8_6_ac
# conv2d_189
# batch_normalization_189
# activation_189
# conv2d_190
# batch_normalization_190
# activation_190
# conv2d_188
# conv2d_191
# batch_normalization_188
# batch_normalization_191
# activation_188
# activation_191
# block8_7_mixed
# block8_7_conv
# block8_7
# block8_7_ac
# conv2d_193
# batch_normalization_193
# activation_193
# conv2d_194
# batch_normalization_194
# activation_194
# conv2d_192
# conv2d_195
# batch_normalization_192
# batch_normalization_195
# activation_192
# activation_195
# block8_8_mixed
# block8_8_conv
# block8_8
# block8_8_ac
# conv2d_197
# batch_normalization_197
# activation_197
# conv2d_198
# batch_normalization_198
# activation_198
# conv2d_196
# conv2d_199
# batch_normalization_196
# batch_normalization_199
# activation_196
# activation_199
# block8_9_mixed
# block8_9_conv
# block8_9
# block8_9_ac
# conv2d_201
# batch_normalization_201
# activation_201
# conv2d_202
# batch_normalization_202
# activation_202
# conv2d_200
# conv2d_203
# batch_normalization_200
# batch_normalization_203
# activation_200
# activation_203
# block8_10_mixed
# block8_10_conv
# block8_10
# conv_7b
# conv_7b_bn
# conv_7b_ac
