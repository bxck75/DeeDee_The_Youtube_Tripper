######################################################################
#
# Deep Dreaming in Keras by Bxck Kooy
# 
# made in jul 2019
#
# Run the script with:
# 
# img='frames/00000001.jpg'; python deep_dreamv2.py $img 'processed_'$img 1 4
# 
# to do a folder of images 
# 
# for file in `ls frames/`; do python deep_dreamv2.py 'frames/'$file 'processed_frames/'$file 4 10; done
# 
######################################################################

from __future__ import print_function

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import scipy
import argparse
import warnings
import imageio
import datetime,os,sys,glob
from keras.applications import inception_v3
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib
# print(dir(K))
# print(dir(tf))
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
                    help='itterations per octave',default=5)
args = parser.parse_args()

# set dream hyper vars
step = 0.03  # Gradient ascent step size
num_octave = args.octaves  # Number of scales at which to run gradient ascent
octave_scale = 1.5  # Size ratio between scales
iterations = args.itterations  # Number of ascent steps per scale
max_loss = 10
# You can tweak these setting to obtain other visual effects.
settings = {
    'features': {
        # 'mixed10': 0.2,
        'average_pooling2d_1': 0.8,
        # 'mixed3': 0.1,
        # 'mixed2': 0.6,
        # 'mixed4': 2.5,
    },
}



#base settings do not change!!!!!!!!!
base_image_path = args.base_image_path
out_image_path = args.out_image_path
Weights_File = 'model/custom_inception_weights.h5'
Model_File = 'model/custom_inception_model.h5'

# proc/cpu limiter
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 15
config.inter_op_parallelism_threads = 15
sess= tf.Session(config=config)

def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate tensors.
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


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

def load_trained_model(weights_path):
    model = load_model(Model_File)
    model.load_weights(weights_path)
    return model



K.set_learning_phase(0)
if os.path.exists(Model_File):
    print('Loading pre-trained model and weights')
    model = load_trained_model(Weights_File)
else:
    print('Load inception_v3 Model')
    # Build the InceptionV3 network with our placeholder.
    # The model will be loaded with pre-trained ImageNet weights.
    model = inception_v3.InceptionV3(weights='imagenet',include_top=False)        
    # save own copy
    model.save(Model_File)
    # #save weights
    model.save_weights(Weights_File)

dream = model.input
print('Model loaded.')

# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])
import collections

# sorted_dict = collections.OrderedDict(layer_dict)
# for aap in sorted_dict:
    # print(aap)

# Define the loss.
loss = K.variable(0.)
for layer_name in settings['features']:
    # Add the L2 norm of the features of a layer to the loss.
    assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
    coeff = settings['features'][layer_name]
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
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x = x + step * grad_values
    return x

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    imageio.imwrite(fname, pil_img)


"""Process:
- Load the original image.
- Define a number of processing scales (i.e. image shapes),
    from smallest to largest.
- Resize the original image to the smallest scale.
- For every scale, starting with the smallest (i.e. current one):
    - Run gradient ascent
    - Upscale image to the next scale
    - Reinject the detail that was lost at upscaling time
- Stop when we are back to the original size.
To obtain the detail lost during upscaling, we simply
take the original image, shrink it down, upscale it,
and compare the result to the (resized) original image.
"""


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
timer=1
# print(config.inter_op_parallelism_threads)
for shape in successive_shapes:
    print('Octave :' + str(timer) + ' Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img = img + lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    timer=timer + 1
print(out_image_path)
save_img(img, fname=out_image_path)

# Cleans up session
sess.close()
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
