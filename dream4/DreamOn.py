'''
Some info on various layers, so you know what to expect
depending on which layer you choose:

layer 1: wavy
layer 2: lines
layer 3: boxes
layer 4: circles?
layer 6: dogs, bears, cute animals.
layer 7: faces, buildings
layer 8: fish begin to appear, frogs/reptilian eyes.
layer 10: Monkies, lizards, snakes, duck

Choosing various parameters like num_iterations, rescale,
and num_repeats really varies on which layer you're doing.


We could probably come up with some sort of formula. The
deeper the layer is, the more iterations and
repeats you will want.

Layer 3: 20 iterations, 0.5 rescale, and 8 repeats is decent start
Layer 10: 40 iterations and 25 repeats is good. 
'''

from deepdreamer import model, load_image, recursive_optimize
import numpy as np
import PIL.Image
import sys,os,warnings
from tensorflow import keras
from tensorflow.keras import layers

in_img = str(sys.argv[1])
out_img = str(sys.argv[2])
tensor_nr= int(sys.argv[3])
octaves = int(sys.argv[4])
iters = int(sys.argv[5])


# choose the tensor
layer_tensor = model.layer_tensors[tensor_nr]
# in file
file_name = in_img
img_result = load_image(filename='{}'.format(file_name))
# optimize
img_result = recursive_optimize(layer_tensor=layer_tensor, image=img_result,
                 # how clear is the dream vs original image        
                 num_iterations=iters, step_size=1.0, rescale_factor=0.5,
                 # How many "passes" over the data. More passes, the more granular the gradients will be.
                 num_repeats=octaves, blend=0.2)
# Save results
img_result = np.clip(img_result, 0.0, 255.0)
img_result = img_result.astype(np.uint8)
result = PIL.Image.fromarray(img_result, mode='RGB')
result.save(out_img)
weights = model.get_weights()
model.save('model/DreamOn_model.h5')
model.summary()

result.show()





