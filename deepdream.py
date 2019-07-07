import math
import numpy as np
import PIL.Image
import tensorflow as tf
import sys,os,warnings
import PIL.ImageEnhance as pie

if not sys.warnoptions:
    print(dir(warnings.simplefilter))
    warnings.simplefilter("default") # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default" # Also affect subprocesses

tf.app.flags.DEFINE_string("model", "tensorflow_inception_graph.pb", "Model")
tf.app.flags.DEFINE_string("input",  sys.argv[1],"inputfile");
tf.app.flags.DEFINE_string("output", sys.argv[2],"outputfile");
tf.app.flags.DEFINE_string("layer",sys.argv[3], "import/mixed4c");
tf.app.flags.DEFINE_integer("feature", "-1", "Individual feature");
tf.app.flags.DEFINE_float("bright_control", "1.0", "brightness control");
tf.app.flags.DEFINE_float("contrast_control", "1.0", "brightness control");
tf.app.flags.DEFINE_integer("frames", "5", "How many frames to run");
tf.app.flags.DEFINE_integer("octaves", "4", "How many mage octaves (scales)");
tf.app.flags.DEFINE_integer("iterations", "6", "How many gradient iterations per octave");
tf.app.flags.DEFINE_float("octave_scale", "1.4", "Octave scaling factor");
tf.app.flags.DEFINE_float("frame_scale", "0.4", "Frame scaling factor");
tf.app.flags.DEFINE_boolean("frame_crop", "false", "Frame crop to original");
tf.app.flags.DEFINE_integer("tilesize", "256", "Size of tiles. Decrease if out of GPU memory. Increase if bad utilization.");

FLAGS = tf.app.flags.FLAGS
# print('../'+FLAGS.model)
# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph, config=tf.ConfigProto(log_device_placement=False))

import inception5h
# get the model
inception5h.maybe_download()
model = inception5h.Inception5h()
tensor_model_size=len(model.layer_tensors)
# print(dir(model.layer_tensors))
graph_def = tf.GraphDef.FromString(open(FLAGS.model).read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

print ("--- Available Layers: ---")
layers = []
for name in (op.name for op in graph.get_operations()):
  layer_shape = graph.get_tensor_by_name(name+':0').get_shape()
  if not layer_shape.ndims: continue
  layers.append((name, int(layer_shape[-1])))
  # print (name, "Features/Channels: ", int(layer_shape[-1]))
print ('Number of layers', len(layers))
print ('Total number of feature channels:', sum((layer[1] for layer in layers)))
print ('Chosen layer: ')
print (graph.get_operation_by_name(FLAGS.layer));

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("%s:0"%layer)

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = map(tf.placeholder, argtypes)
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in xrange(0, max(h-sz//2, sz),sz):
        for x in xrange(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_deepdream(t_obj, img,
                     iter_n=10, step=1.5, octave_n=12, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    # split the image into a number of octaves
    img = img
    octaves = []
    for i in xrange(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)

    # generate details octave by octave
    for octave in xrange(octave_n):
        print (" Octave: ", octave, "Res: ", img.shape)
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in xrange(iter_n):
            g = calc_grad_tiled(img, t_grad, FLAGS.tilesize)
            img += g*(step / (np.abs(g).mean()+1e-7))
    return img

def main(_):
  if FLAGS.input:
    # print(dir(PIL.Image))
    img = np.float32(PIL.Image.open(FLAGS.input));
  else:
    img = np.float32(np.full((1024,1024,3), 128))

  start_shape = img.shape

  # Make RGB if greyscale:
  if len(img.shape)==2 or img.shape[2] == 1:
    img = np.stack([img]*3, axis=2)

  for i_frame in range(FLAGS.frames):
    if FLAGS.frame_scale > 1.0:
      img = resize(img, np.int32(np.float32(img.shape[:2])*FLAGS.frame_scale))
    if FLAGS.frame_crop:
      img = img[img.shape[0]//2-start_shape[0]//2 : img.shape[0]//2-start_shape[0]//2 + start_shape[0],
                img.shape[1]//2-start_shape[1]//2 : img.shape[1]//2-start_shape[1]//2 + start_shape[1],:]

    
    print ("Cycle", i_frame, " Res:", img.shape)
    t_obj = tf.square(T(FLAGS.layer))
    if FLAGS.feature >= 0:
      t_obj = T(FLAGS.layer)[:,:,:,FLAGS.feature]
    img = render_deepdream(t_obj, img,
        iter_n = FLAGS.iterations,
        octave_n = FLAGS.octaves,
        octave_scale = FLAGS.octave_scale)
    print ("Saving ", i_frame)


    img = np.uint8(np.clip(img, 0, 255))

# # brightness controle
#     if FLAGS.bright_control >= 0:
        
#         control = FLAGS.bright_control
#         # print(control)
#         # img.adjust_brightness(img, control)
#         from PIL import Image, ImageEnhance 
#         enhancer = pie.Brightness(img)
#         img = enhancer.enhance(control)

# contrast controle



    filen,ext = os.path.splitext(FLAGS.output)
    out= PIL.Image.fromarray(img)
# contrast controle
    if FLAGS.bright_control >= 0:
        print('Brightness control done! '+ str(FLAGS.bright_control))
        control = FLAGS.bright_control
        enhancer = pie.Brightness(out)
        out = enhancer.enhance(control)
# brightness controle
    if FLAGS.contrast_control >= 0:
        print('Contrast control done! '+ str(FLAGS.contrast_control))
        control = FLAGS.contrast_control
        enhancer = pie.Contrast(out)
        out = enhancer.enhance(control)


    file_name="%s_%05d%s"%(filen, i_frame,ext)
    print(file_name)
    out.save(file_name, "jpeg", quality=50)

if __name__ == "__main__":
    tf.app.run()

# layers 

# (u'import/conv2d0_w', 'Features/Channels: ', 64)
# (u'import/conv2d0_b', 'Features/Channels: ', 64)
# (u'import/conv2d1_w', 'Features/Channels: ', 64)
# (u'import/conv2d1_b', 'Features/Channels: ', 64)
# (u'import/conv2d2_w', 'Features/Channels: ', 192)
# (u'import/conv2d2_b', 'Features/Channels: ', 192)
# (u'import/mixed3a_1x1_w', 'Features/Channels: ', 64)
# (u'import/mixed3a_1x1_b', 'Features/Channels: ', 64)
# (u'import/mixed3a_3x3_bottleneck_w', 'Features/Channels: ', 96)
# (u'import/mixed3a_3x3_bottleneck_b', 'Features/Channels: ', 96)
# (u'import/mixed3a_3x3_w', 'Features/Channels: ', 128)
# (u'import/mixed3a_3x3_b', 'Features/Channels: ', 128)
# (u'import/mixed3a_5x5_bottleneck_w', 'Features/Channels: ', 16)
# (u'import/mixed3a_5x5_bottleneck_b', 'Features/Channels: ', 16)
# (u'import/mixed3a_5x5_w', 'Features/Channels: ', 32)
# (u'import/mixed3a_5x5_b', 'Features/Channels: ', 32)
# (u'import/mixed3a_pool_reduce_w', 'Features/Channels: ', 32)
# (u'import/mixed3a_pool_reduce_b', 'Features/Channels: ', 32)
# (u'import/mixed3b_1x1_w', 'Features/Channels: ', 128)
# (u'import/mixed3b_1x1_b', 'Features/Channels: ', 128)
# (u'import/mixed3b_3x3_bottleneck_w', 'Features/Channels: ', 128)
# (u'import/mixed3b_3x3_bottleneck_b', 'Features/Channels: ', 128)
# (u'import/mixed3b_3x3_w', 'Features/Channels: ', 192)
# (u'import/mixed3b_3x3_b', 'Features/Channels: ', 192)
# (u'import/mixed3b_5x5_bottleneck_w', 'Features/Channels: ', 32)
# (u'import/mixed3b_5x5_bottleneck_b', 'Features/Channels: ', 32)
# (u'import/mixed3b_5x5_w', 'Features/Channels: ', 96)
# (u'import/mixed3b_5x5_b', 'Features/Channels: ', 96)
# (u'import/mixed3b_pool_reduce_w', 'Features/Channels: ', 64)
# (u'import/mixed3b_pool_reduce_b', 'Features/Channels: ', 64)
# (u'import/mixed4a_1x1_w', 'Features/Channels: ', 192)
# (u'import/mixed4a_1x1_b', 'Features/Channels: ', 192)
# (u'import/mixed4a_3x3_bottleneck_w', 'Features/Channels: ', 96)
# (u'import/mixed4a_3x3_bottleneck_b', 'Features/Channels: ', 96)
# (u'import/mixed4a_3x3_w', 'Features/Channels: ', 204)
# (u'import/mixed4a_3x3_b', 'Features/Channels: ', 204)
# (u'import/mixed4a_5x5_bottleneck_w', 'Features/Channels: ', 16)
# (u'import/mixed4a_5x5_bottleneck_b', 'Features/Channels: ', 16)
# (u'import/mixed4a_5x5_w', 'Features/Channels: ', 48)
# (u'import/mixed4a_5x5_b', 'Features/Channels: ', 48)
# (u'import/mixed4a_pool_reduce_w', 'Features/Channels: ', 64)
# (u'import/mixed4a_pool_reduce_b', 'Features/Channels: ', 64)
# (u'import/mixed4b_1x1_w', 'Features/Channels: ', 160)
# (u'import/mixed4b_1x1_b', 'Features/Channels: ', 160)
# (u'import/mixed4b_3x3_bottleneck_w', 'Features/Channels: ', 112)
# (u'import/mixed4b_3x3_bottleneck_b', 'Features/Channels: ', 112)
# (u'import/mixed4b_3x3_w', 'Features/Channels: ', 224)
# (u'import/mixed4b_3x3_b', 'Features/Channels: ', 224)
# (u'import/mixed4b_5x5_bottleneck_w', 'Features/Channels: ', 24)
# (u'import/mixed4b_5x5_bottleneck_b', 'Features/Channels: ', 24)
# (u'import/mixed4b_5x5_w', 'Features/Channels: ', 64)
# (u'import/mixed4b_5x5_b', 'Features/Channels: ', 64)
# (u'import/mixed4b_pool_reduce_w', 'Features/Channels: ', 64)
# (u'import/mixed4b_pool_reduce_b', 'Features/Channels: ', 64)
# (u'import/mixed4c_1x1_w', 'Features/Channels: ', 128)
# (u'import/mixed4c_1x1_b', 'Features/Channels: ', 128)
# (u'import/mixed4c_3x3_bottleneck_w', 'Features/Channels: ', 128)
# (u'import/mixed4c_3x3_bottleneck_b', 'Features/Channels: ', 128)
# (u'import/mixed4c_3x3_w', 'Features/Channels: ', 256)
# (u'import/mixed4c_3x3_b', 'Features/Channels: ', 256)
# (u'import/mixed4c_5x5_bottleneck_w', 'Features/Channels: ', 24)
# (u'import/mixed4c_5x5_bottleneck_b', 'Features/Channels: ', 24)
# (u'import/mixed4c_5x5_w', 'Features/Channels: ', 64)
# (u'import/mixed4c_5x5_b', 'Features/Channels: ', 64)
# (u'import/mixed4c_pool_reduce_w', 'Features/Channels: ', 64)
# (u'import/mixed4c_pool_reduce_b', 'Features/Channels: ', 64)
# (u'import/mixed4d_1x1_w', 'Features/Channels: ', 112)
# (u'import/mixed4d_1x1_b', 'Features/Channels: ', 112)
# (u'import/mixed4d_3x3_bottleneck_w', 'Features/Channels: ', 144)
# (u'import/mixed4d_3x3_bottleneck_b', 'Features/Channels: ', 144)
# (u'import/mixed4d_3x3_w', 'Features/Channels: ', 288)
# (u'import/mixed4d_3x3_b', 'Features/Channels: ', 288)
# (u'import/mixed4d_5x5_bottleneck_w', 'Features/Channels: ', 32)
# (u'import/mixed4d_5x5_bottleneck_b', 'Features/Channels: ', 32)
# (u'import/mixed4d_5x5_w', 'Features/Channels: ', 64)
# (u'import/mixed4d_5x5_b', 'Features/Channels: ', 64)
# (u'import/mixed4d_pool_reduce_w', 'Features/Channels: ', 64)
# (u'import/mixed4d_pool_reduce_b', 'Features/Channels: ', 64)
# (u'import/mixed4e_1x1_w', 'Features/Channels: ', 256)
# (u'import/mixed4e_1x1_b', 'Features/Channels: ', 256)
# (u'import/mixed4e_3x3_bottleneck_w', 'Features/Channels: ', 160)
# (u'import/mixed4e_3x3_bottleneck_b', 'Features/Channels: ', 160)
# (u'import/mixed4e_3x3_w', 'Features/Channels: ', 320)
# (u'import/mixed4e_3x3_b', 'Features/Channels: ', 320)
# (u'import/mixed4e_5x5_bottleneck_w', 'Features/Channels: ', 32)
# (u'import/mixed4e_5x5_bottleneck_b', 'Features/Channels: ', 32)
# (u'import/mixed4e_5x5_w', 'Features/Channels: ', 128)
# (u'import/mixed4e_5x5_b', 'Features/Channels: ', 128)
# (u'import/mixed4e_pool_reduce_w', 'Features/Channels: ', 128)
# (u'import/mixed4e_pool_reduce_b', 'Features/Channels: ', 128)
# (u'import/mixed5a_1x1_w', 'Features/Channels: ', 256)
# (u'import/mixed5a_1x1_b', 'Features/Channels: ', 256)
# (u'import/mixed5a_3x3_bottleneck_w', 'Features/Channels: ', 160)
# (u'import/mixed5a_3x3_bottleneck_b', 'Features/Channels: ', 160)
# (u'import/mixed5a_3x3_w', 'Features/Channels: ', 320)
# (u'import/mixed5a_3x3_b', 'Features/Channels: ', 320)
# (u'import/mixed5a_5x5_bottleneck_w', 'Features/Channels: ', 48)
# (u'import/mixed5a_5x5_bottleneck_b', 'Features/Channels: ', 48)
# (u'import/mixed5a_5x5_w', 'Features/Channels: ', 128)
# (u'import/mixed5a_5x5_b', 'Features/Channels: ', 128)
# (u'import/mixed5a_pool_reduce_w', 'Features/Channels: ', 128)
# (u'import/mixed5a_pool_reduce_b', 'Features/Channels: ', 128)
# (u'import/mixed5b_1x1_w', 'Features/Channels: ', 384)
# (u'import/mixed5b_1x1_b', 'Features/Channels: ', 384)
# (u'import/mixed5b_3x3_bottleneck_w', 'Features/Channels: ', 192)
# (u'import/mixed5b_3x3_bottleneck_b', 'Features/Channels: ', 192)
# (u'import/mixed5b_3x3_w', 'Features/Channels: ', 384)
# (u'import/mixed5b_3x3_b', 'Features/Channels: ', 384)
# (u'import/mixed5b_5x5_bottleneck_w', 'Features/Channels: ', 48)
# (u'import/mixed5b_5x5_bottleneck_b', 'Features/Channels: ', 48)
# (u'import/mixed5b_5x5_w', 'Features/Channels: ', 128)
# (u'import/mixed5b_5x5_b', 'Features/Channels: ', 128)
# (u'import/mixed5b_pool_reduce_w', 'Features/Channels: ', 128)
# (u'import/mixed5b_pool_reduce_b', 'Features/Channels: ', 128)
# (u'import/head0_bottleneck_w', 'Features/Channels: ', 128)
# (u'import/head0_bottleneck_b', 'Features/Channels: ', 128)
# (u'import/nn0_w', 'Features/Channels: ', 1024)
# (u'import/nn0_b', 'Features/Channels: ', 1024)
# (u'import/softmax0_w', 'Features/Channels: ', 1008)
# (u'import/softmax0_b', 'Features/Channels: ', 1008)
# (u'import/head1_bottleneck_w', 'Features/Channels: ', 128)
# (u'import/head1_bottleneck_b', 'Features/Channels: ', 128)
# (u'import/nn1_w', 'Features/Channels: ', 1024)
# (u'import/nn1_b', 'Features/Channels: ', 1024)
# (u'import/softmax1_w', 'Features/Channels: ', 1008)
# (u'import/softmax1_b', 'Features/Channels: ', 1008)
# (u'import/softmax2_w', 'Features/Channels: ', 1008)
# (u'import/softmax2_b', 'Features/Channels: ', 1008)
# (u'import/conv2d0_pre_relu/conv', 'Features/Channels: ', 64)
# (u'import/conv2d0_pre_relu', 'Features/Channels: ', 64)
# (u'import/conv2d0', 'Features/Channels: ', 64)
# (u'import/maxpool0', 'Features/Channels: ', 64)
# (u'import/localresponsenorm0', 'Features/Channels: ', 64)
# (u'import/conv2d1_pre_relu/conv', 'Features/Channels: ', 64)
# (u'import/conv2d1_pre_relu', 'Features/Channels: ', 64)
# (u'import/conv2d1', 'Features/Channels: ', 64)
# (u'import/conv2d2_pre_relu/conv', 'Features/Channels: ', 192)
# (u'import/conv2d2_pre_relu', 'Features/Channels: ', 192)
# (u'import/conv2d2', 'Features/Channels: ', 192)
# (u'import/localresponsenorm1', 'Features/Channels: ', 192)
# (u'import/maxpool1', 'Features/Channels: ', 192)
# (u'import/mixed3a_1x1_pre_relu/conv', 'Features/Channels: ', 64)
# (u'import/mixed3a_1x1_pre_relu', 'Features/Channels: ', 64)
# (u'import/mixed3a_1x1', 'Features/Channels: ', 64)
# (u'import/mixed3a_3x3_bottleneck_pre_relu/conv', 'Features/Channels: ', 96)
# (u'import/mixed3a_3x3_bottleneck_pre_relu', 'Features/Channels: ', 96)
# (u'import/mixed3a_3x3_bottleneck', 'Features/Channels: ', 96)
# (u'import/mixed3a_3x3_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/mixed3a_3x3_pre_relu', 'Features/Channels: ', 128)
# (u'import/mixed3a_3x3', 'Features/Channels: ', 128)
# (u'import/mixed3a_5x5_bottleneck_pre_relu/conv', 'Features/Channels: ', 16)
# (u'import/mixed3a_5x5_bottleneck_pre_relu', 'Features/Channels: ', 16)
# (u'import/mixed3a_5x5_bottleneck', 'Features/Channels: ', 16)
# (u'import/mixed3a_5x5_pre_relu/conv', 'Features/Channels: ', 32)
# (u'import/mixed3a_5x5_pre_relu', 'Features/Channels: ', 32)
# (u'import/mixed3a_5x5', 'Features/Channels: ', 32)
# (u'import/mixed3a_pool', 'Features/Channels: ', 192)
# (u'import/mixed3a_pool_reduce_pre_relu/conv', 'Features/Channels: ', 32)
# (u'import/mixed3a_pool_reduce_pre_relu', 'Features/Channels: ', 32)
# (u'import/mixed3a_pool_reduce', 'Features/Channels: ', 32)
# (u'import/mixed3a', 'Features/Channels: ', 256)
# (u'import/mixed3b_1x1_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/mixed3b_1x1_pre_relu', 'Features/Channels: ', 128)
# (u'import/mixed3b_1x1', 'Features/Channels: ', 128)
# (u'import/mixed3b_3x3_bottleneck_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/mixed3b_3x3_bottleneck_pre_relu', 'Features/Channels: ', 128)
# (u'import/mixed3b_3x3_bottleneck', 'Features/Channels: ', 128)
# (u'import/mixed3b_3x3_pre_relu/conv', 'Features/Channels: ', 192)
# (u'import/mixed3b_3x3_pre_relu', 'Features/Channels: ', 192)
# (u'import/mixed3b_3x3', 'Features/Channels: ', 192)
# (u'import/mixed3b_5x5_bottleneck_pre_relu/conv', 'Features/Channels: ', 32)
# (u'import/mixed3b_5x5_bottleneck_pre_relu', 'Features/Channels: ', 32)
# (u'import/mixed3b_5x5_bottleneck', 'Features/Channels: ', 32)
# (u'import/mixed3b_5x5_pre_relu/conv', 'Features/Channels: ', 96)
# (u'import/mixed3b_5x5_pre_relu', 'Features/Channels: ', 96)
# (u'import/mixed3b_5x5', 'Features/Channels: ', 96)
# (u'import/mixed3b_pool', 'Features/Channels: ', 256)
# (u'import/mixed3b_pool_reduce_pre_relu/conv', 'Features/Channels: ', 64)
# (u'import/mixed3b_pool_reduce_pre_relu', 'Features/Channels: ', 64)
# (u'import/mixed3b_pool_reduce', 'Features/Channels: ', 64)
# (u'import/mixed3b', 'Features/Channels: ', 480)
# (u'import/maxpool4', 'Features/Channels: ', 480)
# (u'import/mixed4a_1x1_pre_relu/conv', 'Features/Channels: ', 192)
# (u'import/mixed4a_1x1_pre_relu', 'Features/Channels: ', 192)
# (u'import/mixed4a_1x1', 'Features/Channels: ', 192)
# (u'import/mixed4a_3x3_bottleneck_pre_relu/conv', 'Features/Channels: ', 96)
# (u'import/mixed4a_3x3_bottleneck_pre_relu', 'Features/Channels: ', 96)
# (u'import/mixed4a_3x3_bottleneck', 'Features/Channels: ', 96)
# (u'import/mixed4a_3x3_pre_relu/conv', 'Features/Channels: ', 204)
# (u'import/mixed4a_3x3_pre_relu', 'Features/Channels: ', 204)
# (u'import/mixed4a_3x3', 'Features/Channels: ', 204)
# (u'import/mixed4a_5x5_bottleneck_pre_relu/conv', 'Features/Channels: ', 16)
# (u'import/mixed4a_5x5_bottleneck_pre_relu', 'Features/Channels: ', 16)
# (u'import/mixed4a_5x5_bottleneck', 'Features/Channels: ', 16)
# (u'import/mixed4a_5x5_pre_relu/conv', 'Features/Channels: ', 48)
# (u'import/mixed4a_5x5_pre_relu', 'Features/Channels: ', 48)
# (u'import/mixed4a_5x5', 'Features/Channels: ', 48)
# (u'import/mixed4a_pool', 'Features/Channels: ', 480)
# (u'import/mixed4a_pool_reduce_pre_relu/conv', 'Features/Channels: ', 64)
# (u'import/mixed4a_pool_reduce_pre_relu', 'Features/Channels: ', 64)
# (u'import/mixed4a_pool_reduce', 'Features/Channels: ', 64)
# (u'import/mixed4a', 'Features/Channels: ', 508)
# (u'import/mixed4b_1x1_pre_relu/conv', 'Features/Channels: ', 160)
# (u'import/mixed4b_1x1_pre_relu', 'Features/Channels: ', 160)
# (u'import/mixed4b_1x1', 'Features/Channels: ', 160)
# (u'import/mixed4b_3x3_bottleneck_pre_relu/conv', 'Features/Channels: ', 112)
# (u'import/mixed4b_3x3_bottleneck_pre_relu', 'Features/Channels: ', 112)
# (u'import/mixed4b_3x3_bottleneck', 'Features/Channels: ', 112)
# (u'import/mixed4b_3x3_pre_relu/conv', 'Features/Channels: ', 224)
# (u'import/mixed4b_3x3_pre_relu', 'Features/Channels: ', 224)
# (u'import/mixed4b_3x3', 'Features/Channels: ', 224)
# (u'import/mixed4b_5x5_bottleneck_pre_relu/conv', 'Features/Channels: ', 24)
# (u'import/mixed4b_5x5_bottleneck_pre_relu', 'Features/Channels: ', 24)
# (u'import/mixed4b_5x5_bottleneck', 'Features/Channels: ', 24)
# (u'import/mixed4b_5x5_pre_relu/conv', 'Features/Channels: ', 64)
# (u'import/mixed4b_5x5_pre_relu', 'Features/Channels: ', 64)
# (u'import/mixed4b_5x5', 'Features/Channels: ', 64)
# (u'import/mixed4b_pool', 'Features/Channels: ', 508)
# (u'import/mixed4b_pool_reduce_pre_relu/conv', 'Features/Channels: ', 64)
# (u'import/mixed4b_pool_reduce_pre_relu', 'Features/Channels: ', 64)
# (u'import/mixed4b_pool_reduce', 'Features/Channels: ', 64)
# (u'import/mixed4b', 'Features/Channels: ', 512)
# (u'import/mixed4c_1x1_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/mixed4c_1x1_pre_relu', 'Features/Channels: ', 128)
# (u'import/mixed4c_1x1', 'Features/Channels: ', 128)
# (u'import/mixed4c_3x3_bottleneck_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/mixed4c_3x3_bottleneck_pre_relu', 'Features/Channels: ', 128)
# (u'import/mixed4c_3x3_bottleneck', 'Features/Channels: ', 128)
# (u'import/mixed4c_3x3_pre_relu/conv', 'Features/Channels: ', 256)
# (u'import/mixed4c_3x3_pre_relu', 'Features/Channels: ', 256)
# (u'import/mixed4c_3x3', 'Features/Channels: ', 256)
# (u'import/mixed4c_5x5_bottleneck_pre_relu/conv', 'Features/Channels: ', 24)
# (u'import/mixed4c_5x5_bottleneck_pre_relu', 'Features/Channels: ', 24)
# (u'import/mixed4c_5x5_bottleneck', 'Features/Channels: ', 24)
# (u'import/mixed4c_5x5_pre_relu/conv', 'Features/Channels: ', 64)
# (u'import/mixed4c_5x5_pre_relu', 'Features/Channels: ', 64)
# (u'import/mixed4c_5x5', 'Features/Channels: ', 64)
# (u'import/mixed4c_pool', 'Features/Channels: ', 512)
# (u'import/mixed4c_pool_reduce_pre_relu/conv', 'Features/Channels: ', 64)
# (u'import/mixed4c_pool_reduce_pre_relu', 'Features/Channels: ', 64)
# (u'import/mixed4c_pool_reduce', 'Features/Channels: ', 64)
# (u'import/mixed4c', 'Features/Channels: ', 512)
# (u'import/mixed4d_1x1_pre_relu/conv', 'Features/Channels: ', 112)
# (u'import/mixed4d_1x1_pre_relu', 'Features/Channels: ', 112)
# (u'import/mixed4d_1x1', 'Features/Channels: ', 112)
# (u'import/mixed4d_3x3_bottleneck_pre_relu/conv', 'Features/Channels: ', 144)
# (u'import/mixed4d_3x3_bottleneck_pre_relu', 'Features/Channels: ', 144)
# (u'import/mixed4d_3x3_bottleneck', 'Features/Channels: ', 144)
# (u'import/mixed4d_3x3_pre_relu/conv', 'Features/Channels: ', 288)
# (u'import/mixed4d_3x3_pre_relu', 'Features/Channels: ', 288)
# (u'import/mixed4d_3x3', 'Features/Channels: ', 288)
# (u'import/mixed4d_5x5_bottleneck_pre_relu/conv', 'Features/Channels: ', 32)
# (u'import/mixed4d_5x5_bottleneck_pre_relu', 'Features/Channels: ', 32)
# (u'import/mixed4d_5x5_bottleneck', 'Features/Channels: ', 32)
# (u'import/mixed4d_5x5_pre_relu/conv', 'Features/Channels: ', 64)
# (u'import/mixed4d_5x5_pre_relu', 'Features/Channels: ', 64)
# (u'import/mixed4d_5x5', 'Features/Channels: ', 64)
# (u'import/mixed4d_pool', 'Features/Channels: ', 512)
# (u'import/mixed4d_pool_reduce_pre_relu/conv', 'Features/Channels: ', 64)
# (u'import/mixed4d_pool_reduce_pre_relu', 'Features/Channels: ', 64)
# (u'import/mixed4d_pool_reduce', 'Features/Channels: ', 64)
# (u'import/mixed4d', 'Features/Channels: ', 528)
# (u'import/mixed4e_1x1_pre_relu/conv', 'Features/Channels: ', 256)
# (u'import/mixed4e_1x1_pre_relu', 'Features/Channels: ', 256)
# (u'import/mixed4e_1x1', 'Features/Channels: ', 256)
# (u'import/mixed4e_3x3_bottleneck_pre_relu/conv', 'Features/Channels: ', 160)
# (u'import/mixed4e_3x3_bottleneck_pre_relu', 'Features/Channels: ', 160)
# (u'import/mixed4e_3x3_bottleneck', 'Features/Channels: ', 160)
# (u'import/mixed4e_3x3_pre_relu/conv', 'Features/Channels: ', 320)
# (u'import/mixed4e_3x3_pre_relu', 'Features/Channels: ', 320)
# (u'import/mixed4e_3x3', 'Features/Channels: ', 320)
# (u'import/mixed4e_5x5_bottleneck_pre_relu/conv', 'Features/Channels: ', 32)
# (u'import/mixed4e_5x5_bottleneck_pre_relu', 'Features/Channels: ', 32)
# (u'import/mixed4e_5x5_bottleneck', 'Features/Channels: ', 32)
# (u'import/mixed4e_5x5_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/mixed4e_5x5_pre_relu', 'Features/Channels: ', 128)
# (u'import/mixed4e_5x5', 'Features/Channels: ', 128)
# (u'import/mixed4e_pool', 'Features/Channels: ', 528)
# (u'import/mixed4e_pool_reduce_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/mixed4e_pool_reduce_pre_relu', 'Features/Channels: ', 128)
# (u'import/mixed4e_pool_reduce', 'Features/Channels: ', 128)
# (u'import/mixed4e', 'Features/Channels: ', 832)
# (u'import/maxpool10', 'Features/Channels: ', 832)
# (u'import/mixed5a_1x1_pre_relu/conv', 'Features/Channels: ', 256)
# (u'import/mixed5a_1x1_pre_relu', 'Features/Channels: ', 256)
# (u'import/mixed5a_1x1', 'Features/Channels: ', 256)
# (u'import/mixed5a_3x3_bottleneck_pre_relu/conv', 'Features/Channels: ', 160)
# (u'import/mixed5a_3x3_bottleneck_pre_relu', 'Features/Channels: ', 160)
# (u'import/mixed5a_3x3_bottleneck', 'Features/Channels: ', 160)
# (u'import/mixed5a_3x3_pre_relu/conv', 'Features/Channels: ', 320)
# (u'import/mixed5a_3x3_pre_relu', 'Features/Channels: ', 320)
# (u'import/mixed5a_3x3', 'Features/Channels: ', 320)
# (u'import/mixed5a_5x5_bottleneck_pre_relu/conv', 'Features/Channels: ', 48)
# (u'import/mixed5a_5x5_bottleneck_pre_relu', 'Features/Channels: ', 48)
# (u'import/mixed5a_5x5_bottleneck', 'Features/Channels: ', 48)
# (u'import/mixed5a_5x5_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/mixed5a_5x5_pre_relu', 'Features/Channels: ', 128)
# (u'import/mixed5a_5x5', 'Features/Channels: ', 128)
# (u'import/mixed5a_pool', 'Features/Channels: ', 832)
# (u'import/mixed5a_pool_reduce_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/mixed5a_pool_reduce_pre_relu', 'Features/Channels: ', 128)
# (u'import/mixed5a_pool_reduce', 'Features/Channels: ', 128)
# (u'import/mixed5a', 'Features/Channels: ', 832)
# (u'import/mixed5b_1x1_pre_relu/conv', 'Features/Channels: ', 384)
# (u'import/mixed5b_1x1_pre_relu', 'Features/Channels: ', 384)
# (u'import/mixed5b_1x1', 'Features/Channels: ', 384)
# (u'import/mixed5b_3x3_bottleneck_pre_relu/conv', 'Features/Channels: ', 192)
# (u'import/mixed5b_3x3_bottleneck_pre_relu', 'Features/Channels: ', 192)
# (u'import/mixed5b_3x3_bottleneck', 'Features/Channels: ', 192)
# (u'import/mixed5b_3x3_pre_relu/conv', 'Features/Channels: ', 384)
# (u'import/mixed5b_3x3_pre_relu', 'Features/Channels: ', 384)
# (u'import/mixed5b_3x3', 'Features/Channels: ', 384)
# (u'import/mixed5b_5x5_bottleneck_pre_relu/conv', 'Features/Channels: ', 48)
# (u'import/mixed5b_5x5_bottleneck_pre_relu', 'Features/Channels: ', 48)
# (u'import/mixed5b_5x5_bottleneck', 'Features/Channels: ', 48)
# (u'import/mixed5b_5x5_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/mixed5b_5x5_pre_relu', 'Features/Channels: ', 128)
# (u'import/mixed5b_5x5', 'Features/Channels: ', 128)
# (u'import/mixed5b_pool', 'Features/Channels: ', 832)
# (u'import/mixed5b_pool_reduce_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/mixed5b_pool_reduce_pre_relu', 'Features/Channels: ', 128)
# (u'import/mixed5b_pool_reduce', 'Features/Channels: ', 128)
# (u'import/mixed5b', 'Features/Channels: ', 1024)
# (u'import/avgpool0', 'Features/Channels: ', 1024)
# (u'import/head0_pool', 'Features/Channels: ', 508)
# (u'import/head0_bottleneck_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/head0_bottleneck_pre_relu', 'Features/Channels: ', 128)
# (u'import/head0_bottleneck', 'Features/Channels: ', 128)
# (u'import/head0_bottleneck/reshape/shape', 'Features/Channels: ', 2)
# (u'import/head0_bottleneck/reshape', 'Features/Channels: ', 2048)
# (u'import/nn0_pre_relu/matmul', 'Features/Channels: ', 1024)
# (u'import/nn0_pre_relu', 'Features/Channels: ', 1024)
# (u'import/nn0', 'Features/Channels: ', 1024)
# (u'import/nn0/reshape/shape', 'Features/Channels: ', 2)
# (u'import/nn0/reshape', 'Features/Channels: ', 1024)
# (u'import/softmax0_pre_activation/matmul', 'Features/Channels: ', 1008)
# (u'import/softmax0_pre_activation', 'Features/Channels: ', 1008)
# (u'import/softmax0', 'Features/Channels: ', 1008)
# (u'import/head1_pool', 'Features/Channels: ', 528)
# (u'import/head1_bottleneck_pre_relu/conv', 'Features/Channels: ', 128)
# (u'import/head1_bottleneck_pre_relu', 'Features/Channels: ', 128)
# (u'import/head1_bottleneck', 'Features/Channels: ', 128)
# (u'import/head1_bottleneck/reshape/shape', 'Features/Channels: ', 2)
# (u'import/head1_bottleneck/reshape', 'Features/Channels: ', 2048)
# (u'import/nn1_pre_relu/matmul', 'Features/Channels: ', 1024)
# (u'import/nn1_pre_relu', 'Features/Channels: ', 1024)
# (u'import/nn1', 'Features/Channels: ', 1024)
# (u'import/nn1/reshape/shape', 'Features/Channels: ', 2)
# (u'import/nn1/reshape', 'Features/Channels: ', 1024)
# (u'import/softmax1_pre_activation/matmul', 'Features/Channels: ', 1008)
# (u'import/softmax1_pre_activation', 'Features/Channels: ', 1008)
# (u'import/softmax1', 'Features/Channels: ', 1008)
# (u'import/avgpool0/reshape/shape', 'Features/Channels: ', 2)
# (u'import/avgpool0/reshape', 'Features/Channels: ', 1024)
# (u'import/softmax2_pre_activation/matmul', 'Features/Channels: ', 1008)
# (u'import/softmax2_pre_activation', 'Features/Channels: ', 1008)
# (u'import/softmax2', 'Features/Channels: ', 1008)
# (u'import/output', 'Features/Channels: ', 1008)
# (u'import/output1', 'Features/Channels: ', 1008)
# (u'import/output2', 'Features/Channels: ', 1008)
# ('Number of layers', 360)
# ('Total number of feature channels:', 87322)
