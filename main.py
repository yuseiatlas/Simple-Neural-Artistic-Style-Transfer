from __future__ import print_function

import time
from PIL import Image
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.applications.vgg16 import VGG16
from scipy.optimize import fmin_l_bfgs_b

import util
from evaluator import Evaluator

height = 512
width = 512
content_weight = 0.025
style_weight = 2.5
total_variation_weight = 1.0

loss = K.variable(0.)

content_image_path = 'images/content/neo.jpg'
content_image = Image.open(content_image_path)
content_image = content_image.resize((width, height))

style_image_path = 'images/styles/scream.jpg'
style_image = Image.open(style_image_path)
style_image = style_image.resize((width, height))

content_array = np.asarray(content_image, dtype='float32')
content_array = np.expand_dims(content_array, axis=0)

style_array = np.asarray(style_image, dtype='float32')
style_array = np.expand_dims(style_array, axis=0)

content_array = util.rbg2bgr(content_array)
style_array = util.rbg2bgr(style_array)

content_image = K.variable(content_array)
style_image = K.variable(style_array)
combination_image = K.placeholder((1, height, width, 3))

input_tensor = K.concatenate([content_image,
                              style_image,
                              combination_image], axis=0)

model = VGG16(input_tensor=input_tensor, weights='imagenet',
              include_top=False)

layers = dict([(layer.name, layer.output) for layer in model.layers])

layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss = loss + content_weight * util.content_loss(content_image_features,
                                                 combination_features)

feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']
size = height * width
for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = util.style_loss(style_features, combination_features, size)
    loss += (style_weight / len(feature_layers)) * sl


loss += total_variation_weight * \
    util.total_variation_loss(combination_image, height, width)

grads = K.gradients(loss, combination_image)

outputs = [loss]
outputs += grads
f_outputs = K.function([combination_image], outputs)

evaluator = Evaluator(height, width, f_outputs)

x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

prev_loss = None
loss_percentage = 100
i = 0

while(loss_percentage > 5 and i < 9):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
    if(prev_loss != None):
        loss_percentage = util.calculate_loss_drop_percentage(
            min_val, prev_loss)
        print('Loss drop percentage: ', loss_percentage, '%')
    prev_loss = min_val
    i += 1

x = util.deprocess_image(x, height, width)
util.imshow(x)
