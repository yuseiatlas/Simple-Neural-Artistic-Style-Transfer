import numpy as np
from keras import backend as K
from PIL import Image
from matplotlib import pyplot as plt

# helper functions


def rbg2bgr(image_array):
    image_array[:, :, :, 0] -= 103.939
    image_array[:, :, :, 1] -= 116.779
    image_array[:, :, :, 2] -= 123.68
    image_array = image_array[:, :, :, ::-1]
    return image_array


def deprocess_image(x, height, width):
    x = x.reshape((height, width, 3))
    x = x[:, :, ::-1]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    return Image.fromarray(x)


def imshow(x):
    plt.imshow(x)
    plt.axis('off')
    plt.show()


def content_loss(content, combination):
    return K.sum(K.square(combination - content))


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination, size):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x, height, width):
    a = K.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = K.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def calculate_loss_drop_percentage(part, whole):
    assert (whole >= part), "the part cannot be greater than the whole"
    return 100 - (100 * part / whole)
