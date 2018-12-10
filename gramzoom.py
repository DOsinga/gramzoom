#!/usr/bin/env python

import argparse
import os

import PIL
import random
import imageio
import numpy as np
import tqdm
from PIL import Image
from keras import backend as K
from keras.applications import vgg16
from keras.preprocessing.image import img_to_array
from scipy import ndimage
from scipy.optimize import fmin_l_bfgs_b


class Evaluator(object):
    def __init__(self, loss_total, result_image, **other):
        grads = K.gradients(loss_total, result_image)
        outputs = [loss_total] + list(other.values()) + grads
        self.iterate = K.function([result_image], outputs)
        self.other = list(other.keys())
        self.other_values = {}
        self.shape = result_image.shape

        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        outs = self.iterate([x.reshape(self.shape)])
        self.loss_value = outs[0]
        self.grad_values = outs[-1].flatten().astype('float64')
        self.other_values = dict(zip(self.other, outs[1:-1]))
        return self.loss_value

    def grads(self, x):
        return np.copy(self.grad_values)

def preprocess_image(image_path, target_size):
    img = PIL.Image.open(image_path)
    w, h = img.size
    min_size = min(img.size)
    new_w = int(target_size * w / min_size)
    new_h = int(target_size * h / min_size)
    img = img.resize((new_w, new_h))
    img = img.crop(((new_w - target_size) // 2,
                    (new_h - target_size) // 2,
                    (new_w + target_size) // 2,
                    (new_h + target_size) // 2))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

def deprocess_image(x, w, h):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, w, h))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((w, h, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


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
    return ndimage.zoom(img, factors, order=1)


def gram_matrix(x):
    if K.image_data_format() != 'channels_first':
        x = K.permute_dimensions(x, (2, 0, 1))
    features = K.batch_flatten(x)
    return K.dot(features - 1, K.transpose(features - 1)) - 1


def style_loss(layer_1, layer_2):
    gr1 = gram_matrix(layer_1)
    gr2 = gram_matrix(layer_2)
    return K.sum(K.square(gr1 - gr2)) / (np.prod(layer_2.shape).value ** 2)


def create_evalutor(model, result_image):
    feature_outputs = [layer.output for layer in model.layers if '_conv' in layer.name]

    loss_style = K.variable(0.)
    for idx, layer_features in enumerate(feature_outputs):
        loss_style += style_loss(layer_features[0, :, :, :], layer_features[1, :, :, :])

    return Evaluator(loss_style, result_image)


def create_input_tensor(image_path, target_size):
    preprocessed = preprocess_image(image_path, target_size=target_size)
    style_image = K.variable(preprocessed)
    result_image = K.placeholder(style_image.shape)
    return K.concatenate([style_image, result_image], axis=0), result_image


def run(evaluator, org_image, num_iter, target_size, zoom_from=None, samples_per_step=10):
    res = []
    image = org_image.copy()
    for i in tqdm.tqdm(list(range(num_iter * samples_per_step))):
        # We roll the image to make the tileable and avoid creases:
        offset = random.randrange(0, target_size)
        image = np.roll(image, (offset, offset), axis=(1, 2))
        image, min_val, info = fmin_l_bfgs_b(evaluator.loss, image.flatten(), fprime=evaluator.grads, maxfun=20)
        image = image.reshape(org_image.shape)
        image = np.roll(image, (-offset, -offset), axis=(1, 2))

        if i % samples_per_step == samples_per_step - 1:
            deprocessed = deprocess_image(image.copy(), target_size, target_size)
            res.append(deprocessed)
            if zoom_from and i >= zoom_from * samples_per_step:
                zoom_size = int(target_size * 1.03)
                image = resize_img(image, (zoom_size, zoom_size))
                image = np.clip(image, -128, 127)
                start = (zoom_size - target_size) // 2
                image = image[:, start: start + target_size, start: start + target_size, :]

                # Blur out a cross:
                center_size = target_size // 16
                center_start = (target_size - center_size) // 2
                image[0,
                      center_start: center_start + center_size,
                      center_start - center_size: center_start + center_size * 2,
                      :] += np.random.uniform(size=(center_size, center_size * 3, 3)) * 16 - 8
                image[0,
                      center_start - center_size: center_start + center_size * 2,
                      center_start: center_start + center_size,
                      :] += np.random.uniform(size=(center_size * 3, center_size, 3)) * 16 - 8

    return res


def write_results(output, frames):
    _, ext = os.path.splitext(output)
    if ext.lower() in ('.mp4', '.mpeg', '.mov'):
        with imageio.get_writer(output, mode='I', fps=25) as writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        PIL.Image.fromarray(frames[-1]).save(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='shells.jpg')
    parser.add_argument('--target_size', type=int, default=512)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--samples_per_step', type=int, default=10)
    parser.add_argument('--zoom_from', type=int, default=2,
                        help='Number of steps after which zooming starts. Also used for the movie cut-off')

    parser.add_argument('--output', type=str, default='shells.mp4')
    args = parser.parse_args()

    input_tensor, result_image = create_input_tensor(args.input, target_size=args.target_size)
    model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    evaluator = create_evalutor(model, result_image)

    x = np.random.uniform(0, 255, result_image.shape) - 128.
    frames = run(evaluator, x, num_iter=args.steps, target_size=args.target_size, zoom_from=args.zoom_from, samples_per_step=args.samples_per_step)

    write_results(args.output, frames[args.zoom_from:])

