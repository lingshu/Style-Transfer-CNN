import os
from argparse import ArgumentParser
import scipy
from sys import stderr
import time
from PIL import Image
import functools
import tensorflow as tf
import numpy as np
import scipy.io
from operator import mul

VggPath = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'
CONTENT_WEIGHT = 5e0
Style_Weight = 5e2
TV_Weight = 1e2
BETA1 = 0.9
BETA2 = 0.999
STYLE_SCALE = 1.0
ITERATIONS = 2
LEARNING_RATE = 1e1
EPSILON = 1e-08
ContentLayers = ('relu4_2', 'relu5_2')
StyleLayers = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

# kc3143
# Incoming parameters on the command line
def build_parser():

    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--styles',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    return parser

# kc3143
# Image processing
def imageProcess(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        img = img[:,:,:3]
    return img

# kc3143
# Save output image
def imageSave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

# kc3143
# Input images, output images and style transfer function
def main():

    parser = build_parser()
    options = parser.parse_args()

    # read content and style images
    content_image = imageProcess(options.content)
    style_images = [imageProcess(style) for style in options.styles]

    style_blend_weights = [1.0/len(style_images) for _ in style_images]

    for iteration, image in stylize(initial=content_image, initial_noiseblend=1.0, content=content_image,
        styles=style_images, network=VggPath, preserve_colors=False, iterations=ITERATIONS,
        learning_rate=LEARNING_RATE, content_weight=CONTENT_WEIGHT, content_weight_blend=1,
        style_weight=Style_Weight, style_layer_weight_exp=1,
        style_blend_weights=style_blend_weights, tv_weight=TV_Weight, beta1=BETA1, beta2=BETA2, epsilon=EPSILON,
        pooling=POOLING):

        outputFile = None
        generate = image
        if iteration is not None:
            outputFile = options.output
        if outputFile:
            imageSave(outputFile, generate)

# kc3143
# Get convolutional layer
def convLayer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                        padding='SAME')
    return tf.nn.bias_add(conv, bias)

# kc3143
# Architecture of VGG19
VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

# kc3143
# Get layers after pooling
def poolLayer(input, pooling):
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')

# kc3143
# load weights in vgg19
def load_vggWeights(weights, input_image, pooling):
    net = {}
    currImage = input_image
    for i, name_of_layers in enumerate(VGG19_LAYERS):
        kind = name_of_layers[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            bias = bias.reshape(-1)
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            currImage = convLayer(currImage, kernels, bias)

        elif kind == 'relu':
            currImage = tf.nn.relu(currImage)

        elif kind == 'pool':
            currImage = poolLayer(currImage, pooling)

        net[name_of_layers] = currImage

    assert len(net) == len(VGG19_LAYERS)
    return net

# kc3143
# Get tensor sizes
def tensorSize(tensor):
    return functools.reduce(mul, (d.value for d in tensor.get_shape()), 1)

# kc3143
# load weights and mean pixel from VGG19 path
def load_net(data_path):
    data = scipy.io.loadmat(data_path)
    # if not all(i in data for i in ('layers', 'classes', 'normalization')):
    #     raise ValueError("You're using the wrong VGG19 data. Please follow the instructions in the README to download the correct data.")
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    return weights, mean_pixel

# kc3143
# Stylize images
def stylize(initial, initial_noiseblend, content, styles, network, preserve_colors, iterations, learning_rate,
        content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
            beta1, beta2, epsilon, pooling,
        print_iterations=None, checkpoint_iterations=None):

    # load vgg weights
    vgg_weights, vggMeanPixel = load_net(network)

    layer_weight = 1.0
    style_layers_weights = {}

    # change the shape of images
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    cont_features = {}
    style_features = [{} for _ in styles]

    # style layer weights normalization
    sum_of_layerWeights = 0
    for i in StyleLayers:
        style_layers_weights[i] = layer_weight
        layer_weight *= style_layer_weight_exp

    for i in StyleLayers:
        sum_of_layerWeights += style_layers_weights[i]

    for i in StyleLayers:
        style_layers_weights[i] /= sum_of_layerWeights

    # compute content features in forward propagation
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = load_vggWeights(vgg_weights, image, pooling)

        contentProcessed = np.array([content - vggMeanPixel])
        for layer in ContentLayers:
            cont_features[layer] = net[layer].eval(feed_dict={image: contentProcessed})

    # compute style features in forward propagation
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=style_shapes[0])
        net = load_vggWeights(vgg_weights, image, pooling)
        styleProcessed = np.array([styles[0] - vggMeanPixel])
        for i in StyleLayers:
            featuresStyle = net[i].eval(feed_dict={image: styleProcessed})
            featuresStyle = np.reshape(featuresStyle, (-1, featuresStyle.shape[3]))
            gram = np.matmul(featuresStyle.T, featuresStyle) / featuresStyle.size
            style_features[0][i] = gram

    contentNoiseCoeff_initial = 1.0 - initial_noiseblend

    # generate stylized image in the process of backpropogation
    with tf.Graph().as_default():
        # generate a white noise image
        initial = np.array([initial - vggMeanPixel])
        initial = initial.astype('float32')
        initial = (initial) * contentNoiseCoeff_initial + (tf.random_normal(shape) * 0.256) * (1.0 - contentNoiseCoeff_initial)
        image = tf.Variable(initial)
        net = load_vggWeights(vgg_weights, image, pooling)

        # Caculate content loss
        contentLayersWeights = {}
        contentLayersWeights['relu4_2'] = content_weight_blend
        contentLayersWeights['relu5_2'] = 1.0 - content_weight_blend

        contentLoss = 0
        contentLosses = []
        for i in ContentLayers:
            contentLosses.append(contentLayersWeights[i] * content_weight * (2 * tf.nn.l2_loss(net[i] -
                                cont_features[i]) / cont_features[i].size))
        contentLoss += functools.reduce(tf.add, contentLosses)

        # Calculate style loss
        styleLoss = 0
        styleLosses = []
        for i in StyleLayers:
            layer = net[i]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_gram = style_features[0][i]
            styleLosses.append(style_layers_weights[i] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
        styleLoss += style_weight * style_blend_weights[0] * functools.reduce(tf.add, styleLosses)

        # Calculate total variation denoising
        tv_ySize = tensorSize(image[:,1:,:,:])
        tv_xSize = tensorSize(image[:,:,1:,:])
        tvLoss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) / tv_ySize) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) / tv_xSize))

        # Calculate total loss
        loss = contentLoss + styleLoss + tvLoss

        # Set up optimizer
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        def print_progress():
            stderr.write('  content loss: %g\n' % contentLoss.eval())
            stderr.write('    style loss: %g\n' % styleLoss.eval())
            stderr.write('       tv loss: %g\n' % tvLoss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())

        # Update generated image
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iterations):
                stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                train_step.run()

                if i == iterations - 1 or (print_iterations and i % print_iterations == 0):
                    print_progress()

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or i == iterations - 1:
                    loss1 = loss.eval()
                    if loss1< best_loss:
                        best_loss = loss1
                        best = image.eval()

                    img_out = best.reshape(shape[1:]) + vggMeanPixel

                    yield (
                        (None if i == iterations - 1 else i),
                        img_out
                    )

if __name__ == '__main__':
    main()
