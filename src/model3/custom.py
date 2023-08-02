from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
import keras

class InstanceNormalization(tf.keras.layers.Layer):

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset


def custom_unet_generator_v2(input_size=(256, 256, 3)):

    inputs = keras.layers.Input(input_size)
    conv1 = keras.layers.Conv2D(64, 3, activation=tf.keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal',use_bias=False)(inputs)
    conv1 = keras.layers.Conv2D(64, 3, activation=tf.keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal',use_bias=False,strides=2)(conv1)

    conv2 = keras.layers.Conv2D(128, 3, activation=tf.keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal',use_bias=False)(conv1)
    conv2 = keras.layers.Conv2D(128, 3, activation=tf.keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal',use_bias=False,strides=2)(conv2)
    conv2 = InstanceNormalization()(conv2)

    conv3 = keras.layers.Conv2D(256, 3, activation=tf.keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal',use_bias=False)(conv2)
    conv3 = keras.layers.Conv2D(256, 3, activation=tf.keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal',use_bias=False,strides=2)(conv3)
    conv3 = InstanceNormalization()(conv3)

    conv4 = keras.layers.Conv2D(512, 3, activation=tf.keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal',use_bias=False)(conv3)
    conv4 = keras.layers.Conv2D(512, 3, activation=tf.keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal',use_bias=False,strides=2)(conv4)
    conv4 = InstanceNormalization()(conv4)

    conv5 = keras.layers.Conv2D(1024, 3, activation=tf.keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal',use_bias=False,strides=2)(conv4)
    conv5 = keras.layers.Conv2D(1024, 3, activation=tf.keras.layers.LeakyReLU(), padding='same', kernel_initializer='he_normal',use_bias=False)(conv5)


    up6 = keras.layers.Conv2DTranspose(512, 2, activation=tf.keras.layers.ReLU(), padding='same', kernel_initializer='he_normal',use_bias=False,strides=2)(conv5)
    merge6 = keras.layers.concatenate([conv4, up6], axis=3)
    conv6 = keras.layers.Conv2D(512, 3, activation=tf.keras.layers.ReLU(), padding='same', kernel_initializer='he_normal',use_bias=False)(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Dropout(0.5)(conv6)

    up7 = keras.layers.Conv2DTranspose(256, 2, activation=tf.keras.layers.ReLU(), padding='same', kernel_initializer='he_normal',use_bias=False,strides=2)(conv6)
    merge7 = keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = keras.layers.Conv2D(256, 3, activation=tf.keras.layers.ReLU(), padding='same', kernel_initializer='he_normal',use_bias=False)(merge7)
    conv7 = keras.layers.Conv2D(256, 3, activation=tf.keras.layers.ReLU(), padding='same', kernel_initializer='he_normal',use_bias=False)(conv7)
    conv7 = Dropout(0.5)(conv7)

    up8 = keras.layers.Conv2DTranspose(128, 2, activation=tf.keras.layers.ReLU(), padding='same', kernel_initializer='he_normal',use_bias=False,strides=2)(conv7)
    merge8 = keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = keras.layers.Conv2D(128, 3, activation=tf.keras.layers.ReLU(), padding='same', kernel_initializer='he_normal',use_bias=False)(merge8)
    conv8 = keras.layers.Conv2D(128, 3, activation=tf.keras.layers.ReLU(), padding='same', kernel_initializer='he_normal',use_bias=False)(conv8)


    up9 = keras.layers.Conv2DTranspose(64, 2, activation=tf.keras.layers.ReLU(), padding='same', kernel_initializer='he_normal',use_bias=False,strides=2)(conv8)
    merge9 = keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = keras.layers.Conv2D(64, 3, activation=tf.keras.layers.ReLU(), padding='same', kernel_initializer='he_normal',use_bias=False)(merge9)
    conv9 = keras.layers.Conv2D(64, 3, activation=tf.keras.layers.ReLU(), padding='same', kernel_initializer='he_normal',use_bias=False)(conv9)
    conv9 = tf.keras.layers.Conv2DTranspose(3, 3, strides=2,padding='same', kernel_initializer='he_normal',activation='tanh')(conv9)  # (bs, 256, 256, 3)

    model = keras.models.Model(inputs=inputs, outputs=conv9)

    return model

# model = custom_unet_generator_v2()
# print(model.summary())


def custom_unet_descriminator_v2(target=True):
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image') #shape=[None, None, 3]
    x = inp
    if target:
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image') #shape=[None, None, 3]
        x = tf.keras.layers.concatenate([inp, tar])

    f_conv1 = keras.layers.Conv2D(32, 4, activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal',use_bias=False, strides=1)(x)
    conv1 = keras.layers.Conv2D(64, 4, activation=tf.keras.layers.LeakyReLU(),kernel_initializer='he_normal', use_bias=False,strides=2)(f_conv1)
    conv2 = keras.layers.Conv2D(128, 4, activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal',
                                use_bias=False, strides=2)(conv1)
    conv2 = InstanceNormalization()(conv2)


    conv3 = keras.layers.Conv2D(256, 4, activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal',
                                use_bias=False, strides=2)(conv2)
    conv3 = InstanceNormalization()(conv3)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(conv3)
    conv4 = tf.keras.layers.Conv2D(
        512, 4, strides=1, activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal',
        use_bias=False)(zero_pad1)
    conv4 = InstanceNormalization()(conv4)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(conv4)

    conv5 = tf.keras.layers.Conv2D(
        512, 4, strides=1, activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal',
        use_bias=False)(zero_pad2)
    conv5 = InstanceNormalization()(conv5)


    conv6 = tf.keras.layers.Conv2D(
        1, 4, strides=1, kernel_initializer='he_normal')(conv5)  # (bs, 24, 24, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=conv6)
    else:
        return tf.keras.Model(inputs=inp, outputs=conv6)

# model = custom_unet_descriminator_v2()
# print(model.summary())

def custom_unet_descriminator_dilated_v2(target=True):
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image') #shape=[None, None, 3]
    x = inp
    if target:
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image') #shape=[None, None, 3]
        x = tf.keras.layers.concatenate([inp, tar])

    f_conv1 = keras.layers.Conv2D(32, 4, activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal',use_bias=False, strides=1)(x)
    conv1 = keras.layers.Conv2D(64, 4, activation=tf.keras.layers.LeakyReLU(),kernel_initializer='he_normal', use_bias=False,strides=2)(f_conv1)
    conv2 = keras.layers.Conv2D(128, 4, activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal',use_bias=False, strides=2)(conv1)
    conv2 = InstanceNormalization()(conv2)

    # Add a dilated convolution layer with dilation rate 2
    dilated_conv1 = tf.keras.layers.Conv2D(
        256, 3, strides=1, dilation_rate=2,
        kernel_initializer='he_normal', use_bias=False)(conv2)  # (bs, 30, 30, 256)
    dilated_conv1 = InstanceNormalization()(dilated_conv1)

    conv3 = keras.layers.Conv2D(512, 4, activation=tf.keras.layers.LeakyReLU(), kernel_initializer='he_normal',
                                use_bias=False, strides=2)(dilated_conv1)
    conv3 = InstanceNormalization()(conv3)

    zero_pad = tf.keras.layers.ZeroPadding2D()(conv3)

    # Add another dilated convolution layer with dilation rate 2
    dilated_conv2 = tf.keras.layers.Conv2D(
        1024, 3, strides=1, dilation_rate=2,
        kernel_initializer='he_normal', use_bias=False)(zero_pad)  # (bs, 16, 16, 1024)
    dilated_conv2 = InstanceNormalization()(dilated_conv2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(dilated_conv2)
    conv4 = tf.keras.layers.Conv2D(
        1, 4, strides=1, kernel_initializer='he_normal')(zero_pad1)  # (bs, 15, 15, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=conv4)
    else:
        return tf.keras.Model(inputs=inp, outputs=conv4)

#
# model = custom_unet_descriminator_dilated_v2()
# print(model.summary())