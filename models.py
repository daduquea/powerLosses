"""Modular U-Net"""

# Installed packages
import tensorflow
if(0):#keras
    import keras
    from keras import backend as K
    from keras.models import Model
    from keras.layers import (
        Input,
        concatenate,
        Conv2D,
        MaxPooling2D,
        UpSampling2D,
        GaussianNoise,
        Dropout,
        Conv2DTranspose,
        SeparableConv2D,
        Activation,
        add,
        BatchNormalization,
    )

else:# tf2
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input,
        concatenate,
        Conv2D,
        MaxPooling2D,
        UpSampling2D,
        GaussianNoise,
        Dropout,
        Conv2DTranspose,
        SeparableConv2D,
        Activation,
        add,
        BatchNormalization,
    )


def unet_block(nb_filters, res, name, batch_norm):
    def fn(tensor):
        x = Conv2D(nb_filters, 3, padding="same", name="%s_1" % name)(tensor)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(nb_filters, 3, padding="same", name="%s_2" % name)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        if res:
            skip = Conv2D(nb_filters, 3, padding="same", name="%s_res" % name)(tensor)
            x = add([x, skip])
        x = Activation("relu")(x)
        return x

    return fn


def unet_down(conv_sampling, nb_filters=None):
    if conv_sampling:
        if nb_filters is None:
            raise ValueError("When conv_sampling is True, nb_filters should be given")

        def fn(tensor):
            x = Conv2D(nb_filters, 3, padding="same", strides=(2, 2))(tensor)
            return x

    else:

        def fn(tensor):
            x = MaxPooling2D(pool_size=(2, 2))(tensor)
            return x

    return fn


def unet_up(conv_sampling, nb_filters):
    if conv_sampling:
        if nb_filters is None:
            raise ValueError("When conv_sampling is True, nb_filters should be given")

        def fn(tensor):
            x = Conv2DTranspose(nb_filters, 3, padding="same", strides=(2, 2))(tensor)
            return x

    else:

        def fn(tensor):
            x = UpSampling2D(size=(2, 2))(tensor)
            return x

    return fn


def u_net_mod(
    shape,
    nb_filters_0=32,
    output_channels=1,
    sigma_noise=0.0,
    drop=0.0,
    skip=True,
    res=False,
    conv_sampling=False,
    batch_norm=False,
):
    """Modular U-Net.

    Note that the dimensions of the input images should be
    multiples of 16.

    Arguments:
    shape: image shape, in the format (x_size, y_size, nb_channels).
    nb_filters_0 : initial number of filters in the convolutional layer.
    output_channels: number of output channels.
    sigma_noise: standard deviation of the gaussian noise layer. If equal to zero, this layer is deactivated.
    drop: dropout rate.
    skip: boolean indicating if skip connections should be used.
    res: boolean indicating if residual blocks should be used.
    conv_sampling: boolean indicating if sub- and up-sampling should be
        done with convolutions
    batch_norm: should we use batch normalization?

    Returns:
    U-Net model - it still needs to be compiled.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI 2015
"""

    x0 = Input(shape)
    x1 = unet_block(nb_filters_0, res, "conv1", batch_norm)(x0)
    x2 = unet_down(conv_sampling, nb_filters_0)(x1)
    x2 = unet_block(nb_filters_0 * 2, res, "conv2", batch_norm)(x2)
    x3 = unet_down(conv_sampling, nb_filters_0 * 2)(x2)
    x3 = unet_block(nb_filters_0 * 4, res, "conv3", batch_norm)(x3)
    x4 = unet_down(conv_sampling, nb_filters_0 * 4)(x3)
    x4 = unet_block(nb_filters_0 * 8, res, "conv4", batch_norm)(x4)
    x5 = unet_down(conv_sampling, nb_filters_0 * 8)(x4)
    x5 = unet_block(nb_filters_0 * 16, res, "conv5", batch_norm)(x5)
    if drop > 0.0:
        x5 = Dropout(drop)(x5)
    x6 = unet_up(conv_sampling, nb_filters_0 * 16)(x5)
    if skip:
        x6 = concatenate([x6, x4], axis=3)
    x6 = unet_block(nb_filters_0 * 8, res, "conv6", batch_norm)(x6)
    x7 = unet_up(conv_sampling, nb_filters_0 * 8)(x6)
    if skip:
        x7 = concatenate([x7, x3], axis=3)
    x7 = unet_block(nb_filters_0 * 4, res, "conv7", batch_norm)(x7)
    x8 = unet_up(conv_sampling, nb_filters_0 * 4)(x7)
    if skip:
        x8 = concatenate([x8, x2], axis=3)
    x8 = unet_block(nb_filters_0 * 2, res, "conv8", batch_norm)(x8)
    x9 = unet_up(conv_sampling, nb_filters_0 * 2)(x8)
    if skip:
        x9 = concatenate([x9, x1], axis=3)
    x9 = unet_block(nb_filters_0, res, "conv9", batch_norm)(x9)
    if sigma_noise > 0:
        x9 = GaussianNoise(sigma_noise)(x9)
    x10 = Conv2D(output_channels, 1, activation="sigmoid", name="out")(x9)

    return Model(x0, x10)

def u_net_mod_lev(
    shape,
    nb_levels = 4,
    nb_filters_0=32,
    output_channels=1,
    sigma_noise=0.0,
    drop=0.0,
    skip=True,
    res=False,
    conv_sampling=False,
    batch_norm=False,
):
    """Modular U-Net.

    Note that the dimensions of the input images should be
    multiples of 16.

    Arguments:
    shape: image shape, in the format (x_size, y_size, nb_channels).
    nb_levels: U "depth", number of downsampling and upsampling steps
    nb_filters_0 : initial number of filters in the convolutional layer.
    output_channels: number of output channels.
    sigma_noise: standard deviation of the gaussian noise layer. If equal to zero, this layer is deactivated.
    drop: dropout rate.
    skip: boolean indicating if skip connections should be used.
    res: boolean indicating if residual blocks should be used.
    conv_sampling: boolean indicating if sub- and up-sampling should be
        done with convolutions
    batch_norm: should we use batch normalization?

    Returns:
    U-Net model - it still needs to be compiled.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI 2015
"""

    x = Input(shape)

    first_layer = []
    first_layer.append(x)
    
    skip_layers = []

    nb_filters = nb_filters_0
    for lev in range(nb_levels):
        conv_name = "conv"+str(lev+1)
        x = unet_block(nb_filters, res, conv_name, batch_norm)(x)

        if(lev < nb_levels -1):#not the last one
            skip_layers.append(x)
            x = unet_down(conv_sampling, nb_filters)(x)
            nb_filters = nb_filters * 2


    if drop > 0.0:
        x = Dropout(drop)(x)

    for lev in range(nb_levels-1):
        x = unet_up(conv_sampling, nb_filters)(x)
        if skip:
            skip_layer = skip_layers.pop()
            x = concatenate([x, skip_layer], axis=3)
        nb_filters= int(nb_filters/2)
        conv_name = "conv"+str(nb_levels+lev+1)
        x = unet_block(nb_filters, res, conv_name, batch_norm)(x)

    if sigma_noise > 0:
        x = GaussianNoise(sigma_noise)(x)
    xFinal = Conv2D(output_channels, 1, activation="sigmoid", name="out")(x)

    x0 = first_layer.pop()
    return Model(x0, xFinal)

