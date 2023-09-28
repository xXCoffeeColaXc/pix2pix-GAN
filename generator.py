import tensorflow as tf

from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Conv2DTranspose, Dropout, Concatenate, Activation
from keras.models import Model
from keras.initializers import RandomNormal


class EncoderBlock:
    def __init__(self, n_filters, batchnorm=True):
        init = RandomNormal(stddev=0.02)
        self.conv = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)
        self.batchnorm = batchnorm

    def __call__(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x


class DecoderBlock:
    def __init__(self, n_filters, dropout=True):
        init = RandomNormal(stddev=0.02)
        self.trans_conv = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)
        self.dropout = dropout

    def __call__(self, x, skip_in):
        x = self.trans_conv(x)
        x = BatchNormalization()(x)
        if self.dropout:
            x = Dropout(0.5)(x)
        x = Concatenate()([x, skip_in])
        x = Activation('relu')(x)
        return x


class Generator:
    def __init__(self, image_shape=(256, 256, 3)):
        init = RandomNormal(stddev=0.02)
        in_image = Input(shape=image_shape)

        self.e1 = EncoderBlock(64, batchnorm=False)
        self.e2 = EncoderBlock(128)
        self.e3 = EncoderBlock(256)
        self.e4 = EncoderBlock(512)
        self.e5 = EncoderBlock(512)
        self.e6 = EncoderBlock(512)
        self.e7 = EncoderBlock(512)

        self.d1 = DecoderBlock(512)
        self.d2 = DecoderBlock(512)
        self.d3 = DecoderBlock(512)
        self.d4 = DecoderBlock(512, dropout=False)
        self.d5 = DecoderBlock(256, dropout=False)
        self.d6 = DecoderBlock(128, dropout=False)
        self.d7 = DecoderBlock(64, dropout=False)

        self.b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)
        self.out_conv = Conv2DTranspose(image_shape[2], (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)

        # build model
        e1 = self.e1(in_image)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)

        b = Activation('relu')(self.b(e7))

        d1 = self.d1(b, e7)
        d2 = self.d2(d1, e6)
        d3 = self.d3(d2, e5)
        d4 = self.d4(d3, e4)
        d5 = self.d5(d4, e3)
        d6 = self.d6(d5, e2)
        d7 = self.d7(d6, e1)

        out_image = Activation('tanh')(self.out_conv(d7))

        self.model = Model(in_image, out_image)

    def get_model(self):
        return self.model


# Usage
#gen = Generator()
#model = gen.get_model()
#model.summary()
