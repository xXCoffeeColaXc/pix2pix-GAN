from keras.models import Model
from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Activation, Concatenate
from keras.optimizers import Adam
from keras.initializers import RandomNormal


class CNNBlock:
    def __init__(self, filters, kernel_size=(4, 4), strides=(2, 2), padding='same', alpha=0.2):
        init = RandomNormal(stddev=0.02)
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=init)
        self.batch_norm = BatchNormalization()
        self.activation = LeakyReLU(alpha=alpha)

    def __call__(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class Discriminator:
    def __init__(self, image_shape):
        # source image input
        in_src_image = Input(shape=image_shape)
        # target image input
        in_target_image = Input(shape=image_shape)

        # concatenate images channel-wise
        merged = Concatenate()([in_src_image, in_target_image])

        # create CNN blocks
        self.block1 = CNNBlock(64)
        self.block2 = CNNBlock(128)
        self.block3 = CNNBlock(256)
        self.block4 = CNNBlock(512)

        # build model
        d = self.block1(merged)
        d = self.block2(d)
        d = self.block3(d)
        d = self.block4(d)

        # second last output layer
        d = Conv2D(512, (4, 4), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        # patch output
        d = Conv2D(1, (4, 4), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d)
        patch_out = Activation('sigmoid')(d)

        # define and compile model
        self.model = Model([in_src_image, in_target_image], patch_out)
        opt = Adam(lr=0.0002, beta_1=0.5)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])

    def get_model(self):
        return self.model


# Usage
#disc = Discriminator((256, 256, 3))
#model = disc.get_model()
#model.summary()
