from keras.layers import Input, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam


class GAN:
    def __init__(self, g_model, d_model, image_shape):
        # make weights in the discriminator not trainable
        for layer in d_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        # define the source image
        self.in_src = Input(shape=image_shape)
        # supply the image as input to the generator
        self.gen_out = g_model(self.in_src)
        # supply the input image and generated image as inputs to the discriminator
        self.dis_out = d_model([self.in_src, self.gen_out])

        # src image as input, generated image and disc. output as outputs
        self.model = Model(self.in_src, [self.dis_out, self.gen_out])

        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        self.model.compile(loss=['binary_crossentropy', 'mae'],
                           optimizer=opt, loss_weights=[1, 100])

    def get_model(self):
        return self.model


# Usage
# Assuming g_model and d_model are previously defined and compiled
# and image_shape is specified
#gan = GAN(g_model, d_model, image_shape)
#model = gan.get_model()
#model.summary()
