import tensorflow as tf
import tensorflow.keras.backend as K
def discriminator_loss(real_output, fake_output):
    real_loss = K.mean(real_output)
    fake_loss = K.mean(fake_output)
    total_loss = fake_loss - real_loss
    return total_loss

def generator_loss(fake_output):
    fake_loss = K.mean(fake_output)
    return -fake_loss

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics, noise_dim, clamp):
        self.generator, self.discriminator = model
        self.generator_optimizer, self.discriminator_optimizer = optimizers
        self.gen_loss, self.disc_loss = metrics
        self.train_dataset = train_dataset
        self.noise_dim = noise_dim
        self.clamp = clamp
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None,100), dtype=tf.float32),
    ])
    def train_g_step(self, noise):
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output)
        self.gen_loss(gen_loss)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 100), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
    ])
    def train_d_step(self, noise, images):
        for v in self.generator.trainable_weights:
            K.update(v, K.clip(v, self.clamp[0], self.clamp[1]))
        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            disc_loss = discriminator_loss(real_output, fake_output)
        self.disc_loss(disc_loss)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    def train(self, epoch,  pic):
        self.gen_loss.reset_states()
        self.disc_loss.reset_states()

        k = 0
        for (batch, (images, labels)) in enumerate(self.train_dataset):
            print('epoch: {}, gen loss: {}, disc loss: {}'.format(epoch, self.gen_loss.result(), self.disc_loss.result()))
            if k < 4:
                k = k + 1
                noise = tf.random.normal([images.shape[0], self.noise_dim])
                self.train_d_step(noise, images)
            else:
                k = 0
                noise = tf.random.normal([images.shape[0], self.noise_dim])
                self.train_d_step(noise, images)
                noise = tf.random.normal([images.shape[0], self.noise_dim])
                self.train_g_step(noise)
                pic.add([self.gen_loss.result().numpy(), self.disc_loss.result().numpy()])
                pic.save()
            if batch % 500 == 0:
                print('epoch: {}, gen loss: {}, disc loss: {}'.format(epoch, self.gen_loss.result(), self.disc_loss.result()))