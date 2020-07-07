import tensorflow as tf
from GAN import load_wrapper
import tensorflow_addons as tfa
import math
import matplotlib.pyplot as plt


def load_iqa(data_range, splits=(1 / 3, 2 / 3), path='../gates/'):
    # Splits is a list [a,b] with 0<=a<=b<=1
    # Returns 3 datasets with filenames in [0,a], [a,b], [b,1] (relative to the data range)
    (a, d) = data_range
    b = int(round(a + (d - a) * splits[0]))
    c = int(round(a + (d - a) * splits[1]))
    datasets = [tf.data.Dataset.list_files([path + 'train/%i.png' % i for i in range(a, b)], shuffle=False),
                tf.data.Dataset.list_files([path + 'train/%i.png' % i for i in range(b, c)], shuffle=False),
                tf.data.Dataset.list_files([path + 'train/%i.png' % i for i in range(c, d)], shuffle=False)]
    for i in range(len(datasets)):
        datasets[i] = datasets[i].map(load_wrapper)
    return datasets


def aug_map(x, y):
    if tf.random.uniform((1,), 0, 2, tf.int32) == 1:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    if tf.random.uniform((1,), 0, 2, tf.int32) == 1:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
    if tf.random.uniform((1,), 0, 2, tf.int32) == 1:
        angle = tf.random.uniform((1,), 0, 2 * math.pi, tf.float32)
        x = tfa.image.rotate(x + 1, angle) - 1
        y = tfa.image.rotate(y + 1, angle) - 1
    return x, y


def aug_ds(ds):
    return ds.map(aug_map)


class NoisyScoreDS():
    'Generates data for Keras'

    def __init__(self, clean_ds, generator, batch_size=32, shuffle=1024, p_noise=1, p_blur=1, crop=0.5,
                 iqa_score='ssim'):
        self.batch_size = batch_size
        self.clean_ds = clean_ds
        self.shuffle = shuffle
        self.generator = generator
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.crop = crop
        if iqa_score == 'ssim':
            self.score = tf.image.ssim
        elif iqa_score == 'psnr':
            self.score = tf.image.psnr
        else:
            raise NameError('iqa_score must be one of ssim or psnr')

        self.ds = clean_ds.shuffle(self.shuffle).batch(self.batch_size).map(self.noise_map)

    def noise_map(self, x, y):
        # Blur the input with probability p_blur
        if tf.random.uniform([1]) < self.p_blur:
            x = tfa.image.gaussian_filter2d(x)

            # Generate noisy input with prob p_noise (random std)
        if tf.random.uniform([1]) < self.p_noise:
            x = x + tf.random.normal(tf.shape(x), 0, tf.random.uniform([1], 0, 32 / 256))
        x = tf.clip_by_value(x, -1, 1)

        # Get image prediction
        prediction = self.generator(x, training=True)

        # Score image
        score = self.score(tf.image.central_crop(y, self.crop), tf.image.central_crop(prediction, self.crop),
                           max_val=2.0)
        return x, score

    def noise_map_all(self, x, y):
        # Blur the input with probability p_blur
        if tf.random.uniform([1]) < self.p_blur:
            noisy_x = tfa.image.gaussian_filter2d(x)
        else:
            noisy_x = x

        # Generate noisy input with prob p_noise (random std)
        if tf.random.uniform([1]) < self.p_noise:
            noisy_x = noisy_x + tf.random.normal(tf.shape(x), 0, tf.random.uniform([1], 0, 32 / 256))
        noisy_x = tf.clip_by_value(noisy_x, -1, 1)

        # Get image prediction
        prediction = self.generator(noisy_x, training=True)

        # Score image
        cropped_y = tf.image.central_crop(y, self.crop)
        cropped_pred = tf.image.central_crop(prediction, self.crop)
        score = self.score(tf.image.central_crop(y, self.crop), tf.image.central_crop(prediction, self.crop),
                           max_val=2.0)
        return x, noisy_x, y, cropped_y, prediction, cropped_pred, score

    def plot_sample(self):
        plt.figure(figsize=(15, 15))
        for x, y in self.clean_ds.shuffle(500).take(1).batch(1):
            x, noisy_x, y, cropped_y, noisy_prediction, noisy_cropped_pred, noisy_score_cropped = self.noise_map_all(x,
                                                                                                                     y)
            prediction = self.generator(x, training=True)
            cropped_pred = tf.image.central_crop(prediction, self.crop)

        display_list = [x, cropped_pred, prediction, '', cropped_y, y, noisy_x,
                        noisy_cropped_pred, noisy_prediction]
        score_cropped = self.score(cropped_y, cropped_pred, max_val=2.0)
        noisy_score = self.score(y, noisy_prediction, max_val=2.0)
        score = self.score(y, prediction, max_val=2.0)
        title = ['Input Image', 'Predicted Image score=%f' % score_cropped,
                 'Predicted Image (not cropped) score=%f' % score, '',
                 'Ground Truth', 'Ground Truth (not cropped)', 'Noisy Image',
                 'Predicted Image score=%f' % noisy_score_cropped,
                 'Predicted Image (not cropped) score=%f' % noisy_score]
        for i in [0, 1, 2, 4, 5, 6, 7, 8]:
            plt.subplot(3, 3, i + 1)
            plt.title(title[i])
            display_im = tf.squeeze(display_list[i])
            plt.imshow(display_im * 0.5 + 0.5, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')

    plt.show()
