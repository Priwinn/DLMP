import tensorflow as tf
from GAN import load_wrapper, eval_model_ds
import tensorflow_addons as tfa
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
from pathlib import Path
import json


class NoisyScoreDS:
    'Generates data for Keras'

    def __init__(self, clean_ds, generator, batch_size=32, shuffle=1024, p_noise=1, p_blur=1, crop=0.5,
                 iqa_score='ssim', prefetch=2):
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

        self.ds = clean_ds.shuffle(self.shuffle).batch(self.batch_size).map(self.noise_map,
                                                                            num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
            2)

    def noise_map(self, x, y):
        # Blur the input with probability p_blur
        if tf.random.uniform([1]) < self.p_blur:
            x = tfa.image.gaussian_filter2d(x)

            # Generate noisy input with prob p_noise (random std)
        if tf.random.uniform([1]) < self.p_noise:
            x = x + tf.random.normal(tf.shape(x), 0, tf.random.uniform([1], 0, 32 / 256))
        x = tf.clip_by_value(x, -1, 1)

        # Get image prediction
        prediction = self.generator.predict(x, training=True)

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
        prediction = self.generator.predcit(noisy_x, training=True)

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

    # Plot histogram, repeat to reduce variance
    def hist(self, repeat=5, kde=False):
        scores = tf.concat([y for x, y in self.ds.repeat(repeat)], axis=0).numpy()
        ax = sns.distplot(scores, 15, kde=kde)
        ax.set_xlabel(self.score.upper())
        if not kde:
            ax.set_ylabel('Counts')
        return ax


class IQATrainer:

    def __init__(self, train_clean_ds, val_clean_ds, generator, iqa, batch_size=8, shuffle=1024, p_noise=1, p_blur=1,
                 crop=0.5, transform='standardize'):
        self.batch_size = batch_size
        self.train_ds = NoisyScoreDS(train_clean_ds, generator, batch_size=batch_size, shuffle=shuffle, p_noise=p_noise,
                                     p_blur=p_blur, crop=crop)
        self.val_ds = NoisyScoreDS(val_clean_ds, generator, batch_size=batch_size, shuffle=shuffle, p_noise=p_noise,
                                   p_blur=p_blur, crop=crop)
        self.shuffle = shuffle
        self.generator = generator
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.crop = crop
        self.iqa = iqa
        print("Computing transformation variables")
        if transform == 'scale':
            scores = tf.concat([y for x, y in self.train_ds.repeat(3)], axis=0).numpy()
            self.offset = np.min(scores)
            self.scale = 1 - self.offset
            self.transform = lambda x, y: (x, (y - self.offset) / self.scale)
            self.detransform = lambda y: y * self.scale + self.offset
        elif transform == 'standardize':
            scores = tf.concat([y for x, y in self.train_ds.repeat(3)], axis=0).numpy()
            self.offset = np.mean(scores)
            self.scale = np.std(scores)
            self.transform = lambda x, y: (x, (y - self.offset) / self.scale)
            self.detransform = lambda y: y * self.scale + self.offset
        else:
            raise NameError('iqa_score must be one of ssim or psnr')
        print('Done')

    def train(self, epochs):
        self.iqa.fit(self.train_ds.map(self.transform, num_parallel_calls=tf.data.experimental.AUTOTUNE), epochs=epochs,
                     validation_data=self.val_ds.map(self.transform, num_parallel_calls=tf.data.experimental.AUTOTUNE),
                     callbacks=tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True
                                                                )
                     )

    def evaluate(self, test_ds=False):
        if not test_ds:
            test_ds = self.val_ds.map(self.transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            test_ds = test_ds.map(self.transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        scores = [(y, tf.convert_to_tensor(self.detransform(self.iqa.predict(x)))) for x, y in test_ds.repeat(5)]
        real = tf.squeeze(tf.concat([y for y, z in scores], axis=0)).numpy()
        predicted = tf.squeeze(tf.concat([z for y, z in scores], axis=0)).numpy()
        spearman = sp.stats.spearmanr(real, predicted)[0]
        pearson = sp.stats.pearsonr(real, predicted)[0]
        print(f'LCC:{pearson}')
        print(f'SROC:{spearman}')
        return pearson, spearman

    def save(self, model_name):
        model_path = f"../models/{model_name}/{datetime.now().strftime('%Y%m%d_%H%M')}"
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.iqa.save(model_path + f"/{model_name}.h5")
        with open(f'{model_path}/parameters.json', 'w+') as file:
            parameters = {'model_name': model_name,
                          'offset': self.offset,
                          'scale': self.scale}
            json.dump(parameters, file)

    def load(self, model_path):
        parameters = json.load(model_path + 'parameters.json')

        self.iqa = tf.keras.models.load_model(model_path + f"/{parameters['model_name']}.h5")


class TrainingTrueGenerator(tf.keras.models.Model):
    def __init__(self, generator):
        super(TrainingTrueGenerator, self).__init__()
        self.generator = generator

    def call(self, inputs):
        return self.generator(inputs, training=True)


class IQA(tf.keras.models.Model):
    def __init__(self, iqa, generator, p_noise=1, p_blur=1, crop=0.5):
        super(IQA, self).__init__()
        self.generator = generator
        self.iqa = iqa
        self.score = tf.image.ssim
        self.crop = 0.5
        self.compile()

    def compile(self):
        super(IQA, self).compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

    def train_step(self, data):

        def standardize(y):
            return (y - 0.88749385) / 0.06297474

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        if tf.random.uniform([1]) < self.p_blur:
            x = tfa.image.gaussian_filter2d(x)
        # Generate noisy input with prob p_noise (random std)
        if tf.random.uniform([1]) < self.p_noise:
            x = x + tf.random.normal(tf.shape(x), 0, tf.random.uniform([1], 0, 32 / 256))
        x = tf.clip_by_value(x, -1, 1)

        prediction = self.generator(x, training=True)
        score = self.score(tf.image.central_crop(y, self.crop), tf.image.central_crop(prediction, self.crop),
                           max_val=2.0)
        score = standardize(score)
        with tf.GradientTape() as tape:
            score_pred = self.iqa(x, training=True)
            loss = self.compiled_loss(score, score_pred)

        _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                  self.iqa.trainable_variables)
        self.compiled_metrics.update_state(score, score_pred)
        return {m.name: m.result() for m in self.metrics}
