import tensorflow as tf
from GAN import load_wrapper


def load_iqa(data_range,splits=[1/3,2/3],path='../gates/'):
# Splits is a list [a,b] with 0<=a<=b<=1
# Returns 3 datasets with filenames in [0,a], [a,b], [b,1] (relative to the data range)
  (a,d)=data_range
  b = int(round(a+(d-a)*splits[0]))
  c = int(round(a+(d-a)*splits[1]))
  datasets=[]
  datasets.append(tf.data.Dataset.list_files([path+'train/%i.png' % i for i in range(a,b)]))
  datasets.append(tf.data.Dataset.list_files([path+'train/%i.png' % i for i in range(b,c)]))
  datasets.append(tf.data.Dataset.list_files([path+'train/%i.png' % i for i in range(c,d)]))
  for i in range(len(datasets)):
    datasets[i] = datasets[i].shuffle(400)
    datasets[i] = datasets[i].map(load_wrapper)
  return datasets


class NoisyScoreDS():
    'Generates data for Keras'

    def __init__(self, imgs, generator, batch_size=32, width=256, height=256,
                 n_channels=3, shuffle=1024, iqa_score='ssim', p_noise=1):
        'Initialization'
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.imgs=imgs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.generator_GAN = generator_GAN
        self.iqa_score = iqa_score
        self.crop_tol = crop_tol
        self.p_noise = p_noise
        self.ds=self.__get_ds__()

    def __getds__(self):
        ds=tf.data.Dataset.from_tensor_slices()

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of x,y in the batch

        x_temp = np.stack([self.x[k, :, :, :] for k in indexes])
        y_temp = np.stack([self.y[k, :, :, :] for k in indexes])
        # Generate data
        noisy_x, score = self.__data_generation(x_temp, y_temp, p_noise=self.p_noise)

        return noisy_x, score

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, x_temp, y_temp, p_noise=1):
        'Generates data containing batch_size samples'

        # Generate noisy input with prob 0.5
        if np.random.choice([0, 1], p=[1 - p_noise, p_noise]):
            noise = np.random.normal(0, np.random.uniform(0, 40 / 256),
                                     x_temp.shape)
        else:
            noise = np.zeros_like(x_temp)

        noisy_x = x_temp + noise

        # Get image prediction
        prediction = self.generator_GAN(noisy_x, training=False)

        # Score image
        if self.iqa_score == 'ssim':
            score = tf.map_fn(lambda z: ssim(*crop_to_interest_area(z[0].numpy(), z[1].numpy(), tol=self.crop_tol),
                                             multichannel=not GRAYSCALE), (y_temp, prediction),
                              dtype=tf.float32).numpy()
        elif self.iqa_score == 'mse':
            score = tf.map_fn(lambda z: mse(*crop_to_interest_area(z[0].numpy(), z[1].numpy(), tol=self.crop_tol)),
                              (y_temp, prediction), dtype=tf.float64).numpy()
        else:
            raise NameError('iqa_score must be one of ssim or mse')
        return noisy_x, score

    def plot_sample(self):
        plt.figure(figsize=(15, 15))
        i = np.random.randint(0, len(self.x))
        noisy_x, noisy_score_cropped = self.__data_generation(np.expand_dims(self.x[i, :, :, :], axis=0),
                                                              np.expand_dims(self.y[i, :, :, :], axis=0))
        prediction = self.generator_GAN(np.expand_dims(self.x[i, :, :, :], axis=0), training=False).numpy()[0, :, :, :]
        tar = self.y[i, :, :, :].numpy()
        prediction_cropped = crop_to_interest_area(tar, prediction, tol=self.crop_tol)[1]
        noisy_prediction = self.generator_GAN(noisy_x, training=False).numpy()[0, :, :, :]
        tar_cropped, noisy_prediction_cropped = crop_to_interest_area(tar, noisy_prediction, tol=self.crop_tol)
        display_list = [self.x[i, :, :, :], prediction_cropped, prediction, '', tar_cropped, tar, noisy_x[0, :, :, :],
                        noisy_prediction_cropped, noisy_prediction]
        score_cropped = ssim(tar_cropped, prediction_cropped, multichannel=not GRAYSCALE)
        noisy_score = ssim(tar, noisy_prediction, multichannel=not GRAYSCALE)
        score = ssim(tar, prediction, multichannel=not GRAYSCALE)
        title = ['Input Image', 'Predicted Image score=%f' % score_cropped,
                 'Predicted Image (not cropped) score=%f' % score, '',
                 'Ground Truth', 'Ground Truth (not cropped)', 'Noisy Image',
                 'Predicted Image score=%f' % noisy_score_cropped,
                 'Predicted Image (not cropped) score=%f' % noisy_score]
        for i in [0, 1, 2, 4, 5, 6, 7, 8]:
            plt.subplot(3, 3, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            display_im = display_list[i]
            if GRAYSCALE:
                plt.imshow(display_im[:, :, 0] * 0.5 + 0.5, cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(display_im * 0.5 + 0.5)
            plt.axis('off')
        plt.show()
