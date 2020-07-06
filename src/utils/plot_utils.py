import matplotlib.pyplot as plt
import tensorflow as tf


def plot_image(img):
    plt.figure()
    plt.imshow(tf.squeeze(img) * 0.5 + 0.5, cmap='gray', vmin=0, vmax=1)


def plot_all(inp, tar, model, crop=0.5):
    if tf.rank(inp) != 4:
        inp = tf.expand_dims(inp, axis=0)
    if tf.rank(tar) != 4:
        tar = tf.expand_dims(tar, axis=0)
    prediction = tf.image.central_crop(model(inp, training=True), crop)
    tar=tf.image.central_crop(tar, crop)
    ssim = tf.image.ssim(tar, prediction, max_val=2.0)
    psnr = tf.image.psnr(tar, prediction, max_val=2.0)
    for j in range(inp.shape[0]):
        plt.figure(figsize=(15, 15))
        display_list = [inp[j], tar[j], prediction[j]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            display_im = display_list[i]
            plt.imshow(display_im * 0.5 + 0.5, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
        plt.show()
        print('Image shown PSNR {:.5}'.format(psnr[j]))
        print('Image shown SSIM {:.5}'.format(ssim[j]))
