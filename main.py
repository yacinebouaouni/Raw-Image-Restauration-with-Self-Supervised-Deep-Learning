from unprocess import *
from process import *
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import imageio
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

import argparse




def main(path='./original_png/',path_s='saved_im_clean/', noise = 0):


    files = os.listdir(path)

    if not os.path.exists('metadata/'):
        os.mkdir('metadata')

    if not os.path.exists('saved_im'):
        os.mkdir('saved_im')

    if not os.path.exists('saved_im_clean'):
        os.mkdir('saved_im_clean')

    for i in tqdm(range(len(files))):

        im_path = path+files[i]
        im = Image.open(im_path)
        im = tf.convert_to_tensor(np.array(im))
        im= tf.cast(im, tf.float32)
        im = im[1:, 1:, :]/255.
        im, meta = unprocess(im)
        a_file = open('metadata/' + files[i][:-4] + '.pkl', "ab")
        pickle.dump(meta, a_file)
        a_file.close()

        #im = add_noise(im, shot_noise=0.0001, read_noise=0.00005)

        for j in range(4):
            imageio.imwrite(path_s+files[i][:-4]+str(j)+'.png', im.numpy()[:, :, j])




def main2(Ids = ['0000','0002']):

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['clean','denoised'], nargs="?", default="clean")
    args = parser.parse_args()

    batch_images = []
    batch_red = []
    batch_blue = []
    batch_cam2rgb = []

    mode = args.mode
    print(mode)
    if mode == "clean":
        im_path_dir = 'saved_im/'

    if mode == "denoised":
        im_path_dir = 'denoised_im/'

    for id in ['0000','0000']:
        image = []
        for i in range(4):
            im_path = im_path_dir+id +str(0)+'.png'
            im = ImageOps.grayscale(Image.open(im_path))
            im = tf.convert_to_tensor(np.array(im))
            im = tf.cast(im, tf.float32)
            image.append(im/255.)

        a_file = open("metadata/"+id+".pkl", "rb")
        metadata = pickle.load(a_file)

        image = tf.stack(image)
        image = tf.transpose(image,[1,2,0])

        batch_images.append(image)
        batch_red.append(metadata['red_gain']/2)
        batch_blue.append(metadata['blue_gain'])
        batch_cam2rgb.append(metadata['cam2rgb'])

    batch_images = tf.stack(batch_images)
    batch_cam2rgb = tf.stack(batch_cam2rgb)


    image = process(batch_images, batch_red, batch_blue, batch_cam2rgb)

    fig,ax = plt.subplots(1,2,figsize=(20,15))
    ax[0].imshow(image[0])
    ax[1].imshow(image[1])
    plt.show()











if __name__ == "__main__":
    #main()
    main2()

