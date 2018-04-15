import os
import numpy as np
import imageio
import downSampling
import binvox_rw


class getData(object):
    def __init__(self, image_path, model_path):
        self.current_point = 0
        self.image_lst = []
        self.model_lst = []
        for _, _, fileList in os.walk(image_path, topdown=False):
            for f in fileList:
                if f[-4:] != ".gif" and ".DS_Store" not in f:
                    self.image_lst.append(f)
        for _, _, fileList in os.walk(model_path, topdown=False):
            for f in fileList:
                if ".DS_Store" not in f:
                    self.model_lst.append(f)
        self.fig_map = {}
        for i in range(len(self.image_lst)):
            num = self.image_lst[i].split(".")[0].split("-")[0]
            self.fig_map[i] = self.model_lst.index(num + ".binvox")

        self.image_array = np.stack([self.read_image(image_path + img) for img in self.image_lst], axis=0)
        self.model_array = np.stack([self.read_3D(model_path + mod) for mod in self.model_lst], axis=0)
        print("Finish Loading data, shape as:")
        print("Image shape:", np.shape(self.image_array))
        print("Model shape:", np.shape(self.model_array))

    def read_image(self, path):
        im = imageio.imread(path)
        return ((np.reshape(downSampling.downsample(im, 8), (64, 64, 1)) / 255.0) - 0.5) / 0.5 # normalize -1 ~ 1

    def read_3D(self, path):
        with open(path, 'rb') as f:
            m1 = binvox_rw.read_as_3d_array(f)
        return (np.reshape(downSampling.downsample_cube(m1.data, 4), (32, 32, 32, 1)) - 0.5) / 0.5 # normalize -1 ~ 1

    def get_batch(self, batch_size):
        if self.current_point + batch_size > len(self.image_lst):
            self.current_point = len(self.image_lst) - batch_size
        x_image = np.stack([self.image_array[i] for i in range(self.current_point, self.current_point + batch_size)], axis=0)
        x_3d = np.stack([self.model_array[self.fig_map[i]] for i in
                             range(self.current_point, self.current_point + batch_size)], axis=0)
        return x_image, x_3d


# if __name__ == '__main__':
#     image_path = "./grey-office-chair-image/"
#     model_path = "./office-chair-model/"
#     dataset = getTrain(image_path, model_path)
#     xs, ys = dataset.get_batch(10)
#     print(np.shape(xs))
#     print(np.shape(ys))
