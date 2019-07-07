from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import sys
from imageio import imread, imwrite
from IPython import embed
from skimage import transform

class EcotaxaDataset(Dataset):
    def __init__(self, csv_file, seed, classes='', weights=''):
        """
        Args:
            csv_file (string): Path to the csv file with image paths and annotations.

        """
        self.img_filepaths = []
        self.img_classes = []
        self.random_state = np.random.RandomState(seed)
        # load file data
        self.input_size = 224
        assert os.path.exists(csv_file); # csv file given doesnt exist
        f = open(csv_file, 'r')

        for line in f.readlines():
            n,l = line.strip().split(',')
            self.img_filepaths.append(n)
            self.img_classes.append(l)

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ])
        self.indexes = np.arange(len(self.img_filepaths))
        if classes == '':
            self.classes = sorted(list(set(self.img_classes)))
        else:
            self.classes = classes
        if weights == '':
            self.find_class_weights()
        else:
            self.weights=weights

        self.img_labels = [self.classes.index(c) for c in self.img_classes]
        self.img_weights = self.weights[np.array(self.img_labels)]

    def find_class_weights(self):
        class_counts = []
        for c in self.classes:
            class_counts.append(np.where(np.array(self.img_classes)==c)[0].shape[0])

        # prevent 0 weight on any by adding .2
        self.class_counts = np.array(class_counts)
        self.weights = 1./self.class_counts

    def __len__(self):
        return len(self.img_filepaths)

    def rotate_image(self, image, max_angle):
        angle = self.random_state.randint(-max_angle, max_angle)
        rotated = transform.rotate(image, angle, resize=False, center=None, order=1, mode='constant', cval=255, clip=True, preserve_range=True)
        return rotated

    def crop_to_size(self, image, h, w):
        # if image is larger than (h,w) randomly crop it
        hh,ww,cc = image.shape
        if hh>h:
          uch = self.random_state.randint(0,hh-h)
          image = image[uch:uch+h]
        if ww > w:
          ucw = self.random_state.randint(0,ww-w)
          image = image[:,ucw:ucw+w]
        return image

    def add_padding(self, image, h, w):
        # blank space should be ones
        # assumes image is same size or smaller than padding size
        assert(image.max() == 255.0)
        hh,ww,cc = image.shape
        uch, ucw = 0,0
        if hh<h:
          uch = self.random_state.randint(0,h-hh)
        if ww<w:
          ucw = self.random_state.randint(0,w-ww)
        canvas = np.ones((h,w,cc),dtype=np.uint8)*255
        canvas[uch:uch+hh,ucw:ucw+ww] = image
        return canvas

    def __getitem__(self, idx):
        filepath = self.img_filepaths[idx]
        class_name = self.img_classes[idx]
        label = self.img_labels[idx]
        image = imread(filepath)
        # images have an annotation that gives the "1 mm" scale of the image
        hh,ww,c = image.shape
        image = image[:hh-20,:]
        image = self.rotate_image(image, max_angle=45)
        image = self.crop_to_size(image, h=self.input_size, w=self.input_size)
        image = self.add_padding(image, h=self.input_size, w=self.input_size)
        image = Image.fromarray(image, mode='RGB')
        image = self.transforms(image)
        return image[0][None], label, filepath, class_name
