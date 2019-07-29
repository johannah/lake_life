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
from skimage.feature import blob_doh
from copy import deepcopy

class UVPDataset(Dataset):
    def __init__(self, csv_file, seed, classes='', weights='', augment=True, run_mean=0.9981, run_std=.0160):
        """
        Args:
            csv_file (string): Path to the csv file with image paths and annotations.

        """
        self.img_filepaths = []
        self.img_classes = []
        self.img_refined_classes = []
        self.random_state = np.random.RandomState(seed)
        # load file data
        self.input_size = 224
        print('loading csv:%s'%csv_file)
        assert os.path.exists(csv_file); # csv file given doesnt exist
        f = open(csv_file, 'r')

        for line in f.readlines():
            ll = line.strip().split(',')
            self.img_filepaths.append(ll[0])
            self.img_classes.append(ll[1])
            #self.refined_img_classes.append(ll[2])
        self.augment = augment
        # TODO - find actual mean/std

        func_transforms = [
            torchvision.transforms.ColorJitter(hue=.1, saturation=.1,
                                               brightness=.1, contrast=.1),
            torchvision.transforms.RandomRotation(45),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
             ]
        self.transforms = torchvision.transforms.Compose(func_transforms)

        self.indexes = np.arange(len(self.img_filepaths))
        if classes == '':
            print('finding classes')
            self.classes = sorted(list(set(self.img_classes)))
        else:
            self.classes = classes
        self.find_class_counts()
        if weights == '':
            print('finding weights')
            self.find_class_weights()
        else:
            self.weights=weights

        self.img_labels = [self.classes.index(c) for c in self.img_classes]
        self.img_weights = self.weights[np.array(self.img_labels)]
        print("CLASS WEIGHTS")
        for cn, cc, cw in zip(self.classes, self.class_counts, self.weights):
            print('class:%s counts:%s weight:%.03f'%(cn, cc, cw))

        if run_mean is None:
             run_mean, run_std = self.find_mean_std()
        func_transforms.append(transforms.Normalize([run_mean], [run_std]))
        self.transforms = torchvision.transforms.Compose(func_transforms)

    def find_mean_std(self):
        limit = int(0.15*self.__len__())
        choices = self.random_state.choice(np.arange(self.__len__()), limit)
        # hmm since my classes aren't balanced - this might be bad
        run_mean = 0
        run_std = 0
        for c in choices:
            image,_,_,_,_ = self.__getitem__(c)
            run_mean += image.mean()
            run_std += image.std()
        run_mean/=float(limit)
        run_std/=float(limit)
        print('found mean/std', run_mean, run_std)
        return run_mean, run_std

    def __len__(self):
        return len(self.img_filepaths)

    def find_class_counts(self):
        class_counts = []
        for c in self.classes:
            class_counts.append(np.where(np.array(self.img_classes)==c)[0].shape[0])
        self.class_counts = np.array(class_counts)

    def find_class_weights(self):
        # prevent 0 weight on any by adding .2
        self.weights = 1.0/self.class_counts

    #def rotate_image(self, image, max_angle, center):
    #    angle = self.random_state.randint(-max_angle, max_angle)
    #    rotated = transform.rotate(image, angle, resize=True, center=center, order=1, mode='constant', cval=255, clip=True, preserve_range=True)
    #    return rotated

    def crop_to_size(self, in_image, h, w, center_y, center_x):
        # if image is larger than (h,w) randomly crop it
        image = deepcopy(in_image)
        hh,ww = image.shape
        if hh>h:
            if self.augment:
                uch = max(0, int((h/2.0)-center_y))
            else:
                uch = max(0, int((hh/2.0)-(h/2.0)))
            image = image[uch:uch+h]
        if ww > w:
            if self.augment:
                ucw = max(0, int((w/2.0)-center_x))
            else:
                ucw = max(0, int((ww/2.0)-(w/2.0)))
            image = image[:,ucw:ucw+w]
        return image

    def add_padding(self, image, h, w):
        # blank space should be ones
        # assumes image is same size or smaller than padding size
        assert(image.max() == 255.0)
        hh,ww = image.shape
        uch, ucw = 0,0
        if hh<h:
          uch = self.random_state.randint(0,h-hh)
        if ww<w:
          ucw = self.random_state.randint(0,w-ww)
        canvas = np.ones((h,w),dtype=np.uint8)*255
        canvas[uch:uch+hh,ucw:ucw+ww] = image
        return canvas

    def get_center(self, image):
        centerx = np.median(np.where(image<255)[0])
        centery = np.median(np.where(image<255)[1])
        return centerx, centery


    def __getitem__(self, idx):
        filepath = self.img_filepaths[idx]
        class_name = self.img_classes[idx]
        try:
            label = self.img_labels[idx]
            #print(idx, filepath, class_name, label)
            image = imread(filepath)[:,:,0]
            # images have an annotation that gives the "1 mm" scale of the image
            hh,ww = image.shape
            # remove label at bottom
            bottom = 45 #np.argmin(image.sum(1))-10
            image = image[:hh-bottom,:]
            centerx,centery = self.get_center(image)
            image = self.crop_to_size(image, self.input_size, self.input_size, centery, centerx)
            # torchvision.transforms.Pad(padding, fill=255,
                                         # padding_mode='constant')
            image = self.add_padding(image, h=self.input_size, w=self.input_size)
            image = Image.fromarray(image, mode='L')
            image = self.transforms(image)
            image = image[0][None]
        except:
            print("COULD NOT LOAD DATA FILE", idx)
            print(filepath)
            # tmp hack - actually that image seems messed up`:w
            # TODO - figureout how to feed failed image to dataloader
            # some uvp images wont load
            rr = self.__getitem__(self.random_state.randint(0, self.__len__()))
            image, label, filepath, class_name, idx = rr
        return image, label, filepath, class_name, idx

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bdir = 'experiments/uvp_big_small'
    train_ds = UVPDataset(csv_file=os.path.join(bdir,'train.csv'), seed=34)
    #valid_ds = EcotaxaDataset(csv_file='valid.csv', seed=334, classes=class_names, weights=class_weights)
    class_names = train_ds.classes
    class_weights = train_ds.weights
    #ds = {'train':train_ds, 'valid':valid_ds}
    ds = {'train':train_ds}
    #for phase in ds.keys():
    for phase in ds.keys():
        if not os.path.exists(phase):
            os.makedirs(phase)
        all_inputs = []
        #for i in range(len(ds[phase])):
        for i in range(1000):
            inputs, label, img_path, class_name, idx = ds[phase][i]
            all_inputs.append(inputs[0].numpy())
            f,ax = plt.subplots(1,2)
            ax[0].imshow(inputs[0].numpy())
            imo = imread(img_path)
            ax[1].imshow(imo[:,:,0])
            img_name = os.path.split(img_path)[1]
            ax[0].set_title(class_names[label])
            ax[1].set_title(label)
            plt.savefig(os.path.join(phase, img_name))
            plt.close()


