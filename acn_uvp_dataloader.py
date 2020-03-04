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
from PIL import Image, ImageChops

class UVPDataset(Dataset):
    def __init__(self, csv_file, seed, classes='', labels='', weights='', valid=False,
                 run_mean=0, run_std=0, find_class_counts=True, limit=1e10, img_size=224):
        #run_mean=0.9981, run_std=.0160):
        """
        Args:
            csv_file (string): Path to the csv file with image paths and annotations.

        """
        self.limit = limit
        self.img_cnts = []
        self.img_filepaths = []
        self.img_classes = []
        self.random_state = np.random.RandomState(seed)
        # load file data
        self.input_size = img_size
        print('loading csv:%s'%csv_file)
        assert os.path.exists(csv_file); # csv file given doesnt exist
        f = open(csv_file, 'r')
        cnt = 0
        for line in f:
            if cnt < self.limit:
                ll = line.strip().split(',')
                dclass = ll[1]
                if dclass != 'none':
                    cnt +=1
                    self.img_cnts = ll[0]
                    self.img_classes.append(dclass)
                    self.img_filepaths.append(ll[2])

        # TODO - find actual mean/std
        print("dataset has %s examples - limit is %s" %(len(self.img_classes), self.limit))
        # warning - if we don't capture all classes in the limit - then we will
        # not call the model correctly - should probably store the class_names
        # and numbers when training the model
        if not valid:
            func_transforms = [
                torchvision.transforms.ColorJitter(hue=.1, saturation=.1,
                                                   brightness=.1, contrast=.1),
                # rotate looks bad when the critter is on the edge
                #torchvision.transforms.RandomRotation(45, expand=False),
                torchvision.transforms.Resize(size=(self.input_size, self.input_size)),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                 ]
        else:
            func_transforms = [
                torchvision.transforms.Resize(size=(self.input_size, self.input_size)),
                transforms.ToTensor(),
                 ]
        self.transforms = torchvision.transforms.Compose(func_transforms)
        self.indexes = np.arange(len(self.img_filepaths))
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/adaptive.html#AdaptiveLogSoftmaxWithLoss
        # when using adaptive loss - largest classes should have the lowest
        # class index
        self.find_class_counts()
        if weights == '':
            print('finding weights')
            self.find_class_weights()
        else:
            print('using provided weights')
            self.weights=weights

        print('finding class indexes')
        self.img_class_nums = np.array([self.classes.index(c) for c in self.img_classes])
        #self.img_label_names = [self.labels.index(l) for l in self.img_labels]
        self.img_weights = self.weights[np.array(self.img_class_nums)]
        print("CLASS WEIGHTS")
        for cn, cc, cw in zip(self.classes, self.class_counts, self.weights):
            print('class:%s counts:%s weight:%.03f'%(cn, cc, cw))

        #if run_mean is None:
        #     run_mean, run_std = self.find_mean_std()
        #func_transforms.append(transforms.Normalize([run_mean], [run_std]))
        self.transforms = torchvision.transforms.Compose(func_transforms)

    def find_mean_std(self):
        limit = int(0.15*self.__len__())
        print("finding mean/std on random %s images"%limit)
        choices = self.random_state.choice(np.arange(self.__len__()), limit)
        # hmm since my classes aren't balanced - this might be bad
        run_mean = 0.0
        run_std = 0.0
        for c in choices:
            r = self.__getitem__(c)
            nonzero = r[0].numpy()
            nonzero = nonzero[nonzero > 0]
            run_mean += nonzero.mean()
            run_std += nonzero.std()
        run_mean/=float(limit)
        run_std/=float(limit)
        print('found mean/std', run_mean, run_std)
        return run_mean, run_std

    def __len__(self):
        return len(self.img_filepaths)

    def find_class_counts(self):
        classes = sorted(list(set(self.img_classes)))
        class_counts = []
        for c in classes:
            class_counts.append(np.where(np.array(self.img_classes)==c)[0].shape[0])
        class_counts = np.array(class_counts)
        # unsorted class counts which are associated with the class names - now
        # sort them from most populous to least
        sorted_zipped_class_counts = [(x,y) for x,y in sorted(zip(class_counts, classes), reverse=True)]
        # now separate them
        print('loaded classes')
        [print(x) for x in sorted_zipped_class_counts]
        self.class_counts = np.array([a for a,b in sorted_zipped_class_counts])
        self.classes = [b for a,b in sorted_zipped_class_counts]
        self.class_nums = [x for x in range(len(self.classes))]
        self.total_samples = np.sum(self.class_counts)

    def find_class_weights(self):
        # make sampling infrequent classes more likely
        self.weights = 1-(self.class_counts/self.total_samples)

    def rotate_image(self, image, max_angle=45):
        #, center):
        angle = self.random_state.randint(-max_angle, max_angle)
        rotated = transform.rotate(image, angle, resize=True, center=None, order=1, mode='constant', cval=0, clip=True, preserve_range=True)
        return rotated

    def crop_to_creature(self, in_image, h, w, center_y, center_x):
        # if image is larger than (h,w) randomly crop it
        image = deepcopy(in_image)
        hh,ww = image.shape
        if hh>h:
            uch = max(0, int((h/2.0)-center_y))
            image = image[uch:uch+h]
        if ww > w:
            ucw = max(0, int((w/2.0)-center_x))
            image = image[:,ucw:ucw+w]
        return image


    def crop_to_size(self, in_image, h, w, center_y, center_x):
        # if image is larger than (h,w) randomly crop it
        image = deepcopy(in_image)
        hh,ww = image.shape
        if hh>self.input_size:
            uch = max(0, int((h/2.0)-center_y))
            image = image[uch:uch+h]
        if ww > self.input_size:
            ucw = max(0, int((w/2.0)-center_x))
            image = image[:,ucw:ucw+w]
        return image

    def downscale(self, image):
        ldim = np.argmax(image.shape)
        l = np.max(image.shape)
        if l > self.input_size:
            scale = self.input_size/float(l+1)
            image = transform.rescale(image, scale, preserve_range=True, multichannel=False).astype(np.uint8)
        return image

    def add_padding(self, image):
        # blank space should be ones
        # assumes image is same size or smaller than padding size
        hh,ww = image.shape
        uch, ucw = 0,0
        if hh<self.input_size:
          uch = self.random_state.randint(0,self.input_size-hh)
        if ww<self.input_size:
          ucw = self.random_state.randint(0,self.input_size-ww)
        canvas = np.ones((self.input_size,self.input_size), dtype=np.uint8)
        canvas[uch:uch+hh,ucw:ucw+ww] = image
        return canvas

    def get_center(self, image):
        centery = np.median(np.where(image>0)[1])
        centerx = np.median(np.where(image>0)[0])
        return centery, centerx

    def trim(self, image):
        xs = np.where(image>0)[0]
        ys = np.where(image>0)[1]
        leftx = max([0, np.min(ys)-10])
        rightx = min([np.max(ys)+10, image.shape[0]])
        # firstdim
        topy = min([np.max(xs)+10, image.shape[1]])
        boty = max([np.min(xs)-10, 0])
        return image[leftx:rightx, boty:topy]

    def __getitem__(self, idx):
        filepath = self.img_filepaths[idx]
        class_name = self.img_classes[idx]
        class_num = self.classes.index(class_name)
        try:
            #print(idx, filepath, class_name, label)
            image = imread(filepath)[:,:,0]
        except:
            print("unable to load file", filepath)
            #return self.__getitem__(self.random_state.randint(0, self.__len__()))
        # images have an annotation that gives the "1 mm" scale of the image
        hh,ww = image.shape
        # remove label at bottom
        bottom = 45 #np.argmin(image.sum(1))-10
        # flip to enable rotation which infills with 0
        image = (255-image[:hh-bottom,:])
        #image = (image[:hh-bottom,:])
        image = image[:hh-bottom,:]
        image = self.rotate_image(image)
        image = self.trim(image)
        image = self.downscale(image)
        image = self.add_padding(image)
        #center_y, center_x = self.get_center(image)
        #image = self.crop_to_size(image, self.input_size, self.input_size, center_y, center_x)
        #print("crop size", image.shape)
        ## turn into PIL
        image = Image.fromarray(image)
        #image = self.trim(image, 0)
        ## normalize between 0 and 1 seems to hurt
        timage = self.transforms(image)
        return timage, class_num, filepath, idx

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config import exp_dir, img_size
    rs = np.random.RandomState(3)
    #train_ds = UVPDataset(csv_file=os.path.join(exp_dir,'valid.csv'), seed=34)
    train_ds = UVPDataset(csv_file=os.path.join(exp_dir,'train.csv'), seed=34, img_size=img_size)
    #valid_ds = EcotaxaDataset(csv_file='valid.csv', seed=334, classes=class_names, weights=class_weights)
    class_names = train_ds.classes
    class_weights = train_ds.weights
    #ds = {'train':train_ds, 'valid':valid_ds}
    ds = {'train':train_ds}
    #for phase in ds.keys():
    for phase in ds.keys():
        exdir = os.path.join(exp_dir, 'example_images', phase)
        if not os.path.exists(exdir):
            os.makedirs(exdir)
        indexes = rs.choice(np.arange(len(ds[phase])), 10)

        for i in indexes:
            image, class_num, filepath, idx = ds[phase][i]
            imo = imread(filepath)
            h,w,c = imo.shape
            f,ax = plt.subplots(1,2)
            ax[0].imshow(image[0].numpy())
            ax[1].imshow(imo[:,:,0])
            img_name = os.path.split(filepath)[1]
            ax[0].set_title("%s" %(train_ds.classes[class_num]))
            #ax[1].set_title("%s %s" %(train_ds.large_classes[large_class_num], train_ds.small_classes[small_class_num]))
            outpath = os.path.join(exdir, img_name)
            print(outpath)
            plt.savefig(outpath)
            plt.close()


