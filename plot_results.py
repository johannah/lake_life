import matplotlib
matplotlib.use("Agg")
import numpy as np
from glob import glob
import os
import sys
import matplotlib.pyplot as plt
import torch
from IPython import embed
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch

# make histogram plot
# train on everything
#
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from ecotaxa_dataloader import EcotaxaDataset
np.set_printoptions(precision=2)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, filename='confusion.png'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print("starting plotting confusion", filename)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [classes[n] for n in list(unique_labels(y_true, y_pred))]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(filename)
    print("finished plotting confusion", filename)
    return ax

def evaluate_model(model, dataloaders, basename=''):
    model.eval()
    for phase in ['valid', 'train']:
    #for phase in ['valid']:
        y_true = []
        y_pred = []

        for inputs, labels, img_path, class_name in dataloaders[phase]:
             inputs = inputs.to(device)
             labels = labels.to(device)
             outputs = model(inputs)
             _, preds = torch.max(outputs, 1)
             y_true.extend(list(labels.detach().numpy()))
             y_pred.extend(list(preds.detach().numpy()))
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(y_true, y_pred, classes=class_names,
                              title='Confusion matrix, without normalization',
                              filename=basename+'_'+phase+'_'+'unnormalized_confusion.png')

        # Plot normalized confusion matrix
        plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True,
                              title='Normalized confusion matrix',
                              filename=basename+'_'+phase+'_'+'normalized_confusion.png')


def load_latest_checkpoint(loaddir='checkpoints'):
    search = sorted(glob(os.path.join(loaddir, '*.pth')))
    print('found %s checkpoints'%len(search))
    ckpt = search[-1]
    print('loading %s' %ckpt)
    return ckpt, torch.load(ckpt)

if __name__ == '__main__':
    ckpt_name, ckpt_dict = load_latest_checkpoint()

    bname = 'limited'
    datadir = './'
    batch_size = 32
    train_ds = EcotaxaDataset(csv_file=os.path.join(bname, 'train.csv'),seed=34)
    class_names = train_ds.classes
    class_weights = train_ds.weights
    valid_ds = EcotaxaDataset(csv_file=os.path.join(bname, 'valid.csv'),seed=334, classes=class_names, weights=class_weights)
    train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True, num_workers=1
        )
    valid_dl = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=True, num_workers=1
        )
    dataloaders = {'train':train_dl, 'valid':valid_dl}


    # Load the pretrained model from pytorch
    rmodel = torchvision.models.resnet50(pretrained=True)
    num_epochs = 1000
    # Number of classes in the dataset
    num_classes = len(class_names)

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False
    # print(rmodel)
    # last layer: (fc): Linear(in_features=2048, out_features=1000, bias=True)
    # need to reshape
    rmodel.fc = nn.Linear(2048, num_classes)
    rmodel.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print('using', device)
    rmodel = rmodel.to(device)
    print("loading state dict")
    rmodel.load_state_dict(ckpt_dict['state_dict'])
    evaluate_model(rmodel, dataloaders, bname)

