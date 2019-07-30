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
from imageio import imread
# make histogram plot
# train on everything
#
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from ecotaxa_dataloader import EcotaxaDataset
from uvp_dataloader import UVPDataset
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
    fig, ax = plt.subplots(figsize=(20,20))
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
    plt.close()
    print("finished plotting confusion", filename)
    return ax

def plot_error(iinput, img_filename, label, predicted, filename, suptitle):
    timg = imread(img_filename)
    plt.figure()
    f,ax = plt.subplots(1,2)
    ax[0].imshow(iinput)
    ax[0].set_title('P%02d-%s'%(predicted, class_names[predicted]))
    ax[1].imshow(timg[:,:,0])
    ax[1].set_title('T%02d-%s'%(label, class_names[label]))
    f.suptitle(suptitle)
    plt.savefig(filename.replace('.jpg', '_P%s_T%s.png'%(class_names[predicted], class_names[label])))
    plt.close()

def evaluate_model(model, dataloaders, basename=''):
    #model.eval()
    model.train()
    cnt = 0
    for phase in ['valid', 'train']:
    #for phase in ['valid']:
    #for phase in ['train']:
        with torch.no_grad():
            error_dir = os.path.join(basename, phase)
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            error_inputs = []
            error_filenames = []
            error_labels = []
            error_preds = []
            y_true = []
            y_pred = []

            cnt = 0
            for inputs, labels, img_path, class_name, didx in dataloaders[phase]:
                 inputs = inputs.to(device)
                 labels = labels.to(device)
                 outputs = model(inputs)
                 _, preds = torch.max(outputs, 1)
                 llist = list(labels.detach().numpy())
                 lpred = list(preds.detach().numpy())

                 ninputs = inputs.detach().numpy()
                 y_true.extend(llist)
                 y_pred.extend(lpred)

                 ### keep track of everything we got wrong
                 #wrong_inds = [ind for ind,(lp,l) in enumerate(zip(lpred, llist)) if not lp==l]
                 ##if False:
                 #if cnt < 200:
                 #    for wi in wrong_inds:
                 #        name = os.path.join(error_dir, 'C%05d_%02d'%(cnt,wi) + 'D%05d'%didx[wi] + os.path.split(img_path[wi])[1])
                 #        plot_error(ninputs[wi,0], img_path[wi], llist[wi], lpred[wi], name, img_path[wi])
                 #        error_inputs.append(ninputs[wi,0])
                 #        error_filenames.append(img_path[wi])
                 #        error_labels.append(llist[wi])
                 #        error_preds.append(lpred[wi])
                 #        #print(llist[wi], lpred[wi], outputs[wi])
                 cnt+=inputs.shape[0]
                 print(cnt)
                 if cnt > 5000:
                     break

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(y_true, y_pred, classes=class_names,
                              title='Confusion matrix, without normalization',
                              filename=basename+'_'+phase+'_'+'unnormalized_confusion.png')

        # Plot normalized confusion matrix
        plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True,
                              title='Normalized confusion matrix',
                              filename=basename+'_'+phase+'_'+'normalized_confusion.png')


def load_latest_checkpoint(loaddir='experiment_name', search='*.pth'):
    search_path = os.path.join(loaddir, 'checkpoints', search)
    print("searching", search_path)
    search = sorted(glob(search_path))
    print('found %s checkpoints'%len(search))
    ckpt = search[-1]
    print('loading %s' %ckpt)
    return ckpt, torch.load(ckpt)

def plot_history(history_dict, filename):
    try:
        # old code saved as pt
        history_dict['valid'] = np.array([x.cpu().numpy() for x in history_dict['valid']])
        history_dict['train'] = np.array([x.cpu().numpy() for x in history_dict['train']])
    except:
        pass
    plt.figure()
    _ind1 = np.argmax(history_dict['valid'])
    _ind2 = np.argmin(history_dict['valid'])
    rs = [history_dict['valid'][_ind1], history_dict['valid'][_ind2]]
    plt.plot(history_dict['train'], label='train')
    plt.plot(history_dict['valid'], label='valid')
    plt.scatter([_ind1, _ind2], rs, marker='x', s=30)
    plt.legend()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    exp_name = 'uvp_big_small_v2_noliving'
    exp_path = os.path.join('experiments', exp_name)
    print(sys.argv)
    if len(sys.argv)>1:
        search  = sys.argv[1]
    else:
        search = '*.pth'
    ckpt_name, ckpt_dict = load_latest_checkpoint(exp_path, search)
    bname = ckpt_name.replace('.pth', '')
    datadir = './'
    plot_history(ckpt_dict['loss'], bname+'_loss.png')
    plot_history(ckpt_dict['accuracy'], bname+'_accuracy.png')
    batch_size = 32
    train_ds = UVPDataset(csv_file=os.path.join(exp_path, 'train.csv'), seed=34, augment=False)
    class_names = train_ds.classes
    class_weights = train_ds.weights
    valid_ds = UVPDataset(csv_file=os.path.join(exp_path, 'valid.csv'), seed=334, classes=class_names, weights=class_weights, augment=False)
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
    print("loading state dict")
    rmodel.load_state_dict(ckpt_dict['state_dict'])
    del ckpt_dict
    rmodel = rmodel.to(device)
    evaluate_model(rmodel, dataloaders, bname)

