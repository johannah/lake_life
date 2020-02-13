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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
from utils import get_model, get_dataset, set_model_mode
from train import forward_pass
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
    acc = accuracy_score(y_true, y_pred)
    if not title:
        if normalize:
            title = 'Normalized Acc %.02f'%acc
        else:
            title = 'Acc %.02f'%acc

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

def write_results(basename, phase, idxs, class_names, y_true, y_pred):
    ''' idxs is idx into tsv file ''''
    path = basename+'_'+phase+'_'+'predictions.csv'
    print('writing', rpath)
    fo = open(rpath, 'w')
    fo.write('idx,true_name,true_num,pred_name,pred_num\n')
    for line in range(len(idxs)):
        fo.write('%s,%s,%s,%s,%s\n'%(idxs[line], class_names[y_true[line]], y_true[line],
                                           class_names[y_pred[line]], y_pred[line]))
    fo.close()

def evaluate_model(model_dict, dataloaders, class_names, basename='', device='cpu'):
    cnt = 0
    model_dict = set_model_mode(model_dict, 'eval')
    results = {}
    for phase in ['train']:
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
            idxs = []
            print("starting phase", phase)
            for data in dataloaders[phase]:
                print('evaluating', phase, cnt)
                if True:
                ##if cnt > 200000:
                #    continue
                #else:
                    images, class_num, filepath, idx = data
                    model_dict, class_num, predictions, loss = forward_pass(model_dict, images, class_num, device)
                    y_true.extend(list(class_num.detach().cpu().numpy()))
                    y_pred.extend(list(predictions.detach().cpu().numpy()))
                    idxs.extend(list(class_num.detach().cpu().numpy()))
                    cnt +=len(class_num)
        write_results(basename, phase, idxs, class_names, y_true, y_pred)
    return class_names, y_true, y_pred


def find_latest_checkpoint(loaddir='experiment_name', search='*.pth'):
    search_path = os.path.join(loaddir, 'checkpoints', search)
    print("searching", search_path)
    search = sorted(glob(search_path))
    print('found %s checkpoints'%len(search))
    ckpt = search[-1]
    print('loading %s' %ckpt)
    return ckpt

def plot_history(history_dict, filename):
    for key in sorted(history_dict.keys()):
        _amax = np.argmax(history_dict[key])
        _amin = np.argmax(history_dict[key])
        rs = [history_dict[key][_amax], history_dict[key][_amin]]
        plt.plot(history_dict[key], label=key)
        plt.scatter([_amax, _amin], rs, marker='x', s=30)
    plt.legend()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    # can give model or checkpoints dir to search for model in
    device = 'cuda'
    use_dir = sys.argv[-1]
    if not use_dir.split('.')[-1] == '.pt':
        load_path = sorted(glob(os.path.join(use_dir, '**.pt')))[-1]
    else:
        load_path = use_dir
    checkpoints_dir = os.path.split(load_path)[0]
    exp_dir = os.path.split(checkpoints_dir)[0]

    dataloaders, class_names = get_dataset(exp_dir, 64, num_workers=1, evaluation=True)
    num_classes = len(class_names)
    model_dict, cnt_start, epoch_cnt, all_accuracy, all_losses = get_model(load_path, num_classes, 512, device)
    bname = load_path.replace('.pt', '')
    loss_path = bname+'_loss.png'
    acc_path = bname+'_accuracy.png'
    if not os.path.exists(loss_path):
        plot_history(all_losses, loss_path)
    if not os.path.exists(acc_path):
        plot_history(all_accuracy, acc_path)

    class_names, y_true, y_pred = evaluate_model(model_dict, dataloaders, class_names, bname, device)
    print('plotting confusion matrices')
    # TODO - load from file
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=class_names,
                          filename=basename+'_'+phase+'_'+'unnormalized_confusion.png')

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True,
                          filename=basename+'_'+phase+'_'+'normalized_confusion.png')
#

