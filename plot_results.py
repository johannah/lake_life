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

def evaluate_model(model_dict, dataloaders, basename='', device='cpu'):
    cnt = 0
    model_dict = set_model_mode(model_dict, 'eval')
    #for phase in ['valid', 'train']:
    for phase in ['valid', 'train']:
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
            num_large_classes = len(dataloaders['train'].dataset.large_classes)-1 # -1 for the wrong_class_size class
            class_names = dataloaders['train'].dataset.large_classes[:-1] +  dataloaders['train'].dataset.small_classes

            # TODOOOOOO it takes forever to load train phase - something is hosed
            print("starting phase", phase)
            for data in dataloaders[phase]:
                print('evaluating', phase, cnt)
                if cnt > 10000:
                    break
                else:
                    belongs_in_small_class_idx = dataloaders[phase+'_large_wrong_class_idx']
                    inputs, class_num, large_class_num, small_class_num, filepath, idx = data
                    out = forward_pass(model_dict, inputs, large_class_num, small_class_num, belongs_in_small_class_idx, device, phase='eval')
                    model_dict, large_outputs, small_outputs, _,  pred_small_class_num, large_predictions, small_predictions, pred_small, some_small  = out
                    # large_class_num[pred_small] should ideally ==  belongs_in_small_class_idx
                    true_class_num = []
                    pred_class_num = []
                    lpred = list(large_predictions.detach().numpy())
                    small_class_num = small_class_num.cpu().numpy()
                    large_class_num = large_class_num.cpu().numpy()
                    if some_small:
                        spred = list(small_predictions.detach().numpy())
                    else:
                        spred = []
                    for idx in range(len(lpred)):
                        if lpred[idx] != belongs_in_small_class_idx:
                            pred_class_num.append(lpred[idx])
                        else:
                            pred_class_num.append(spred.pop(0)+num_large_classes)
                    for idx, cn in enumerate(large_class_num):
                        if cn == belongs_in_small_class_idx:
                            true_class_num.append(small_class_num[idx]+num_large_classes)
                        else:
                            true_class_num.append(cn)

                    y_true.extend(true_class_num)
                    y_pred.extend(pred_class_num)
                    cnt +=len(pred_class_num)


                     ### keep track of everything we got wrong
                     # ninputs = inputs.detach().numpy()
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
#                     cnt+=inputs.shape[0]
#                     print(cnt)
#                     if cnt > 1000:
#                         break
#
#        # Plot non-normalized confusion matrix
        plot_confusion_matrix(y_true, y_pred, classes=class_names,
                              title='Confusion matrix, without normalization',
                              filename=basename+'_'+phase+'_'+'unnormalized_confusion.png')

        # Plot normalized confusion matrix
        plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True,
                              title='Normalized confusion matrix',
                              filename=basename+'_'+phase+'_'+'normalized_confusion.png')
#

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
    use_dir = sys.argv[-1]
    if not use_dir.split('.')[-1] == '.pt':
        load_path = sorted(glob(os.path.join(use_dir, '**.pt')))[-1]
    else:
        load_path = use_dir
    checkpoints_dir = os.path.split(load_path)[0]
    exp_dir = os.path.split(checkpoints_dir)[0]

    dataloaders, large_class_names, small_class_names = get_dataset(exp_dir, 64, num_workers=1, evaluation=True)
    large_num_classes = len(large_class_names)
    small_num_classes = len(small_class_names)
    model_dict, cnt_start, epoch_cnt, all_accuracy, all_losses = get_model(load_path, large_num_classes, small_num_classes, 'cpu')
    bname = load_path.replace('.pt', '')
    loss_path = bname+'_loss.png'
    acc_path = bname+'_accuracy.png'
    if not os.path.exists(loss_path):
        plot_history(all_losses, loss_path)
    if not os.path.exists(acc_path):
        plot_history(all_accuracy, acc_path)
    evaluate_model(model_dict, dataloaders, bname)

