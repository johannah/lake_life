"""
Associative Compression Network based on https://arxiv.org/pdf/1804.02476v2.pdf
determined this architecture based on experiments in github.com/johannah/ACN

"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
from glob import glob
import numpy as np
from copy import deepcopy, copy

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision import transforms
from torch.nn.utils.clip_grad import clip_grad_value_

from utils import create_new_info_dict, save_checkpoint, seed_everything
from utils import plot_example, plot_losses, count_parameters
from utils import set_model_mode, kl_loss_function, write_log_files
from utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic
from acn_models import tPTPriorNetwork, ACNVQVAEres
from uvp_dataloader import UVPDataset

import torchvision
from torchvision import datasets, models, transforms
from config import exp_dir, exp_name, get_checkpoints_dir, img_size
from utils import get_model, get_dataset, save_model, set_model_mode
from IPython import embed

# setup loss-specific parameters for data
# data going into dml should be bt -1 and 1
# atari data coming in is uint 0 to 255
norm_by = 255.0
rescale = lambda x: ((x / norm_by) * 2.)-1
rescale_inv = lambda x: 255*((x+1)/2.)

def create_models(info, model_loadpath=''):
    '''
    load details of previous model if applicable, otherwise create new models
    '''
    train_cnt = 0
    epoch_cnt = 0

    # use argparse device no matter what info dict is loaded
    preserve_args = ['device', 'batch_size', 'save_every_epochs',
                     'base_filepath', 'model_loadpath', 'perplexity',
                     'use_pred']
    largs = info['args']
    # load model if given a path
    if model_loadpath !='':
        _dict = torch.load(model_loadpath, map_location=lambda storage, loc:storage)
        dinfo = _dict['info']
        pkeys = info.keys()
        for key in dinfo.keys():
            if key not in preserve_args or key not in pkeys:
                info[key] = dinfo[key]
        train_cnt = info['train_cnts'][-1]
        epoch_cnt = info['epoch_cnt']
        info['args'].append(largs)

    # pixel cnn architecture is dependent on loss
    # for dml prediction, need to output mixture of size nmix
    info['nmix'] =  (2*info['nr_logistic_mix']+info['nr_logistic_mix'])*info['target_channels']
    info['output_dim']  = info['nmix']
    # setup models
    # acn prior with vqvae embedding
    vq_acn_model = ACNVQVAEres(code_len=info['code_length'],
                               input_size=info['input_channels'],
                               output_size=info['output_dim'],
                               hidden_size=info['hidden_size'],
                               num_clusters=info['num_vqk'],
                               num_z=info['num_z'],
                               ).to(info['device'])

    prior_model = tPTPriorNetwork(size_training_set=info['size_training_set'],
                               code_length=info['code_length'], k=info['num_k']).to(info['device'])
    prior_model.codes = prior_model.codes.to(info['device'])

    model_dict = {'vq_acn_model':vq_acn_model, 'prior_model':prior_model}
    parameters = []
    for name,model in model_dict.items():
        parameters+=list(model.parameters())
        print('created %s model with %s parameters' %(name,count_parameters(model)))

    model_dict['opt'] = optim.Adam(parameters, lr=info['learning_rate'])

    if model_loadpath !='':
       for name,model in model_dict.items():
            model_dict[name].load_state_dict(_dict[name+'_state_dict'])
    return model_dict, info, train_cnt, epoch_cnt

def account_losses(loss_dict):
    ''' return avg losses for each loss sum in loss_dict based on total examples in running key'''
    loss_avg = {}
    for key in loss_dict.keys():
        if key != 'running':
            loss_avg[key] = loss_dict[key]/loss_dict['running']
    return loss_avg

def clip_parameters(model_dict, clip_val=10):
    for name, model in model_dict.items():
        if 'model' in name:
            clip_grad_value_(model.parameters(), clip_val)
    return model_dict

def forward_pass(model_dict, images, index_indexes, phase, info):
    # prepare data in appropriate way
    bs,c,h,w = images.shape
    images = images.to(info['device'], non_blocking=True)
    images = F.dropout(images, p=info['dropout_rate'], training=True, inplace=False)
    # put action cond and reward cond in and project to same size as state
    z, u_q = model_dict['vq_acn_model'](images)
    u_q_flat = u_q.view(bs, info['code_length'])
    if phase == 'train':
        # check that we are getting what we think we are getting from the replay
        # buffer
        assert index_indexes.max() < model_dict['prior_model'].codes.shape[0]
        model_dict['prior_model'].update_codebook(index_indexes, u_q_flat.detach())
    rec_dml, z_e_x, z_q_x, latents =  model_dict['vq_acn_model'].decode(z)
    u_p, s_p = model_dict['prior_model'](u_q_flat)
    u_p = u_p.view(bs, 3, 8, 8)
    s_p = s_p.view(bs, 3, 8, 8)
    return model_dict, images, rec_dml, u_q, u_p, s_p, rec_dml, z_e_x, z_q_x, latents

def run(train_cnt, model_dict, data_dict, phase, info):
    st = time.time()
    loss_dict = {'running': 0,
             'kl':0,
             'rec_%s'%info['rec_loss_type']:0,
             'vq':0,
             'commit':0,
             'loss':0,
              }
    #print('starting', phase, 'cuda', torch.cuda.memory_allocated(device=None))
    data_loader = data_dict[phase]
    #data_loader.reset_unique()
    num_batches = len(data_loader)//info['batch_size']
    set_model_mode(model_dict, phase)
    torch.set_grad_enabled(phase=='train')
    #while data_loader.unique_available:
    batch_cnt = 0
    for data in data_loader:
        for key in model_dict.keys():
            model_dict[key].zero_grad()
        images, class_num, filepath, idx = data
        fp_out = forward_pass(model_dict, images, idx, phase, info)
        model_dict, images, rec_dml, u_q, u_p, s_p, rec_dml, z_e_x, z_q_x, latents = fp_out
        bs,c,h,w = images.shape
        if batch_cnt == 0:
            log_ones = torch.zeros(bs, info['code_length']).to(info['device'])
        if bs != log_ones.shape[0]:
            log_ones = torch.zeros(bs, info['code_length']).to(info['device'])
        kl = kl_loss_function(u_q.view(bs, info['code_length']), log_ones,
                              u_p.view(bs, info['code_length']), s_p.view(bs, info['code_length']),
                              reduction=info['reduction'])
        rec_loss = discretized_mix_logistic_loss(rec_dml, images, nr_mix=info['nr_logistic_mix'], reduction=info['reduction'])
        vq_loss = F.mse_loss(z_q_x, z_e_x.detach(), reduction=info['reduction'])
        commit_loss = F.mse_loss(z_e_x, z_q_x.detach(), reduction=info['reduction'])
        commit_loss *= info['vq_commitment_beta']
        loss = kl+rec_loss+commit_loss+vq_loss
        loss_dict['running']+=bs
        loss_dict['loss']+=loss.detach().cpu().item()
        loss_dict['kl']+= kl.detach().cpu().item()
        loss_dict['vq']+= vq_loss.detach().cpu().item()
        loss_dict['commit']+= commit_loss.detach().cpu().item()
        loss_dict['rec_%s'%info['rec_loss_type']]+=rec_loss.detach().cpu().item()
        if phase == 'train':
            model_dict = clip_parameters(model_dict)
            loss.backward()
            model_dict['opt'].step()
            train_cnt+=bs
        if batch_cnt == num_batches-3:
            # store example near end for plotting
            rec_yhat = sample_from_discretized_mix_logistic(rec_dml, info['nr_logistic_mix'], only_mean=info['sample_mean'], sampling_temperature=info['sampling_temperature'])
            example = {
                       'target':images.detach().cpu(),
                       'rec':rec_yhat.detach().cpu(),
                       }
        if not batch_cnt % 100:
            print(train_cnt, batch_cnt, account_losses(loss_dict))
            print(phase, 'cuda', torch.cuda.memory_allocated(device=None))
        batch_cnt+=1

    loss_avg = account_losses(loss_dict)
    print("finished %s after %s secs at cnt %s"%(phase,
                                                time.time()-st,
                                                train_cnt,
                                                ))
    return loss_avg, example


def train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info, rescale_inv):
    print('starting training routine')
    base_filepath = info['base_filepath']
    base_filename = os.path.split(info['base_filepath'])[1]
    while train_cnt < info['num_examples_to_train']:
        print('starting epoch %s on %s'%(epoch_cnt, info['device']))

        with torch.autograd.profiler.profile(use_cuda=info['device']=='cuda') as prof:
            train_loss_avg, train_example = run(train_cnt,
                                                model_dict,
                                                data_dict,
                                                phase='train', info=info)
            print(prof)
        epoch_cnt +=1
        train_cnt +=info['size_training_set']
        if not epoch_cnt % info['save_every_epochs'] or epoch_cnt == 1:
            # make a checkpoint
            print('starting valid phase')
            #model_dict, data_dict, valid_loss_avg, valid_example = run(train_cnt,
            valid_loss_avg, valid_example = run(train_cnt,
                                                 model_dict,
                                                 data_dict,
                                                 phase='valid', info=info)
            for loss_key in valid_loss_avg.keys():
                for lphase in ['train_losses', 'valid_losses']:
                    if loss_key not in info[lphase].keys():
                        info[lphase][loss_key] = []
                info['valid_losses'][loss_key].append(valid_loss_avg[loss_key])
                info['train_losses'][loss_key].append(train_loss_avg[loss_key])

            # store model
            state_dict = {}
            for key, model in model_dict.items():
                state_dict[key+'_state_dict'] = model.state_dict()

            info['train_cnts'].append(train_cnt)
            info['epoch_cnt'] = epoch_cnt
            state_dict['info'] = info

            ckpt_filepath = os.path.join(base_filepath, "%s_%010dex.pt"%(base_filename, train_cnt))
            train_img_filepath = os.path.join(base_filepath,"%s_%010d_train_rec.png"%(base_filename, train_cnt))
            valid_img_filepath = os.path.join(base_filepath, "%s_%010d_valid_rec.png"%(base_filename, train_cnt))
            plot_filepath = os.path.join(base_filepath, "%s_%010d_loss.png"%(base_filename, train_cnt))
            plot_example(train_img_filepath, train_example, num_plot=10)
            plot_example(valid_img_filepath, valid_example, num_plot=10)
            save_checkpoint(state_dict, filename=ckpt_filepath)

            plot_losses(info['train_cnts'],
                        info['train_losses'],
                        info['valid_losses'], name=plot_filepath, rolling_length=1)


def call_plot(model_dict, data_dict, info, sample, tsne, pca):
    from acn_utils import tsne_plot
    from acn_utils import pca_plot
    # always be in eval mode - so we dont swap neighbors
    model_dict = set_model_mode(model_dict, 'valid')
    with torch.no_grad():
        for phase in ['valid', 'train']:
            data_loader = data_dict[phase]
            for data in data_loader:
                images, class_num, filepath, idx = data
                fp_out = forward_pass(model_dict, images, idx, phase, info)
                model_dict, states, actions, rewards, target, u_q, u_p, s_p, rec_dml, z_e_x, z_q_x, latents = fp_out
                rec_yhat = sample_from_discretized_mix_logistic(rec_dml, info['nr_logistic_mix'], only_mean=info['sample_mean'], sampling_temperature=info['sampling_temperature'])
                f,ax = plt.subplots(10,3)
                ax[0,0].set_title('prev')
                ax[0,1].set_title('true')
                ax[0,2].set_title('pred')
                for i in range(10):
                    ax[i,0].matshow(states[i,-1])
                    ax[i,1].matshow(target[i,-1])
                    ax[i,2].matshow(rec_yhat[i,-1])
                    ax[i,0].axis('off')
                    ax[i,1].axis('off')
                    ax[i,2].axis('off')
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.tight_layout()

                plt_path = info['model_loadpath'].replace('.pt', '_%s_plt.png'%phase)
                print('plotting', plt_path)
                plt.savefig(plt_path)
                bs = states.shape[0]
                u_q_flat = u_q.view(bs, info['code_length'])
                X = u_q_flat.cpu().numpy()
                color = index_indexes
                images = target[:,0].cpu().numpy()
                if tsne:
                    param_name = '_tsne_%s_P%s.html'%(phase, info['perplexity'])
                    html_path = info['model_loadpath'].replace('.pt', param_name)
                    tsne_plot(X=X, images=images, color=color,
                          perplexity=info['perplexity'],
                          html_out_path=html_path, serve=False)
                if pca:
                    param_name = '_pca_%s.html'%(phase)
                    html_path = info['model_loadpath'].replace('.pt', param_name)
                    pca_plot(X=X, images=images, color=color,
                              html_out_path=html_path, serve=False)
                break


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='train acn')
    # operatation options
    parser.add_argument('-l', '--model_loadpath', default='', help='load model to resume training or sample')
    parser.add_argument('-ll', '--load_last_model', default='',  help='load last model from directory from directory')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=394)
    parser.add_argument('--num_threads', default=2)
    parser.add_argument('--tf_train_last_frame', action='store_true', default=False)
    parser.add_argument('-se', '--save_every_epochs', default=5, type=int)
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--input_channels', default=1, type=int, help='num of channels of input')
    parser.add_argument('--target_channels', default=1, type=int, help='num of channels of target')
    parser.add_argument('--num_examples_to_train', default=50000000, type=int)
    parser.add_argument('-e', '--exp_name', default='mid_1e6trfreeway_spvq_twgrad', help='name of experiment')
    parser.add_argument('-dr', '--dropout_rate', default=0.2, type=float)
    parser.add_argument('-r', '--reduction', default='sum', type=str, choices=['sum', 'mean'])
    parser.add_argument('--rec_loss_type', default='dml', type=str, help='name of loss. options are dml', choices=['dml'])
    parser.add_argument('--nr_logistic_mix', default=10, type=int)
    parser.add_argument('--small', action='store_true', default=False, help='create and operate on smaller, 20x20 maxpooled frames  as opposed to default 40x40')
    parser.add_argument('--big', action='store_true', default=False, help='operate on orig 84x84 frames')
    # acn model setup
    parser.add_argument('-cl', '--code_length', default=192, type=int)
    parser.add_argument('-k', '--num_k', default=5, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    # vq model setup
    parser.add_argument('--vq_commitment_beta', default=0.25, help='scale for loss 3 in vqvae - how hard to enforce commitment to cluster')
    parser.add_argument('--num_vqk', default=512, type=int)
    parser.add_argument('--num_z', default=64, type=int)
    #parser.add_argument('-kl', '--kl_beta', default=.5, type=float, help='scale kl loss')
    parser.add_argument('--last_layer_bias', default=0.0, help='bias for output decoder - should be 0 for dml')
    parser.add_argument('-sm', '--sample_mean', action='store_true', default=False)
    # dataset setup
    parser.add_argument('--size_training_set', default=60000, type=int, help='number of random examples from base_train_buffer_path to use')
    # sampling info
    parser.add_argument('-s', '--sample', action='store_true', default=False)
    parser.add_argument('-tf', '--teacher_force_prev', action='store_true', default=False)
    parser.add_argument('-st', '--sampling_temperature', default=0.1, help='temperature to sample dml')
    # latent pca/tsne info
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--tsne', action='store_true', default=False)
    parser.add_argument('-p', '--perplexity', default=10, type=int, help='perplexity used in scikit-learn tsne call')


    args = parser.parse_args()
    checkpoints_dir = get_checkpoints_dir()
    # note - when reloading model, this will use the seed given in args - not
    # the original random seed
    seed_everything(args.seed, args.num_threads)
    # get naming scheme
    if args.load_last_model != '':
        # load last model from this dir
        base_filepath = args.load_last_model
        args.model_loadpath = sorted(glob(os.path.join(base_filepath, '*.pt')))[-1]
        print('loading last model....')
        print(args.model_loadpath)
    elif args.model_loadpath != '':
        # use full path to model
        base_filepath = os.path.split(args.model_loadpath)[0]
    else:
        # create new base_filepath
        base_filepath = get_checkpoints_dir()
    print('base filepath is %s'%base_filepath)
    info = create_new_info_dict(vars(args), base_filepath, __file__)
    model_dict, info, train_cnt, epoch_cnt = create_models(info, args.model_loadpath)
    train_ds = UVPDataset(csv_file=os.path.join(exp_dir, 'train.csv'), seed=args.seed,
                          valid=False, img_size=img_size)
    valid_ds = UVPDataset(csv_file=os.path.join(exp_dir, 'test.csv'),
                          seed=args.seed, valid=True, img_size=img_size)
    train_sam = torch.utils.data.sampler.SubsetRandomSampler(indices=np.arange(len(train_ds)))
    valid_sam = torch.utils.data.sampler.SubsetRandomSampler(indices=np.arange(len(train_ds)))
    data_dict = {'train':torch.utils.data.DataLoader(train_ds,
                                                     batch_size=args.batch_size,
                                                     sampler=train_sam,
                                                     num_workers=4),
                 'valid':torch.utils.data.DataLoader(valid_ds,
                                                     batch_size=args.batch_size,
                                                     sampler=valid_sam,
                                                     num_workers=2)}
    if max([args.sample, args.tsne, args.pca]):
        call_plot(model_dict, data_dict, info, args.sample, args.tsne, args.pca)
    else:
        # only train if we weren't asked to do anything else
        write_log_files(info)
        train_acn(train_cnt, epoch_cnt, model_dict, data_dict, info, rescale_inv)
