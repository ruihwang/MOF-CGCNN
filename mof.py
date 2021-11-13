import os
import shutil
import sys
import time
import warnings
import csv
from random import sample
from typing import Dict, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from mofcgcnn.data import CIFData
from mofcgcnn.data import collate_pool
from mofcgcnn.model import CrystalGraphConvNet


class Normalizer(object):
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
class MOF_CGCNN(object):
    def __init__(self, 
            cuda : bool = False,
            task : str = 'regression',
            works : int = 0,
            start_epoch : int = 0,
            epoch : int = 200,
            batch_size : int = 24,
            lr : float = 0.008,
            optim : str = 'Adam',
            lr_milestones : list = [100],
            momentum : float = 0.9,
            weight_decay : float = 0.0,
            print_freq : int = 10,
            dropout : float = 0.0,
            atom_fea_len :int = 64,
            n_conv : int = 5,
            h_fea_len : int = 364,
            resume : str = None,
            root_file : str = None,
            trainset : List = None,
            valset : List = None,
            testset : List = None):
        self.cuda = cuda
        self.task = task
        self.works = works
        self.epoch = epoch
        self.start_epoch = start_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.optim = optim
        self.lr_milestones = lr_milestones
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.print_freq = print_freq
        self.dropout = dropout
        self.atom_fea_len = atom_fea_len
        self.n_conv = n_conv
        self.h_fea_len = h_fea_len
        self.resume = resume
        self.trainset = CIFData(root_file,trainset)
        self.valset = CIFData(root_file,valset)
        self.testset = CIFData(root_file,testset)
        self.train_loader = DataLoader(self.trainset, batch_size=self.batch_size,
        num_workers=self.works,shuffle=True,collate_fn=collate_pool, pin_memory=self.cuda)
        self.val_loader = DataLoader(self.valset, batch_size=self.batch_size,
        num_workers=self.works,shuffle=False,
        collate_fn=collate_pool, pin_memory=self.cuda)
        self.test_loader = DataLoader(self.testset, batch_size=self.batch_size,
        num_workers=self.works,shuffle=False,
        collate_fn=collate_pool, pin_memory=self.cuda)

    def train_MOF(self):
        if self.task == 'classification':
            normalizer = Normalizer(torch.zeros(2))
            normalizer.load_state_dict({'mean': 0., 'std': 1.})
        else:
            if len(self.trainset) < 4000:
                warnings.warn('Dataset has less than 4000 data points. '
                                        'Lower accuracy is expected. ')
                sample_data_list = [self.trainset[i] for i in range(len(self.trainset))]
            else:
                sample_data_list = [self.trainset[i] for i in sample(range(len(self.trainset)), 4000)]
            _, sample_target, _ = collate_pool(sample_data_list)
            normalizer = Normalizer(sample_target)
        if self.task == 'regression':
            best_mae_error = 1e10
        else:
            best_mae_error = 0.
        structures, targets, _ = self.trainset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        self.n_p = len(targets)
        print("Predicting ", self.n_p, " properties!!")
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=self.atom_fea_len,
                                n_conv=self.n_conv,
                                h_fea_len=self.h_fea_len,
                                n_p= self.n_p,
                                classification=True if self.task ==
                                'classification' else False,dropout=self.dropout)
        if self.cuda == True:
            model.cuda()
        if self.task == 'classification':
            criterion = nn.NLLLoss()
        else:
            criterion = nn.MSELoss()
        if self.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), self.lr,
                              momentum=self.momentum,
                              weight_decay=self.weight_decay)
        elif self.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), self.lr,
                                weight_decay=self.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed ')
        if self.resume:
            if os.path.isfile(self.resume):
                print("=> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                best_mae_error = checkpoint['best_mae_error']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                normalizer.load_state_dict(checkpoint['normalizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.resume))
        scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones,
                            gamma=0.1)
        for epoch in range(self.epoch):
        # train for one epoch
            self.train_model(self.train_loader, model, criterion, optimizer, epoch, normalizer)
        # evaluate on validation set
            mae_error = self.validate(self.val_loader, model, criterion, normalizer)
            if mae_error != mae_error:
                print('Exit due to NaN')
                sys.exit(1)
            scheduler.step()
        # remember the best mae_eror and save checkpoint
            if self.task == 'regression':
                is_best = mae_error < best_mae_error
                best_mae_error = min(mae_error, best_mae_error)
            else:
                is_best = mae_error > best_mae_error
                best_mae_error = max(mae_error, best_mae_error)
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'orig_atom_fea_len':orig_atom_fea_len,
            'nbr_fea_len':nbr_fea_len,
            'atom_fea_len': self.atom_fea_len,
            'n_conv':self.n_conv,
            'h_fea_len':self.h_fea_len,
            'n_p': self.n_p,
            'task':self.task}, is_best)
    # test best model
        print('---------Evaluate Model on Test Set---------------')
        best_checkpoint = torch.load('model_best.pth.tar')
        model.load_state_dict(best_checkpoint['state_dict'])
        self.validate(self.test_loader, model, criterion, normalizer, test=True)

    def pred_MOF(self,
        root_file : str = None,
        predset : List = None,
        modelpath : str = None,):

        if os.path.isfile(modelpath):
            print("=> loading model params '{}'".format(modelpath))
            model_checkpoint = torch.load(modelpath,
                                  map_location=lambda storage, loc: storage)
            print("=> loaded model params '{}'".format(modelpath))
        else:
            print("=> no model params found at '{}'".format(modelpath))
        pred_data = CIFData(root_file,predset,pred=True)
        test_loader = DataLoader(pred_data, batch_size=self.batch_size, shuffle=False,
                             num_workers=self.works, collate_fn=collate_pool,
                             pin_memory=self.cuda)
    # build model
        model = CrystalGraphConvNet(orig_atom_fea_len=model_checkpoint['orig_atom_fea_len'],
                             nbr_fea_len=model_checkpoint['nbr_fea_len'],
                                atom_fea_len=model_checkpoint['atom_fea_len'],
                                n_conv=model_checkpoint['n_conv'],
                                h_fea_len=model_checkpoint['h_fea_len'],
                                n_p = model_checkpoint['n_p'],
                                classification=True if model_checkpoint['task'] ==
                                'classification' else False)
        if self.cuda:
            model.cuda()
    # define loss func and optimizer
        normalizer = Normalizer(torch.zeros(3))
    # optionally resume from a checkpoint
        if os.path.isfile(modelpath):
            print("=> loading model '{}'".format(modelpath))
            checkpoint = torch.load(modelpath,
                                 map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded model '{}' (epoch {}, validation {})"
              .format(modelpath, checkpoint['epoch'],
                      checkpoint['best_mae_error']))
        else:
            print("=> no model found at '{}'".format(modelpath))
        self.pred_fun(test_loader, model, normalizer) 
    def cal_loss(self,lossall,number):
        if number == 1:
            loss = lossall[0]
        elif number == 2 :
            loss = (lossall[0] + lossall[1])/2
        elif number == 3:
            loss = (lossall[0] + lossall[1] + lossall[2])/3
        elif number == 4:
            loss = (lossall[0] + lossall[1] + lossall[2] + lossall[3])/4
        elif number == 5:
            loss = (lossall[0] + lossall[1] + lossall[2] + lossall[3] + lossall[4])/5
        elif number ==6:
            loss = (lossall[0] + lossall[1] + lossall[2] + lossall[3] + lossall[4] + lossall[5])/6
        return loss
    def train_model(self,train_loader, model, criterion, optimizer, epoch, normalizer):
        losses = AverageMeter()
        if self.task == 'regression':
            mae_errors = AverageMeter()
        else:
            accuracies = AverageMeter()
            precisions = AverageMeter()
            recalls = AverageMeter()
            fscores = AverageMeter()
            auc_scores = AverageMeter()
    # switch to train mode
        model.train()
        for i, (input, target, _) in enumerate(train_loader):
            if self.cuda:
                input_var = (Variable(input[0].cuda()),
                         Variable(input[1].cuda()),
                         input[2].cuda(),
                         input[3].cuda(),
                         [crys_idx.cuda() for crys_idx in input[4]],
                         Variable(input[5].cuda()))
            else:
                input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3],
                         input[4],
                         Variable(input[5]))
        # normalize target
            if self.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            if self.cuda:
                target_var = Variable(target_normed.cuda())
            else:
                target_var = Variable(target_normed)

        # compute output

            output = model(*input_var)
            mse_loss = [criterion(output[:,i], target_var[:,i]) for i in range(self.n_p)]
            loss = self.cal_loss(mse_loss,self.n_p)
        # measure accuracy and record loss
            if self.task == 'regression':
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
            else:
                accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))
        # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % self.print_freq == 0:
                if self.task == 'regression':
                    print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), loss=losses, mae_errors=mae_errors)
                )
                else:
                    print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores)
                )
        
    def validate(self,val_loader, model, criterion, normalizer, test=False):
        losses = AverageMeter()
        if self.task == 'regression':
            mae_errors = AverageMeter()
        else:
            accuracies = AverageMeter()
            precisions = AverageMeter()
            recalls = AverageMeter()
            fscores = AverageMeter()
            auc_scores = AverageMeter()
        if test:
            test_targets = []
            test_preds = []
            test_cif_ids = []
        else:
            val_targets = []
            val_preds = []
            val_cif_ids = []
    # switch to evaluate mode
        model.eval()
        for i, (input, target, batch_cif_ids) in enumerate(val_loader):
            if self.cuda:
                with torch.no_grad():
                    input_var = (Variable(input[0].cuda()),
                             Variable(input[1].cuda()),
                             input[2].cuda(),
                             input[3].cuda(),
                             [crys_idx.cuda() for crys_idx in input[4]],
                             Variable(input[5].cuda()))
            else:
                with torch.no_grad():
                    input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3],
                             input[4],
                             Variable(input[5]))
            if self.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            if self.cuda:
                with torch.no_grad():
                    target_var = Variable(target_normed.cuda())
            else:
                with torch.no_grad():
                    target_var = Variable(target_normed)
        # compute output
            output = model(*input_var)
            mse_loss = [criterion(output[:,i], target_var[:,i]) for i in range(self.n_p)]
            loss = self.cal_loss(mse_loss,self.n_p)
         
        # measure accuracy and record loss
            if self.task == 'regression':
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                mae_errors.update(mae_error, target.size(0))
                if test:
                    test_pred = normalizer.denorm(output.data.cpu())
                    test_target = target
                    test_preds += test_pred
                    test_targets += test_target
                    test_cif_ids += batch_cif_ids
                else:
                    val_pred = normalizer.denorm(output.data.cpu())
                    val_target = target
                    val_preds += val_pred
                    val_targets += val_target
                    val_cif_ids += batch_cif_ids
            else:
                accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))
                if test:
                    test_pred = torch.exp(output.data.cpu())
                    test_target = target
                    assert test_pred.shape[1] == 2
                    test_preds += test_pred[:, 1].tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids
                else:
                    val_pred = torch.exp(output.data.cpu())
                    val_target = target
                    assert val_pred.shape[1] == 2
                    val_preds += val_pred[:, 1].tolist()
                    val_targets += val_target.view(-1).tolist()
                    val_cif_ids += batch_cif_ids
            if i % self.print_freq == 0:
                if self.task == 'regression':
                    print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader),  loss=losses,
                    mae_errors=mae_errors))
                else:
                    print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader),  loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))
        if test:
            star_label = '**'
            with open('test_results.csv', 'w') as f:
                for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                    target_ = ['%.2f' %t for t in target[:]]
                    pred_ = ['%.2f' %t for t in pred[:]]
                    T_P = target_ + pred_
                    f.write(str(cif_id)+',')
                    for i in range(len(T_P)):
                        f.write(T_P[i]+',')
                    f.write('\n')
        else:
            star_label = '*'
            with open('val_results.csv', 'w') as f:
                for cif_id, target, pred in zip(val_cif_ids, val_targets,val_preds):
                    target_ = ['%.2f' %t for t in target[:]]
                    pred_ = ['%.2f' %t for t in pred[:]]
                    T_P = target_ + pred_
                    f.write(str(cif_id)+',')
                    for i in range(len(T_P)):
                        f.write(T_P[i]+',')
                    f.write('\n')
        if self.task == 'regression':
            print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
            return mae_errors.avg
        else:
            print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
            return auc_scores.avg 

    def transfer_learning(self,modelpath : str = None,fix_layer_lr: float = 0.0005,flex_layer_lr : float = 0.002):
        if self.task == 'classification':
            normalizer = Normalizer(torch.zeros(2))
            normalizer.load_state_dict({'mean': 0., 'std': 1.})
        else:
            if len(self.trainset) < 4000:
                warnings.warn('Dataset has less than 4000 data points. '
                                        'Lower accuracy is expected. ')
                sample_data_list = [self.trainset[i] for i in range(len(self.trainset))]
            else:
                sample_data_list = [self.trainset[i] for i in sample(range(len(self.trainset)), 4000)]
            _, sample_target, _ = collate_pool(sample_data_list)
            normalizer = Normalizer(sample_target)
        if self.task == 'regression':
            best_mae_error = 1e10
        else:
            best_mae_error = 0.
        structures, targets, _ = self.trainset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        self.n_p = len(targets)
        print("Predicting ", self.n_p, " properties!!")
        ### load Frame of graph
        if os.path.isfile(modelpath):
            print("=> loading model params '{}'".format(modelpath))
            model_checkpoint = torch.load(modelpath,
                                  map_location=lambda storage, loc: storage)
            print("=> loaded model params '{}'".format(modelpath))
        else:
            print("=> no model params found at '{}'".format(modelpath))
    # reduction graph
        if orig_atom_fea_len == model_checkpoint['orig_atom_fea_len']:
            print('test loading orig_atom_fea_len',model_checkpoint['orig_atom_fea_len'])
        if nbr_fea_len == model_checkpoint['nbr_fea_len']:
            print('test loading nbr_fea_len',model_checkpoint['nbr_fea_len'])
        print('loading transfer_learning model atom_fea_len',model_checkpoint['atom_fea_len'])
        print('loading transfer_learning model n_conv',model_checkpoint['n_conv'])
        print('loading transfer_learning model h_fea_len',model_checkpoint['h_fea_len'])
        model = CrystalGraphConvNet(orig_atom_fea_len=model_checkpoint['orig_atom_fea_len'],
                             nbr_fea_len=model_checkpoint['nbr_fea_len'],
                                atom_fea_len=model_checkpoint['atom_fea_len'],
                                n_conv=model_checkpoint['n_conv'],
                                h_fea_len=model_checkpoint['h_fea_len'],
                                n_p = model_checkpoint['n_p'],
                                classification=True if model_checkpoint['task'] ==
                                'classification' else False,dropout=self.dropout)

        ###load Parameters of graph
        if os.path.isfile(modelpath):
            print("=> loading model '{}'".format(modelpath))
            checkpoint = torch.load(modelpath,
                                 map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded model '{}' (epoch {}, validation {})"
              .format(modelpath, checkpoint['epoch'],
                      checkpoint['best_mae_error']))
        else:
            print("=> no model found at '{}'".format(modelpath))
    ##transfer-learning
        if self.task == 'classification':
            criterion = nn.NLLLoss()
        else:
            criterion = nn.MSELoss() 
        ignored_params = list(map(id, model.fc_out.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer = optim.Adam([{'params': base_params, 'lr': fix_layer_lr},{'params': model.fc_out.parameters(), 'lr': flex_layer_lr}],weight_decay = self.weight_decay)
        #for param in model.parameters():
        #    param.requires_grad =  False
        if self.cuda:
            model.cuda()
        scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones,
                            gamma=0.1)
        for epoch in range(self.epoch):
        # train for one epoch
            self.train_model(self.train_loader, model, criterion, optimizer, epoch, normalizer)
        # evaluate on validation set
            mae_error = self.validate(self.val_loader, model, criterion, normalizer)
            if mae_error != mae_error:
                print('Exit due to NaN')
                sys.exit(1)
            scheduler.step()
        # remember the best mae_eror and save checkpoint
            if self.task == 'regression':
                is_best = mae_error < best_mae_error
                best_mae_error = min(mae_error, best_mae_error)
            else:
                is_best = mae_error > best_mae_error
                best_mae_error = max(mae_error, best_mae_error)
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'orig_atom_fea_len':orig_atom_fea_len,
            'nbr_fea_len':nbr_fea_len,
            'atom_fea_len': self.atom_fea_len,
            'n_conv':self.n_conv,
            'h_fea_len':self.h_fea_len,
            'n_p': self.n_p,
            'task':self.task}, is_best)
    # test best model
        print('---------Evaluate Model on Test Set---------------')
        best_checkpoint = torch.load('model_best.pth.tar')
        model.load_state_dict(best_checkpoint['state_dict'])
        self.validate(self.test_loader, model, criterion, normalizer, test=True)
    def pred_fun(self,val_loader, model, normalizer):
        test_targets = []
        test_preds = []
        test_cif_ids = []
    # switch to evaluate mode
        model.eval()
        for i, (input,target, batch_cif_ids) in enumerate(val_loader):
            if self.cuda:
                with torch.no_grad():
                    input_var = (Variable(input[0].cuda()),
                             Variable(input[1].cuda()),
                             input[2].cuda(),
                             input[3].cuda(),
                             [crys_idx.cuda() for crys_idx in input[4]],
                             Variable(input[5].cuda()))
            else:
                with torch.no_grad():
                    input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3],
                             input[4],
                             Variable(input[5]))
        # compute output
            output = model(*input_var)
            if self.task == 'regression':
                test_pred = normalizer.denorm(output.data.cpu())
                test_preds += test_pred
                test_cif_ids += batch_cif_ids       
            else:
                test_pred = torch.exp(output.data.cpu())
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_cif_ids += batch_cif_ids
            with open('pred_results.csv', 'w') as f:
                for cif_id,  pred in zip(test_cif_ids,test_preds):
                    pred_ = ['%.2f' %t for t in pred[:]]
                    f.write(str(cif_id)+',')
                    for i in range(len(pred_)):
                        f.write(pred_[i]+',')
                    f.write('\n')


