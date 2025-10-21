from __future__ import print_function
import os
import sys
import argparse
import datetime
import random
import torch
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path to import goat package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from goat.utils.core.common import *
from goat.trainer import DisparityTrainer

cudnn.benchmark = True

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, os.path.join(opt.outf,filename))
    if is_best:
        torch.save(state, os.path.join(opt.outf,'model_best.pth'))

'''  Main Function to train the model'''
def main(opt):
    
    # distributed training.
    
    # load the training loss scheme
    loss_json = load_loss_scheme(opt.loss)
    train_round = loss_json["round"]
    loss_scale = loss_json["loss_scale"]
    loss_weights = loss_json["loss_weights"]
    epoches = loss_json["epoches"]
    if opt.local_rank==0:
        logger.info(loss_weights)
    
    # initialize a trainer
    trainer = DisparityTrainer(opt)
    
    # validate the pretrained model on test data
    best_EPE = -1
    best_index = 0
    summary_writer = SummaryWriter(opt.save_logdir)
    start_epoch = opt.startEpoch 

    if trainer.is_pretrain:
        pass
        # best_EPE = trainer.validate(summary_writer=summary_writer,epoch=start_epoch)

    iterations = 0
    
    for r in range(opt.startRound, train_round):
        
        # Set your loss here
        criterion = None
        trainer.set_criterion(criterion)
        end_epoch = epoches[r]

        if opt.local_rank==0:
            logger.info('round %d: %s' % (r, str(loss_weights[r])))
            logger.info('num of epoches: %d' % end_epoch)
            logger.info('\t'.join(['epoch', 'time_stamp', 'train_loss', 'train_EPE', 'EPE', 'lr']))
        
        for i in range(start_epoch, end_epoch):
            # val_EPE = trainer.validate(summary_writer,i,local_rank=opt.local_rank)
            avg_loss, avg_EPE,iterations = trainer.train_one_epoch(i, r,iterations,summary_writer,opt.local_rank)
            val_EPE = trainer.validate(summary_writer,i,local_rank=opt.local_rank)
            is_best = best_EPE < 0 or val_EPE < best_EPE

            if is_best:
                best_EPE = val_EPE
                best_index = i
            
            # only save at the GPU 0
            if opt.local_rank==0:
                save_checkpoint({
                    'round': r + 1,
                    'epoch': i + 1,
                    'arch': 'dispnet',
                    'state_dict': trainer.get_model(),
                    'best_EPE': best_EPE,
                    }, is_best, '%s_%d_%d_%.3f.pth' % (opt.net, r, i, val_EPE))
                
            if opt.local_rank==0:
                logger.info('Validation[epoch:%d]: '%i+'\t'.join([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(avg_loss), str(avg_EPE), str(val_EPE), str(trainer.current_lr)]))
                logger.info("min epe from %d epoch" % (best_index))
        
        start_epoch = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicates the name of net', default='simplenet')
    parser.add_argument('--loss', type=str, help='indicates the loss scheme', default='simplenet_flying')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--test_batch', type=int, default=4, help='test batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd, alpha parameter for adam. default=0.9')
    parser.add_argument('--beta', type=float, default=0.999, help='beta parameter for adam. default=0.999')
    parser.add_argument('--cuda', action='store_true', help='enables, cuda')
    parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--model', type=str, help='model for finetuning', default='')
    parser.add_argument('--startRound', type=int, help='the round number to start training, useful of lr scheduler', default='0')
    parser.add_argument('--startEpoch', type=int, help='the epoch number to start training, useful of lr scheduler', default='0')
    parser.add_argument('--logFile', type=str, help='logging file', default='./train.log')
    parser.add_argument('--showFreq', type=int, help='display frequency', default='100')
    parser.add_argument('--flowDiv', type=float, help='the number by which the flow is divided.', default='1.0')
    parser.add_argument('--maxdisp', type=int, help='disparity search range.', default='-1')
    parser.add_argument('--dataset', type=str, help='provide the dataset name', default='sceneflow')
    parser.add_argument('--datapath', type=str, help='provide the root path of the data', default='/spyder/sceneflow/')
    parser.add_argument('--trainlist', type=str, help='provide the train file (with file list)', default='FlyingThings3D_release_TRAIN.list')
    parser.add_argument('--vallist', type=str, help='provide the val file (with file list)', default='FlyingThings3D_release_TEST.list')
    parser.add_argument('--augment', type=int, help='if augment data in training', default=0)
    parser.add_argument('--trainmodel', type=str, help='denote the specific model to train')
    parser.add_argument('--use_deform',action='store_true',help='use deform conv or not in features extraction period')
    parser.add_argument('--save_logdir',type=str,help='tensorboard log files saved path',default='experiments_logdirs')
    parser.add_argument('--pretrain',type=str,help='Load pretrain model for fine-tuning',default='None')
    parser.add_argument('--initial_pretrain',type=str,help='Load Part of the weight',default='None')
    parser.add_argument("--local_rank",help='local device id on current node',type=int)
    parser.add_argument("--datathread",help='numbers of workers in dataloader',type=int)
    
    opt = parser.parse_args()
    print("Use Deformable Conv ? :",opt.use_deform)
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    hdlr = logging.FileHandler(opt.logFile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    if opt.local_rank==0:
        logger.info('Configurations: %s', opt)
    
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    if opt.local_rank==0:
        logger.info("Random Seed: %s", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    if torch.cuda.is_available() and not opt.cuda:
        logger.warning("WARNING: You should run with --cuda since you have a CUDA device.")
    main(opt)