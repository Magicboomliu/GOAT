import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np
from torchvision import transforms as torch_transforms
from torch.utils.data import DataLoader
from goat.utils.core.AverageMeter import AverageMeter
from data.dataloaders.kitti.preprocess import scale_disp
from data.dataloaders.kitti.SceneflowLoaderOcc import StereoDatasetOcc
from goat.utils.core.metric import P1_metric, P1_Value, D1_metric, Disparity_EPE_Loss
from data.dataloaders.kitti import transforms

from goat.losses.modules.disparity_sequence_loss import sequence_lossV2
import os
# select the networks
from goat.utils.core.common import logger
from goat.models.networks.Methods.GOAT_T import GOAT_T
# metric
from goat.utils.core.metric import compute_iou, Occlusion_EPE
from goat.utils.core.visual import save_images, disp_error_img
import time

# IMAGENET NORMALIZATION
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



class DisparityTrainer(object):
    def __init__(self,args):
        super(DisparityTrainer, self).__init__()
        
        self.lr = args.lr
        self.datathread = args.datathread
        self.initial_pretrain = args.initial_pretrain
        self.current_lr =args.lr
        self.devices = args.devices
        self.devices = [int(item) for item in self.devices.split(',')]
        ngpu = len(self.devices)
        self.ngpu = ngpu
        self.local_rank = args.local_rank
        self.trainlist = args.trainlist
        self.vallist = args.vallist
        self.dataset = args.dataset
        self.datapath = args.datapath
        self.batch_size = args.batch_size
        self.test_batch = args.test_batch
        self.pretrain = args.pretrain 
        self.maxdisp = args.maxdisp
        self.use_deform= args.use_deform
        self.criterion = None
        self.epe = Disparity_EPE_Loss
        self.p1_error = P1_metric
        self.model = args.model
        self.initialize()
    
    # Get Dataset Here
    def _prepare_dataset(self):
        datathread = self.datathread
        if os.environ.get('datathread') is not None:
            datathread = int(os.environ.get('datathread'))
        
        if self.local_rank == 0:
            logger.info("Use %d processes to load data..." % datathread)
            
        if self.dataset == 'sceneflow':
            train_transform_list = [transforms.RandomCrop(320, 640),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]
            train_transform = transforms.Compose(train_transform_list)

            val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                         ]
            val_transform = transforms.Compose(val_transform_list)
            
            train_dataset = StereoDatasetOcc(data_dir=self.datapath,train_datalist=self.trainlist,test_datalist=self.vallist,
                                    dataset_name='SceneFlow',mode='train',transform=train_transform)
            test_dataset = StereoDatasetOcc(data_dir=self.datapath,train_datalist=self.trainlist,test_datalist=self.vallist,
                                    dataset_name='SceneFlow',mode='val',transform=val_transform)

        self.img_height, self.img_width = train_dataset.get_img_size()

        self.scale_height, self.scale_width = test_dataset.get_scale_size()

        # define the train sampler for distributed training
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        self.test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, \
                                        pin_memory=True, num_workers=datathread, sampler=self.train_sampler)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, \
                                        pin_memory=True, num_workers=datathread, sampler=self.test_sampler)
        self.num_batches_per_epoch = len(self.train_loader)
        

    def _build_net(self):
        # Build the Network architecture according to the model name
        if self.model=="GOAT_T_Origin":
            self.net = GOAT_T(radius=3,num_levels=4,sample_points=4,dropout=0,refine_type='g',up_scale=3)
        else:
            raise NotImplementedError
        
        self.is_pretrain = False
        # loaded the model by distributed model
        device = torch.device("cuda", self.local_rank)
        self.net.cuda(device)
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[self.local_rank],find_unused_parameters=True)
        if self.local_rank==0:
            logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.net.parameters()])))


        if self.pretrain == 'none':
            if self.local_rank==0:
                logger.info('Initial a new model...')
            if self.initial_pretrain !='none':
                pretrain_ckpt = self.initial_pretrain
                ckpt = torch.load(pretrain_ckpt)
                current_model_dict = self.net.state_dict()
                useful_dict ={k:v for k,v in ckpt['state_dict'].items() if k in current_model_dict.keys()}
                current_model_dict.update(useful_dict)
                self.net.load_state_dict(current_model_dict)
        else:
            if os.path.isfile(self.pretrain):
                model_data = torch.load(self.pretrain)
                logger.info('Load pretrain model: %s', self.pretrain)
                if 'state_dict' in model_data.keys():
                    self.net.load_state_dict(model_data['state_dict'])
                else:
                    self.net.load_state_dict(model_data)
                self.is_pretrain = True
            else:
                logger.warning('Can not find the specific model %s, initial a new model...', self.pretrain)

    def _build_optimizer(self):
        beta = 0.999
        momentum = 0.9
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), self.lr,
                                        betas=(momentum, beta), amsgrad=True)

    def initialize(self):
        # Specific the backen
        dist.init_process_group('nccl',world_size=self.ngpu,rank=self.local_rank)
        # distrubute the GPU: equals CUDA_VISIBLE_DEVICES
        torch.cuda.set_device(self.local_rank)
        if self.local_rank==0:
            logger.info(">> Training with distributed parallel.............")
            
        self._prepare_dataset()
        self._build_net()
        self._build_optimizer()

    def adjust_learning_rate(self, epoch):
        if epoch>=0 and epoch<=10:
            cur_lr = 3e-4
        elif epoch > 10 and epoch<45:
            cur_lr = 1e-4
        elif epoch>=40 and epoch<50:
            cur_lr = 5e-5
        elif epoch>=50 and epoch<60:
            cur_lr = 3e-5
        elif epoch>=60:
            cur_lr =1.5e-5
        else:
            cur_lr = self.lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
        self.current_lr = cur_lr
        return cur_lr

    def set_criterion(self, criterion):
        self.criterion = criterion


    def train_one_epoch(self, epoch, round,iterations,summary_writer,local_rank):
        
        # Data Summary
        batch_time = AverageMeter()
        data_time = AverageMeter()    
        losses = AverageMeter()
        flow2_EPEs = AverageMeter()
        
        disp_loss_meter = AverageMeter()
        occlusion_loss_meter = AverageMeter()
        occlusion_epe_meter = AverageMeter()
        occlusion_mIOU_meter = AverageMeter()

        
        # switch to train mode
        self.net.train()
        end = time.time()
        cur_lr = self.adjust_learning_rate(epoch)
        if local_rank==0:
            logger.info("learning rate of epoch %d: %f." % (epoch, cur_lr))
        summary_writer.add_scalar("Learning_Rate",cur_lr,epoch+1)
        
        # each card get the random data
        self.train_sampler.set_epoch(epoch)
        
        for i_batch, sample_batched in enumerate(self.train_loader):
            left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(local_rank), requires_grad=False)
            right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(local_rank), requires_grad=False)

            # Here the Traget disparity is [320*640]
            target_disp = sample_batched['gt_disp'].unsqueeze(1)
            target_disp = target_disp.cuda(local_rank)
            target_disp = torch.autograd.Variable(target_disp, requires_grad=False)

            target_occ = sample_batched['occu_left'].unsqueeze(1)
            target_occ = target_occ.cuda(local_rank)
            target_occ = torch.autograd.Variable(target_occ,requires_grad=False) 

            data_time.update(time.time() - end)
            self.optimizer.zero_grad()
            show_multi_loss =False
            occlusion_pred_or_not =False

            
            if self.model=='GOAT_T_Origin':
                outputs,pred_occlusion =  self.net(left_input, right_input,iters=12)
                loss, disp_loss, occlusion_loss = sequence_lossV2(outputs,target_disp,pred_occlusion,
                                                              target_occ)
                show_multi_loss=True
                occlusion_pred_or_not=True


            # Get the higher resolution
            output = outputs[-1]
            # Loss Meterization
            if type(loss) is list or type(loss) is tuple:
                loss = np.sum(loss)
            if type(output) is list or type(output) is tuple: 
                flow2_EPE = self.epe(output[-1], target_disp)    
            else:
                if output.size(-1)!= target_disp.size(-1):
                    output = F.interpolate(output,scale_factor=8.0,mode='bilinear',align_corners=False) * 8.0
                assert (output.size(-1) == target_disp.size(-1))
                flow2_EPE = self.epe(output, target_disp)

            if show_multi_loss and occlusion_pred_or_not:
                # Compute the occlusion EPE
                occlusion_epe = Occlusion_EPE(predicted_occlusion=pred_occlusion,target_occlusion=target_occ,
                                          disp_gt=target_disp)
                # Compute the occlusion mIOU
                occlusion_mIOU = compute_iou(pred=pred_occlusion,occ_mask=target_occ,target_disp=target_disp)
                
                disp_loss_meter.update(disp_loss.data.item(),left_input.size(0))
                occlusion_loss_meter.update(occlusion_loss.data.item(),left_input.size(0))
                occlusion_epe_meter.update(occlusion_epe.data.item(),left_input.size(0))
                occlusion_mIOU_meter.update(occlusion_mIOU.data.item(),left_input.size(0))
                summary_writer.add_scalar("disp_sequence_loss",disp_loss_meter.val,iterations+1)
                summary_writer.add_scalar("occlusion_loss",occlusion_loss_meter.val,iterations+1)
                summary_writer.add_scalar("occlusion_mIOU_on_train",occlusion_mIOU_meter.val,iterations+1)
                summary_writer.add_scalar("occlusion_epe_on_train",occlusion_epe_meter.val,iterations+1)
                
                
            # Record loss and EPE in the tfboard
            losses.update(loss.data.item(), left_input.size(0))
            flow2_EPEs.update(flow2_EPE.data.item(), left_input.size(0))
            summary_writer.add_scalar("total_loss",losses.val,iterations+1)
            summary_writer.add_scalar("disp_EPE_on_train",flow2_EPEs.val,iterations+1)


            loss.backward()
            self.optimizer.step()
            iterations = iterations+1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if self.local_rank==0:
                if occlusion_pred_or_not:
                    if i_batch % 10 == 0:
                        logger.info('this is round %d', round)
                        logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                        'Disp_Loss {disp_loss.val:.3f} ({disp_loss.avg:.3f})\t'
                        'Occ_Loss {occ_loss.val:.3f} ({occ_loss.avg:.3f})\t'
                        'Occ_mIOU {occ_mIOU.val:.3f} ({occ_mIOU.avg:.3f})\t'
                        'Disp_EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                        epoch, i_batch, self.num_batches_per_epoch, 
                        batch_time=batch_time,
                        disp_loss = disp_loss_meter, 
                        occ_loss = occlusion_loss_meter,
                        occ_mIOU = occlusion_mIOU_meter, 
                        data_time=data_time, loss=losses,flow2_EPE=flow2_EPEs))
                
                else:
                    if i_batch % 10 == 0:
                        logger.info('this is round %d', round)
                        logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                        'Disp_EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                        epoch, i_batch, self.num_batches_per_epoch, 
                        batch_time=batch_time,
                        data_time=data_time, loss=losses,flow2_EPE=flow2_EPEs))

        return losses.avg, flow2_EPEs.avg,iterations
    
    
    def validate(self,summary_writer,epoch,vis=False,local_rank=0):
        
        # print("OOOk here")
        
        batch_time = AverageMeter()
        flow2_EPEs = AverageMeter()
        P1_errors = AverageMeter()    
        losses = AverageMeter()

        occlusion_epe_meter = AverageMeter()
        occlusion_mIOU_meter = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        end = time.time()
        inference_time = 0
        img_nums = 0
        nums_samples = len(self.test_loader)
        test_count = 0
        for i, sample_batched in enumerate(self.test_loader):
            left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(local_rank), requires_grad=False)
            right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(local_rank), requires_grad=False)
            input = torch.cat((left_input, right_input), 1)
            input_var = torch.autograd.Variable(input, requires_grad=False)
            target_disp = sample_batched['gt_disp'].unsqueeze(1)
            target_disp = target_disp.cuda(local_rank)
            target_disp = torch.autograd.Variable(target_disp, requires_grad=False)

            target_occ = sample_batched['occu_left'].unsqueeze(1)
            target_occ = target_occ.cuda(local_rank)
            target_occ = torch.autograd.Variable(target_occ,requires_grad=False) 
            
            occlusion_pred_or_not =False
            with torch.no_grad():
                
                start_time = time.perf_counter()
                if self.model=="GOAT_T_Origin":
                    cur_disp,output,pred_occlusion = self.net(left_input, right_input,iters=12,test_mode=True)
                    pred_occ = scale_disp(pred_occlusion,(output.size()[0], self.img_height, self.img_width))
                    occlusion_pred_or_not = True

                
                inference_time += time.perf_counter() - start_time
                img_nums += left_input.shape[0]
                # Here Need to be Modification
                output = scale_disp(output, (output.size()[0], self.img_height, self.img_width))                
                loss = self.epe(output, target_disp)
                flow2_EPE = self.epe(output, target_disp)
                P1_error = self.p1_error(output, target_disp)
                
                if occlusion_pred_or_not:
                    occ_mIOU_data = compute_iou(pred_occ,target_occ,target_disp)
                    occ_epe_data = Occlusion_EPE(pred_occ,target_occ,target_disp)
                    occlusion_epe_meter.update(occ_epe_data.data.item(),input_var.size(0))
                    occlusion_mIOU_meter.update(occ_mIOU_data.data.item(),input_var.size(0))


            if loss.data.item() == loss.data.item():
                losses.update(loss.data.item(), input_var.size(0))
            if flow2_EPE.data.item() == flow2_EPE.data.item():
                flow2_EPEs.update(flow2_EPE.data.item(), input_var.size(0))
            if P1_error.data.item() == P1_error.data.item():
                P1_errors.update(P1_error.data.item(), input_var.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if self.local_rank==0:
                if i % 10 == 0:
                    logger.info('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                          .format(i, len(self.test_loader), batch_time.val, flow2_EPEs.val))
        
        if self.local_rank==0:
            logger.info(' * DISP EPE {:.3f}'.format(flow2_EPEs.avg))
            logger.info(' * P1_error {:.3f}'.format(P1_errors.avg))
            
            if occlusion_pred_or_not:
                logger.info(" * Occlusion EPE {:.3f}".format(occlusion_epe_meter.avg))
                logger.info(" * Occlusion mIOU {:.3f}".format(occlusion_mIOU_meter.avg))
            
            logger.info(' * avg inference time {:.3f}'.format(inference_time / img_nums))
            
        return flow2_EPEs.avg


    def get_model(self):
        return self.net.state_dict()