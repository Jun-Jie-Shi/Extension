#coding=utf-8
import argparse
import copy
import csv
import logging
import os
import random
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from data.data_utils import init_fn
from data.datasets_nii import (Brats_loadall_metaval_nii_idt,
                               Brats_loadall_test_nii,
                               Brats_loadall_train_nii_idt,
                               Brats_loadall_train_nii_pdt,
                               Brats_loadall_val_nii)
from data.transforms import *
from models import m2ftrans, mmformer, rfnet
from models.ensemble import ensemble_inc
# import torch.optim
from optim.gmd import GMD
from options import args_parser
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import criterions
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from utils.parser import set_seed, setup
from utils.predict import AverageMeter, test_dice_hd95_softmax

## parse arguments
args = args_parser()
## training setup
setup(args, 'training')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
## checkpoints saving path
ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks_test))
mask_name = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

# masks_valid = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
#          [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
#          [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
#          [True, True, True, True]]
# masks_valid_torch = torch.from_numpy(np.array(masks_valid))
# masks_valid_array = np.array(masks_valid)
masks_all = [True, True, True, True]
masks_all_torch = torch.from_numpy(np.array(masks_all))

# mask_name_valid = ['t2', 't1c', 't1', 'flair',
#             't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
#             'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
#             'flairt1cet1t2']
# mask_name_single = ['flair', 't1c', 't1', 't2']
# print (masks_valid_torch.int())

def main():
    ##########setting seed
    set_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BraTS/BRATS2021', 'BraTS/BRATS2020', 'BraTS/BRATS2018']:
        num_cls = 4
    else:
        print ('dataset is error')
        exit(0)

    if args.model == 'mmformer':
        model = mmformer.Model(num_cls=num_cls)
    elif args.model == 'rfnet':
        model = rfnet.Model(num_cls=num_cls)
    elif args.model == 'm2ftrans':
        model = m2ftrans.Model(num_cls=num_cls)
    elif args.model == 'ensemble':
        model = ensemble_inc.Ensemble(in_channels=4, out_channels=num_cls, width_ratio=0.5)

    print (model)
    model = torch.nn.DataParallel(model).cuda()
    # model.module.mask_type = args.mask_type
    # model.module.use_passion = args.use_passion
    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    pc_adam = GMD(optimizer, reduction='mean', writer=writer)
    # params_enc = []
    # params_sep = []
    # params_fuse = []
    # for name, param in model.named_parameters():
    #     if ('_encoder' in name) or ('decoder_sep' in name):
    #         params_enc.append({'params': [param], 'lr': args.lr, 'weight_decay':args.weight_decay})
    #         # record_names_enc.append((name, param))
    #         continue
    #     # elif ('decoder_sep' in name):
    #     #     params_sep.append({'params': [param], 'lr': args.lr, 'weight_decay':args.weight_decay})
    #     #     continue
    #     else:
    #         params_fuse.append({'params': [param], 'lr': args.lr, 'weight_decay':args.weight_decay})
    #         continue
    # opt_enc = torch.optim.AdamW(params_enc,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    # opt_sep = torch.optim.AdamW(params_sep,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    # optimizer = torch.optim.AdamW(params_fuse,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    # pc_opt_enc = GMD(opt_enc, reduction='sum', writer=writer)
    # pc_opt_sep = GMD(opt_sep, reduction='sum', writer=writer)
    # pc_opt_fuse = GMD(opt_enc, reduction='sum', writer=writer)

    temp = args.temp

    ##########Setting data
        ####BRATS2020
    if args.dataname == 'BraTS/BRATS2020':
        train_file = os.path.join(args.datarootPath, args.imbmrpath)
        test_file = os.path.join(args.datasetPath, 'test.txt')
        # valid_file = os.path.join(args.datasetPath, 'val.txt')
    #### Other Datasets Setting (Like BraTS2020)
    # elif args.dataname == 'BraTS/BRATS2018':
    #     ####BRATS2018 contains three splits (1,2,3)
    #     train_file = os.path.join(args.datarootPath, 'BraTS/brats_split/Brats2018_imb_split_mr2468.csv')
    #     test_file = os.path.join(args.datasetPath, 'test1.txt')
    #   # valid_file = os.path.join(args.datasetPath, 'val1.txt')
    # elif args.dataname == 'BraTS/BRATS2021':
    #     ####BRATS2021

    logging.info(str(args))
    set_seed(args.seed)
    # if args.mask_type in ['pdt', 'idt', 'idt_drop']:
    #     train_set = Brats_loadall_train_nii_idt(transforms=args.train_transforms, root=args.datasetPath, num_cls=num_cls, mask_type=args.mask_type, train_file=train_file)
    # else:
    #     print ('training setting is error')
    #     exit(0)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datasetPath, test_file=test_file)
    # valid_set = Brats_loadall_val_nii(transforms=args.train_transforms, root=args.datasetPath, num_cls=num_cls, train_file=valid_file)
    meta_set = [Brats_loadall_metaval_nii_idt(transforms=args.train_transforms, root=args.datasetPath, mask_value=i, num_cls=num_cls, train_file=train_file) for i in range(15)]
    meta_loader = [MultiEpochsDataLoader(
        dataset=meta_set[i],
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn) for i in range(15)]
    # train_loader = MultiEpochsDataLoader(
    #     dataset=train_set,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     pin_memory=True,
    #     shuffle=True,
    #     worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    # valid_loader = MultiEpochsDataLoader(
    #     dataset=valid_set,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     pin_memory=True,
    #     shuffle=True,
    #     worker_init_fn=init_fn)

    #### Whether use pretrained model
    if args.resume is not None and args.use_pretrain:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # logging.info('pretrained_dict: {}'.format(pretrained_dict))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info('load ok')


    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)

    meta_iters = [iter(meta_loader[i]) for i in range(15)]
    meta_list = list(range(15))
    # np.random.shuffle(meta_list)
    meta_list_iter = iter(meta_list)

    iter_per_epoch = args.iter_per_epoch

    # iter_per_epoch = len(train_loader)
    # train_iter = iter(train_loader)

    ### IDT Init-Imb-Weight Setting
    imb_mr_csv_data = pd.read_csv(train_file)
    modal_num = torch.tensor((0,0,0,0), requires_grad=False).cuda().float()
    for sample_mask in imb_mr_csv_data['mask']:
        modal_num += torch.tensor(eval(sample_mask), requires_grad=False).cuda().float()
    logging.info('Training Imperfect Datasets with Mod.Flair-{:d}, Mod.T1c-{:d}, Mod.T1-{:d}, Mod.T2-{:d}'\
    .format(int(modal_num[0].item()), int(modal_num[1].item()), int(modal_num[2].item()), int(modal_num[3].item())))

    modal_weight = torch.tensor((1,1,1,1), requires_grad=False).cuda().float()
    # modal_weight = (iter_per_epoch/modal_num).cuda().float()
    # valid_iter = iter(valid_loader)
    imb_beta = torch.tensor((1,1,1,1), requires_grad=False).cuda().float()
    eta = 0.01
    eta_ext = 1.5
    phi_tilde = [p.clone() for p in model.parameters()]
    epsilon = 0.1
    if args.use_passion:
        if args.mask_type == 'pdt':
            logging.info('#############PASSION-PDT-Training############')
        elif args.mask_type == 'idt':
            logging.info('#############PASSION-IDT-Training############')
        for epoch in range(args.num_epochs):
            step_lr = lr_schedule(optimizer, epoch)
            epsilon = 0.1 + 0.9 * (epoch / args.num_epochs)
            writer.add_scalar('lr', step_lr, global_step=(epoch+1))
            epoch_fuse_losses = torch.zeros(1).cpu().float()
            # epoch_sep_losses = torch.zeros(1).cpu().float()
            # epoch_prm_losses = torch.zeros(1).cpu().float()
            # epoch_kl_losses = torch.zeros(1).cpu().float()
            # epoch_proto_losses = torch.zeros(1).cpu().float()
            # epoch_dist_losses = torch.zeros(1).cpu().float()
            epoch_losses = torch.zeros(1).cpu().float()
            epoch_sep_m = torch.zeros(4).cpu().float()
            # epoch_kl_m = torch.zeros(4).cpu().float()
            # epoch_proto_m = torch.zeros(4).cpu().float()
            # epoch_dist_m = torch.zeros(4).cpu().float()

            b = time.time()
            for i in range(iter_per_epoch):
                step = (i+1) + epoch*iter_per_epoch
                ###Data load
                try:
                    meta_mask_value = next(meta_list_iter)
                except:
                    np.random.shuffle(meta_list)
                    meta_list_iter = iter(meta_list)
                    meta_mask_value = next(meta_list_iter)

                    with torch.no_grad():
                        pp = [pp for pp in model.parameters()]
                        for p, g in zip(model.parameters(), phi_tilde):
                            p.data.copy_(g + epsilon * (p - g))
                        phi_tilde = [p.clone() for p in model.parameters()]

                try:
                    data = next(meta_iters[meta_mask_value])
                except:
                    meta_iters[meta_mask_value] = iter(meta_loader[meta_mask_value])
                    data = next(meta_iters[meta_mask_value])
                x, target, mask, name = data
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                mask = mask.cuda(non_blocking=True)

                model.module.is_training = True
                model.train()

                # kl_loss_m = torch.zeros(4).cuda().float()
                sep_loss_m = torch.zeros(4).cuda().float()
                # proto_loss_m = torch.zeros(4).cuda().float()
                # dist_m = torch.zeros(4).cuda().float()
                # prm_loss = torch.zeros(1).cuda().float()
                # fuse_loss = torch.zeros(1).cuda().float()
                # rp_iter = torch.zeros(4).cuda().float()

                # fuse_pred, prm_loss_bs, sep_loss_m_bs, kl_loss_m_bs, proto_loss_m_bs, dist_m_bs = model(x, mask, target=target, temp=temp)
                out_bs, de_fe_bs = model(x, mask, target=target)
                losses = []
                losses_summary = []
                # for out_bs_ava in out_bs:
                #     losses.append(criterions.DiceCoef(out_bs_ava, target, num_cls=num_cls))
                # cnt = 0
                for modc in range(4):
                    if mask[0,modc]:
                        losses.append(criterions.DiceCoef(out_bs[modc], target, num_cls=num_cls))
                        losses_summary.append(criterions.DiceCoef(out_bs[modc], target, num_cls=num_cls))
                        # cnt += 1
                    else:
                        losses_summary.append(torch.tensor(0).cuda().float())
                losses.append(criterions.DiceCoef(out_bs[4], target, num_cls=num_cls))
                losses_summary.append(criterions.DiceCoef(out_bs[4], target, num_cls=num_cls))
                # cnt += 1
                # logging.info(losses_summary)
                epoch_summary = torch.stack(losses_summary, dim=0)
                loss = torch.mean(epoch_summary)
                # losses /= cnt

                pc_adam.zero_grad()
                pc_adam.pc_backward(losses)
                pc_adam.step()


                epoch_losses += (loss/iter_per_epoch).detach().cpu()
                epoch_sep_m += (epoch_summary[0:4]/iter_per_epoch).detach().cpu()
                epoch_fuse_losses += (epoch_summary[4]/iter_per_epoch).detach().cpu()

                msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
                msg += 'fuse_loss:{:.4f}, '.format(epoch_summary[4].item())
                # msg += 'sep_loss:{:.4f}, '.format(sep_loss.item())
                # msg += 'kl_loss:{:.4f}, proto_loss:{:.4f},'.format(kl_loss.item(), proto_loss.item())
                msg += 'seplist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(epoch_summary[0].item(), epoch_summary[1].item(), epoch_summary[2].item(), epoch_summary[3].item())
                # msg += 'kllist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(kl_loss_m[0].item(), kl_loss_m[1].item(), kl_loss_m[2].item(), kl_loss_m[3].item())
                # msg += 'protolist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(proto_loss_m[0].item(), proto_loss_m[1].item(), proto_loss_m[2].item(), proto_loss_m[3].item())
                # msg += 'distlist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(dist_m[0].item(), dist_m[1].item(), dist_m[2].item(), dist_m[3].item())
                # for bs_n in range(x.size(0)):
                #     msg += '{:>20}, '.format(name[bs_n])
                # msg += 'rp_iter[{:.2f},{:.2f},{:.2f},{:.2f}]'.format(rp_iter[0].item(), rp_iter[1].item(), rp_iter[2].item(), rp_iter[3].item())
                logging.info(msg)
            b_train = time.time()
            logging.info('train time per epoch: {}'.format(b_train - b))

            # epoch_dist_avg = (sum(epoch_dist_m)/4.0).cpu().float()
            # # rp_epoch = ((epoch_dist_avg - epoch_dist_m) / epoch_dist_avg)
            # rp_epoch = ((epoch_dist_m - epoch_dist_avg) / epoch_dist_avg)
            # if epoch < args.region_fusion_start_epoch:
            #     imb_beta = imb_beta.cuda()
            # else:
            #     if epoch % 100 == 0:
            #         eta = eta * eta_ext
            #     imb_beta = imb_beta.cpu() - eta * rp_epoch
            #     imb_beta = torch.clamp(imb_beta, min=0.1, max=4.0)
            #     imb_beta = 2 * imb_beta / (sum(imb_beta**2)**(0.5))
            #     imb_beta = imb_beta.cuda()


            # logging.info('rp_epoch:[{:.4f},{:.4f},{:.4f},{:.4f}]'.format(rp_epoch[0].item(), rp_epoch[1].item(), rp_epoch[2].item(), rp_epoch[3].item()))
            # logging.info('imb_beta:[{:.4f},{:.4f},{:.4f},{:.4f}]'.format(imb_beta[0].item(), imb_beta[1].item(), imb_beta[2].item(), imb_beta[3].item()))


            writer.add_scalar('epoch_losses', epoch_losses.item(), global_step=(epoch+1))
            writer.add_scalar('epoch_fuse_losses', epoch_fuse_losses.item(), global_step=(epoch+1))
            # writer.add_scalar('epoch_prm_losses', epoch_prm_losses.item(), global_step=(epoch+1))
            # writer.add_scalar('epoch_sep_losses', epoch_sep_losses.item(), global_step=(epoch+1))
            # writer.add_scalar('epoch_kl_losses', epoch_kl_losses.item(), global_step=(epoch+1))
            # # writer.add_scalar('epoch_dist_losses', epoch_dist_losses.item(), global_step=(epoch+1))
            # writer.add_scalar('epoch_proto_losses', epoch_proto_losses.item(), global_step=(epoch+1))
            for m in range(4):
                # writer.add_scalar('kl_m{}'.format(m), epoch_kl_m[m].item(), global_step=(epoch+1))
                writer.add_scalar('sep_m{}'.format(m), epoch_sep_m[m].item(), global_step=(epoch+1))
                # writer.add_scalar('proto_m{}'.format(m), epoch_proto_m[m].item(), global_step=(epoch+1))
                # writer.add_scalar('dist_m{}'.format(m), epoch_dist_m[m].item(), global_step=(epoch+1))
                # writer.add_scalar('rp_m{}'.format(m), rp_epoch[m].item(), global_step=(epoch+1))


            #########model save
            file_name = os.path.join(ckpts, 'model_last.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)

            if (epoch+1) % 100 == 0 or (epoch>=(args.num_epochs-5)):
                file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    },
                    file_name)
    else:
        if args.mask_type == 'pdt':
            logging.info('#############NO-PASSION-PDT-Training############')
        else:
            logging.info('#############NO-PASSION-IDT-Training############')
        for epoch in range(args.num_epochs):
            step_lr = lr_schedule(optimizer, epoch)
            writer.add_scalar('lr', step_lr, global_step=(epoch+1))

            epoch_fuse_losses = torch.zeros(1).cpu().float()
            epoch_sep_losses = torch.zeros(1).cpu().float()
            epoch_prm_losses = torch.zeros(1).cpu().float()
            epoch_losses = torch.zeros(1).cpu().float()
            epoch_sep_m = torch.zeros(4).cpu().float()

            b = time.time()
            for i in range(iter_per_epoch):
                step = (i+1) + epoch*iter_per_epoch
                ###Data load
                try:
                    data = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    data = next(train_iter)
                x, target, mask, name = data
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                mask = mask.cuda(non_blocking=True)

                model.module.is_training = True

                sep_loss_m = torch.zeros(4).cuda().float()
                prm_loss = torch.zeros(1).cuda().float()
                fuse_loss = torch.zeros(1).cuda().float()

                fuse_pred, prm_loss_bs, sep_loss_m_bs = model(x, mask, target=target, temp=temp)

                ###Loss compute
                fuse_loss_bs = criterions.softmax_weighted_loss_bs(fuse_pred, target, num_cls=num_cls) + criterions.dice_loss_bs(fuse_pred, target, num_cls=num_cls)
                fuse_loss = torch.sum(fuse_loss_bs)

                prm_loss = torch.sum(prm_loss_bs)

                if args.mask_type == 'pdt':
                    sep_loss_m = torch.sum(sep_loss_m_bs, dim=0)
                    sep_loss = sep_loss_m.sum()
                    ## warmup with shared sep-decoder like rfnet
                    if epoch < args.region_fusion_start_epoch:                        
                        loss = fuse_loss * 0.0 + sep_loss + prm_loss * 0.0
                    else:
                        loss = fuse_loss + sep_loss + prm_loss

                    # ## without warmup and without shared sep-decoder
                    # sep_loss = (rp_mask * imb_beta * sep_loss_m).sum()
                    # loss = fuse_loss + sep_loss * 0.0 + prm_loss + kl_loss * 0.5 + proto_loss * 0.1
                else: ### idt or idt_moddrop
                    # masks_sum = torch.clamp(torch.sum(mask, dim=0).float(), min=0.005, max=args.batch_size)
                    sep_loss_m = torch.sum(sep_loss_m_bs*mask, dim=0)
                    sep_loss = sep_loss_m.sum()
                    # sep_loss = (modal_weight * sep_loss_m).sum()

                    ## warmup with shared sep-decoder like rfnet
                    if epoch < args.region_fusion_start_epoch:
                        loss = fuse_loss * 0.0 + sep_loss + prm_loss * 0.0
                    else:
                        loss = fuse_loss + sep_loss + prm_loss

                # ## without warmup and without shared sep-decoder
                # sep_loss = (rp_mask * imb_beta * modal_weight * sep_loss_m).sum()
                # loss = fuse_loss + sep_loss * 0.0 + prm_loss + kl_loss * 0.5 + proto_loss * 0.1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses += (loss/iter_per_epoch).detach().cpu()
                epoch_fuse_losses += (fuse_loss/iter_per_epoch).detach().cpu()
                epoch_prm_losses += (prm_loss/iter_per_epoch).detach().cpu()
                epoch_sep_losses += (sep_loss/iter_per_epoch).detach().cpu()

                if args.mask_type == 'idt':
                    epoch_sep_m += (sep_loss_m/modal_num).detach().cpu()
                else:
                    epoch_sep_m += (sep_loss_m/iter_per_epoch).detach().cpu()


                msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
                msg += 'fuse_loss:{:.4f}, prm_loss:{:.4f}, '.format(fuse_loss.item(), prm_loss.item())
                msg += 'sep_loss:{:.4f}, '.format(sep_loss.item())
                msg += 'seplist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(sep_loss_m[0].item(), sep_loss_m[1].item(), sep_loss_m[2].item(), sep_loss_m[3].item())

                logging.info(msg)
                ###log
            # #########Validate this epoch model
            # if args.use_valid:
            #     with torch.no_grad():
            #         logging.info('#############validation############')
            #         score_modality = torch.zeros(16)
            #         for j, masks in enumerate(masks_valid_array):
            #             logging.info('{}'.format(mask_name_valid[j]))
            #             for i in range(len(valid_loader)):
            #             # step = (i+1) + epoch*iter_per_epoch
            #             ###Data load
            #                 try:
            #                     data = next(valid_iter)
            #                 except:
            #                     valid_iter = iter(valid_loader)
            #                     data = next(valid_iter)
            #                 x, target= data[:2]
            #                 x = x.cuda(non_blocking=True)
            #                 target = target.cuda(non_blocking=True)
            #                 batchsize=x.size(0)


            #                 mask = torch.unsqueeze(torch.from_numpy(masks), dim=0)
            #                 mask = mask.repeat(batchsize,1)
            #                 mask = mask.cuda(non_blocking=True)

            #                 model.module.is_training = True
            #                 # fuse_pred, sep_preds, prm_preds = model(x, mask)

            #                 fuse_pred, sep_preds, prm_preds = model(x, mask)

            #                 ###Loss compute
            #                 fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            #                 fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            #                 fuse_loss = fuse_cross_loss + fuse_dice_loss

            #                 sep_cross_loss = torch.zeros(1).cuda().float()
            #                 sep_dice_loss = torch.zeros(1).cuda().float()
            #                 for sep_pred in sep_preds:
            #                     sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
            #                     sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            #                 sep_loss = sep_cross_loss + sep_dice_loss

            #                 prm_cross_loss = torch.zeros(1).cuda().float()
            #                 prm_dice_loss = torch.zeros(1).cuda().float()
            #                 for prm_pred in prm_preds:
            #                     prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
            #                     prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            #                 prm_loss = prm_cross_loss + prm_dice_loss

            #                 loss = fuse_loss + sep_loss + prm_loss

            #                 # loss = fuse_loss
            #                 # score -= loss
            #                 score_modality[j] -= loss.item()
            #                 score_modality[15] -= loss.item()
            #         score_modality[15] = score_modality[15] / len(masks_valid_array)
            #         if epoch == 0:
            #             best_score = score_modality[15]
            #             best_epoch = epoch
            #         elif score_modality[15] > best_score:
            #             best_score = score_modality[15]
            #             best_epoch = epoch
            #             file_name = os.path.join(ckpts, 'model_best.pth')
            #             torch.save({
            #                 'epoch': epoch,
            #                 'state_dict': model.state_dict(),
            #                 'optim_dict': optimizer.state_dict(),
            #                 },
            #                 file_name)

            #         for z, _ in enumerate(masks_valid_array):
            #             writer.add_scalar('{}'.format(mask_name_valid[z]), score_modality[z].item(), global_step=epoch+1)
            #         writer.add_scalar('score_average', score_modality[15].item(), global_step=epoch+1)
            #         logging.info('epoch total score: {}'.format(score_modality[15].item()))
            #         logging.info('best score: {}'.format(best_score.item()))
            #         logging.info('best epoch: {}'.format(best_epoch + 1))
            #         logging.info('validate time per epoch: {}'.format(time.time() - b_train))

            b_train = time.time()
            logging.info('train time per epoch: {}'.format(b_train - b))

            writer.add_scalar('epoch_losses', epoch_losses.item(), global_step=(epoch+1))
            writer.add_scalar('epoch_fuse_losses', epoch_fuse_losses.item(), global_step=(epoch+1))
            writer.add_scalar('epoch_prm_losses', epoch_prm_losses.item(), global_step=(epoch+1))
            writer.add_scalar('epoch_sep_losses', epoch_sep_losses.item(), global_step=(epoch+1))

            for m in range(4):
                writer.add_scalar('sep_m{}'.format(m), epoch_sep_m[m].item(), global_step=(epoch+1))

            #########model save
            file_name = os.path.join(ckpts, 'model_last.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)

            if (epoch+1) % 100 == 0 or (epoch>=(args.num_epochs-5)):
                file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    },
                    file_name)

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Test the last epoch model

    test_dice_score = AverageMeter()
    test_hd95_score = AverageMeter()
    csv_name = os.path.join(ckpts, '{}.csv'.format(args.model))
    with torch.no_grad():
        logging.info('###########test last epoch model###########')
        file = open(csv_name, "a+")
        csv_writer = csv.writer(file)
        csv_writer.writerow(['WT Dice', 'TC Dice', 'ET Dice','ETPro Dice', 'WT HD95', 'TC HD95', 'ET HD95' 'ETPro HD95'])
        file.close()
        for i, mask in enumerate(masks_test[::-1]):
            logging.info('{}'.format(mask_name[::-1][i]))
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow([mask_name[::-1][i]])
            file.close()
            dice_score, hd95_score = test_dice_hd95_softmax(
                            test_loader,
                            model,
                            dataname = args.dataname,
                            feature_mask = mask,
                            mask_name = mask_name[::-1][i],
                            csv_name = csv_name,
                            )
            test_dice_score.update(dice_score)
            test_hd95_score.update(hd95_score)

        logging.info('Avg Dice scores: {}'.format(test_dice_score.avg))
        logging.info('Avg HD95 scores: {}'.format(test_hd95_score.avg))


    # ##########Test the best epoch model
    # file_name = os.path.join(ckpts, 'model_best.pth')
    # checkpoint = torch.load(file_name)
    # logging.info('best epoch: {}'.format(checkpoint['epoch']+1))
    # model.load_state_dict(checkpoint['state_dict'])
    # test_best_score = AverageMeter()
    # with torch.no_grad():
    #     logging.info('###########test validate best model###########')
    #     for i, mask in enumerate(masks_test[::-1]):
    #         logging.info('{}'.format(mask_name[::-1][i]))
    #         dice_best_score = test_softmax(
    #                         test_loader,
    #                         model,
    #                         dataname = args.dataname,
    #                         feature_mask = mask,
    #                         mask_name = mask_name[::-1][i])
    #         test_best_score.update(dice_best_score)
    #     logging.info('Avg scores: {}'.format(test_best_score.avg))

if __name__ == '__main__':
    main()
