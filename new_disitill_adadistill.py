import os
import time
import torch
import logging
import argparse
import losses
#from config import configKD
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss, MSELoss
#from backbones.iresnet import iresnet18, iresnet34
from dataset import MXFaceDataset, DataLoaderX
from utils_logging import AverageMeter, init_logging
from torch.nn.parallel.distributed import DistributedDataParallel
from utils_callbacks import CallBackVerification, CallBackLoggingKD, CallBackModelCheckpointKD
from arc2face import CLIPTextModelWrapper, project_face_embs

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import torch.nn as nn
from onnx2pytorch import ConvertModel
import onnx
import cv2
from insightface.app.common import Face
from insightface.utils import face_align
import torch.optim as optim

if __name__ == '__main__':


    torch.backends.cudnn.benchmark = True

    local_rank = "cuda:0"
    # torch.cuda.set_device(local_rank)
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    log_root = logging.getLogger()
    output="output/FaceRecognitionKD"
    init_logging(log_root,output)
    trainset = MXFaceDataset(root_dir="./faces_emore", local_rank="cuda")
    num_classes = 85742
    number_image = 5822653
    #batch_size = 8
    embedding_size = 512
    momentum = 0.9
    weight_decay = 5e-4
    batch_size = 100
    lr = 0.1
    scale=1.0
    global_step=0
    s=64.0
    m=0.5
    w=10000
    world_size=1
    eval_step=5856
    output = "./model_file"
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [4, 6, 9, 11] if m - 1 <= epoch])
    lr_func = lr_step_func
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #         trainset, shuffle=True)
    num_epoch = 12
    train_loader = DataLoaderX(
            local_rank=local_rank, dataset=trainset, batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    tea_model_onnx = app.models['recognition'].model_file
    tea_model = onnx.load(tea_model_onnx)
    tea_recognition = ConvertModel(tea_model, experimental=True)
    tea_recognition = tea_recognition.cuda()
    tea_recognition.eval()


 
    #app_student = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    #app_student = FaceAnalysis(name='buffalo_s', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app_student = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    #antelopev2
    app_student.prepare(ctx_id=0, det_size=(640, 640))


    stu_model_onnx = app_student.models['recognition'].model_file
    onnx_model = onnx.load(stu_model_onnx)
    stu_recognition = ConvertModel(onnx_model, experimental=True)
    #stu_recognition.load_state_dict(torch.load('./model_file/4001backbone.pth'))
    #stu_recognition.load_state_dict(torch.load('stu_recognition_ada.pth', map_location="cuda"))
    #stu_recognition = torch.load('stu_recognition_ada.pth')
    stu_recognition = stu_recognition.cuda()


    # #break

    stu_recognition.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)


    stu_recognition.train()

    #header = losses.ArcFace(in_features=embedding_size, out_features=num_classes, s=s, m=m).to(local_rank)
    header = losses.AdaptiveAArcFace(in_features=embedding_size, out_features=num_classes, s=s, m=m,  adaptive_weighted_alpha=True).to(local_rank)
    #header = torch.load('header_Mine.pth')
    #header.train()
    header.eval()


    opt_backbone_student = torch.optim.SGD(
        params=[{'params': stu_recognition.parameters()}],
        lr=lr / 512 * batch_size * world_size,
        momentum=0.9, weight_decay=weight_decay)
    # opt_header = torch.optim.SGD(
    #     params=[{'params': header.parameters()}],
    #     lr=lr / 512 * batch_size * world_size,
    #     momentum=0.9, weight_decay=weight_decay)

    scheduler_backbone_student = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone_student, lr_lambda=lr_func)
    # scheduler_header = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer=opt_header, lr_lambda=lr_func)        

    criterion = CrossEntropyLoss()
    criterion2 = MSELoss()

    start_epoch = 0
    total_step = int(len(trainset) / batch_size / world_size * num_epoch)

    #callback_verification = CallBackVerification(eval_step, rank, configKD.cfg.val_targets, configKD.cfg.rec) # 2000
    callback_logging = CallBackLoggingKD(50, 0, total_step, batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpointKD(output)

    loss = AverageMeter()
    loss1 = AverageMeter()
    loss2 = AverageMeter()
    global_step = global_step




    for epoch in range(start_epoch, num_epoch):
        print(f'Epoch: {epoch}')
        #train_sampler.set_epoch(epoch)
        for step, (_, img, label) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)

            label = label.cuda(local_rank, non_blocking=True)#.unsqueeze(1)
            #print(label.size())
            with torch.no_grad():
                features_teacher = F.normalize(tea_recognition(img))
            features_student = F.normalize(stu_recognition(img))
            #print(features_teacher)
            #print(features_student.size())
            
            #break
            #thetas = header(features_student, label)
            thetas ,target_logit_mean, lma, cos_theta_tmp = header(features_student,features_teacher, label)
            loss_v1 = criterion(thetas, label)
            loss_v2 = w*criterion2(features_student, features_teacher)
            loss_v = loss_v1 + loss_v2*0
            loss_v.backward()
            #print(loss_v)
            #break
            # Clips gradient norm of an iterable of parameters.
            # The norm is computed over all gradients together,
            # as if they were concatenated into a single vector. Gradients are modified in-place.
            clip_grad_norm_(stu_recognition.parameters(), max_norm=5, norm_type=2)

            opt_backbone_student.step()
            #opt_header.step()

            opt_backbone_student.zero_grad()
            #opt_header.zero_grad()

            loss.update(loss_v.item(), 1)
            loss1.update(loss_v1.item(), 1)
            loss2.update(loss_v2.item(), 1)

            callback_logging(global_step, loss, loss1, loss2, epoch)
            #callback_verification(global_step, stu_recognition)

            if step ==500:
                callback_checkpoint(global_step, stu_recognition, header)
                torch.save(stu_recognition, 'stu_recognition_ada.pth')
                torch.save(header, 'header_ada.pth')
                #torch.save(stu_recognition.state_dict(), 'stu_recognition_Mine.pth')
                
        #break
        scheduler_backbone_student.step()
        #scheduler_header.step()

        callback_checkpoint(global_step, stu_recognition, header)