# coding=utf-8

import sys
sys.path.append("../utils")

import torch
import torch.nn as nn

import config.yolov3_config as cfg
from utils import tools


class FocalLoss(nn.Module):
    # https://arxiv.org/abs/1708.02002
    def __init__(self, gamma = 2.0, alpha = 1.0, reduction = "mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction = reduction)

    def forward(self, input, target):
        loss = self.__loss(input = input, target = target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)
        return loss


class Loss_yolov3(nn.Module):
    def __init__(self, strides, iou_threshold_loss = 0.5):
        super(Loss_yolov3, self).__init__()
        self.__strides = strides
        self.__iou_threshold_loss = iou_threshold_loss

    def forward(self, p, p_d, label_s, label_m, label_l, bboxes_s, bboxes_m, bboxes_l):
        """
        """
        loss_s, loss_giou_s, loss_conf_s, loss_cls_s = self.__cal_loss_per_layer(p[0], p_d[0], label_s, bboxes_s, self.__strides[0])
        loss_m, loss_giou_m, loss_conf_m, loss_cls_m = self.__cal_loss_per_layer(p[1], p_d[1], label_m, bboxes_m, self.__strides[1])
        loss_l, loss_giou_l, loss_conf_l, loss_cls_l = self.__cal_loss_per_layer(p[2], p_d[2], label_l, bboxes_l, self.__strides[2])

        loss = loss_l + loss_m + loss_s
        loss_giou = loss_giou_s + loss_giou_m + loss_giou_l
        loss_conf = loss_conf_s + loss_conf_m + loss_conf_l
        loss_cls = loss_cls_s + loss_cls_m + loss_cls_l

        return loss, loss_giou, loss_conf, loss_cls

    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):
        """
        """
        BCE = nn.BCEWithLogitsLoss(reduction = "none")
        FOCAL = FocalLoss(gamma = 2, alpha = 1.0, reduction = "none")

        # p: torch.size([batch_size, num_grids(h), num_grids(w), num_anchors, (tx, ty, tw, th, to, num_classes)])
        batch_size, num_grids = p.shape[:2]
        img_size = num_grids * stride

        pred_xywh = p_d[..., 0:4] # 取实际值
        pred_conf = p[..., 4:5] # 取未经sigmoid的
        pred_cls = p[..., 5:] # 取未经sigmoid的

        label_xywh = label[..., 0:4]
        label_obj_mask = label[..., 4:5]
        label_mix = label[..., 5:6]
        label_cls = label[..., 6:]

        # ##### loss giou #####
        giou = tools.GIOU_xywh_torch(pred_xywh, label_xywh).unsqueeze(-1)

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)
        loss_giou = label_obj_mask * bbox_loss_scale * (1.0 - giou) * label_mix

        # ##### loss confidence #####
        iou = tools.iou_xywh_torch(pred_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim = True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.__iou_threshold_loss).float() # the grid has no obj and iou_max < iou_thresh

        loss_conf = (label_obj_mask * FOCAL(input = pred_conf, target = label_obj_mask) +               # obj conf
                     label_noobj_mask * FOCAL(input = pred_conf, target = label_obj_mask)) * label_mix  # noobj conf

        # ##### loss classes #####
        loss_cls = label_obj_mask * BCE(input = pred_cls, target = label_cls) * label_mix

        loss_giou = (torch.sum(loss_giou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss = loss_giou + loss_conf + loss_cls

        return loss, loss_giou, loss_conf, loss_cls



if __name__ == "__main__":
    from model.yolov3 import Model_yolov3
    net = Model_yolov3(20)

    p, p_d = net(torch.rand(3, 3, 416, 416))
    label_s = torch.rand(3, 13, 13, 3, 26)
    label_m = torch.rand(3, 26, 26, 3, 26)
    label_l = torch.rand(3, 52, 52, 3, 26)
    bboxes_s = torch.rand(3, 60, 4)
    bboxes_m = torch.rand(3, 60, 4)
    bboxes_l = torch.rand(3, 60, 4)

    loss, loss_giou, loss_conf, loss_cls = Loss_yolov3(cfg.MODEL["STRIDES"])(p, p_d, label_s, label_m, label_l, bboxes_s, bboxes_m, bboxes_l)
    print(loss)

