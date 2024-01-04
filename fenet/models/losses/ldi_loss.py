import torch
import pdb

def line_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, shape: (num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    # pred torch.Size([192, 72])

    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    # ovr torch.Size([192, 3, 72])
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    # iou torch.Size([192, 3])
    return iou

def cal_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, shape: (num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
        left_ovr = target - torch.max(px1, tx1)
        left_union = length
        right_ovr = torch.min(px2, tx2) - target 
        right_union = length
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))
        left_ovr = target - torch.max(px1[:, None, :], tx1[None, ...])
        left_union = length
        right_ovr = torch.min(px2[:, None, :], tx2[None, ...]) - target 
        right_union = length

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    left_ovr[invalid_masks] = 0.
    right_ovr[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)

    all_shape = 72 
    left_iou = left_ovr.sum(dim=-1) / (left_union*all_shape + 1e-9)
    right_iou = right_ovr.sum(dim=-1) / (right_union*all_shape + 1e-9)

    return iou,left_iou,right_iou


def ldi_iou_loss(pred, target, img_w, length=15):
    lane_iou,left_iou,right_iou = cal_iou(pred, target, img_w, length)
    lane_iou = (1 - lane_iou).mean()
    left_iou = (1 - left_iou).mean()
    right_iou = (1 - right_iou).mean()
    return lane_iou,left_iou,right_iou