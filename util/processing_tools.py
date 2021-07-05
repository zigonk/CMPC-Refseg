from __future__ import absolute_import, division, print_function

import numpy as np

def generate_spatial_batch(N, featmap_H, featmap_W):
    spatial_batch_val = np.zeros((N, featmap_H, featmap_W, 8), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w+1) / featmap_W * 2 - 1
            xctr = (xmin+xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h+1) / featmap_H * 2 - 1
            yctr = (ymin+ymax) / 2
            spatial_batch_val[:, h, w, :] = \
                [xmin, ymin, xmax, ymax, xctr, yctr, 1/featmap_W, 1/featmap_H]
    return spatial_batch_val

def generate_bilinear_filter(stride):
    # Bilinear upsampling filter
    f = np.concatenate((np.arange(0, stride), np.arange(stride, 0, -1))) / stride
    return np.outer(f, f).astype(np.float32)[:, :, np.newaxis, np.newaxis]

def compute_accuracy(scores, labels):
    is_pos = (labels != 0)
    is_neg = np.logical_not(is_pos)
    num_pos = np.sum(is_pos)
    num_neg = np.sum(is_neg)
    num_all = num_pos + num_neg

    is_correct = np.logical_xor(scores < 0, is_pos)
    accuracy_all = np.sum(is_correct) / num_all
    accuracy_pos = np.sum(is_correct[is_pos]) / (num_pos + 1)
    accuracy_neg = np.sum(is_correct[is_neg]) / num_neg
    return accuracy_all, accuracy_pos, accuracy_neg

def compute_meanIoU(scores, labels):
    gt = (labels != 0)
    pred = (scores > 0)
    union = pred | gt
    intersect = pred & gt
    return np.sum(intersect) / np.sum(union)
    

def spatial_feature_from_bbox(bboxes, imsize):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))
    im_w, im_h = imsize
    assert(np.all(bboxes[:, 0] < im_w) and np.all(bboxes[:, 2] < im_w))
    assert(np.all(bboxes[:, 1] < im_h) and np.all(bboxes[:, 3] < im_h))

    feats = np.zeros((bboxes.shape[0], 8))
    feats[:, 0] = bboxes[:, 0] * 2.0 / im_w - 1  # x1
    feats[:, 1] = bboxes[:, 1] * 2.0 / im_h - 1  # y1
    feats[:, 2] = bboxes[:, 2] * 2.0 / im_w - 1  # x2
    feats[:, 3] = bboxes[:, 3] * 2.0 / im_h - 1  # y2
    feats[:, 4] = (feats[:, 0] + feats[:, 2]) / 2  # x0
    feats[:, 5] = (feats[:, 1] + feats[:, 3]) / 2  # y0
    feats[:, 6] = feats[:, 2] - feats[:, 0]  # w
    feats[:, 7] = feats[:, 3] - feats[:, 1]  # h
    return feats

def bbox_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / (union_area + 1e-6)
    # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss

def preprocess_true_boxes(bboxes, train_input_size, anchors, stride=8, anchor_per_scale=3, max_bbox_per_scale=3):
    train_output_size = train_input_size // stride
    label = np.zeros((train_output_size, train_output_size, anchor_per_scale,
                        5))
    bboxes_xywh = np.zeros((max_bbox_per_scale, 4))
    bbox_count = 0

    for bbox in bboxes:
        bbox_coor = bbox[:4]
        print(bbox)
        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        bbox_xywh_scaled = 1.0 * bbox_xywh / stride

        iou = []
        exist_positive = False
        
        anchors_xywh = np.zeros((anchor_per_scale, 4))
        anchors_xywh[:,0:2] = np.floor(bbox_xywh_scaled[0:2]).astype(np.int32) + 0.5
        anchors_xywh[:,2:4] = anchors

        iou_scale = bbox_iou(bbox_xywh_scaled[np.newaxis, :], anchors_xywh)
        iou.append(iou_scale)
        iou_mask = iou_scale > 0.3

        if np.any(iou_mask):
            xind, yind = np.floor(bbox_xywh_scaled[0:2]).astype(np.int32)
            xind = np.clip(xind, 0, train_output_size - 1)     
            yind = np.clip(yind, 0, train_output_size - 1)     
            # This will mitigate errors generated when the location computed by this is more the grid cell location. 
            # e.g. For 52x52 grid cells possible values of xind and yind are in range [0-51] including both. 
            # But sometimes the coomputation makes it 52 and then it will try to find that location in label array 
            # which is not present and throws error during training.

            label[yind, xind, iou_mask, :] = 0
            label[yind, xind, iou_mask, 0:4] = bbox_xywh
            label[yind, xind, iou_mask, 4:5] = 1.0

            bbox_ind = int(bbox_count % max_bbox_per_scale)
            bboxes_xywh[bbox_ind, :4] = bbox_xywh
            bbox_count += 1

            exist_positive = True

        if not exist_positive:
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_anchor = int(best_anchor_ind % anchor_per_scale)
            xind, yind = np.floor(bbox_xywh_scaled[0:2]).astype(np.int32)
            xind = np.clip(xind, 0, train_output_size - 1)     
            yind = np.clip(yind, 0, train_output_size - 1)     
            # This will mitigate errors generated when the location computed by this is more the grid cell location. 
            # e.g. For 52x52 grid cells possible values of xind and yind are in range [0-51] including both. 
            # But sometimes the coomputation makes it 52 and then it will try to find that location in label array 
            # which is not present and throws error during training.

            label[yind, xind, best_anchor, :] = 0
            label[yind, xind, best_anchor, 0:4] = bbox_xywh
            label[yind, xind, best_anchor, 4:5] = 1.0

            bbox_ind = int(bbox_count % max_bbox_per_scale)
            bboxes_xywh[bbox_ind, :4] = bbox_xywh
            bbox_count += 1
    return label, bboxes_xywh