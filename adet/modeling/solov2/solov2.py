# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms, cat, paste_masks_in_image
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import (
    permute_all_cls_and_box_to_N_HWA_K_and_concat,
    permute_to_N_HWA_K,
)
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n


from detectron2.layers.DFConv2d import DFConv2d


__all__ = ["SOLOv2"]


@META_ARCH_REGISTRY.register()
class SOLOv2(nn.Module):
    """
    SOLOv2 model. Creates FPN backbone, instance branch for class-specific kernels,
    mask branch for class-agnostic masks.
    Calculates and applies proper losses to class and masks.
    """

    def __init__(self, cfg):
        super().__init__()

        # get the device of the model
        self.device = torch.device(cfg.MODEL.DEVICE)

        # Instance parameters.
        self.num_classes = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.num_kernels = cfg.MODEL.SOLOV2.NUM_KERNELS
        self.num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS
        self.instance_in_features = cfg.MODEL.SOLOV2.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.SOLOV2.INSTANCE_CHANNELS

        # Mask parameters.
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_in_features = cfg.MODEL.SOLOV2.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.SOLOV2.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.SOLOV2.MASK_CHANNELS
        self.num_masks = cfg.MODEL.SOLOV2.NUM_MASKS

        # Inference parameters.
        self.max_before_nms = cfg.MODEL.SOLOV2.NMS_PRE
        self.score_threshold = cfg.MODEL.SOLOV2.SCORE_THR
        self.update_threshold = cfg.MODEL.SOLOV2.UPDATE_THR
        self.mask_threshold = cfg.MODEL.SOLOV2.MASK_THR
        self.max_per_img = cfg.MODEL.SOLOV2.MAX_PER_IMG
        self.nms_kernel = cfg.MODEL.SOLOV2.NMS_KERNEL
        self.sigma = cfg.MODEL.SOLOV2.SIGMA

        # build the backbone.
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()

        # build the ins head.
        instance_shapes = [backbone_shape[f] for f in self.instance_in_features]
        self.ins_head = SOLOv2InsHead(cfg, instance_shapes)

        # build the mask head.
        mask_shapes = [backbone_shape[f] for f in self.mask_in_features]
        self.mask_head = SOLOv2MaskHead(cfg, mask_shapes)

        # image transform
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.
            For now, each item in the list is a dict that contains:
                image: Tensor, image in (C, H, W) format.
                instances: Instances
                Other information that's included in the original dicts, such as:
                    "height", "width" (int): the output resolution of the model, used in inference.
                        See :meth:`postprocess` for details.
         Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        # ins branch
        ins_features = [features[f] for f in self.instance_in_features]
        ins_features = self.split_feats(ins_features)
        cate_pred, kernel_pred = self.ins_head(ins_features)

        # mask branch
        mask_features = [features[f] for f in self.mask_in_features]
        mask_pred = self.mask_head(mask_features)

        if self.training:
            """
            get_ground_truth.
            return loss and so on.
            """
            pass
        else:
            # point nms.
            cate_pred = [self.point_nms(cate_p.sigmoid(), kernel=2).permute(0, 2, 3, 1)
                         for cate_p in cate_pred]
            # do inference for results.
            results = self.inference(cate_pred, kernel_pred, mask_pred, images.image_sizes, batched_inputs)
            return results

    @torch.no_grad()
    def point_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        # mmdetection 是RGB训练
        permute = [2, 1, 0]
        images = [x[permute, :, :] for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def split_feats(feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    @staticmethod
    def matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores, sigma=2.0, kernel='gaussian'):
        n_samples = len(cate_labels)
        if n_samples == 0:
            return []

        seg_masks = seg_masks.reshape(n_samples, -1).float()
        # inter.
        inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
        # union.
        sum_masks_x = sum_masks.expand(n_samples, n_samples)
        # iou.
        iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
        # label_specific matrix.
        cate_labels_x = cate_labels.expand(n_samples, n_samples)
        label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

        # IoU compensation
        compensate_iou, _ = (iou_matrix * label_matrix).max(0)
        compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

        # IoU decay / soft nms
        delay_iou = iou_matrix * label_matrix

        # matrix nms
        if kernel == 'linear':
            delay_matrix = (1 - delay_iou) / (1 - compensate_iou)
            delay_coefficient, _ = delay_matrix.min(0)
        else:
            delay_matrix = torch.exp(-1 * sigma * (delay_iou ** 2))
            compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
            delay_coefficient, _ = (delay_matrix / compensate_matrix).min(0)

        # update the score.
        cate_scores_update = cate_scores * delay_coefficient

        return cate_scores_update

    def inference(self, pred_cates, pred_kernels, pred_masks, cur_sizes, images):
        """
        Arguments:
            pred_cates, pred_kernels, pred_masks.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(pred_cates) == len(pred_kernels)

        results = []
        num_ins_levels = len(pred_cates)
        for img_idx in range(len(images)):
            # image size.
            ori_img = images[img_idx]
            height, width = ori_img["height"], ori_img["width"]
            ori_size = (height, width)

            # prediction.
            pred_cate = [pred_cates[i][img_idx].view(-1, self.num_classes).detach()
                          for i in range(num_ins_levels)]
            pred_kernel = [pred_kernels[i][img_idx].permute(1, 2, 0).view(-1, self.num_kernels).detach()
                            for i in range(num_ins_levels)]
            pred_mask = pred_masks[img_idx, ...].unsqueeze(0)

            pred_cate = torch.cat(pred_cate, dim=0)
            pred_kernel = torch.cat(pred_kernel, dim=0)

            # inference for single image.
            result = self.inference_single_image(pred_cate, pred_kernel, pred_mask,
                                                 cur_sizes[img_idx], ori_size)
            results.append({"instances": result})

        return results

    def inference_single_image(
            self, cate_preds, kernel_preds, seg_preds, cur_size, ori_size
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            pred_logits (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (AxHxW, K)
            pred_deltas (list[Tensor]): Same shape as 'pred_logits' except that K becomes 4.
            pred_masks (list[list[Tensor]]): List of #feature levels, each is a list of #anchors.
                Each entry contains tensor of size (M_i*M_i, H, W). `None` if mask_on=False.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """

        # overall info.
        h, w = cur_size
        f_h, f_w = seg_preds.size()[-2:]
        ratio = math.ceil(h/f_h)
        upsampled_size_out = (int(f_h*ratio), int(f_w*ratio))

        # process.
        inds = (cate_preds > self.score_threshold)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.num_grids)
        strides[:size_trans[0]] *= self.instance_strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.instance_strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        N, I = kernel_preds.shape
        kernel_preds = kernel_preds.view(N, I, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()

        # mask.
        seg_masks = seg_preds > self.mask_threshold
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_before_nms:
            sort_inds = sort_inds[:self.max_before_nms]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = self.matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                                      sigma=self.sigma, kernel=self.nms_kernel)
        # filter.
        keep = cate_scores >= self.update_threshold
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_per_img:
            sort_inds = sort_inds[:self.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # reshape to original size.
        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_size,
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > self.mask_threshold

        results = Instances(ori_size)
        results.pred_classes = cate_labels
        results.scores = cate_scores
        results.pred_masks = seg_masks
        return results


class SOLOv2InsHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        SOLOv2 Instance Head.
        """
        super().__init__()
        # fmt: off
        self.num_classes = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.num_kernels = cfg.MODEL.SOLOV2.NUM_KERNELS
        self.num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS
        self.instance_in_features = cfg.MODEL.SOLOV2.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.SOLOV2.INSTANCE_CHANNELS
        # Convolutions to use in the towers
        self.type_dcn = cfg.MODEL.SOLOV2.TYPE_DCN
        self.num_levels = len(self.instance_in_features)
        assert self.num_levels == len(self.instance_strides), \
            print("Strides should match the features.")
        # fmt: on

        head_configs = {"cate": (cfg.MODEL.SOLOV2.NUM_INSTANCE_CONVS,
                                 cfg.MODEL.SOLOV2.USE_DCN_IN_INSTANCE,
                                 False),
                        "kernel": (cfg.MODEL.SOLOV2.NUM_INSTANCE_CONVS,
                                   cfg.MODEL.SOLOV2.USE_DCN_IN_INSTANCE,
                                   cfg.MODEL.SOLOV2.USE_COORD_CONV)
                        }

        norm = None if cfg.MODEL.SOLOV2.NORM == "none" else cfg.MODEL.SOLOV2.NORM
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, \
            print("Each level must have the same channel!")
        in_channels = in_channels[0]
        assert in_channels == cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS, \
            print("In channels should equal to tower in channels!")

        for head in head_configs:
            tower = []
            num_convs, use_deformable, use_coord = head_configs[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    if self.type_dcn == "DCN":
                        conv_func = DFConv2d
                    else:
                        raise NotImplementedError
                else:
                    conv_func = nn.Conv2d
                if i == 0:
                    if use_coord:
                        chn = self.instance_in_channels + 2
                    else:
                        chn = self.instance_in_channels
                else:
                    chn = self.instance_channels

                tower.append(conv_func(
                        chn, self.instance_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                tower.append(nn.ReLU(inplace=True))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cate_pred = nn.Conv2d(
            self.instance_channels, self.num_classes,
            kernel_size=3, stride=1, padding=1
        )
        self.kernel_pred = nn.Conv2d(
            self.instance_channels, self.num_kernels,
            kernel_size=3, stride=1, padding=1
        )

        for modules in [
            self.cate_tower, self.kernel_tower,
            self.cate_pred, self.kernel_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.SOLOV2.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cate_pred.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        cate_pred = []
        kernel_pred = []

        for idx, feature in enumerate(features):
            ins_kernel_feat = feature
            # concat coord
            x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
            y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # individual feature.
            kernel_feat = ins_kernel_feat
            seg_num_grid = self.num_grids[idx]
            kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')
            cate_feat = kernel_feat[:, :-2, :, :]

            # kernel
            kernel_feat = self.kernel_tower(kernel_feat)
            kernel_pred.append(self.kernel_pred(kernel_feat))

            # cate
            cate_feat = self.cate_tower(cate_feat)
            cate_pred.append(self.cate_pred(cate_feat))

        return cate_pred, kernel_pred


class SOLOv2MaskHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        SOLOv2 Mask Head.
        """
        super().__init__()
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        self.num_masks = cfg.MODEL.SOLOV2.NUM_MASKS
        self.mask_in_features = cfg.MODEL.SOLOV2.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.SOLOV2.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.SOLOV2.MASK_CHANNELS
        self.num_levels = len(input_shape)
        assert self.num_levels == len(self.mask_in_features), \
            print("Input shape should match the features.")
        # fmt: on
        norm = None if cfg.MODEL.SOLOV2.NORM == "none" else cfg.MODEL.SOLOV2.NORM

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.num_levels):
            convs_per_level = nn.Sequential()
            if i == 0:
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_in_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(i), nn.Sequential(*conv_tower))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.mask_in_channels + 2 if i == 3 else self.mask_in_channels
                    conv_tower = list()
                    conv_tower.append(nn.Conv2d(
                        chn, self.mask_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                    ))
                    if norm == "GN":
                        conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                    conv_tower.append(nn.ReLU(inplace=False))
                    convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                    upsample_tower = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), upsample_tower)
                    continue

                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                upsample_tower = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), upsample_tower)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.mask_channels, self.num_masks,
                kernel_size=1, stride=1,
                padding=0, bias=norm is None),
            nn.GroupNorm(32, self.num_masks),
            nn.ReLU(inplace=True)
        )

        for modules in [self.convs_all_levels, self.conv_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        assert len(features) == self.num_levels, \
            print("The number of input features should be equal to the supposed level.")

        # bottom features first.
        feature_add_all_level = self.convs_all_levels[0](features[0])
        for i in range(1, self.num_levels):
            mask_feat = features[i]
            if i == 3:  # add for coord.
                x_range = torch.linspace(-1, 1, mask_feat.shape[-1], device=mask_feat.device)
                y_range = torch.linspace(-1, 1, mask_feat.shape[-2], device=mask_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([mask_feat.shape[0], 1, -1, -1])
                x = x.expand([mask_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                mask_feat = torch.cat([mask_feat, coord_feat], 1)
            # add for top features.
            feature_add_all_level += self.convs_all_levels[i](mask_feat)

        mask_pred = self.conv_pred(feature_add_all_level)

        return mask_pred
