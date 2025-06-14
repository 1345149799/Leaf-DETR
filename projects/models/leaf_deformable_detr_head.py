import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean, bbox_overlaps)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.detr_head import DETRHead

import sys
import numpy as np
from mmcv.ops import batched_nms
from mmdet.core import bbox_mapping_back, merge_aug_proposals

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


@HEADS.register_module()
class LeafDeformDETRHead(DETRHead):
    def __init__(self,
                 *args,
                 max_pos_coords=300,
                 lambda_1=1,
                 with_box_refine=False,
                 as_two_stage=False,
                 mixed_selection=False,
                 transformer=None,
                 use_zero_padding=False,
                 **kwargs):
        self.max_pos_coords = max_pos_coords
        self.lambda_1 = lambda_1
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.mixed_selection = mixed_selection
        self.use_zero_padding = use_zero_padding
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if self.mixed_selection:
            transformer['mixed_selection'] = self.mixed_selection
        super(LeafDeformDETRHead, self).__init__(
            *args, transformer=transformer, **kwargs)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        self.downsample = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, self.embed_dims)
        )
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        self.num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers
        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, self.num_pred)
            self.reg_branches = _get_clones(reg_branch, self.num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred)])
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
        elif self.mixed_selection:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))

        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage or self.mixed_selection:
            query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references, \
            enc_outputs_class, enc_outputs_coord, enc_outputs = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            return_encoder_output=True
        )
        outs = []
        num_level = len(mlvl_feats)
        start = 0
        for lvl in range(num_level):
            bs, c, h, w = mlvl_feats[lvl].shape
            end = start + h * w
            feat = enc_outputs[start:end].permute(1, 2, 0).contiguous()
            start = end
            outs.append(feat.reshape(bs, c, h, w))

        outs.append(self.downsample(outs[-1]))
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        if self.as_two_stage:
            return outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord.sigmoid(), outs
        else:
            return outputs_classes, outputs_coords, None, None, outs
    def forward_aux(self, mlvl_feats, img_metas, aux_targets, head_idx):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """
        aux_coords, aux_labels, aux_targets, aux_label_weights, aux_bbox_weights, aux_feats, attn_masks = aux_targets
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']

        img_masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))
        query_embeds = None
        hs, init_reference, inter_references = self.transformer.forward_aux(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            aux_coords,
            pos_feats=aux_feats,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            return_encoder_output=True,
            attn_masks=attn_masks,
            head_idx=head_idx
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        return outputs_classes, outputs_coords, None, None

    def loss_single_aux(self,
                        cls_scores,
                        bbox_preds,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        img_metas,
                        gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        num_q = cls_scores.size(1)
        try:
            labels = labels.reshape(num_imgs * num_q)
            label_weights = label_weights.reshape(num_imgs * num_q)
            bbox_targets = bbox_targets.reshape(num_imgs * num_q, 4)
            bbox_weights = bbox_weights.reshape(num_imgs * num_q, 4)
        except:
            return cls_scores.mean() * 0, cls_scores.mean() * 0, cls_scores.mean() * 0

        bg_class_ind = self.num_classes
        num_total_pos = len(((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1))
        num_total_neg = num_imgs * num_q - num_total_pos
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls * self.lambda_1, loss_bbox * self.lambda_1, loss_iou * self.lambda_1
    def get_aux_targets(self, pos_coords, img_metas, mlvl_feats, head_idx, atss_inds):
        coords, labels, targets = pos_coords[:3]
        head_name = pos_coords[-1]
        bs, c = len(coords), mlvl_feats[0].shape[1]
        max_num_coords = 0
        all_feats = []
        for i in range(bs):
            label = labels[i]
            feats = [feat[i].reshape(c, -1).transpose(1, 0) for feat in mlvl_feats]
            feats = torch.cat(feats, dim=0)
            bg_class_ind = self.num_classes
            pos_inds = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
            max_num_coords = max(max_num_coords, len(pos_inds))
            all_feats.append(feats)
        max_num_coords = min(self.max_pos_coords, max_num_coords)
        max_num_coords = max(9, max_num_coords)

        if self.use_zero_padding:
            attn_masks = []
            label_weights = coords[0].new_zeros([bs, max_num_coords])
        else:
            attn_masks = None
            label_weights = coords[0].new_ones([bs, max_num_coords])

        bbox_weights = coords[0].new_zeros([bs, max_num_coords, 4])

        aux_coords, aux_labels, aux_targets, aux_feats = [], [], [], []

        for i in range(bs):
            coord, label, target = coords[i], labels[i], targets[i]
            feats = all_feats[i]
            if 'rcnn' in head_name:
                feats = pos_coords[-2][i]
                num_coords_per_point = 1
            else:
                feats = feats[atss_inds]
                num_coords_per_point = coord.shape[0] // feats.shape[0]

            feats = feats.unsqueeze(1).repeat(1, num_coords_per_point, 1)
            feats = feats.reshape(feats.shape[0] * num_coords_per_point, feats.shape[-1])
            img_meta = img_metas[i]
            img_h, img_w, _ = img_meta['img_shape']

            factor = coord.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            bg_class_ind = self.num_classes
            pos_inds = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
            neg_inds = ((label == bg_class_ind)).nonzero().squeeze(1)
            if pos_inds.shape[0] > max_num_coords:
                indices = torch.randperm(pos_inds.shape[0])[:max_num_coords].cuda()
                pos_inds = pos_inds[indices]
            coord = bbox_xyxy_to_cxcywh(coord[pos_inds] / factor)
            label = label[pos_inds]
            target = bbox_xyxy_to_cxcywh(target[pos_inds] / factor)
            feat = feats[pos_inds]
            if self.use_zero_padding:
                label_weights[i][:len(label)] = 1
                bbox_weights[i][:len(label)] = 1
                attn_mask = torch.zeros([max_num_coords, max_num_coords, ]).bool().to(coord.device)
            else:
                bbox_weights[i][:len(label)] = 1
            if coord.shape[0] < max_num_coords:
                padding_shape = max_num_coords - coord.shape[0]
                if self.use_zero_padding:
                    padding_coord = coord.new_zeros([padding_shape, 4])
                    padding_label = label.new_ones([padding_shape]) * self.num_classes
                    padding_target = target.new_zeros([padding_shape, 4])
                    padding_feat = feat.new_zeros([padding_shape, c])
                    attn_mask[coord.shape[0]:, 0: coord.shape[0], ] = True
                    attn_mask[:, coord.shape[0]:, ] = True
                else:
                    indices = torch.randperm(neg_inds.shape[0])[:padding_shape].cuda()
                    neg_inds = neg_inds[indices]
                    padding_coord = bbox_xyxy_to_cxcywh(coords[i][neg_inds] / factor)
                    padding_label = labels[i][neg_inds]
                    padding_target = bbox_xyxy_to_cxcywh(targets[i][neg_inds] / factor)
                    padding_feat = feats[neg_inds]
                coord = torch.cat((coord, padding_coord), dim=0)
                label = torch.cat((label, padding_label), dim=0)
                target = torch.cat((target, padding_target), dim=0)
                feat = torch.cat((feat, padding_feat), dim=0)
            if self.use_zero_padding:
                attn_masks.append(attn_mask.unsqueeze(0))
            aux_coords.append(coord.unsqueeze(0))
            aux_labels.append(label.unsqueeze(0))
            aux_targets.append(target.unsqueeze(0))
            aux_feats.append(feat.unsqueeze(0))

        if self.use_zero_padding:
            attn_masks = torch.cat(attn_masks, dim=0).unsqueeze(1).repeat(1, 8, 1, 1)
            attn_masks = attn_masks.reshape(bs * 8, max_num_coords, max_num_coords)
        else:
            attn_mask = None

        aux_coords = torch.cat(aux_coords, dim=0)
        aux_labels = torch.cat(aux_labels, dim=0)
        aux_targets = torch.cat(aux_targets, dim=0)
        aux_feats = torch.cat(aux_feats, dim=0)
        aux_label_weights = label_weights
        aux_bbox_weights = bbox_weights
        return (aux_coords, aux_labels, aux_targets, aux_label_weights, aux_bbox_weights, aux_feats, attn_masks)
    def forward_train_aux(self,
                          x,
                          img_metas,
                          gt_bboxes,
                          gt_labels=None,
                          gt_bboxes_ignore=None,
                          pos_coords=None,
                          head_idx=0,
                          atss_inds=None,
                          **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        aux_targets = self.get_aux_targets(pos_coords, img_metas, x, head_idx, atss_inds)
        outs = self.forward_aux(x[:-1], img_metas, aux_targets, head_idx)
        outs = outs + aux_targets
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:

            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)

        losses = self.loss_aux(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss_aux(self,
                 all_cls_scores,
                 all_bbox_preds,
                 enc_cls_scores,
                 enc_bbox_preds,
                 aux_coords,
                 aux_labels,
                 aux_targets,
                 aux_label_weights,
                 aux_bbox_weights,
                 aux_feats,
                 attn_masks,
                 gt_bboxes_list,
                 gt_labels_list,
                 img_metas,
                 gt_bboxes_ignore=None):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_labels = [aux_labels for _ in range(num_dec_layers)]
        all_label_weights = [aux_label_weights for _ in range(num_dec_layers)]
        all_bbox_targets = [aux_targets for _ in range(num_dec_layers)]
        all_bbox_weights = [aux_bbox_weights for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single_aux, all_cls_scores, all_bbox_preds,
            all_labels, all_label_weights, all_bbox_targets,
            all_bbox_weights, img_metas_list, all_gt_bboxes_ignore_list)
        loss_dict = dict()
        loss_dict['loss_cls_aux'] = losses_cls[-1]
        loss_dict['loss_bbox_aux'] = losses_bbox[-1]
        loss_dict['loss_iou_aux'] = losses_iou[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_aux'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_aux'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou_aux'] = loss_iou_i
            num_dec_layer += 1

        return loss_dict
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        enc_outputs = outs[-1]
        return losses, enc_outputs

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             enc_cls_scores,
             enc_bbox_preds,
             enc_outputs,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels]. # decoder??class head?? [6,bs,300,80]
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].  # decoder??reg head [6,bs,300,4]
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.  [bs,sum(hw),80]
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None. [bs,sum(hw),4]
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single,
            all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            img_metas_list,
            all_gt_bboxes_ignore_list)
        loss_dict = dict()
        if enc_cls_scores is not None:
            binary_labels_list = [torch.zeros_like(gt_labels_list[i]) for i in range(len(img_metas))]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i

            num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   enc_cls_scores,
                   enc_bbox_preds,
                   enc_outputs,
                   img_metas,
                   rescale=False,
                   with_nms=False
                   ):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        result_list = []
        attn_mask = [
            self.cache_dict['query_next'][-1][
                img_id * 8][0]
            for img_id in range(len(img_metas))
        ]
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id][attn_mask[img_id]]
            bbox_pred = bbox_preds[img_id][attn_mask[img_id]]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale, with_nms)
            result_list.append(proposals)
        return result_list

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)  # bs
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]  # ?????
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]  # ?????
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)  # ??????????
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        # ???loss
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        num_imgs = len(cls_scores_list)  # bs
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [
                gt_bboxes_ignore_list for _ in range(num_imgs)
            ]
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        ori_gt_bboxes_ignore = gt_bboxes_ignore
        gt_bboxes_ignore = None
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False,
                           with_nms=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        max_per_img = min(max_per_img,len(cls_score))
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)

        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)
        return det_bboxes, det_labels


    def aug_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes with test-time augmentation.
        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[ndarray]: bbox results of each class
        """
        raise ValueError('Not implemented')

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        with_nms = self.test_cfg.get('nms', None)
        with_nms = True if with_nms is not None else False
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale, with_nms=with_nms)
        return results_list
