import warnings

import torch
import torch.nn as nn

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector


@DETECTORS.register_module()
class LeafDETR(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 query_head=None,
                 rpn_head=None,
                 roi_head=[None],
                 bbox_head=[None],
                 train_cfg=[None, None],
                 test_cfg=[None, None],
                 pretrained=[None, None],
                 init_cfg=None,
                 with_pos_coord=True,
                 with_attn_mask=True,
                 eval_module='detr',
                 eval_index=0):

        super(LeafDETR, self).__init__(init_cfg)
        self.with_pos_coord = with_pos_coord
        self.with_attn_mask = with_attn_mask
        self.eval_module = eval_module
        self.eval_index = eval_index
        self.backbone = build_backbone(backbone)

        head_idx = 0

        if neck is not None:
            self.neck = build_neck(neck)

        if query_head is not None:
            query_head.update(
                train_cfg=train_cfg[head_idx] if (train_cfg is not None and train_cfg[head_idx] is not None) else None)
            query_head.update(test_cfg=test_cfg[head_idx])
            self.query_head = build_head(query_head)
            self.query_head.init_weights()
            head_idx += 1

        if rpn_head is not None:
            rpn_train_cfg = train_cfg[head_idx].rpn if (
                    train_cfg is not None and train_cfg[head_idx] is not None) else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg[head_idx].rpn)
            self.rpn_head = build_head(rpn_head_)
            self.rpn_head.init_weights()

        self.roi_head = nn.ModuleList()
        for i in range(len(roi_head)):
            if roi_head[i]:
                rcnn_train_cfg = train_cfg[i + head_idx].rcnn if (
                        train_cfg and train_cfg[i + head_idx] is not None) else None
                roi_head[i].update(train_cfg=rcnn_train_cfg)
                roi_head[i].update(test_cfg=test_cfg[i + head_idx].rcnn)
                self.roi_head.append(build_head(roi_head[i]))
                self.roi_head[-1].init_weights()

        self.bbox_head = nn.ModuleList()
        for i in range(len(bbox_head)):
            if bbox_head[i]:
                bbox_head[i].update(train_cfg=train_cfg[i + head_idx + len(self.roi_head)] if (
                        train_cfg and train_cfg[i + head_idx + len(self.roi_head)] is not None) else None)
                bbox_head[i].update(test_cfg=test_cfg[i + head_idx + len(self.roi_head)])
                self.bbox_head.append(build_head(bbox_head[i]))
                self.bbox_head[-1].init_weights()

        self.head_idx = head_idx
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cqr_cfg = dict(type='nms', iou_threshold=0.8)
        cache_dict = dict()
        for m in self.modules():
            m.cache_dict = cache_dict
        self.cache_dict['query_static'] = [0, 0]
        self.cache_dict['query_next'] = []
        self.cache_dict['cls_branches'] = self.query_head.cls_branches
        self.cache_dict['num_heads'] = self.query_head.transformer.decoder.\
            layers[0].attentions[0].attn.num_heads
        self.cache_dict['cqr_cfg'] = self.cqr_cfg
    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_query_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'query_head') and self.query_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head) > 0

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head[0].with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head) > 0)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None and len(self.bbox_head) > 0))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return (hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0 and self.roi_head[0].with_mask)

    # 抽取出特征
    def extract_feat(self, img, img_metas=None):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape


        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.query_head(x, dummy_img_metas)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        if not self.with_attn_mask:
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]
        x = self.extract_feat(img, img_metas)

        losses = dict()
        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k, v in losses.items():
                new_k = '{}{}'.format(k, idx)
                if isinstance(v, list) or isinstance(v, tuple):
                    new_losses[new_k] = [i * weight for i in v]
                else:
                    new_losses[new_k] = v * weight
            return new_losses
        if self.with_query_head:
            bbox_losses, x = self.query_head.forward_train(x, img_metas,
                                                           gt_bboxes, gt_labels, gt_bboxes_ignore)
            losses.update(bbox_losses)
        if self.with_rpn:
            proposal_cfg = self.train_cfg[self.head_idx].get('rpn_proposal', self.test_cfg[self.head_idx].rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)

            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        positive_coords = []

        for i in range(len(self.roi_head)):
            roi_losses = self.roi_head[i].forward_train(x, img_metas,
                                                        proposal_list,
                                                        gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks,
                                                        **kwargs)
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))
            else:
                if 'pos_coords' in roi_losses.keys():
                    tmp = roi_losses.pop('pos_coords')
            roi_losses = upd_loss(roi_losses, idx=i)
            losses.update(roi_losses)
        for i in range(len(self.bbox_head)):
            bbox_losses = self.bbox_head[i].forward_train(x, img_metas,
                                                          gt_bboxes, gt_labels, gt_bboxes_ignore)
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)
            else:
                if 'pos_coords' in bbox_losses.keys():
                    tmp = bbox_losses.pop('pos_coords')

            bbox_losses = upd_loss(bbox_losses, idx=i + len(self.roi_head))
            losses.update(bbox_losses)
        if self.with_pos_coord and len(positive_coords) > 0:
            coords_list = distinct_aux_coords(positive_coords,img_metas)
            positive_coords = filter_pos_coords_by_indices(positive_coords,coords_list)
            for i in range(len(positive_coords)):
                bbox_losses = self.query_head.forward_train_aux(x, img_metas, gt_bboxes,
                                                                gt_labels, gt_bboxes_ignore,
                                                                positive_coords[i], i, coords_list[1])

                bbox_losses = upd_loss(bbox_losses, idx=i)

                losses.update(bbox_losses)

        return losses

    def simple_test_roi_head(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask:
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-1]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head[self.eval_index].simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def simple_test_query_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        index = 0
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask:  # remove attn mask for LSJ
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        results_list = self.query_head.simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test_bbox_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask:
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-1]
        results_list = self.bbox_head[self.eval_index].simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head[self.eval_index].num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.eval_module in ['detr', 'one-stage', 'two-stage']
        if self.with_bbox and self.eval_module=='one-stage':
            return self.simple_test_bbox_head(img, img_metas, proposals, rescale)
        if self.with_roi_head and self.eval_module=='two-stage':
            return self.simple_test_roi_head(img, img_metas, proposals, rescale)
        return self.simple_test_query_head(img, img_metas, proposals, rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.query_head, 'aug_test'), \
            f'{self.query_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.query_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.query_head.forward_onnx(x, img_metas)[:2]
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        if len(outs) == 2:
            outs = (*outs, None)
        det_bboxes, det_labels = self.query_head.onnx_export(*outs, img_metas)
        return det_bboxes, det_labels
def distinct_aux_coords(pos_coords, img_metas):
    from mmdet.core import bbox_xyxy_to_cxcywh

    from torchvision.ops import nms
    coords_list = []
    img_meta = img_metas[0]
    for i in range(len(pos_coords)):
        if pos_coords[i][-1] == 'rcnn':
            coords = pos_coords[i][0]
            img_h, img_w, _ = img_meta['img_shape']
            factor = coords.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            coord = bbox_xyxy_to_cxcywh(coords / factor).squeeze(0)
            coords_list.append(coord)
        else:
            coords = pos_coords[i][0][0]
            img_h, img_w, _ = img_meta['img_shape']
            factor = coords.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            coord = bbox_xyxy_to_cxcywh(coords / factor)
            coords_list.append(coord)
    def cxcyhw_to_xyxy(boxes):
        cx, cy, w, h = boxes.unbind(dim=1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    all_boxes_xyxy = torch.cat([cxcyhw_to_xyxy(b) for b in coords_list], dim=0)
    scores = torch.rand(all_boxes_xyxy.size(0)).to(all_boxes_xyxy)
    iou_threshold = 0.8
    keep_global_indices = nms(all_boxes_xyxy, scores, iou_threshold)
    source_ids = []
    original_indices = []
    for i, boxes in enumerate(coords_list):
        n = len(boxes)
        source_ids.extend([i] * n)
        original_indices.extend(list(range(n)))
    source_ids = torch.tensor(source_ids)
    original_indices = torch.tensor(original_indices)
    keep_source_ids = source_ids[keep_global_indices]
    keep_original_indices = original_indices[keep_global_indices]
    keep_indices_list = []
    for i in range(len(coords_list)):
        mask = (keep_source_ids == i)
        keep_local = keep_original_indices[mask]
        keep_indices_list.append(keep_local)
    return keep_indices_list

def filter_pos_coords_by_indices(pos_coords, keep_indices):
    filtered_pos_coords = []

    for i, data in enumerate(pos_coords):
        keep_idx = keep_indices[i]

        if i == 0:
            filtered_group = []
            for item in data[:-1]:
                B, N = item.shape[:2]
                flat_item = item.view(B * N, *item.shape[2:])
                selected = flat_item[torch.arange(B)[:, None] * N + keep_idx]
                filtered = selected.view(B, len(keep_idx), *item.shape[2:])
                filtered_group.append(filtered)
            filtered_group.append(data[-1])
            filtered_pos_coords.append(filtered_group)

        elif i == 1:
            filtered_group = []
            for group in data[:-1]:
                filtered_batch = []
                for b in range(len(group)):
                    item = group[b]
                    if len(item) > 0:
                        filtered = item[keep_idx]
                    else:
                        filtered = item.new_empty((0,) + item.shape[1:])
                    filtered_batch.append(filtered)
                filtered_group.append(filtered_batch)
            filtered_group.append(data[-1])
            filtered_pos_coords.append(filtered_group)

    return filtered_pos_coords