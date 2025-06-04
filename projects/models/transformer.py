import math

import copy
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence

from mmdet.models.utils.transformer import Transformer, DeformableDetrTransformer, DeformableDetrTransformerDecoder
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.ops import batched_nms
from mmdet.core import bbox_cxcywh_to_xyxy

def padding_to(inputs, max=300):
    if max is None:
        return inputs
    num_padding = max - len(inputs)
    if inputs.dim() > 1:
        padding = inputs.new_zeros(num_padding,
                                   *inputs.size()[1:],
                                   dtype=inputs.dtype)
    else:
        padding = inputs.new_zeros(num_padding, dtype=inputs.dtype)
    inputs = torch.cat([inputs, padding], dim=0)
    return inputs
def align_tensor(inputs, max_len=None):
    if max_len is None:
        max_len = max([len(item) for item in inputs])

    return torch.stack([padding_to(item, max_len) for item in inputs])

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# Co-DETR的decoder
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LeafDeformableDetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, look_forward_twice=False, **kwargs):

        super(LeafDeformableDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                                         torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                                         valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                                                    ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(
                    new_reference_points
                    if self.look_forward_twice
                    else reference_points
                )
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)
        return output, reference_points


@TRANSFORMER.register_module()
class LeafDeformableDetrTransformer(DeformableDetrTransformer):
    """Implements the DeformableDETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 mixed_selection=True,
                 with_pos_coord=True,
                 with_coord_feat=True,
                 num_co_heads=1,
                 **kwargs):
        self.mixed_selection = mixed_selection
        self.with_pos_coord = with_pos_coord
        self.with_coord_feat = with_coord_feat
        self.num_co_heads = num_co_heads
        super(LeafDeformableDetrTransformer, self).__init__(**kwargs)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                self.head_pos_embed = nn.Embedding(self.num_co_heads, 1, 1, self.embed_dims)
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(nn.Linear(self.embed_dims * 2, self.embed_dims * 2))
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims * 2))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        num_pos_feats = self.embed_dims // 2
        scale = 2 * math.pi
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                cls_branches=None,
                return_encoder_output=False,
                attn_masks=None,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.


        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)
        reference_points = self.get_reference_points(spatial_shapes,
                                                     valid_ratios,
                                                     device=feat.device)
        feat_flatten = feat_flatten.permute(1, 0, 2)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes)
            enc_outputs_class = cls_branches[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            topk = query_embed.shape[0]
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            if not self.mixed_selection:
                query_pos, query = torch.split(pos_trans_out, c, dim=2)
            else:
                query = query_embed.unsqueeze(0).expand(bs, -1, -1)
                query_pos, _ = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            attn_masks=attn_masks,
            aux=False,
            **kwargs)

        inter_references_out = inter_references
        if self.as_two_stage:
            if return_encoder_output:
                return inter_states, init_reference_out, \
                    inter_references_out, enc_outputs_class, \
                    enc_outputs_coord_unact, memory
            return inter_states, init_reference_out, \
                inter_references_out, enc_outputs_class, \
                enc_outputs_coord_unact
        if return_encoder_output:
            return inter_states, init_reference_out, \
                inter_references_out, None, None, memory
        return inter_states, init_reference_out, \
            inter_references_out, None, None

    def forward_aux(self,
                    mlvl_feats,
                    mlvl_masks,
                    query_embed,
                    mlvl_pos_embeds,
                    pos_anchors,
                    pos_feats=None,
                    reg_branches=None,
                    cls_branches=None,
                    return_encoder_output=False,
                    attn_masks=None,
                    head_idx=0,
                    **kwargs):
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        feat_flatten = feat_flatten.permute(1, 0, 2)
        memory = feat_flatten
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        topk = pos_anchors.shape[1]
        topk_coords_unact = inverse_sigmoid((pos_anchors))
        reference_points = pos_anchors
        init_reference_out = reference_points
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](self.get_proposal_pos_embed(topk_coords_unact)))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](self.pos_feats_trans[head_idx](pos_feats))
                query_pos = query_pos + self.head_pos_embed.weight[head_idx]
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            attn_masks=attn_masks,
            aux=True,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


def build_MLP(input_dim, hidden_dim, output_dim, num_layers):
    assert num_layers > 1, \
        f'num_layers should be greater than 1 but got {num_layers}'
    h = [hidden_dim] * (num_layers - 1)
    layers = list()
    for n, k in zip([input_dim] + h[:-1], h):
        layers.extend((nn.Linear(n, k), nn.ReLU()))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DinoTransformerDecoder(DeformableDetrTransformerDecoder):

    def __init__(self, *args, **kwargs):
        super(DinoTransformerDecoder, self).__init__(*args, **kwargs)
        self._init_layers()
    def _init_layers(self):
        self.ref_point_head = build_MLP(self.embed_dims * 2, self.embed_dims,
                                        self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    @staticmethod
    def gen_sineembed_for_position(pos_tensor, pos_feat):
        scale = 2 * math.pi
        dim_t = torch.arange(
            pos_feat, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / pos_feat)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack(
                (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack(
                (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
                pos_tensor.size(-1)))
        return pos
    def be_distinct(self, ref_points, query, self_attn_mask, lid):
        num_imgs = len(ref_points)
        dis_start, num_dis = self.cache_dict['query_static']
        dis_mask = self_attn_mask[:, dis_start: dis_start + num_dis, \
                   dis_start: dis_start + num_dis]
        scores = self.cache_dict['cls_branches'][lid](
            query[:, dis_start:dis_start + num_dis]).sigmoid().max(-1).values
        proposals = ref_points[:, dis_start:dis_start + num_dis]
        proposals = bbox_cxcywh_to_xyxy(proposals)

        attn_mask_list = []
        for img_id in range(num_imgs):
            single_proposals = proposals[img_id]
            single_scores = scores[img_id]
            attn_mask = ~dis_mask[img_id * self.cache_dict['num_heads']][0]
            ori_index = attn_mask.nonzero().view(-1)
            _, keep_idxs = batched_nms(single_proposals[ori_index],
                                       single_scores[ori_index],
                                       torch.ones(len(ori_index)),
                                       self.cache_dict['cqr_cfg'])

            real_keep_index = ori_index[keep_idxs]

            attn_mask = torch.ones_like(dis_mask[0]).bool()
            attn_mask[real_keep_index] = False
            attn_mask[:, real_keep_index] = False

            attn_mask = attn_mask[None].repeat(self.cache_dict['num_heads'], 1,
                                               1)
            attn_mask_list.append(attn_mask)
        attn_mask = torch.cat(attn_mask_list)
        self_attn_mask = copy.deepcopy(self_attn_mask)
        self_attn_mask[:, dis_start: dis_start + num_dis, \
            dis_start: dis_start + num_dis] = attn_mask
        self.cache_dict['query_next'].append(~attn_mask)
        return self_attn_mask
    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                aux=None,
                **kwargs):
        output = query
        attn_masks = kwargs.pop('attn_masks',None)
        if attn_masks is None:
            attn_masks = torch.zeros(
                (query.size(0), query.size(0))).bool().cuda()
        attn_masks = attn_masks[None].repeat(query.size(1) * 8, 1, 1)
        self.cache_dict['query_next'] = []
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = self.gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.embed_dims // 2)
            query_pos = self.ref_point_head(query_sine_embed)

            query_pos = query_pos.permute(1, 0, 2)
            output = layer(
                output,
                *args,
                query_pos=query_pos,
                reference_points=reference_points_input,
                attn_masks=attn_masks,
                **kwargs)
            output = output.permute(1, 0, 2)
            if not self.training:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(reference_points,
                                                             eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                if aux == False:
                    if lid < (len(self.layers) - 1):
                        attn_masks = self.be_distinct(reference_points, output,
                                                          attn_masks, lid)
            else:
                num_crowd = self.cache_dict['num_crowd_queries']
                tmp_crowd = reg_branches[lid](output[:, :-num_crowd])
                tmp = self.aux_reg_branches[lid](output[:, -num_crowd:])
                tmp = torch.cat([tmp_crowd, tmp], dim=1)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(reference_points,
                                                             eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                if aux == False:
                    if lid < (len(self.layers) - 1):
                        attn_masks = self.be_distinct(reference_points, output,
                                                          attn_masks, lid)
            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_reference_points.append(new_reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points



# DINO的transformer
@TRANSFORMER.register_module()
class LeafDinoTransformer(LeafDeformableDetrTransformer):

    def __init__(self, *args,**kwargs):
        super(LeafDinoTransformer, self).__init__(*args, **kwargs)
    def init_layers(self):
        """Initialize layers of the DinoTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.query_embed = nn.Embedding(self.two_stage_num_proposals,
                                        self.embed_dims)
        self.query_map = nn.Linear(self.embed_dims, self.embed_dims)
    def _init_layers(self):
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(nn.Linear(self.embed_dims * 2, self.embed_dims))
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        super().init_weights()

        nn.init.xavier_uniform_(self.query_embed.weight)

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        assert self.as_two_stage and query_embed is None, \
            'as_two_stage must be True for DINO'

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)
        enc_outputs_class = cls_branches[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](
            output_memory) + output_proposals
        if self.training:
            crowd_enc_outputs_class = cls_branches[-1](
                output_memory)
            crowd_enc_outputs_coord_unact = reg_branches[-1](
                output_memory) + output_proposals

        topk = self.two_stage_num_proposals
        crowd_topk = int(topk * 1.5)
        proposals = enc_outputs_coord_unact.sigmoid()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        scores = enc_outputs_class.max(-1)[0].sigmoid()
        if self.training:
            crowd_proposals = crowd_enc_outputs_coord_unact.sigmoid()
            crowd_proposals = bbox_cxcywh_to_xyxy(crowd_proposals)
            crowd_scores = crowd_enc_outputs_class.max(-1)[0].sigmoid()
        num_imgs = len(scores)
        topk_score = []
        topk_coords_unact = []
        query = []
        crowd_topk_score = []
        crowd_topk_coords_unact = []
        crowd_topk_anchor = []
        crowd_query = []

        for img_id in range(num_imgs):
            single_proposals = proposals[img_id]
            single_scores = scores[img_id]
            _, keep_idxs = batched_nms(single_proposals, single_scores,
                                       torch.ones(len(single_scores)),
                                       self.cache_dict['cqr_cfg'])
            if self.training:
                crowd_single_proposals = crowd_proposals[img_id]
                crowd_single_scores = crowd_scores[img_id]
                _, crowd_keep_idxs = batched_nms(
                    crowd_single_proposals, crowd_single_scores,
                    torch.ones(len(crowd_single_scores)), None)

                crowd_topk_score.append(crowd_enc_outputs_class[img_id]
                                        [crowd_keep_idxs][:crowd_topk])
                crowd_topk_coords_unact.append(
                    crowd_enc_outputs_coord_unact[img_id][crowd_keep_idxs]
                    [:crowd_topk])

            topk_score.append(enc_outputs_class[img_id][keep_idxs][:topk])
            topk_coords_unact.append(
                enc_outputs_coord_unact[img_id][keep_idxs][:topk])
            map_memory = self.query_map(memory[img_id].detach())
            query.append(map_memory[keep_idxs][:topk])
            if self.training:
                crowd_query.append(map_memory[crowd_keep_idxs][:crowd_topk])
        topk_score = align_tensor(topk_score, topk)
        topk_coords_unact = align_tensor(topk_coords_unact, topk)
        query = align_tensor(query, topk)
        if self.training:
            crowd_topk_score = align_tensor(crowd_topk_score)
            crowd_topk_coords_unact = align_tensor(crowd_topk_coords_unact)
            crowd_query = align_tensor(crowd_query)
            num_crowd_queries = crowd_query.size(1)
        if self.training:
            query = torch.cat([query, crowd_query], dim=1)
            topk_coords_unact = torch.cat(
                [topk_coords_unact, crowd_topk_coords_unact], dim=1)
        topk_anchor = topk_coords_unact.sigmoid()
        if self.training:
            crowd_topk_anchor = topk_anchor[:, -num_crowd_queries:]
            topk_anchor = topk_anchor[:, :-num_crowd_queries]
        topk_coords_unact = topk_coords_unact.detach()

        if dn_label_query is not None:
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
            ori_size = attn_mask.size(-1)
            new_size = attn_mask.size(-1) + num_crowd_queries
            new_dn_mask = attn_mask.new_ones((new_size, new_size)).bool()
            crowd_mask = torch.zeros(num_crowd_queries,
                                     num_crowd_queries).bool()
            self.cache_dict['query_static'] = [dn_label_query.size(1), self.two_stage_num_proposals]
            new_dn_mask[ori_size:, ori_size:] = crowd_mask
            new_dn_mask[:ori_size, :ori_size] = attn_mask
            attn_mask = new_dn_mask
            self.cache_dict['num_crowd_queries'] = num_crowd_queries
            self.decoder.aux_reg_branches = kwargs['aux_reg_branches']
        else:
            self.cache_dict['query_static'] = [0, self.two_stage_num_proposals]
            reference_points = topk_coords_unact

        reference_points = reference_points.sigmoid()
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references= self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=attn_mask,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            aux=False,
            **kwargs)

        inter_references_out = inter_references

        return inter_states, inter_references_out, topk_score, topk_anchor, memory, crowd_topk_score, crowd_topk_anchor
    def forward_aux(self,
                    mlvl_feats,
                    mlvl_masks,
                    query_embed,
                    mlvl_pos_embeds,
                    pos_anchors,
                    pos_feats=None,
                    reg_branches=None,
                    cls_branches=None,
                    return_encoder_output=False,
                    attn_masks=None,
                    head_idx=0,
                    **kwargs):
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)
        feat_flatten = feat_flatten.permute(1, 0, 2)
        memory = feat_flatten
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        topk = pos_anchors.shape[1]
        topk_coords_unact = inverse_sigmoid((pos_anchors))
        reference_points = (pos_anchors)
        init_reference_out = reference_points
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](self.get_proposal_pos_embed(topk_coords_unact)))
            query = pos_trans_out
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](self.pos_feats_trans[head_idx](pos_feats))
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=None,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            aux=True,
            **kwargs)
        inter_references_out = inter_references

        return inter_states, inter_references_out

