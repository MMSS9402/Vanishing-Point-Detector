import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer import build_transformer
from .matchers import build_matcher
from .label_macher import build_label_matcher


class GPTran(nn.Module):
    def __init__(self, backbone, transformer, num_queries, 
                 aux_loss=False, use_structure_tensor=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        
        self.use_structure_tensor = use_structure_tensor

        hidden_dim = transformer.d_model
        self.vp1_embed = nn.Linear(hidden_dim, 3)
        self.vp2_embed = nn.Linear(hidden_dim, 3)
        self.vp3_embed = nn.Linear(hidden_dim, 3)
        self.vp1_class_embed = nn.Linear(hidden_dim, 1)
        self.vp2_class_embed = nn.Linear(hidden_dim, 1)
        self.vp3_class_embed = nn.Linear(hidden_dim, 1)
        
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        #self.line_embed = nn.Embedding(512, hidden_dim)
        line_dim = 3
        if self.use_structure_tensor:
            line_dim = 6        
        self.input_line_proj = nn.Linear(line_dim, hidden_dim)        
        self.backbone = backbone
        self.aux_loss = aux_loss   

    def forward(self, samples: NestedTensor, extra_samples):
#     def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels            
        """
        extra_info = {}

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        lines = extra_samples['lines']
        lmask = ~extra_samples['line_mask'].squeeze(2).bool()
        
        # vlines [bs, n, 3]
        if self.use_structure_tensor:
            lines = self._to_structure_tensor(lines)
                
        hs, memory, enc_attn, dec_self_attn, dec_cross_attn = (
            self.transformer(src=self.input_proj(src), mask=mask,
                             query_embed=self.query_embed.weight,
                             tgt=self.input_line_proj(lines), 
                             tgt_key_padding_mask=lmask,
                             pos_embed=pos[-1]))#,line_embed = self.line_embed.weight))
        # ha [n_dec_layer, bs, num_query, ch]
        extra_info['enc_attns'] = enc_attn
        extra_info['dec_self_attns'] = dec_self_attn
        extra_info['dec_cross_attns'] = dec_cross_attn

        outputs_vp1 = self.vp1_embed(hs[:,:,0,:]) # [n_dec_layer, bs, 3]
        outputs_vp1 = F.normalize(outputs_vp1, p=2, dim=-1)

        outputs_vp2 = self.vp2_embed(hs[:,:,1,:]) # [n_dec_layer, bs, 3]
        outputs_vp2 = F.normalize(outputs_vp2, p=2, dim=-1)

        outputs_vp3 = self.vp3_embed(hs[:,:,2,:]) # [n_dec_layer, bs, 3]
        outputs_vp3 = F.normalize(outputs_vp3, p=2, dim=-1)  

        outputs_vp1_class = self.vp1_class_embed(hs[:,:,3:,:])
        outputs_vp2_class = self.vp2_class_embed(hs[:,:,3:,:])
        outputs_vp3_class = self.vp3_class_embed(hs[:,:,3:,:])

        out = {
            'pred_vp1': outputs_vp1[-1], 
            'pred_vp2': outputs_vp2[-1],
            'pred_vp3': outputs_vp3[-1],
            'pred_vp1_logits': outputs_vp1_class[-1], # [bs, n, 1]
            'pred_vp2_logits': outputs_vp2_class[-1], # [bs, n, 1]
            'pred_vp3_logits': outputs_vp3_class[-1], # [bs, n, 1]
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_vp1, 
                                                    outputs_vp2, 
                                                    outputs_vp3, 
                                                    outputs_vp1_class,
                                                    outputs_vp2_class,
                                                    outputs_vp3_class)
        return out, extra_info

    @torch.jit.unused
    def _set_aux_loss(self, outputs_vp1, outputs_vp2, outputs_vp3, 
                      outputs_vp1_class, outputs_vp2_class,outputs_vp3_class):
        return [{'pred_vp1': a, 'pred_vp2': b, 'pred_vp3': c, 
                 'pred_vp1_logits': d, 'pred_vp2_logits': e,'pred_vp3_logits':f}
                for a, b, c, d, e, f in zip(outputs_vp1[:-1], 
                                      outputs_vp2[:-1], 
                                      outputs_vp3[:-1], 
                                      outputs_vp1_class[:-1],
                                      outputs_vp2_class[:-1],
                                      outputs_vp3_class[:-1])]

    def _to_structure_tensor(self, params):    
        (a,b,c) = torch.unbind(params, dim=-1)
        return torch.stack([a*a, a*b,
                            b*b, b*c,
                            c*c, c*a], dim=-1)
    
    def _evaluate_whls_zvp(self, weights, vlines):
        vlines = F.normalize(vlines, p=2, dim=-1)
        u, s, v = torch.svd(weights * vlines)
        return v[:, :, :, -1]
    
class SetCriterion(nn.Module):
    def __init__(self,label_matcher,matcher,weight_dict, losses, 
                       line_pos_angle, line_neg_angle):
        super().__init__()
        #self.matcher = matcher
        self.weight_dict = weight_dict        
        self.losses = losses
        self.matcher = matcher
        self.label_matcher = label_matcher
        self.thresh_line_pos = np.cos(np.radians(line_pos_angle), dtype=np.float32) # near 0.0
        self.thresh_line_neg = np.cos(np.radians(line_neg_angle), dtype=np.float32) # near 0.0
        
    
    def loss_vp1(self, outputs, targets,indices,indices2, **kwargs):
        #print(outputs.keys())
        assert 'pred_vp1' in outputs
        assert 'pred_vp2' in outputs
        assert 'pred_vp3' in outputs
        pred_vp1 = outputs['pred_vp1']
        pred_vp2 = outputs['pred_vp2']
        pred_vp3 = outputs['pred_vp3']
        pred_vp = torch.cat([torch.cat([pred_vp1.unsqueeze(1),pred_vp2.unsqueeze(1)],dim=1),pred_vp3.unsqueeze(1)],dim=1)
        tgt_vp = torch.cat([v["vp"] for v in targets])
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        cos_sim = F.cosine_similarity(pred_vp[src_idx], tgt_vp[tgt_idx], dim=-1).abs()    
        loss_vp1_cos = (1.0 - cos_sim).mean()
                
        losses = {'loss_vp1': loss_vp1_cos}
        return losses
    

    def line_label(self, outputs, targets):
        src_logits = outputs['pred_vp1_logits']
        target_lines = torch.stack([t['lines'] for t in targets], dim=0)
        target_mask = torch.stack([t['line_mask'] for t in targets], dim=0)
        target_vp1 = torch.stack([t['vp1'] for t in targets], dim=0) # [bs, 3]
        target_vp2 = torch.stack([t['vp2'] for t in targets], dim=0) # [bs, 3]
        target_vp3 = torch.stack([t['vp3'] for t in targets], dim=0) # [bs, 3]

        target_vp1 = target_vp1.unsqueeze(1)
        target_vp2 = target_vp2.unsqueeze(1)
        target_vp3 = target_vp3.unsqueeze(1)

        with torch.no_grad():
            
            cos_sim_zvp = F.cosine_similarity(target_lines, target_vp1, dim=-1).abs()
            cos_sim_hvp1 = F.cosine_similarity(target_lines, target_vp2, dim=-1).abs()
            cos_sim_hvp2 = F.cosine_similarity(target_lines, target_vp3, dim=-1).abs()
            cos_sim_zvp = cos_sim_zvp.unsqueeze(-1)
            cos_sim_hvp1 = cos_sim_hvp1.unsqueeze(-1)
            cos_sim_hvp2 = cos_sim_hvp2.unsqueeze(-1)
            
            ones = torch.ones_like(src_logits)
            zeros = torch.zeros_like(src_logits)

            cos_class_1 = torch.where(cos_sim_zvp < self.thresh_line_pos, ones, zeros)
            cos_class_2 = torch.where(cos_sim_hvp1 < self.thresh_line_pos, ones, zeros)
            cos_class_3 = torch.where(cos_sim_hvp2 < self.thresh_line_pos, ones, zeros)
            
            mask_zvp = torch.where(torch.gt(cos_class_1, self.thresh_line_pos) &
                            torch.lt(cos_class_1, self.thresh_line_neg),  
                            zeros, ones)
            mask_hvp1 = torch.where(torch.gt(cos_class_2, self.thresh_line_pos) &
                            torch.lt(cos_class_2, self.thresh_line_neg),  
                            zeros, ones)
            mask_hvp2 = torch.where(torch.gt(cos_class_3, self.thresh_line_pos) &
                            torch.lt(cos_class_3, self.thresh_line_neg),  
                            zeros, ones)
            cos_sim = torch.where(cos_class_1==1, 1, 0) + torch.where(cos_class_2==1, 2, 0) + torch.where(cos_class_3==1, 3, 0)

            overlaps = cos_sim >= 4
            
            if overlaps.any():
                values, indices = torch.stack([cos_sim_zvp, cos_sim_hvp1, cos_sim_hvp2], dim=-1)[overlaps].min(dim=-1)
            
                cos_sim[overlaps] = indices + 1

            cos_class_1 = cos_sim == 1
            cos_class_2 = cos_sim == 2
            cos_class_3 = cos_sim == 3
            
            cos_class_1 = cos_class_1.float()
            cos_class_2 = cos_class_2.float()
            cos_class_3 = cos_class_3.float()

        return cos_class_1, cos_class_2,cos_class_3,mask_zvp,mask_hvp1,mask_hvp2
    
    def loss_vp1_labels(self, outputs, targets, indices, indices2, **kwargs):
        
        src_logits1 = outputs["pred_vp1_logits"]
        src_logits2 = outputs["pred_vp2_logits"]
        src_logits3 = outputs["pred_vp3_logits"]
        src_logits = torch.cat([src_logits1.unsqueeze(1),src_logits2.unsqueeze(1),src_logits3.unsqueeze(1)],dim=1)

        target_mask = torch.stack([t["line_mask"] for t in targets], dim=0)
        target_mask = target_mask.unsqueeze(1).repeat(1,3,1,1)
        
        class_zvp,class_hvp1,class_hvp2,mask_zvp,mask_hvp1,mask_hvp2 = self.line_label(outputs,targets)
        class_vp = torch.cat([class_zvp.unsqueeze(1),class_hvp1.unsqueeze(1),class_hvp2.unsqueeze(1)],dim=1)
        mask_vp = torch.cat([mask_zvp.unsqueeze(1),mask_hvp1.unsqueeze(1),mask_hvp2.unsqueeze(1)],dim=1)
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        with torch.no_grad():

            target_classes = class_vp[tgt_idx]

            mask = target_mask[tgt_idx]*mask_vp[tgt_idx]
        
        loss_ce = F.binary_cross_entropy_with_logits(
            src_logits[src_idx], target_classes, reduction='none')
        loss_ce = mask*loss_ce
        loss_ce = loss_ce.sum(dim=1)/mask.sum(dim=1)
        
        losses = {
            "loss_vp1_ce": loss_ce.mean(),
        }
        return losses
        
    


    
 
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    
    def get_loss(self, loss, outputs, targets,indices,indices2,**kwargs):
        loss_map = {            
            'vp1': self.loss_vp1,
            # 'vp2': self.loss_vp2,
            # 'vp3': self.loss_vp3,
            'vp1_labels': self.loss_vp1_labels, 
            # 'vp2_labels': self.loss_vp2_labels,
            # 'vp3_labels': self.loss_vp3_labels
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets,indices,indices2, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)
        indices2 = self.label_matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets,indices,indices2))
            # losses.update(self.get_loss(loss, outputs, targets,indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets,indices,indices2, **kwargs)
                    # l_dict = self.get_loss(loss, aux_outputs, targets,indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg, train=True):
    device = torch.device(cfg.DEVICE)

    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)
    matcher = build_matcher(cfg)
    label_matcher = build_label_matcher(cfg)

    model = GPTran(
        backbone,
        transformer,        
        num_queries=cfg.MODELS.TRANSFORMER.NUM_QUERIES,
        aux_loss=cfg.LOSS.AUX_LOSS,
        use_structure_tensor=cfg.MODELS.USE_STRUCTURE_TENSOR,
    )
    weight_dict = dict(cfg.LOSS.WEIGHTS)
    
    # TODO this is a hack
    if cfg.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.MODELS.TRANSFORMER.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = cfg.LOSS.LOSSES    
    criterion = SetCriterion(label_matcher=label_matcher,matcher=matcher,weight_dict=weight_dict,
                             losses=losses,
                             line_pos_angle=cfg.LOSS.LINE_POS_ANGLE,
                             line_neg_angle=cfg.LOSS.LINE_NEG_ANGLE)
    criterion.to(device)    

    return model, criterion
