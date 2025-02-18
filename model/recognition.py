import torch
from torch import nn
import torch.utils.checkpoint
# from .encoder import Encoder
# from .decoder import Decoder
from model.attention import CrossAttention, SelfAttention
from model.layers import CoordinateMapping, FeedForwardLayer
from model.visual_head import VisualHead
from model.layers import StaticPositionalEncoding
from model.utils import create_attention_mask

class KeypointsEncoderLayer(nn.Module):
    def __init__(self, joint_idx, in_dim, out_dim, ff_dim, attention_heads, dropout=0.2):
        super().__init__()
        
        self.joint_idx = joint_idx
        
        if out_dim != in_dim:
            self.conv_x = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2)
            self.conv_y = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2)
            self.mask_mapping = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(2, 2))
        else:
            self.mask_mapping = nn.Identity()
            self.conv_x = nn.Identity(  )
            self.conv_y = nn.Identity()
        
        self.coordinate_mapping = CoordinateMapping(out_dim, out_dim)
        
        self.first_norm_x = nn.LayerNorm(out_dim)
        self.first_norm_y = nn.LayerNorm(out_dim)
        
        self.self_attn_x = SelfAttention(out_dim, attention_heads, dropout)
        self.self_attn_y = SelfAttention(out_dim, attention_heads, dropout)
        
        self.cross_attn_x = CrossAttention(out_dim, attention_heads, dropout)
        self.cross_attn_y = CrossAttention(out_dim, attention_heads, dropout)
        
        self.ffn_x = FeedForwardLayer(out_dim, ff_dim, dropout)
        self.ffn_y = FeedForwardLayer(out_dim, ff_dim, dropout)
        self.ffn_cross = FeedForwardLayer(out_dim, ff_dim, dropout)
        
        self.self_attn_x_layer_norm = nn.LayerNorm(out_dim)
        self.self_attn_y_layer_norm = nn.LayerNorm(out_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(out_dim)
        self.ffn_layer_norm = nn.LayerNorm(out_dim)
        
        self.pos_emb = StaticPositionalEncoding(out_dim)
        
    
    def forward(self, x_coord, y_coord, attention_mask):
        
        x_coord = self.conv_x(x_coord.permute(0, 2, 1))
        y_coord = self.conv_y(y_coord.permute(0, 2, 1))
        
        attention_mask = self.mask_mapping(attention_mask)
        
        x_embed, y_embed = self.coordinate_mapping(x_coord.permute(0, 2, 1), y_coord.permute(0, 2, 1))
    
        
        x_embed = self.pos_emb(x_embed)
        y_embed = self.pos_emb(y_embed)
        
        x_embed = self.first_norm_x(x_embed)
        y_embed = self.first_norm_y(y_embed)
        
        res_x, res_y = x_embed, y_embed
        
        x_embed = self.self_attn_x(x_embed, attention_mask)
        y_embed = self.self_attn_y(y_embed, attention_mask)
        
        x_embed = res_x + x_embed
        y_embed = res_y + y_embed
        
        x_embed = self.self_attn_x_layer_norm(x_embed)
        y_embed = self.self_attn_y_layer_norm(y_embed)
        
        x_embed = self.ffn_x(x_embed)
        y_embed = self.ffn_y(y_embed)
        
        x_embed = self.cross_attn_x(x_embed, y_embed, attention_mask)
        y_embed = self.cross_attn_y(y_embed, x_embed, attention_mask)

        if x_embed.dtype == torch.float16 and (
            torch.isinf(x_embed).any() or torch.isnan(x_embed).any()
        ):
            clamp_value = torch.finfo(x_embed.dtype).max - 1000
            x_embed = torch.clamp(x_embed, min=-clamp_value, max=clamp_value)
            
        if y_embed.dtype == torch.float16 and (
            torch.isinf(y_embed).any() or torch.isnan(y_embed).any()
        ):
            clamp_value = torch.finfo(y_embed.dtype).max - 1000
            y_embed = torch.clamp(y_embed, min=-clamp_value, max=clamp_value)

    
        return x_embed, y_embed, attention_mask
    
class KeypointsEncoder(nn.Module):
    def __init__(self, joint_idx, net):
        super().__init__()
        
        self.coordinate_mapping = CoordinateMapping(len(joint_idx), net[0][0])
        
        self.layers = []
        
        for i, (in_dim, out_dim, ff_dim, attention_heads) in enumerate(net):
            self.layers.append(KeypointsEncoderLayer(joint_idx, in_dim, out_dim, ff_dim, attention_heads, dropout=i*0.05))
        
        self.layers = nn.ModuleList(self.layers)

        self.cross_attention = CrossAttention(net[-1][0], attention_heads, dropout=0.2)
    
    def forward(self, keypoints, attention_mask, return_mask=False):
        
        attention_mask = create_attention_mask(attention_mask, keypoints.dtype)
        
        x_embed, y_embed = self.coordinate_mapping(keypoints[:, :, :, 0], keypoints[:, :, :, 1])
        for encoder_layer in self.layers:
            x_embed, y_embed, attention_mask = encoder_layer(x_embed, y_embed, attention_mask)
        
        x = self.cross_attention(x_embed, y_embed, attention_mask)
        if return_mask:
            return x, attention_mask
        else:
            return x

class RecognitionNetwork(nn.Module):
    def __init__(self, cfg, gloss_tokenizer):
        super().__init__()
        self.cfg = cfg
        self.cross_distillation = cfg['cross_distillation']
        
        self.body_encoder = KeypointsEncoder(cfg['body_idx'], cfg['net'])
        self.left_encoder = KeypointsEncoder(cfg['left_idx'], cfg['net'])
        self.right_encoder = KeypointsEncoder(cfg['right_idx'], cfg['net'])
        self.face_encoder = KeypointsEncoder(cfg['face_idx'], cfg['net'])
        
        self.body_visual_head = VisualHead(**cfg['body_visual_head'], cls_num=len(gloss_tokenizer))
        self.left_visual_head = VisualHead(**cfg['left_visual_head'], cls_num=len(gloss_tokenizer))
        self.right_visual_head = VisualHead(**cfg['right_visual_head'], cls_num=len(gloss_tokenizer))
        self.fuse_visual_head = VisualHead(**cfg['fuse_visual_head'], cls_num=len(gloss_tokenizer))
        
        self.loss_fn = nn.CTCLoss(
            blank=0,
            zero_infinity=True,
            reduction='sum'
        )

    def forward(self, src_input):
        keypoints = src_input['keypoints']
        mask = src_input['mask']
        
        body_embed = self.body_encoder(keypoints[:, :, self.cfg['body_idx'], :], mask)
        left_embed = self.left_encoder(keypoints[:, :, self.cfg['left_idx'], :], mask)
        right_embed = self.right_encoder(keypoints[:, :, self.cfg['right_idx'], :], mask)
        face_embed, mask = self.face_encoder(keypoints[:, :, self.cfg['face_idx'], :], mask,     return_mask=True)
        
        fuse_ouput = torch.cat([body_embed, left_embed, right_embed, face_embed], dim=-1)        
        left_ouput = torch.cat([left_embed, face_embed], dim=-1)
        right_ouput = torch.cat([right_embed, face_embed], dim=-1)

        valid_len_in = src_input['valid_len_in']
        
        body_head = self.body_visual_head(body_embed, mask, valid_len_in)  
        left_head = self.left_visual_head(left_ouput, mask, valid_len_in)  
        right_head = self.right_visual_head(right_ouput, mask, valid_len_in)  
        fuse_head = self.fuse_visual_head(fuse_ouput, mask, valid_len_in)
        
        head_outputs = {'ensemble_last_gloss_logits': (left_head['gloss_probabilities'] + right_head['gloss_probabilities'] +
                                                           body_head['gloss_probabilities']+fuse_head['gloss_probabilities']).log(),
                            'fuse': fuse_ouput,
                            'fuse_gloss_logits': fuse_head['gloss_logits'],
                            'fuse_gloss_probabilities_log': fuse_head['gloss_probabilities_log'],
                            'body_gloss_logits': body_head['gloss_logits'],
                            'body_gloss_probabilities_log': body_head['gloss_probabilities_log'],
                            'left_gloss_logits': left_head['gloss_logits'],
                            'left_gloss_probabilities_log': left_head['gloss_probabilities_log'],
                            'right_gloss_logits': right_head['gloss_logits'],
                            'right_gloss_probabilities_log': right_head['gloss_probabilities_log'],
                            }

        head_outputs['ensemble_last_gloss_probabilities_log'] = head_outputs['ensemble_last_gloss_logits'].log_softmax(2)
        head_outputs['ensemble_last_gloss_probabilities'] = head_outputs['ensemble_last_gloss_logits'].softmax(2)
        self.cfg['gloss_feature_ensemble'] = self.cfg.get('gloss_feature_ensemble', 'gloss_feature')
        head_outputs['gloss_feature'] = fuse_head[self.cfg['gloss_feature_ensemble']]
        outputs = {**head_outputs,
                   'input_lengths':src_input['valid_len_in']}

        for k in ['left', 'right', 'fuse', 'body']:
            outputs[f'recognition_loss_{k}'] = self.compute_loss(
                gloss_labels=src_input['gloss_labels'],
                gloss_lengths=src_input['gloss_lengths'],
                gloss_probabilities_log=head_outputs[f'{k}_gloss_probabilities_log'],
                input_lengths=src_input['valid_len_in'])
        outputs['recognition_loss'] = outputs['recognition_loss_left'] + outputs['recognition_loss_right'] + \
                                      outputs['recognition_loss_fuse'] + outputs['recognition_loss_body']
        if 'cross_distillation' in self.cfg:
            loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            for student in ['left', 'right', 'fuse', 'body']:
                teacher_prob = outputs['ensemble_last_gloss_probabilities']
                teacher_prob = teacher_prob.detach()
                student_log_prob = outputs[f'{student}_gloss_probabilities_log']
                outputs[f'{student}_distill_loss'] = loss_func(input=student_log_prob, target=teacher_prob)
                outputs['recognition_loss'] += outputs[f'{student}_distill_loss']
        return outputs

        # return outputs

    def compute_loss(self, gloss_labels, gloss_lengths, gloss_probabilities_log, input_lengths):
        loss = self.loss_fn(
            log_probs = gloss_probabilities_log.permute(1,0,2),
            targets = gloss_labels,
            input_lengths = input_lengths,
            target_lengths = gloss_lengths
        )
        loss = loss/gloss_probabilities_log.shape[0]
        return loss


if __name__ == "__main__":
    import yaml
    with open("./configs/phoenix-2014t.yaml") as f:
        config = yaml.safe_load(f)
        
    model = RecognitionNetwork(config['model'])
    
    keypoints = torch.randn(2, 20, 75, 2)
    attention_mask = torch.ones(2, 20)
    labels = torch.randint(0, 10, (2, 20))
    logits = model(keypoints, attention_mask, labels)
    print(logits)

