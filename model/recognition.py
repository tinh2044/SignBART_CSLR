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
from model.encoder import EncoderLayer, Encoder
from model.decoder import DecoderLayer, Decoder
class KeypointsEncoderLayer(nn.Module):
    def __init__(self, joint_idx, in_dim, out_dim, ff_dim, attention_heads, dropout=0.1):
        super().__init__()
        
        self.joint_idx = joint_idx
        
        self.coordinate_mapping = CoordinateMapping(in_dim, out_dim)
        
        self.first_norm_x = nn.LayerNorm(out_dim)
        self.first_norm_y = nn.LayerNorm(out_dim)
        
        self.self_attn_x = EncoderLayer(out_dim, attention_heads, ff_dim, dropout)
        self.self_attn_y = EncoderLayer(out_dim, attention_heads, ff_dim, dropout)
        
        self.cross_attn_x = DecoderLayer(out_dim, attention_heads, ff_dim, dropout)
        self.cross_attn_y = DecoderLayer(out_dim, attention_heads, ff_dim, dropout)
        
        self.pos_emb = StaticPositionalEncoding(out_dim)
    
        
    
    def forward(self, x_coord, y_coord, attention_mask):
        
        x_embed = self.pos_emb(x_coord)
        y_embed = self.pos_emb(y_coord)
        
        x_embed = self.first_norm_x(x_embed)
        y_embed = self.first_norm_y(y_embed)
        
        
        x_embed = self.self_attn_x(x_embed, attention_mask)
        y_embed = self.self_attn_y(y_embed, attention_mask)
    
        
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

    
        return x_embed, y_embed
    
class KeypointsEncoder(nn.Module):
    def __init__(self, joint_idx, cfg):
        super().__init__()
        
        self.coordinate_mapping = CoordinateMapping(len(joint_idx), cfg['d_model'])
        
        self.trans_encoder = Encoder(cfg)
        self.trans_decoder = Decoder(cfg)

        
        # for i, (in_dim, out_dim, ff_dim, attention_heads) in enumerate(net):
        #     self.encoder_layers.append(KeypointsEncoderLayer(joint_idx, in_dim, out_dim, ff_dim, attention_heads, dropout=i*0.05))
        
        # self.layers = nn.ModuleList(self.layers)

        # self.cross_attention = DecoderLayer(net[-1][0], attention_heads, net[-1][2])
    
    def forward(self, keypoints, attention_mask):
        
        
        x_embed, y_embed = self.coordinate_mapping(keypoints[:, :, :, 0], keypoints[:, :, :, 1])
        # for encoder_layer in self.layers:
        #     x_embed, y_embed = encoder_layer(x_embed, y_embed, attention_mask)
        
        # x = self.cross_attention(x_embed, y_embed, attention_mask)
        
        x_embed = self.trans_encoder( x_embed=x_embed, attention_mask=attention_mask)
        output = self.trans_decoder(encoder_hidden_states=x_embed, encoder_attention_mask=attention_mask, 
                               y_embed=y_embed, attention_mask=attention_mask)
        
        return output

class RecognitionNetwork(nn.Module):
    def __init__(self, cfg, gloss_tokenizer):
        super().__init__()
        self.cfg = cfg
        self.cross_distillation = cfg['cross_distillation']
        
        self.body_encoder = KeypointsEncoder(cfg['body_idx'], cfg)
        self.left_encoder = KeypointsEncoder(cfg['left_idx'], cfg)
        self.right_encoder = KeypointsEncoder(cfg['right_idx'], cfg)
        self.face_encoder = KeypointsEncoder(cfg['face_idx'], cfg)
        
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
        face_embed = self.face_encoder(keypoints[:, :, self.cfg['face_idx'], :], mask)
        
        fuse_ouput = torch.cat([left_embed, right_embed, body_embed, face_embed], dim=-1)        
        left_ouput = torch.cat([left_embed, face_embed], dim=-1)
        right_ouput = torch.cat([right_embed, face_embed], dim=-1)

        valid_len_in = src_input['valid_len_in']
        mask_head = src_input['mask_head']
        
        body_head = self.body_visual_head(body_embed, mask_head, valid_len_in)  
        left_head = self.left_visual_head(left_ouput, mask_head, valid_len_in)  
        right_head = self.right_visual_head(right_ouput, mask_head, valid_len_in)  
        fuse_head = self.fuse_visual_head(fuse_ouput, mask_head, valid_len_in)
        
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

