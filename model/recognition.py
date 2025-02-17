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
        self.coordinate_mapping = CoordinateMapping(in_dim, out_dim)
        
        # self.first_norm_x = nn.LayerNorm(out_dim)
        # self.first_norm_y = nn.LayerNorm(out_dim)
        
        # self.self_attn_x = SelfAttention(out_dim, attention_heads, dropout)
        # self.self_attn_y = SelfAttention(out_dim, attention_heads, dropout)
        
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
        x_embed, y_embed = self.coordinate_mapping(x_coord, y_coord)
        
        x_embed = self.pos_emb(x_embed)
        y_embed = self.pos_emb(y_embed)
        
        # x_embed = self.first_norm_x(x_embed)
        # y_embed = self.first_norm_y(y_embed)
        
        # res_x, res_y = x_embed, y_embed
        
        # x_embed = self.self_attn_x(x_embed, attention_mask)
        # y_embed = self.self_attn_y(y_embed, attention_mask)
        
        # x_embed = res_x + x_embed
        # y_embed = res_y + y_embed
        
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

    
        return x_embed, y_embed
    
class KeypointsEncoder(nn.Module):
    def __init__(self, joint_idx, net):
        super().__init__()
        
        self.coordinate_mapping = CoordinateMapping(len(joint_idx), net[0][0])
        
        self.layers = []
        
        for i, (in_dim, out_dim, ff_dim, attention_heads) in enumerate(net):
            self.layers.append(KeypointsEncoderLayer(joint_idx, in_dim, out_dim, ff_dim, attention_heads, dropout=i*0.05))
        
        self.layers = nn.ModuleList(self.layers)

        self.cross_attention = CrossAttention(net[-1][0], attention_heads, dropout=0.2)
    
    def forward(self, keypoints, attention_mask):
        
        attention_mask = create_attention_mask(attention_mask, keypoints.dtype)
        
        x_embed, y_embed = self.coordinate_mapping(keypoints[:, :, :, 0], keypoints[:, :, :, 1])
        for encoder_layer in self.layers:
            x_embed, y_embed = encoder_layer(x_embed, y_embed, attention_mask)
        
        x = self.cross_attention(x_embed, y_embed, attention_mask)
        
        return x

class RecognitionNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cross_distillation = cfg['cross_distillation']
    
        self.body_encoder = KeypointsEncoder(cfg['body_idx'], cfg['net'])
        self.left_encoder = KeypointsEncoder(cfg['left_idx'], cfg['net'])
        self.right_encoder = KeypointsEncoder(cfg['right_idx'], cfg['net'])
        self.face_encoder = KeypointsEncoder(cfg['face_idx'], cfg['net'])
        
        self.visual_head = VisualHead(**cfg['visual_head'], cls_num=cfg['num_labels'])
        
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
        rigt_embed = self.right_encoder(keypoints[:, :, self.cfg['right_idx'], :], mask)
        face_embed = self.face_encoder(keypoints[:, :, self.cfg['face_idx'], :], mask)
        fuse = torch.cat([body_embed, left_embed, rigt_embed, face_embed], dim=-1)        
        
        # decoder_outputs = self.visual_head(keypoints, mask)

        valid_len_in = src_input['valid_len_in']
        
        head_outputs = self.visual_head(fuse, mask, valid_len_in=valid_len_in)        
        
        head_outputs['last_gloss_logits'] = head_outputs['gloss_probabilities'].log()
        head_outputs['last_gloss_probabilities_log'] =  head_outputs['last_gloss_logits'].log_softmax(2)
        head_outputs['last_gloss_probabilities'] =  head_outputs['last_gloss_logits'].softmax(2)

        
        outputs = {
            **head_outputs,
            'valid_len_in': src_input['valid_len_in']
            
        }
        outputs['recognition_loss'] = self.compute_loss(
            gloss_labels = src_input['gloss_labels'],
            gloss_lengths = src_input['gloss_length'],
            gloss_probabilities_log = head_outputs['last_gloss_probabilities_log'],
            input_lengths = src_input['valid_len_in']
        )
        if self.cross_distillation:
            loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            
            teacher_prob = outputs['last_gloss_probabilities']
            teacher_prob = teacher_prob.detach()
            student_log_prob = outputs['gloss_probabilities_log']
            outputs['distill_loss'] = loss_func(input=student_log_prob, target=teacher_prob)
            outputs['recognition_loss'] += outputs['distill_loss']
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

