import torch
from torch import nn
import torch.utils.checkpoint
from .encoder import Encoder
from .decoder import Decoder
from .layers import Projection
from .visual_head import VisualHead

class RecognitionNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.joint_idx = cfg['joint_idx']
        self.cross_distillation = cfg['cross_distillation']

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        
        self.visual_head = VisualHead(**cfg['visual_head'], cls_num=cfg['num_labels'])
        
        self.projection = Projection(cfg)
        self.loss_fn = nn.CTCLoss(
            blank=0,
            zero_infinity=True,
            reduction='sum'
        )

    def forward(self, src_input):
        keypoints = src_input['keypoints']
        mask = src_input['mask']
        
        
        b = keypoints.shape[0]
        keypoints = keypoints[:, :, self.joint_idx, :]
        x_embed, y_embed = self.projection(keypoints)

        encoder_outputs = self.encoder(x_embed=x_embed, attention_mask=mask)

        decoder_outputs = self.decoder(
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=mask,
            attention_mask=mask,
            y_embed=y_embed)

        
        # last_indices = (attention_mask == 1).float().cumsum(dim=1).argmax(dim=1)

        head_outputs = self.visual_head(decoder_outputs, mask)        
        
        head_outputs['last_gloss_logits'] = head_outputs['gloss_probabilities'].log()
        head_outputs['last_gloss_probabilities_log'] =  head_outputs['last_gloss_logits'].log_softmax(2)
        head_outputs['last_gloss_probabilities'] =  head_outputs['last_gloss_logits'].softmax(2)

        
        outputs = {
            **head_outputs,
            'input_lengths': src_input['length_keypoints']
            
        }
        outputs['recognition_loss'] = self.compute_loss(
            gloss_labels = src_input['gloss_labels'],
            gloss_lengths = src_input['gloss_length'],
            gloss_probabilities_log = head_outputs['last_gloss_probabilities_log'],
            input_lengths = src_input['length_keypoints']
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

