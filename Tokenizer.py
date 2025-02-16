import pickle
import torch, pickle, json
from collections import defaultdict
from transformers import MBartTokenizer


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, ignore_index: int=-100):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)
    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    for ii,ind in enumerate(index_of_eos.squeeze(-1)):
        input_ids[ii, ind:] = ignore_index
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens


class GlossTokenizer:
    def __init__(self, tokenizer_cfg):
        with open(tokenizer_cfg['gloss2id_file'], 'r') as f:
            self.gloss2id = json.load(f)
        self.gloss2id = defaultdict(lambda: self.gloss2id['<unk>'], self.gloss2id)
        
        self.id2gloss = {v:k for k,v in self.gloss2id.items()}
        self.id2gloss = defaultdict(lambda: self.id2gloss['<unk>'], self.id2gloss)
        
        self.lower_case = False
        self.split = tokenizer_cfg.get('split', ' ')
        
        if '<s>' in self.gloss2id:
            self.start_token = '<s>'
            self.start_id = self.gloss2id[self.start_token]
            
        if "<pad>" in self.gloss2id:
            self.pad_token = '<pad>'
            self.pad_id = self.gloss2id[self.pad_token]
        else:
            raise ValueError("pad token not in gloss2id")

    def encode(self, _input, max_len=None, has_split=True, return_length=False):
        if not has_split:
            _input = _input.split(self.split)
        attention_mask = torch.ones(len(_input), dtype=torch.long)
        inputs_ids = torch.tensor([self.gloss2id[gls.lower() if self.lower_case else gls] for gls in _input])
        if max_len is not None:
            attention_mask = torch.concat(
                (attention_mask, torch.zeros(max_len - len(_input), dtype=torch.long)),
                dim=0)

            inputs_ids = torch.concat(
                (inputs_ids, torch.ones(max_len - len(_input), dtype=torch.long) * self.pad_id),
                dim=0)
        
        return {
            'input_ids': inputs_ids,
            'attention_mask': attention_mask,
            'length': len(_input)
            }
        
    def batch_encode(self, batch, max_len=None, return_length=False):
        batch = [x.split(self.split) for x in batch]
        if max_len is None:
            max_len = max([len(x) for x in batch])
        
        input_ids, attention_mask = [], []
        
        if return_length:
            lengths = []
        
        for seq in batch:
            output = self.encode(seq, max_len, return_length=return_length)
            input_ids.append(output['input_ids'])
            attention_mask.append(output['attention_mask'])
            
            if return_length:
                lengths.append(output['length'])
        
        
        
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        if return_length:
            lengths = torch.tensor(lengths)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'length': lengths
            }
    
    def decode(self, _input, skip_special_tokens=True):
        if type(_input) is dict:
            tokens = _input['input_ids']
        if type(_input) is list:
            tokens = torch.tensor(_input)
        else:
            tokens = _input
            
        if skip_special_tokens:
            tokens = tokens[tokens != self.pad_id]
        
        return " ".join([self.id2gloss[int(x)] for x in tokens])
        
    def batch_decode(self, batch, skip_special_tokens=True):
        return [self.decode(x, skip_special_tokens) for x in batch]
    
    def __call__(self, batch):
        
        batch = [x.split(" ") for x in batch]
        max_len = max([len(x) for x in batch])
        
        gloss_labels = [], []
        
        for i, seq in enumerate(batch):
            gloss_label = [self.gloss2id[gls.lower() if self.lower_case else gls] for gls in seq]
            gls_ids = gls_ids + (max_len - len(gls_ids)) * [self.pad_id]
            gloss_label.append(gls_ids)

        gloss_labels = torch.tensor(gloss_labels)
        return {'gloss_labels': gloss_labels}

    def __len__(self):
        
        return len(self.id2gloss)
    
if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    cgf = {
        "gloss2id_file": "./data/phoenix-2014-T/gloss2id.json",
        "split": " "
    }
    tokenizer = GlossTokenizer(cgf)
    
    df = pd.read_csv("./data/phoenix-2014-T/annotations/all.csv")
    
    sentences = df['orth'].tolist()
    for s in tqdm(sentences):
        oputput = tokenizer.encode(s, has_split=False)
        decode = tokenizer.decode(oputput)
        assert s == decode, (f"{s} != {decode}")