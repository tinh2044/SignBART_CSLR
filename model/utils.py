import torch

def create_attention_mask(mask, dtype, tgt_len = None):

        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
    
    
def create_causal_attention_mask(attention_mask, input_shape, inputs_embeds):
    
    batch_size, query_length = input_shape[0], input_shape[1]

    expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, query_length, query_length).to(
        dtype=inputs_embeds.dtype
    )
    inverted_mask = 1.0 - expanded_mask
    expanded_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(inputs_embeds.dtype).min)
    
    causal_mask = torch.tril(torch.ones((query_length, query_length), device=inputs_embeds.device, dtype=inputs_embeds.dtype))
    expanded_mask += causal_mask[None, None, :, :]

    return expanded_mask