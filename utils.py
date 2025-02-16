import os

import numpy as np
import torch
from tqdm import tqdm
import tensorflow as tf
from itertools import groupby


total_body_idx = 33
total_hand = 42

body_idx = list(range(11, 17))
lefthand_idx = [x + total_body_idx for x in range(0, 21)]

righthand_idx = [x + 21 for x in lefthand_idx]

total_idx = body_idx + lefthand_idx + righthand_idx

def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def top_k_accuracy(logits, labels, k=5):
    top_k_preds = torch.topk(logits, k, dim=1).indices
    correct = (top_k_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
    total = labels.size(0)
    return correct / total


def save_checkpoints(model, optimizer, path_dir, epoch, name=None):
    if not os.path.exists(path_dir):
        print(f"Making directory {path_dir}")
        os.makedirs(path_dir)
    if name is None:
        filename = f'{path_dir}/checkpoints_{epoch}.pth'
    else:
        filename = f'{path_dir}/checkpoints_{epoch}_{name}.pth'
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, filename)


def load_checkpoints(model, optimizer, path, resume=True):
    if not os.path.exists(path):
        raise FileNotFoundError
    if os.path.isdir(path):
        epoch = max([int(x[x.index("_") + 1:len(x) - 4]) for x in os.listdir(path)])
        filename = f'{path}/checkpoints_{epoch}.pth'
        print(f'Loaded latest checkpoint: {epoch}')

        checkpoints = torch.load(filename)

    else:
        print(f"Load checkpoint from file : {path}")
        checkpoints = torch.load(path)

    model.load_state_dict(checkpoints['model'])
    optimizer.load_state_dict(checkpoints['optimizer'])
    if resume:
        return checkpoints['epoch'] + 1
    else:
        return 1


def train_epoch(model, dataloader, optimizer, scheduler=None, epoch=0, epochs=0):
    all_loss, all_acc, all_top_5_acc = 0.0, 0.0, 0.0
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True, desc=f"Training epoch {epoch + 1}/{epochs}: ")
    for i, data in loop:
        labels = data["labels"]
        optimizer.zero_grad()
        loss, logits = model(**data)
        loss.backward()
        optimizer.step()
        all_loss += loss.item()

        acc = accuracy(logits, labels)
        top_5_acc = top_k_accuracy(logits, labels, k=5)

        all_acc += acc
        all_top_5_acc += top_5_acc

        loop.set_postfix_str(f"Loss: {loss.item():.3f}, Acc: {acc:.3f}, Top 5 Acc: {top_5_acc:.3f}")

    if scheduler:
        scheduler.step(loss)

    all_loss /= len(dataloader)
    all_acc /= len(dataloader)
    all_top_5_acc /= len(dataloader)

    return all_loss, all_acc, all_top_5_acc


def evaluate(model, dataloader, epoch=0, epochs=0):
    all_loss, all_acc, all_top_5_acc = 0.0, 0.0, 0.0
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True,
                desc=f"Evaluation epoch {epoch + 1}/{epochs}: ")

    for i, data in loop:
        labels = data["labels"]
        loss, logits = model(**data)
    
        all_loss += loss.item()
        acc = accuracy(logits, labels)
        top_5_acc = top_k_accuracy(logits, labels, k=5)

        all_acc += acc
        all_top_5_acc += top_5_acc

        loop.set_postfix_str(f"Loss: {loss.item():.3f}, Acc: {acc:.3f}, Top 5 Acc: {top_5_acc:.3f}")

    all_loss /= len(dataloader)
    all_acc /= len(dataloader)
    all_top_5_acc /= len(dataloader)

    return all_loss, all_acc, all_top_5_acc


def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def ctc_decode(gloss_logits, beam_size, input_lengths):
        gloss_logits = gloss_logits.permute(1, 0, 2)
        gloss_logits = gloss_logits.cpu().detach().numpy()
        tf_gloss_logits = np.concatenate(
            (gloss_logits[:, :, 1:], gloss_logits[:, :, 0, None]),
            axis=-1,
        )
        ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
            inputs=tf_gloss_logits,
            sequence_length=input_lengths.cpu().detach().numpy(),
            beam_width=beam_size,
            top_paths=1,
        )
        ctc_decode = ctc_decode[0]
        tmp_gloss_sequences = [[] for i in range(input_lengths.shape[0])]
        for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
            tmp_gloss_sequences[dense_idx[0]].append(
                ctc_decode.values[value_idx].numpy() + 1
            )
        decoded_gloss_sequences = []
        for seq_idx in range(0, len(tmp_gloss_sequences)):
            decoded_gloss_sequences.append(
                [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
            )
        return decoded_gloss_sequences
    
if __name__ == "__main__":
    print(body_idx)
    print(lefthand_idx)
    print(righthand_idx)