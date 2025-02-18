import random

import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from PIL import Image
import os
from Tokenizer import GlossTokenizer
import numpy as np
import pickle
from augmentation import rotate_keypoints, flip_keypoints


class SLR_Dataset(Dataset.Dataset):
    def __init__(self, root, cfg, split, gloss_tokenizer=None, text_tokenizer=None):
        self.cfg = cfg
        self.max_len = cfg['max_len']
        self.split = split
        self.normalize = cfg['normalize']
        self.joint_parts = cfg['joint_parts']
        self.task = cfg['task']
        assert self.task in ["S2T", "S2G"], f"Task {self.task} not supported"
        
        _dir = f"{root}/{split}"
        assert os.path.exists(_dir), f"Path {_dir} does not exist"
        
        self.paths = [f"{_dir}/{x}" for x in os.listdir(f"{_dir}")]
        
        if cfg['shuffle'] is True:
            random.shuffle(self.paths)
        
        
        self.gloss_tokenizer = gloss_tokenizer
        if self.task == "S2T":
            self.text_tokenizer = text_tokenizer
        
        if self.split == "train":
            self.min_rate = 0.5
            self.max_rate = 1.5
        else:
            self.min_rate = 1.0
            self.max_rate = 1.0        
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        
        with open(path, "rb") as f:
            sample = pickle.load(f)
        
        keypoints = sample['keypoints'][:, :, :-2]
        gloss = sample['gloss']
        text = sample['text']
        name = sample['name']
        
        return keypoints, gloss, text, name
    
    def data_collator(self, batch):
        keypoints_batch, length_keypoints_batch, gloss_batch, text_batch, name_batch = [], [], [], [], []
        
        for keypoints, gloss, text, name in batch:
            
            keypoints = self.preprocess_keypoints(keypoints)
            keypoints = torch.from_numpy(keypoints).float()
            
            length_keypoints_batch.append(keypoints.shape[0])
            keypoints_batch.append(keypoints)
            gloss_batch.append(gloss)
            text_batch.append(text)
            name_batch.append(name)
        
        padding_keypoints_batch = []
        max_len = max(length_keypoints_batch)
        
        for keypoints, length_keypoints in zip(keypoints_batch, length_keypoints_batch):
            if length_keypoints < max_len:
                padding = torch.zeros((max_len - length_keypoints, keypoints.shape[1], keypoints.shape[2]))
                keypoints = torch.cat((keypoints, padding), dim=0)
                padding_keypoints_batch.append(keypoints)
            else:
                padding_keypoints_batch.append(keypoints)
                            
        keypoints_batch = torch.stack(padding_keypoints_batch)
        length_keypoints_batch = torch.tensor(length_keypoints_batch)
        
        attention_mask = torch.zeros((keypoints_batch.shape[0], max_len), dtype=torch.long)
        for i in range(keypoints_batch.shape[0]):
            attention_mask[i, :length_keypoints_batch[i]] = 1

        gloss_output = self.gloss_tokenizer.batch_encode(gloss_batch, return_length=True)
        
        new_src_lengths = (((length_keypoints_batch - 1) / 2)).long()
        new_src_lengths = (((new_src_lengths - 1) / 2)).long()
        
        src_input = {
            "name": name_batch,
            "keypoints": keypoints_batch,
            "valid_len_in": new_src_lengths,
            "mask": attention_mask,
            
            "gloss_labels": gloss_output['input_ids'],
            "gloss_lengths": gloss_output['length'],
            "gloss_input": gloss_batch,
            "text_input": text_batch
        }

        if self.task == "S2T":
            text_output = self.text_tokenizer.encode_batch(text_batch, return_length=True)
            src_input['translation_inputs'] = {**text_output}
            src_input['translation_inputs']['text'] = text_batch
            src_input['translation_inputs']['gloss_ids'] = gloss_output['input_ids']
            src_input['translation_inputs']['gloss_length'] = gloss_output['length']
        
        return src_input
    
    def preprocess_keypoints(self, keypoints):
        
        keypoints = self.select_frames(keypoints)
        
        if self.split == "train" and np.random.rand() < 0.5:
            keypoints = self.apply_augmentation(keypoints)
        if self.normalize:
            keypoints = self.normalize_keypoints(keypoints)

        return keypoints
    
    def normalize_keypoints(self, keypoints):
        for i in range(keypoints.shape[0]):
            for part in self.joint_parts:
                keypoints[i, part, :] = self.normalize_part(keypoints[i, part, :])
            
        return keypoints
        
    def normalize_part(self, keypoint):   
        assert keypoint.shape[-1] == 2, "Keypoints must have x, y"
        x_coords = keypoint[:, 0]
        y_coords = keypoint[:, 1]
        min_x, min_y = min(x_coords), min(y_coords)
        max_x, max_y = max(x_coords), max(y_coords)
        w = max_x - min_x
        h = max_y - min_y

        if w > h:
            delta_x = 0.05 * w
            delta_y = delta_x + ((w - h) / 2)
        else:
            delta_y = 0.05 * h
            delta_x = delta_y + ((h - w) / 2)

        s_point = [max(0, min(min_x - delta_x, 1)), max(0, min(min_y - delta_y, 1))]
        e_point = [max(0, min(max_x + delta_x, 1)), max(0, min(max_y + delta_y, 1))]

        assert s_point[0] >= 0, f"Starting point[0] < 0 is {s_point[0]}"
        assert s_point[1] >= 0, f"Starting point[1] < 0 is {s_point[1]}"
        assert e_point[0] >= 0, f"Ending_point point[0] < 0 is {e_point[0]}"
        assert e_point[1] >= 0, f"Ending_point point[1] < 0 is {e_point[1]}"

        if (e_point[0] - s_point[0]) != 0.0:
            keypoint[:, 0] = (keypoint[:, 0] - s_point[0]) / (e_point[0] - s_point[0])
        if (e_point[1] - s_point[1]):
            keypoint[:, 1] = (keypoint[:, 1] - s_point[1]) / (e_point[1] - s_point[1])

        return keypoint
         
    def apply_augmentation(self, keypoints):
        aug = False
        while not aug:
            if np.random.uniform(0, 1) < 0.5:
                angle = np.random.uniform(-15, 15)
                keypoints = rotate_keypoints(keypoints, (0, 0), angle)
                aug = True
            if np.random.uniform(0, 1) < 0.5:
                keypoints = flip_keypoints(keypoints)
                aug = True
        
        return keypoints
    def select_frames(self, keypoints):
        n_frames = keypoints.shape[0]
        if self.split != "train":
            if n_frames <= self.max_len:
                frames_idx = np.arange(n_frames)
            else:
                frames_idx = np.arange(n_frames)
                f_s = (n_frames - self.max_len) // 2
                f_e = n_frames - self.max_len - f_s
                
                frames_idx = frames_idx[f_s:-f_e]
            
            assert len(frames_idx) <= self.max_len, f"len(frames_idx) = {len(frames_idx)} > {self.max_len}"
            return keypoints[frames_idx]
        else:
            min_len = min(int(self.min_rate * n_frames), self.max_len)
            max_len = min(int(self.max_rate * n_frames), self.max_len)
            tgt_len = random.randint(min_len, max_len+1)
            
            if tgt_len <= n_frames:
                frames_idx = np.arange(n_frames)
                frames_idx = sorted(np.random.permutation(frames_idx)[:tgt_len])
            else:
                copy_idx = np.random.randint(0, n_frames, tgt_len - n_frames)
                frames_idx = np.arange(n_frames)
                frames_idx = sorted(np.concatenate([frames_idx, copy_idx]))

            assert len(frames_idx) == tgt_len, f"len(frames_idx) = {len(frames_idx)} != {tgt_len}"
            return keypoints[frames_idx]