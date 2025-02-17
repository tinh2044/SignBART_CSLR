import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import time
import argparse, json, datetime
import numpy as np
import yaml
import random
from pathlib import Path
from loguru import logger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from optimizer import build_optimizer, build_scheduler
from Tokenizer import GlossTokenizer
from dataset import SLR_Dataset
from model import SignLanguageModel
from opt import train_one_epoch, evaluate_fn
import utils

def get_args_parser():
    parser = argparse.ArgumentParser('Distributed Training for Sign Language Recognition', add_help=False)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--finetune', default='', help='Finetune from checkpoint')
    parser.add_argument('--world_size', default=2, type=int, help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--device', default='cuda', help='Device to use for training/testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--test_on_last_epoch', action='store_true', help='Perform evaluation on last epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cfg_path', type=str, required=True, help='Path to config file')
    parser.add_argument('--log_all', action='store_true', help='Log in all processes, otherwise only in rank 0')
    parser.add_argument('--print_freq', default=10, type=int, help='Print frequency')
    return parser

def init_distributed(args):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method=args.dist_url)
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    args.is_main = args.rank == 0
    if args.is_main:
        logger.info(f"Using {args.world_size} GPUs for training")

def main(args, cfg):
    init_distributed(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    
    if args.is_main:
        print(f"Creating dataset:")
    cfg_data = cfg['data']
    gloss_tokenizer = GlossTokenizer(cfg['gloss_tokenizer'])
    train_data = SLR_Dataset(root=cfg_data['root'], gloss_tokenizer=gloss_tokenizer, cfg=cfg_data, split='train')
    dev_data = SLR_Dataset(root=cfg_data['root'], gloss_tokenizer=gloss_tokenizer, cfg=cfg_data, split='dev')
    test_data = SLR_Dataset(root=cfg_data['root'], gloss_tokenizer=gloss_tokenizer, cfg=cfg_data, split='test')
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                  collate_fn=train_data.data_collator, shuffle=False,
                                  pin_memory=True, drop_last=True, sampler=train_sampler)
    
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                collate_fn=dev_data.data_collator, pin_memory=True)
    
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                 collate_fn=test_data.data_collator, pin_memory=True)
    
    if args.is_main:
        print(f"Creating model:")
    model = SignLanguageModel(cfg=cfg, gloss_tokenizer=gloss_tokenizer).cuda(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    n_parameters = utils.count_model_parameters(model)
    # print(model)
    print(f"Number of parameters: {n_parameters}")
    
    optimizer = build_optimizer(cfg['training']['optimization'], model)
    scheduler, _ = build_scheduler(cfg['training']['optimization'], optimizer)
    
    output_dir = Path(cfg['training']['model_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
    
    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        dev_stats = evaluate_fn(args, config, dev_dataloader, model, gloss_tokenizer, epoch=0, beam_size=5,
                              generate_cfg=config['training']['validation']['translation'],
                              do_translation=config['do_translation'], do_recognition=config['do_recognition'],
                              print_freq=args.print_freq)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate_fn(args, config, test_dataloader, model, gloss_tokenizer, epoch=0, beam_size=5,
                              generate_cfg=config['testing']['translation'],
                              do_translation=config['do_translation'], do_recognition=config['do_recognition'],
                              print_freq=args.print_freq)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        return

    
    if args.is_main:
        print(f"Starting training for {args.epochs} epochs")
    
    print(f"Trainining on {device}")
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = 200
    bleu_4 = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        train_results = train_one_epoch(args, model, train_dataloader, optimizer, epoch, print_freq=args.print_freq)
        scheduler.step()
        checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
        # for checkpoint_path in checkpoint_paths:
        #    utils.save_on_master({
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'scheduler': scheduler.state_dict(),
        #         'epoch': epoch,
        #     }, checkpoint_path)
        print()
        test_results = evaluate_fn(args, config, dev_dataloader, model, gloss_tokenizer, epoch,
                              beam_size=config['training']['validation']['recognition']['beam_size'],
                              generate_cfg=config['training']['validation']['translation'],
                              do_translation=config['do_translation'], do_recognition=config['do_recognition'], print_freq=args.print_freq)
        if config['task'] == "S2T":
            if bleu_4 < test_results["bleu4"]:
                bleu_4 = test_results["bleu4"]
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
            }, checkpoint_path)

            print(f"* DEV BLEU-4 {test_results['bleu4']:.3f} Max DEV BLEU-4 {bleu_4}")
        else:
            if min_loss > test_results["wer"]:
                min_loss = test_results["wer"]
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch,
                        }, checkpoint_path)
            print(f"* DEV wer {test_results['wer']:.3f} Min DEV WER {min_loss}")

        log_results = {**{f'train_{k}': v for k, v in train_results.items()},
                     **{f'test_{k}': v for k, v in test_results.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        print()
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_results) + "\n")


    if args.test_on_last_epoch:
        checkpoint = torch.load(str(output_dir) + '/best_checkpoint.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        dev_stats = evaluate_fn(args, config, dev_dataloader, model, gloss_tokenizer, epoch=0, beam_size=config['testing']['recognition']['beam_size'],
                             generate_cfg=config['training']['validation']['translation'],
                             do_translation=config['do_translation'], do_recognition=config['do_recognition'], print_freq=args.print_freq)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")
        test_stats = evaluate_fn(args, config, test_dataloader, model, gloss_tokenizer, epoch=0, beam_size=config['testing']['recognition']['beam_size'],
                              generate_cfg=config['testing']['translation'],
                              do_translation=config['do_translation'], do_recognition=config['do_recognition'], print_freq=args.print_freq)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        if config['do_recognition']:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps({'Dev WER:': dev_stats['wer'],
                                    'Test WER:': test_stats['wer']}) + "\n")
        if config['do_translation']:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps({'Dev Bleu-4:': dev_stats['bleu4'],
                                    'Test Bleu-4:': test_stats['bleu4']}) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('Sign Language Recognition Training', parents=[get_args_parser()])
    args = parser.parse_args()
    with open(args.cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    main(args, config)
