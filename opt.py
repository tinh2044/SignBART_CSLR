import torch
import numpy as np
from collections import defaultdict
import wandb
import math
import sys
from metrics import wer_list, bleu, rouge

from logger import MetricLogger, SmoothedValue
from utils import ctc_decode


def train_one_epoch(args, model, gloss_tokenizer, data_loader, optimizer, epoch, print_freq= 1):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Traing epoch: [{epoch}/{args.epochs}]'
    for step, (src_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        output = model(src_input)
        loss = output['total_loss']
        loss_value = loss.item()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        optimizer.step()
        model.zero_grad()
        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Averaged resluts:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate_fn(args, config, dataloader, model, tokenizer, epoch, beam_size=1, generate_cfg={}, do_translation=False,
             do_recognition=True, print_freq = 1):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Test epoch: [{epoch}/{args.epochs}]'
    print_freq = 10
    results = defaultdict(dict)

    with torch.no_grad():
        for _, (src_input) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
            output = model(src_input)
            if do_recognition:
                logits = output['gloss_logits']
                ctc_decode_output = ctc_decode(gloss_logits=logits, 
                                               beam_size=beam_size,
                                                input_lengths=output['input_lengths'])
                # # print(ctc_decode_output)
                
                batch_pred_gls = tokenizer.batch_decode(ctc_decode_output)
                # print(batch_pred_gls)
                lower_case = tokenizer.lower_case
                for name, gls_hyp, gls_ref in zip(src_input['name'], batch_pred_gls, src_input['gloss_input']):
                    gls_hyp = gls_hyp.upper() if lower_case else gls_hyp
                    # gls_hyp = ' '.join(gls_hyp).upper() if lower_case else ' '.join(gls_hyp)
                    gls_ref = gls_ref.upper() if tokenizer.lower_case \
                        else gls_ref
                    results[name]['gls_hyp'] = gls_hyp
                    results[name]['gls_ref'] = gls_ref
                    
            if do_translation:
                generate_output = model.generate_txt(
                    transformer_inputs=output['transformer_inputs'],
                    generate_cfg=generate_cfg)
                for name, txt_hyp, txt_ref in zip(src_input['name'], generate_output['decoded_sequences'],
                                                  src_input['text']):
                    results[name]['txt_hyp'], results[name]['txt_ref'] = txt_hyp, txt_ref
            # metric_logger.update(loss=output['total_loss'].item())
            metric_logger.update(loss=0.0)
            
        if do_recognition:
            evaluation_results = {}
            evaluation_results['wer'] = 200
            
            if config['data']['dataset_name'].lower() == 'phoenix-2014t':
                gls_ref = [results[n]['gls_ref'] for n in results]
                gls_hyp = [results[n]["gls_hyp"] for n in results]
            elif config['data']['dataset_name'].lower() == 'phoenix-2014':
                gls_ref = [results[n]['gls_ref'] for n in results]
                gls_hyp = [results[n]["gls_hyp"] for n in results]
            elif config['data']['dataset_name'].lower() == 'csl-daily':
                gls_ref = [results[n]['gls_ref'] for n in results]
                gls_hyp = [results[n]["gls_hyp"] for n in results]
    
            wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
            evaluation_results['wer_list'] = wer_results
            evaluation_results['wer'] = min(wer_results['wer'], evaluation_results['wer'])
            metric_logger.update(wer=evaluation_results['wer'])

        if do_translation:
            txt_ref = [results[n]['txt_ref'] for n in results]
            txt_hyp = [results[n]['txt_hyp'] for n in results]
            bleu_dict = bleu(references=txt_ref, hypotheses=txt_hyp, level=config['data']['level'])
            rouge_score = rouge(references=txt_ref, hypotheses=txt_hyp, level=config['data']['level'])
            for k, v in bleu_dict.items():
                print('{} {:.2f}'.format(k, v))
            print('ROUGE: {:.2f}'.format(rouge_score))
            evaluation_results['rouge'], evaluation_results['bleu'] = rouge_score, bleu_dict
            
            metric_logger.update(bleu1=bleu_dict['bleu1'])
            metric_logger.update(bleu2=bleu_dict['bleu2'])
            metric_logger.update(bleu3=bleu_dict['bleu3'])
            metric_logger.update(bleu4=bleu_dict['bleu4'])
            metric_logger.update(rouge=rouge_score)

    # if args.run:
    #     args.run.log(
    #         {'epoch': epoch + 1, 'epoch/dev_loss': output['recognition_loss'].item(), 'wer': evaluation_results['wer']})
    print("* Averaged resluts:", metric_logger)
    print('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}