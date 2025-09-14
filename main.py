import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple
import logging
import pandas as pd

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc
from lora import Linear, MergedLinear, ConvLoRA, lora_state_dict
from segment_anything_training import sam_model_registry
from segment_anything_training.modeling.transformer import Attention
from segment_anything_training.modeling.image_encoder import Attention as EncoderAttention


def get_args_parser():
    home = os.path.expanduser('~')
    
    parser = argparse.ArgumentParser('spt', add_help=False)
    parser.add_argument("--output", type=str, required=True, help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, required=True, help="The path to the pretrained SAM checkpoint.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on.")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=16, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=1, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int, help="number of distributed processes")
    parser.add_argument('--local-rank', type=int, help='local rank for dist')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument("--restore-model", type=str, help="Path to trained checkpoint for testing.")
    parser.add_argument("--val-data", default='data/ad', type=str)
    parser.add_argument("--val-point-prompt", default=-1, type=int)
    parser.add_argument("--lora-r", default=8, type=int)
    parser.add_argument("--vatt_alpha", default=0.0, type=float)
    parser.add_argument("--vatt_mask_detach", default=0, type=int)
    parser.add_argument("--vatt_init", default="zero", type=str, choices=["zero", "random"]) 
    parser.add_argument("--vatt_pos", default=-1, type=int)

    return parser.parse_args()


def prepare_lora(model_type, model: nn.Module, r):
    for name, module in model.named_children():
        if 'neck' in name:
            continue
        if isinstance(module, Attention):
            q_proj = module.q_proj
            v_proj = module.v_proj
            new_q_proj = Linear(q_proj.in_features, q_proj.out_features, r=r)
            new_v_proj = Linear(v_proj.in_features, v_proj.out_features, r=r)
            setattr(module, 'q_proj', new_q_proj)
            setattr(module, 'v_proj', new_v_proj)
        elif isinstance(module, EncoderAttention):
            qkv = module.qkv
            setattr(module, 'qkv', MergedLinear(qkv.in_features, qkv.out_features, r, enable_lora=[True, False, True]))
        elif ('rep' in model_type) and isinstance(module, nn.Conv2d) and module.kernel_size[0] == 1 and module.groups==1:
            setattr(model, name, ConvLoRA(module, module.in_channels, module.out_channels, 1, r=r))
        else:
            prepare_lora(model_type, module, r)     

def freeze_bn_stats(model: nn.Module):
    for _, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

def main(net, valid_datasets, args):
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, _ = create_dataloaders(valid_im_gt_list, my_transforms = [Resize(args.input_size)], batch_size=args.batch_size_valid, training=False)
    logging.info(f"{len(valid_dataloaders)} valid dataloaders created")
    
    net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)
    net_without_ddp = net.module

    logging.info(f"restore model from: {args.restore_model}")
    net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"), strict=False)

    evaluate(args, net, valid_dataloaders, valid_datasets)


def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)



@torch.no_grad()
def evaluate(args, net, valid_dataloaders, valid_datasets):
    net.eval()
    logging.info("Validating...")
    test_stats = {}

    draft_df = pd.DataFrame([], columns=['Dataset', 'IoU, BIoU'])
    avg_ious_draft = []
    avg_bious_draft = []

    df = pd.DataFrame([], columns=['Dataset', 'IoU, BIoU'])
    
    avg_ious = []
    avg_bious = []

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        logging.info(f"valid_dataloader len: {len(valid_dataloader)}")

        for data_val in metric_logger.log_every(valid_dataloader,100, logger=logging):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            
            labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
            
            if args.val_point_prompt == -1:
                input_keys = ['box']
            else:
                input_keys = ['point']
                labels_points = misc.masks_sample_points(labels_val[:, 0, :, :], args.val_point_prompt)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=net.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                batched_output, interm_embeddings = net(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)

            masks_draft = [batched_output[i_l]['low_res_logits_draft'] for i_l in range(batch_len)]
            masks_draft = torch.cat(masks_draft, 0)

            masks_final = [batched_output[i_l]['low_res_logits'] for i_l in range(batch_len)]
            masks_final = torch.cat(masks_final, 0)
                
            iou_draft = compute_iou(masks_draft,labels_ori)
            boundary_iou_draft = compute_boundary_iou(masks_draft,labels_ori)

            iou_final = compute_iou(masks_final,labels_ori)
            boundary_iou_final = compute_boundary_iou(masks_final,labels_ori)
            
            dataset = data_val["ori_im_path"][0].split("/")[2]

            loss_dict = {"draft_val_iou_"+str(k): iou_draft, "draft_val_boundary_iou_"+str(k): boundary_iou_draft, "val_iou_"+str(k): iou_final, "val_boundary_iou_"+str(k): boundary_iou_final,}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)


        logging.info('============================')
        metric_logger.synchronize_between_processes()
        logging.info(f"Averaged stats: {metric_logger}")
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)
        avg_ious_draft.append(resstat['draft_val_iou_'+str(k)])
        avg_bious_draft.append(resstat['draft_val_boundary_iou_'+str(k)])

        avg_ious.append(resstat['val_iou_'+str(k)])
        avg_bious.append(resstat['val_boundary_iou_'+str(k)])

        draft_df.loc[len(draft_df.index)] = [valid_datasets[k]['name'], f'{round(avg_ious_draft[-1], 3)}, {round(avg_bious_draft[-1], 3)}']  

        df.loc[len(df.index)] = [valid_datasets[k]['name'], f'{round(avg_ious[-1], 3)}, {round(avg_bious[-1], 3)}'] 

    avg_ious_draft = np.array(avg_ious_draft)
    avg_bious_draft = np.array(avg_bious_draft)
    logging.info(f'\n {draft_df.sort_values("Dataset").to_markdown()}')
    logging.info(f"[{args.model_type}] Draft results: IoU, BIoU: {round(avg_ious_draft.mean(), 3)}, {round(avg_bious_draft.mean(), 3)}")

    avg_ious = np.array(avg_ious)
    avg_bious = np.array(avg_bious)
    logging.info(f'\n {df.sort_values("Dataset").to_markdown()}')
    logging.info(f"[{args.model_type}] Final results: IoU, BIoU: {round(avg_ious.mean(), 3)}, {round(avg_bious.mean(), 3)}")

    return test_stats, avg_ious.mean()


def gather_all_val(val_data):
    val_datasets = []
    for d in os.listdir(val_data):
        val_datasets.append({
                "name": f"{val_data}/{d}",
                "im_dir": f"{val_data}/{d}/test",
                "gt_dir": f"{val_data}/{d}/test",
                "im_ext": ".jpg",
                "gt_ext": ".png"
            })
    return val_datasets


if __name__ == "__main__":
    args = get_args_parser()
    misc.init_distributed_mode(args)
    
    if args.rank == 0:
        os.makedirs(args.output, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"{args.output}/{'train.log' if not args.eval else 'val.log'}" ),
                logging.StreamHandler()
            ]
        )
    else:
        logging.info = lambda x: x

    logging.info('rank: {}'.format(args.rank))
    logging.info('local_rank: {}'.format(args.local_rank))
    logging.info("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True

    valid_datasets = gather_all_val(args.val_data)
    
    net = sam_model_registry[args.model_type](checkpoint=args.checkpoint, args=args)
    state = net.state_dict()
    prepare_lora(args.model_type, net, args.lora_r)
    results = net.load_state_dict(state, strict=False)

    for n, p in net.named_parameters():
        if 'lora' not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True
            # logging.info(n)

    main(net, valid_datasets, args)
