# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from time import time
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from losses.SwinUNETR_loss import Loss
from models.ssl_head import SSLHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import get_loader
from utils.ops import aug_rand, rot_rand
from utils.dist_utils import init_distributed_mode, is_main_process, has_batchnorms
from data.mri_dataset import get_fpaths
import icecream
from icecream import install, ic
install()
ic.configureOutput(includeContext=True)
ic.disable()

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--dataset', type=str,
        help='Please specify dataset name')
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=4e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument('--gpu', type=int, default=0, help="GPU id")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--distributed", action="store_true", help="Set True for Distributed Training")
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument("--deterministic", help="set seed for deterministic training", action="store_true")
    parser.add_argument("--num_workers", type=int)

    return parser

def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args, global_step, train_loader, val_best, val_best_total, scaler):
        model.train()
        loss_train = []
        loss_train_rot = []
        loss_train_contrast = []
        loss_train_recon = []

        for step, batch in enumerate(train_loader):
            t1 = time()
            # x = batch["image"].cuda()
            x = batch.cuda(non_blocking=True)
            x1, rot1 = rot_rand(args, x)
            x2, rot2 = rot_rand(args, x)
            x1_augment = aug_rand(args, x1)
            x2_augment = aug_rand(args, x2)
            x1_augment = x1_augment
            x2_augment = x2_augment
            ic(x1.size(), rot1.size())
            ic(x2.size(), rot2.size())
            ic(x1_augment.size(), x2_augment.size())
            with autocast(enabled=args.amp):
                rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
                rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rots = torch.cat([rot1, rot2], dim=0)
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                imgs = torch.cat([x1, x2], dim=0)
                loss, losses_tasks = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
            loss_train.append(loss.item())
            loss_train_rot.append(losses_tasks[0].item())
            loss_train_contrast.append(losses_tasks[1].item())
            loss_train_recon.append(losses_tasks[2].item())
            ic(loss.item())
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            if args.distributed:
                if dist.get_rank() == 0:
                    print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss.item(), time() - t1))
            else:
                print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss.item(), time() - t1))

            global_step += 1
            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            if val_cond:
                val_loss, val_loss_rot, val_loss_contrast, val_loss_recon, img_list = validation(args, test_loader)
                if is_main_process():
                    writer.add_scalar("Validation/loss_total", scalar_value=val_loss, global_step=global_step)
                    writer.add_scalar("Validation/loss_rot", scalar_value=val_loss_rot, global_step=global_step)
                    writer.add_scalar("Validation/loss_contrast", scalar_value=val_loss_contrast, global_step=global_step)
                    writer.add_scalar("Validation/loss_recon", scalar_value=val_loss_recon, global_step=global_step)
                    writer.add_scalar("train/loss_total", scalar_value=np.mean(loss_train), global_step=global_step)
                    writer.add_scalar("train/loss_rot", scalar_value=np.mean(loss_train_rot), global_step=global_step)
                    writer.add_scalar("train/loss_contrast", scalar_value=np.mean(loss_train_contrast), global_step=global_step)
                    writer.add_scalar("train/loss_recon", scalar_value=np.mean(loss_train_recon), global_step=global_step)

                    writer.add_image("Validation/x1_gt", img_list[0], global_step, dataformats="HW")
                    writer.add_image("Validation/x1_aug", img_list[1], global_step, dataformats="HW")
                    writer.add_image("Validation/x1_recon", img_list[2], global_step, dataformats="HW")

                    if val_loss_recon < val_best:
                        val_best = val_loss_recon
                        checkpoint = {
                            "global_step": global_step,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "val_loss": val_loss,
                            "val_loss_recon": val_loss_recon,
                        }
                        save_ckp(checkpoint, logdir + "/model_bestValRMSE.pt")
                        print(
                            "Model was saved ! Best Recon. Val Loss: {:.4f}, Recon. Val Loss: {:.4f}".format(
                                val_best, val_loss_recon
                            )
                        )
                    else:
                        print(
                            "Model was not saved ! Best Recon. Val Loss: {:.4f} Recon. Val Loss: {:.4f}".format(
                                val_best, val_loss_recon
                            )
                        )

                    if val_loss < val_best_total:
                        val_best_total = val_loss
                        checkpoint = {
                            "global_step": global_step,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "val_loss": val_loss,
                            "val_loss_recon": val_loss_recon,
                        }
                        save_ckp(checkpoint, logdir + "/model_bestVal.pt")
                        print(
                            "Model was saved ! Best Recon. Val Loss: {:.4f}, Recon. Val Loss: {:.4f}".format(
                                val_best_total, val_loss
                            )
                        )
                    else:
                        print(
                            "Model was not saved ! Best Recon. Val Loss: {:.4f} Recon. Val Loss: {:.4f}".format(
                                val_best_total, val_loss
                            )
                        )

        return global_step, loss, val_best, val_best_total

    def validation(args, test_loader):
        model.eval()
        loss_val = []
        loss_val_rot = []
        loss_val_contrast = []
        loss_val_recon = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(test_loader)):
                # val_inputs = batch["image"].cuda()
                val_inputs = batch.cuda()
                x1, rot1 = rot_rand(args, val_inputs)
                x2, rot2 = rot_rand(args, val_inputs)
                x1_augment = aug_rand(args, x1)
                x2_augment = aug_rand(args, x2)
                with autocast(enabled=args.amp):
                    rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
                    rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
                    rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                    rots = torch.cat([rot1, rot2], dim=0)
                    imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                    imgs = torch.cat([x1, x2], dim=0)
                    loss, losses_tasks = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
                loss_step_recon = losses_tasks[2].item()
                loss_val.append(loss.item())
                loss_val_rot.append(losses_tasks[0].item())
                loss_val_contrast.append(losses_tasks[1].item())
                loss_val_recon.append(losses_tasks[2].item())
                x_gt = x1.detach().cpu().numpy()
                x_gt = (x_gt - np.min(x_gt)) / (np.max(x_gt) - np.min(x_gt))
                xgt = x_gt[0][0][:, :, 48] * 255.0
                xgt = xgt.astype(np.uint8)
                x1_augment = x1_augment.detach().cpu().numpy()
                x1_augment = (x1_augment - np.min(x1_augment)) / (np.max(x1_augment) - np.min(x1_augment))
                x_aug = x1_augment[0][0][:, :, 48] * 255.0
                x_aug = x_aug.astype(np.uint8)
                rec_x1 = rec_x1.detach().cpu().numpy()
                rec_x1 = (rec_x1 - np.min(rec_x1)) / (np.max(rec_x1) - np.min(rec_x1))
                recon = rec_x1[0][0][:, :, 48] * 255.0
                recon = recon.astype(np.uint8)
                img_list = [xgt, x_aug, recon]
                # if is_main_process():
                print("Validation step:{}, Loss:{:.4f}, Loss Reconstruction:{:.4f}".format(step, loss, loss_step_recon))

        return np.mean(loss_val), np.mean(loss_val_rot), np.mean(loss_val_contrast), np.mean(loss_val_recon), img_list

    parser = get_parser()

    args = parser.parse_args()
    logdir = "./runs/" + args.logdir
    args.amp = not args.noamp
    
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    
    init_distributed_mode(args)

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    seed = args.seed + dist.get_rank()

    if args.deterministic:
        torch.manual_seed(seed)
        torch.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    if dist.get_rank() == 0:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
    else:
        writer = None

    model = SSLHead(args)
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters())
    print("number of params: {}".format(n_parameters))
    # exit()
    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    
    global_step = 0
    if args.resume:
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        model.load_state_dict(model_dict["state_dict"])
        global_step = model_dict["global_step"]
        optimizer.load_state_dict(model_dict["optimizer"])

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    loss_function = Loss(args.batch_size * args.sw_batch_size, args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.gpu],
                                        output_device=args.gpu,)
                                        # find_unused_parameters=False)
        # model._set_static_graph()
    # For BraTS2020, use single modality (t1ce) for SSL fine-tuning
    data_list = get_fpaths(args.data_path, modality='t1ce')
    random.shuffle(data_list)
    train_list, val_list = data_list[:int(0.8*len(data_list))], data_list[int(0.8*len(data_list)):]
    ic(len(train_list), len(val_list))
    print(f"BraTS2020 SSL fine-tuning: {len(train_list)} train, {len(val_list)} val samples")
    train_loader, test_loader = get_loader(train_list, val_list=val_list, num_workers=args.num_workers,
                                        a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
                                        roi_x=args.roi_x, roi_y=args.roi_y, roi_z=args.roi_z, sw_batch_size=args.sw_batch_size,
                                        batch_size=args.batch_size, distributed=args.distributed, cache_dataset=args.cache_dataset, smartcache_dataset=args.smartcache_dataset)

    ic(len(train_loader), len(train_loader.dataset))
    best_val, best_val_total = 1e8, 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    while global_step < args.num_steps:
        global_step, loss, best_val, best_val_total = train(args, global_step, train_loader, best_val, best_val_total, scaler)
    checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "val_loss": best_val_total,
                    "val_loss_recon": best_val}

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), logdir + "final_model.pth")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), logdir + "final_model.pth")
    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")


if __name__ == "__main__":
    main()