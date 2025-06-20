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

import torch
from torch.nn import functional as F
import sys
sys.path.append('../')
from utils.dist_utils import is_dist_avail_and_initialized, reduce_tensor
import icecream
from icecream import install, ic
install()
ic.configureOutput(includeContext=True)
# from ..utils import dist_utils
# from dist_utils import is_dist_avail_and_initialized, reduce_tensor

class Contrast(torch.nn.Module):
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.expected_batch_size = batch_size  # 保存预期批次大小用于调试
        self.register_buffer("temp", torch.tensor(temperature)) #.to(torch.device(f"cuda:{args.local_rank}")))
        # 移除固定的neg_mask，改为动态创建

    def forward(self, x_i, x_j):
        ic(f"输入张量形状: x_i={x_i.size()}, x_j={x_j.size()}")

        # 动态计算实际批次大小
        actual_batch_size = x_i.size(0)
        ic(f"预期批次大小: {self.expected_batch_size}, 实际批次大小: {actual_batch_size}")

        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        ic(f"相似度矩阵形状: {sim.size()}")

        # 使用实际批次大小计算对角线
        sim_ij = torch.diag(sim, actual_batch_size)
        sim_ji = torch.diag(sim, -actual_batch_size)
        ic(f"对角线张量形状: sim_ij={sim_ij.size()}, sim_ji={sim_ji.size()}")

        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)

        # 动态创建neg_mask
        total_size = actual_batch_size * 2
        neg_mask = (~torch.eye(total_size, total_size, dtype=bool, device=sim.device)).float()
        ic(f"动态创建的neg_mask形状: {neg_mask.size()}")

        exp_sim = torch.exp(sim / self.temp)
        ic(f"张量形状检查: pos={pos.size()}, nom={nom.size()}, neg_mask={neg_mask.size()}, exp_sim={exp_sim.size()}")

        denom = neg_mask * exp_sim
        loss = torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * actual_batch_size)

        ic(f"对比学习损失: {loss.item()}")
        return loss


class Loss(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(args, batch_size).cuda()
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        ic(output_rot.size(), target_rot.size())
        ic(output_contrastive.size(), target_contrastive.size())
        ic(output_recons.size(), target_recons.size())
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss

        if is_dist_avail_and_initialized():
            return reduce_tensor(total_loss), (reduce_tensor(rot_loss), reduce_tensor(contrast_loss), reduce_tensor(recon_loss))
        
        return total_loss, (rot_loss, contrast_loss, recon_loss)