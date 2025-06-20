import pandas as pd
import numpy as np
from tqdm import tqdm
from adrd.nn import UNet3DModel
from icecream import ic
import torch
from .imgdataload import ImageDataset
from torch.utils.data import DataLoader
from monai.networks.nets.swin_unetr import SwinUNETR
import os
import random
import functools
from monai.utils.type_conversion import convert_to_tensor

ckpts_dict = {
    # 'unet3d': '~/adrd_tool/img_pretrained_ckpt/model_best.pkl',
    'SwinUNETR': '/root/github/nmed2024/dev/ssl_mri/runs/results/brats2020_ssl_swinunetr/model_bestVal.pt'
}

device = 'cuda'

def get_mri_dataloader(feature, df, transforms=None, stripped=False):
    test_envs=[1,2,3]

    dataset = ImageDataset(feature=feature, task='mri_dg', root_dir='', domain_name="NACC_ADNI_NC_MCI_AD", domain_label=0, transform=transforms, indices=None, test_envs=test_envs, df=df, num_classes=3)
    
    def monai_collate_singles(samples_list, dataset, dtype=torch.half, return_dict=False, labels=None, multilabel=False):
        orig_len = len(samples_list)
        for s in samples_list:
            fname, img = s
            if s is None or img is None or img.shape[0] != 1 or torch.isnan(s[1]).any():
                samples_list.remove(s)

        # samples_list = [s for s in samples_list if not ]
        diff = orig_len - len(samples_list)
        ic(diff)
        if diff > 0:
            ic('recursive call')  
            return monai_collate_singles(samples_list + [dataset[random.randint(0, len(dataset)-1)] for _ in range(diff)], dataset, return_dict=return_dict, labels=labels, multilabel=multilabel)

        if return_dict:
            collated_dict = {"image": torch.stack([convert_to_tensor(s["image"]) for s in samples_list])}
            if labels:
                if multilabel:
                    collated_dict["label"] = {k: torch.Tensor([s["label"][k] for s in samples_list]) for k in labels}
                else:
                    collated_dict["label"] = torch.Tensor([s["label"] for s in samples_list])
            return collated_dict
        
        else:
            if isinstance(samples_list[0], tuple):
                # return fnames, imgs
                fnames_list = [s[0] for s in samples_list]
                imgs_list = [convert_to_tensor(s[1]) for s in samples_list]
                return fnames_list, torch.stack(imgs_list)
            return torch.stack([convert_to_tensor(s) for s in samples_list])

    collate_fn = functools.partial(monai_collate_singles, dataset=dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=False, collate_fn=collate_fn)

    return dataloader

def load_model(arch: str,
               ckpt_path: str = None,
               ckpt_key: str | None = "state_dict"
               ):
    if ckpt_path is None:
        ckpt_path = ckpts_dict.get(arch, ckpts_dict['SwinUNETR'])
    print(f'Loading {arch} model from checkpoint: {ckpt_path} ...')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # ic(state_dict.keys())
    if arch == 'SwinUNETR':
        model = SwinUNETR(
            in_channels=1,
            out_channels=1,
            img_size=128,
            feature_size=48,
            use_checkpoint=True,
        )
        # 不需要替换键名，直接使用原始的state_dict
        # ckpt[ckpt_key] = {k.replace("swinViT.", "module."): v for k, v in ckpt[ckpt_key].items()}
        ic(list(ckpt[ckpt_key].keys())[:5])
        # 直接加载state_dict而不是使用load_from
        model.load_state_dict(ckpt[ckpt_key], strict=False)
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.swinunetr = model

            def forward(self, x_in):
                try:
                    # 尝试使用完整的前向传播
                    hidden_states_out = self.swinunetr.swinViT(x_in, self.swinunetr.normalize)
                    ic([h.size() for h in hidden_states_out])

                    # 使用最深层的hidden state作为特征
                    # hidden_states_out[-1] 是最深层的特征
                    return hidden_states_out[-1]

                except Exception as e:
                    print(f"Error in SwinViT forward pass: {e}")
                    # 如果SwinViT失败，尝试使用更简单的方法
                    # 直接使用encoder1的输出作为特征
                    try:
                        enc0 = self.swinunetr.encoder1(x_in)
                        return enc0
                    except Exception as e2:
                        print(f"Error in encoder1: {e2}")
                        # 最后的备选方案：使用全局平均池化
                        return torch.mean(x_in, dim=[2, 3, 4], keepdim=True)

        img_model = ModelWrapper(model)
    elif 'unet3d' in arch.lower():
        state_dict = ckpt[ckpt_key]
        img_model = UNet3DModel(num_classes=3)
        img_model.load_checkpoint(state_dict)
    else:
        raise NotImplementedError(f"Model {arch} not implemented.")

    return img_model

def mri_emb(img_model, data):
    # print(data)
    fnames = data[0]
    # ic(len(fnames), fnames[0])
    x = data[1].float()
    y = data[2].long()
    # print(x.size(), y.size())
    x = x.to(device)
    y = y.to(device)
    # feats, output = img_model.predict(x, stage='get_features', attention=False)
    feats = img_model.extract_features(x, attention=True)
    # print(feats[1].shape)
    # break
    return np.array(feats[0].cpu())

def save_emb(feature, df):
    dataloader = get_mri_dataloader(feature, df)
    img_model = load_model()
    embeddings = []
    img_model.to(device)
    img_model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            try:
                filename = data[0][0].split('/')[-1]
                if os.path.exists('./MRI_emb/' + filename):
                    continue
                emb = mri_emb(img_model, data)
                # print(emb.shape)
                np.save('./MRI_emb/' + filename, emb)
            except:
                continue

def get_emb(feature, df, savedir=None, arch='unet3d', transforms=None, stripped=False, ckpt_path=None, ckpt_key="state_dict"):
    print('------------get_emb()------------')
    dataloader = get_mri_dataloader(feature, df, transforms=transforms, stripped=stripped)
    img_model = load_model(arch, ckpt_path, ckpt_key)
    # device = 'cuda'
    img_model.to(device)
    img_model.eval()
    embeddings = {}
    if savedir:
        os.makedirs(savedir, exist_ok=True)
    print(f"Dataloader length: {len(dataloader)}")
    with torch.no_grad():
        for fnames, data in tqdm(dataloader):
            print(f"Processing batch with {len(fnames)} files")
            try:
                if torch.isnan(data).any() or data.size(1) != 1:
                    continue
                data = data.float().to(device)
                # print(data.size())
                # 处理每个文件单独，避免批处理时的尺寸不匹配问题
                for idx, fname in enumerate(fnames):
                    print(fname)
                    if ('localizer' in fname.lower()) | ('localiser' in fname.lower()) |  ('LOC' in fname) | ('calibration' in fname.lower()) | ('field_mapping' in fname.lower()) | (fname.lower()[:-4].endswith('_ph_stripped')):
                        continue
                    # if 'DWI' in fname:
                    #     continue
                    if 'NACC' in fname:
                            filename = fname.split('/')[6] + '@' + '@'.join(fname.split('/')[-2:]).replace('.nii', '.npy')
                    elif 'FHS' in fname:
                        filename = 'FHS_' + '_'.join(fname.split('/')[-2:]).replace('.nii.gz', '.npy')
                    elif 'BMC' in fname:
                        filename = '_'.join(fname.split('/')[-3:]).replace('.nii', '.npy')
                    else:
                        filename = fname.split('/')[-1].replace('.nii', '.npy')
                    if savedir and os.path.exists(savedir + filename):
                        continue

                    # 单独处理每个样本
                    single_data = data[idx:idx+1]  # 取单个样本
                    if arch == 'SwinUNETR':
                        try:
                            single_emb = img_model(single_data)
                            # 对于SwinUNETR，single_emb 是一个张量，不是列表
                            # 应用全局平均池化来处理不同尺寸
                            # single_emb shape: [1, channels, H, W, D]
                            pooled_emb = torch.mean(single_emb, dim=[2, 3, 4])  # [1, channels]

                            embeddings[filename] = pooled_emb[0,:].cpu().detach().numpy()
                            if savedir:
                                np.save(savedir + filename, embeddings[filename])
                                print(f"Saved embedding for {filename} with shape {embeddings[filename].shape}")
                        except Exception as e:
                            print(f"Error processing {fname}: {e}")
                            continue
                    else:
                        emb_tensor = mri_emb(img_model, single_data)
                        embeddings[filename] = emb_tensor[0,:,:,:,:].cpu().detach().numpy()
                        if savedir:
                            np.save(savedir + filename, embeddings[filename])
            except Exception as e:
                print(f"Error processing {fnames}: {e}")
                continue
    print("Embeddings saved to ", savedir)
    print("Done.")
    if savedir:
        exit()
    return embeddings


