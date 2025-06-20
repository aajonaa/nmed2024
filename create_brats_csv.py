#!/usr/bin/env python3
import pandas as pd
import os

# 创建BraTS2020数据的CSV文件
brats_dir = 'dev/ssl_mri/data/MRI/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
mri_files = []

print(f"Checking directory: {brats_dir}")
print(f"Directory exists: {os.path.exists(brats_dir)}")

if os.path.exists(brats_dir):
    subjects = sorted(os.listdir(brats_dir))
    print(f"Found {len(subjects)} subjects")
    
    for subject_dir in subjects:
        if subject_dir.startswith('BraTS20_Training_'):
            t1ce_path = os.path.join(brats_dir, subject_dir, f'{subject_dir}_t1ce.nii')
            if os.path.exists(t1ce_path):
                mri_files.append({
                    'filename': t1ce_path,
                    'subject_id': subject_dir,
                    'label': 0  # 虚拟标签，因为我们只是生成embeddings
                })

print(f'Found {len(mri_files)} MRI files')

if len(mri_files) > 0:
    # 创建训练、验证、测试分割
    total = len(mri_files)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)

    train_df = pd.DataFrame(mri_files[:train_size])
    val_df = pd.DataFrame(mri_files[train_size:train_size+val_size])
    test_df = pd.DataFrame(mri_files[train_size+val_size:])

    # 保存CSV文件
    os.makedirs('data/train_vld_test_split_updated', exist_ok=True)
    train_df.to_csv('data/train_vld_test_split_updated/demo_train.csv', index=False)
    val_df.to_csv('data/train_vld_test_split_updated/demo_vld.csv', index=False)
    test_df.to_csv('data/train_vld_test_split_updated/nacc_test_with_np_cli.csv', index=False)

    print(f'Created train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)} samples')
    print("CSV files created successfully!")
else:
    print("No MRI files found!")
