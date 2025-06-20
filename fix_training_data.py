#!/usr/bin/env python3
"""
修复训练数据脚本
1. 删除主数据文件中不必要的ID列
2. 修正subject_id列，使其与filename一致
3. 重新生成train/vld/test分割文件作为主文件的子集
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def extract_subject_id_from_filename(filename):
    """从filename中提取subject_id"""
    # 从路径中提取BraTS样本ID
    # 例如: "dev/ssl_mri/.../BraTS20_Training_001/BraTS20_Training_001_t1ce.nii" -> "BraTS20_Training_001"
    basename = os.path.basename(filename)
    if basename.endswith('_t1ce.nii'):
        return basename.replace('_t1ce.nii', '')
    else:
        # 如果格式不匹配，从路径中提取
        parts = filename.split('/')
        for part in parts:
            if part.startswith('BraTS20_Training_'):
                return part
    return basename.replace('.nii', '')

def fix_main_data_file():
    """修复主数据文件"""
    print("修复主数据文件...")
    
    # 读取主数据文件
    main_file = "data/training_cohorts/new_nacc_revised_selection.csv"
    df = pd.read_csv(main_file)
    
    print(f"原始数据形状: {df.shape}")
    print(f"原始列数: {len(df.columns)}")
    
    # 删除ID列（如果存在）
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
        print("已删除ID列")
    
    # 修正subject_id列
    if 'filename' in df.columns:
        # 从filename提取subject_id
        df['subject_id'] = df['filename'].apply(extract_subject_id_from_filename)
        print("已更新subject_id列")
    
    # 重新排列列顺序，将filename和subject_id放在前面
    cols = df.columns.tolist()
    if 'filename' in cols and 'subject_id' in cols:
        # 移除这两列
        cols.remove('filename')
        cols.remove('subject_id')
        # 重新排列：filename, subject_id, 其他列
        new_cols = ['filename', 'subject_id'] + cols
        df = df[new_cols]
    
    print(f"修复后数据形状: {df.shape}")
    print(f"修复后列数: {len(df.columns)}")
    
    # 保存修复后的文件
    df.to_csv(main_file, index=False)
    print(f"已保存修复后的主数据文件: {main_file}")
    
    return df

def create_train_vld_test_splits(main_df):
    """创建train/vld/test分割文件作为主文件的子集"""
    print("创建train/vld/test分割文件...")
    
    # 设置随机种子以确保可重现性
    np.random.seed(42)
    
    # 获取所有样本
    total_samples = len(main_df)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    # 计算分割点
    train_size = int(0.7 * total_samples)  # 70% 训练
    vld_size = int(0.15 * total_samples)   # 15% 验证
    # 剩余15%作为测试
    
    train_indices = indices[:train_size]
    vld_indices = indices[train_size:train_size + vld_size]
    test_indices = indices[train_size + vld_size:]
    
    # 创建分割数据
    train_df = main_df.iloc[train_indices].copy()
    vld_df = main_df.iloc[vld_indices].copy()
    test_df = main_df.iloc[test_indices].copy()
    
    print(f"训练集样本数: {len(train_df)}")
    print(f"验证集样本数: {len(vld_df)}")
    print(f"测试集样本数: {len(test_df)}")
    
    # 保存分割文件
    train_df.to_csv("data/train_vld_test_split_updated/demo_train.csv", index=False)
    vld_df.to_csv("data/train_vld_test_split_updated/demo_vld.csv", index=False)
    test_df.to_csv("data/train_vld_test_split_updated/nacc_test_with_np_cli.csv", index=False)
    
    print("已保存所有分割文件")
    
    return train_df, vld_df, test_df

def verify_data_consistency():
    """验证数据一致性"""
    print("验证数据一致性...")

    # 读取所有文件
    main_df = pd.read_csv("data/training_cohorts/new_nacc_revised_selection.csv")
    train_df = pd.read_csv("data/train_vld_test_split_updated/demo_train.csv")
    vld_df = pd.read_csv("data/train_vld_test_split_updated/demo_vld.csv")
    test_df = pd.read_csv("data/train_vld_test_split_updated/nacc_test_with_np_cli.csv")

    print(f"主文件形状: {main_df.shape}")
    print(f"训练文件形状: {train_df.shape}")
    print(f"验证文件形状: {vld_df.shape}")
    print(f"测试文件形状: {test_df.shape}")

    # 检查列是否一致
    main_cols = set(main_df.columns)
    train_cols = set(train_df.columns)
    vld_cols = set(vld_df.columns)
    test_cols = set(test_df.columns)

    if main_cols == train_cols == vld_cols == test_cols:
        print("✓ 所有文件的列结构一致")
    else:
        print("✗ 文件列结构不一致")
        print(f"主文件列数: {len(main_cols)}")
        print(f"训练文件列数: {len(train_cols)}")

    # 检查样本总数
    total_split_samples = len(train_df) + len(vld_df) + len(test_df)
    if total_split_samples == len(main_df):
        print("✓ 分割文件样本总数与主文件一致")
    else:
        print(f"✗ 样本数不匹配: 主文件{len(main_df)}, 分割总数{total_split_samples}")

    # 检查subject_id格式
    sample_subject_ids = main_df['subject_id'].head(5).tolist()
    print(f"样本subject_id: {sample_subject_ids}")

    # 检查filename格式
    sample_filenames = main_df['filename'].head(3).tolist()
    print(f"样本filename: {sample_filenames}")

    print("数据一致性验证完成")

def main():
    """主函数"""
    try:
        print("开始修复训练数据...")

        # 1. 修复主数据文件
        main_df = fix_main_data_file()

        # 2. 创建train/vld/test分割
        train_df, vld_df, test_df = create_train_vld_test_splits(main_df)

        # 3. 验证数据一致性
        verify_data_consistency()

        print("\n修复完成！现在所有文件都是一致的：")
        print("- 主文件删除了不必要的ID列")
        print("- subject_id与filename保持一致")
        print("- train/vld/test文件是主文件的真正子集")
        print("- 所有文件具有相同的列结构")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
