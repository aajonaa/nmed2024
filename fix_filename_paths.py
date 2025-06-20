#!/usr/bin/env python3
"""
修复filename路径脚本
将数据文件中的filename从.nii路径转换为对应的.npy嵌入文件路径
"""

import pandas as pd
import os
import glob

def convert_nii_to_npy_path(nii_path):
    """将.nii文件路径转换为对应的.npy嵌入文件路径"""
    # 从完整路径中提取文件名
    # 例如: "dev/ssl_mri/.../BraTS20_Training_001_t1ce.nii" -> "BraTS20_Training_001_t1ce.npy"
    basename = os.path.basename(nii_path)
    if basename.endswith('.nii'):
        npy_filename = basename.replace('.nii', '.npy')
        # 返回MRI_emb文件夹中的路径
        return f"MRI_emb/{npy_filename}"
    return nii_path

def fix_filename_paths_in_file(file_path):
    """修复指定文件中的filename路径"""
    print(f"修复文件: {file_path}")
    
    # 读取文件
    df = pd.read_csv(file_path)
    
    print(f"  原始filename示例: {df['filename'].iloc[0]}")
    
    # 转换filename列
    df['filename'] = df['filename'].apply(convert_nii_to_npy_path)
    
    print(f"  修复后filename示例: {df['filename'].iloc[0]}")
    
    # 验证对应的.npy文件是否存在
    existing_count = 0
    for filename in df['filename']:
        if os.path.exists(filename):
            existing_count += 1
    
    print(f"  {existing_count}/{len(df)} 个嵌入文件存在")
    
    # 保存修复后的文件
    df.to_csv(file_path, index=False)
    print(f"  已保存修复后的文件")
    
    return existing_count, len(df)

def main():
    """主函数"""
    try:
        print("开始修复filename路径...")
        
        # 需要修复的文件列表
        files_to_fix = [
            "data/training_cohorts/new_nacc_revised_selection.csv",
            "data/train_vld_test_split_updated/demo_train.csv",
            "data/train_vld_test_split_updated/demo_vld.csv",
            "data/train_vld_test_split_updated/nacc_test_with_np_cli.csv"
        ]
        
        total_existing = 0
        total_samples = 0
        
        for file_path in files_to_fix:
            if os.path.exists(file_path):
                existing, samples = fix_filename_paths_in_file(file_path)
                total_existing += existing
                total_samples += samples
                print()
            else:
                print(f"文件不存在: {file_path}")
                print()
        
        # 验证MRI_emb文件夹中的文件
        mri_emb_files = glob.glob("MRI_emb/*.npy")
        print("=" * 50)
        print("修复结果总结:")
        print("=" * 50)
        print(f"MRI_emb文件夹中的嵌入文件数量: {len(mri_emb_files)}")
        print(f"数据文件中引用的嵌入文件数量: {total_samples}")
        print(f"实际存在的嵌入文件数量: {total_existing}")
        
        if total_existing == len(mri_emb_files):
            print("✓ 所有嵌入文件都被正确引用")
        else:
            print("⚠ 部分嵌入文件可能未被引用或路径不匹配")
        
        if total_existing > 0:
            print("✓ 修复完成！现在可以运行训练脚本")
        else:
            print("✗ 没有找到有效的嵌入文件，请检查路径")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
