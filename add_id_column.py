#!/usr/bin/env python3
"""
为数据文件添加ID列的脚本
"""

import pandas as pd
import os

def add_id_column_to_file(file_path):
    """为指定文件添加ID列"""
    print(f"处理文件: {file_path}")
    
    # 读取文件
    df = pd.read_csv(file_path)
    
    print(f"  原始列数: {len(df.columns)}")
    print(f"  原始样本数: {len(df)}")
    
    # 检查是否已经有ID列
    if 'ID' in df.columns:
        print(f"  文件已经有ID列，跳过")
        return
    
    # 添加ID列作为第一列，使用行索引作为ID
    df.insert(0, 'ID', range(len(df)))
    
    print(f"  添加ID列后列数: {len(df.columns)}")
    print(f"  ID列范围: {df['ID'].min()} - {df['ID'].max()}")
    
    # 保存修改后的文件
    df.to_csv(file_path, index=False)
    print(f"  已保存修改后的文件")

def main():
    """主函数"""
    try:
        print("开始为数据文件添加ID列...")
        
        # 需要处理的文件列表
        files_to_process = [
            "data/training_cohorts/new_nacc_revised_selection.csv",
            "data/train_vld_test_split_updated/demo_train.csv",
            "data/train_vld_test_split_updated/demo_vld.csv",
            "data/train_vld_test_split_updated/nacc_test_with_np_cli.csv"
        ]
        
        for file_path in files_to_process:
            if os.path.exists(file_path):
                add_id_column_to_file(file_path)
                print()
            else:
                print(f"文件不存在: {file_path}")
                print()
        
        print("=" * 50)
        print("ID列添加完成！")
        print("=" * 50)
        
        # 验证结果
        for file_path in files_to_process:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"✓ {file_path}: {len(df)} 样本, {len(df.columns)} 列")
                if 'ID' in df.columns:
                    print(f"  ID列: {df['ID'].min()} - {df['ID'].max()}")
                else:
                    print(f"  ✗ 缺少ID列")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
