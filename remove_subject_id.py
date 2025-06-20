#!/usr/bin/env python3
"""
删除subject_id列的脚本
因为subject_id在TOML配置文件中没有定义，所以应该从所有数据文件中删除
"""

import pandas as pd
import os

def remove_subject_id_from_file(file_path):
    """从指定文件中删除subject_id列"""
    print(f"处理文件: {file_path}")
    
    # 读取文件
    df = pd.read_csv(file_path)
    
    print(f"  原始列数: {len(df.columns)}")
    print(f"  原始前5列: {list(df.columns[:5])}")
    
    # 检查是否存在subject_id列
    if 'subject_id' in df.columns:
        # 删除subject_id列
        df = df.drop('subject_id', axis=1)
        print(f"  已删除subject_id列")
    else:
        print(f"  未找到subject_id列")
    
    print(f"  修复后列数: {len(df.columns)}")
    print(f"  修复后前5列: {list(df.columns[:5])}")
    
    # 保存修复后的文件
    df.to_csv(file_path, index=False)
    print(f"  已保存修复后的文件")
    
    return len(df.columns)

def main():
    """主函数"""
    try:
        print("开始删除subject_id列...")
        
        # 需要处理的文件列表
        files_to_process = [
            "data/training_cohorts/new_nacc_revised_selection.csv",
            "data/train_vld_test_split_updated/demo_train.csv",
            "data/train_vld_test_split_updated/demo_vld.csv",
            "data/train_vld_test_split_updated/nacc_test_with_np_cli.csv"
        ]
        
        results = {}
        
        for file_path in files_to_process:
            if os.path.exists(file_path):
                column_count = remove_subject_id_from_file(file_path)
                results[file_path] = column_count
                print()
            else:
                print(f"文件不存在: {file_path}")
                print()
        
        # 验证结果
        print("=" * 50)
        print("修复结果总结:")
        print("=" * 50)
        
        expected_columns = 405  # TOML配置中定义的特征数量
        
        for file_path, column_count in results.items():
            status = "✓" if column_count == expected_columns else "✗"
            print(f"{status} {file_path}: {column_count} 列")
        
        # 检查所有文件是否都有相同的列数
        unique_counts = set(results.values())
        if len(unique_counts) == 1 and list(unique_counts)[0] == expected_columns:
            print(f"\n✓ 所有文件都有正确的列数 ({expected_columns} 列)")
            print("✓ 数据修复完成！")
        else:
            print(f"\n✗ 文件列数不一致或不正确")
            print(f"期望列数: {expected_columns}")
            print(f"实际列数: {unique_counts}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
