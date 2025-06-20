#!/usr/bin/env python3
"""
生成合成训练数据脚本
基于现有的MRI嵌入和TOML配置文件创建完整的训练数据集
"""

import pandas as pd
import numpy as np
import os
import glob
import toml
import random
from pathlib import Path

def load_config(config_path):
    """加载TOML配置文件"""
    return toml.load(config_path)

def get_mri_files():
    """获取所有MRI嵌入文件"""
    mri_emb_dir = "MRI_emb"
    mri_files = glob.glob(os.path.join(mri_emb_dir, "*.npy"))
    
    # 提取对应的原始MRI文件路径
    original_paths = []
    for emb_file in mri_files:
        # 从嵌入文件名提取原始文件路径
        basename = os.path.basename(emb_file).replace('.npy', '.nii')
        # 构建原始MRI文件路径（基于现有的demo文件格式）
        original_path = f"dev/ssl_mri/data/MRI/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/{basename.replace('_t1ce.nii', '')}/{basename}"
        original_paths.append(original_path)
    
    return original_paths, mri_files

def generate_categorical_feature(num_samples, num_categories, feature_name):
    """生成分类特征数据"""
    # 为不同类型的特征设置不同的分布
    if 'SEX' in feature_name:
        # 性别：大致平衡
        return np.random.choice(range(num_categories), num_samples, p=[0.45, 0.55])
    elif 'RACE' in feature_name:
        # 种族：模拟真实分布
        if num_categories == 6:
            return np.random.choice(range(num_categories), num_samples, 
                                  p=[0.7, 0.15, 0.05, 0.05, 0.03, 0.02])
    elif 'EDUC' in feature_name or 'education' in feature_name.lower():
        # 教育：偏向较高教育水平
        return np.random.choice(range(num_categories), num_samples,
                              p=np.array([0.1, 0.2, 0.3, 0.25, 0.15])[:num_categories])
    elif any(x in feature_name for x in ['DIABETES', 'HYPERTEN', 'ANXIETY']):
        # 疾病：较低患病率
        if num_categories == 3:
            return np.random.choice(range(num_categories), num_samples,
                                  p=[0.7, 0.2, 0.1])  # 无、轻度、重度
        elif num_categories == 2:
            return np.random.choice(range(num_categories), num_samples,
                                  p=[0.75, 0.25])  # 无、有
    
    # 默认：均匀分布
    return np.random.choice(range(num_categories), num_samples)

def generate_numerical_feature(num_samples, feature_name):
    """生成数值特征数据"""
    if 'AGE' in feature_name:
        # 年龄：65-90岁，正态分布
        return np.clip(np.random.normal(75, 8, num_samples), 50, 95)
    elif 'EDUC' in feature_name:
        # 教育年限：8-20年
        return np.clip(np.random.normal(14, 3, num_samples), 8, 20)
    elif 'BMI' in feature_name:
        # BMI：18-35
        return np.clip(np.random.normal(25, 4, num_samples), 18, 40)
    elif 'HEIGHT' in feature_name:
        # 身高：150-190cm
        return np.clip(np.random.normal(170, 10, num_samples), 150, 190)
    elif 'WEIGHT' in feature_name:
        # 体重：45-120kg
        return np.clip(np.random.normal(70, 15, num_samples), 45, 120)
    elif 'BPSYS' in feature_name:
        # 收缩压：100-180
        return np.clip(np.random.normal(130, 20, num_samples), 100, 180)
    elif 'BPDIAS' in feature_name:
        # 舒张压：60-110
        return np.clip(np.random.normal(80, 10, num_samples), 60, 110)
    elif 'HRATE' in feature_name:
        # 心率：50-100
        return np.clip(np.random.normal(70, 10, num_samples), 50, 100)
    elif 'MMSE' in feature_name:
        # MMSE分数：0-30
        return np.clip(np.random.normal(24, 5, num_samples), 0, 30)
    elif 'MOCA' in feature_name:
        # MoCA分数：0-30
        return np.clip(np.random.normal(22, 6, num_samples), 0, 30)
    elif 'BIRTHYR' in feature_name:
        # 出生年：1930-1970
        return np.random.randint(1930, 1971, num_samples)
    elif any(x in feature_name for x in ['YEAR', 'YR']):
        # 其他年份：1990-2020
        return np.random.randint(1990, 2021, num_samples)
    elif any(x in feature_name for x in ['SCORE', 'TOT']):
        # 各种评分：0-100
        return np.clip(np.random.normal(50, 20, num_samples), 0, 100)
    else:
        # 默认：0-10的正态分布
        return np.clip(np.random.normal(5, 2, num_samples), 0, 10)

def generate_labels(num_samples):
    """生成多标签分类标签"""
    labels = ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']
    
    # 创建标签数据框
    label_data = {}
    
    # 设置标签分布（模拟真实的疾病分布）
    label_probs = {
        'NC': 0.4,    # 正常对照
        'MCI': 0.25,  # 轻度认知障碍
        'DE': 0.3,    # 痴呆
        'AD': 0.2,    # 阿尔茨海默病
        'LBD': 0.05,  # 路易体痴呆
        'VD': 0.08,   # 血管性痴呆
        'PRD': 0.03,  # 帕金森相关痴呆
        'FTD': 0.04,  # 额颞叶痴呆
        'NPH': 0.02,  # 正常压力脑积水
        'SEF': 0.02,  # 癫痫相关
        'PSY': 0.03,  # 精神疾病
        'TBI': 0.02,  # 创伤性脑损伤
        'ODE': 0.03   # 其他痴呆
    }
    
    for label in labels:
        # 生成二元标签（0或1）
        prob = label_probs.get(label, 0.1)
        label_data[label] = np.random.choice([0, 1], num_samples, p=[1-prob, prob])
    
    return label_data

def generate_synthetic_data():
    """生成完整的合成训练数据"""
    print("开始生成合成训练数据...")

    # 设置随机种子以确保可重现性
    np.random.seed(42)
    random.seed(42)

    # 加载配置文件
    config_path = "dev/data/toml_files/default_conf_new.toml"
    config = load_config(config_path)

    # 获取MRI文件
    mri_paths, mri_emb_files = get_mri_files()
    num_samples = len(mri_paths)

    print(f"找到 {num_samples} 个MRI嵌入文件")

    # 初始化数据字典
    data = {}

    # 生成ID列
    data['ID'] = [f"NACC_{i+1:04d}" for i in range(num_samples)]

    # 处理特征
    print("生成特征数据...")
    # 检查配置文件结构
    feature_section = config.get('feature', {})
    print(f"找到 {len(feature_section)} 个特征定义")

    for feature_name, feature_config in feature_section.items():
        feature_type = feature_config['type']

        if feature_type == 'imaging':
            # 成像特征：使用MRI文件路径
            data[feature_name] = mri_paths
        elif feature_type == 'categorical':
            # 分类特征
            num_categories = feature_config['num_categories']
            data[feature_name] = generate_categorical_feature(num_samples, num_categories, feature_name)
        elif feature_type == 'numerical':
            # 数值特征
            data[feature_name] = generate_numerical_feature(num_samples, feature_name)
        else:
            print(f"警告：未知特征类型 {feature_type} for {feature_name}")

    # 生成标签
    print("生成标签数据...")
    label_data = generate_labels(num_samples)
    data.update(label_data)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 确保数值特征为适当的数据类型
    for feature_name, feature_config in config['feature'].items():  # 修复：使用'feature'而不是'features'
        if feature_config['type'] == 'numerical':
            df[feature_name] = df[feature_name].astype(float)
        elif feature_config['type'] == 'categorical':
            df[feature_name] = df[feature_name].astype(int)

    # 确保标签为整数类型
    for label_name in config['label'].keys():  # 修复：使用'label'而不是'labels'
        if label_name in df.columns:
            df[label_name] = df[label_name].astype(int)

    return df

def main():
    """主函数"""
    try:
        # 生成合成数据
        df = generate_synthetic_data()

        # 保存主要训练文件
        output_path = "data/training_cohorts/new_nacc_revised_selection.csv"
        df.to_csv(output_path, index=False)
        print(f"合成训练数据已保存到: {output_path}")

        # 打印数据统计信息
        print(f"\n数据统计:")
        print(f"样本数量: {len(df)}")
        print(f"特征数量: {len(df.columns)}")

        # 打印标签分布
        print(f"\n标签分布:")
        label_columns = ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']
        for label in label_columns:
            if label in df.columns:
                count = df[label].sum()
                percentage = (count / len(df)) * 100
                print(f"{label}: {count} ({percentage:.1f}%)")

        print(f"\n前5行数据预览:")
        print(df.head())

        print(f"\n数据生成完成！")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
