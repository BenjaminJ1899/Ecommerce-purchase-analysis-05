"""快速数据探查脚本。

打印数据的基本统计信息（维度、类型、缺失值、分类分布等），
用于在正式分析前快速了解数据结构。

用法：从 utils/ 目录运行
    cd utils && python explore.py
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_raw_data

df = load_raw_data()
print(f'\nShape: {df.shape}')
print(f'\nDtypes:\n{df.dtypes}')
print(f'\nMissing:\n{df.isnull().sum()}')
print(f'\nNumeric describe:\n{df.describe().to_string()}')

cat_cols = ['Gender','ProductCategory','PreferredDevice','Region','ReferralSource',
            'CustomerSegment','LoyaltyProgram','PurchaseStatus']
for col in cat_cols:
    if col in df.columns:
        print(f'\n--- {col} ---')
        print(df[col].value_counts())

print(f'\nDuplicate rows: {df.duplicated().sum()}')
print(f'\nAge range: {df["Age"].min()} - {df["Age"].max()}')
print(f'Income range: {df["AnnualIncome"].min():.0f} - {df["AnnualIncome"].max():.0f}')
print(f'Purchase rate: {df["PurchaseStatus"].mean()*100:.2f}%')
print(f'Satisfaction mean: {df["CustomerSatisfaction"].mean():.2f}')
