"""共享数据加载与清洗模块。

三个 notebook 和 analysis.py 统一通过此模块加载清洗后的数据，
避免清洗逻辑重复。清洗规则变更时只需修改此文件。
"""

import os
import pandas as pd


def get_project_root():
    """返回项目根目录（utils/ 的上一级）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_raw_data():
    """加载原始 CSV，不做任何清洗。"""
    csv_path = os.path.join(get_project_root(), 'customerData_500k.csv')
    return pd.read_csv(csv_path)


def load_cleaned_data():
    """加载并清洗数据。

    清洗步骤：
    1. 负值修复：NumberOfPurchases / TimeSpentOnWebsite /
       CustomerTenureYears / LastPurchaseDaysAgo 中的负值 → 0
    2. 年龄过滤：移除 Age < 17（有独立收入但未成年的矛盾数据）

    Returns:
        pd.DataFrame: 清洗后的数据
    """
    df = load_raw_data()

    # 1) 负值修复
    neg_cols = ['NumberOfPurchases', 'TimeSpentOnWebsite',
                'CustomerTenureYears', 'LastPurchaseDaysAgo']
    for col in neg_cols:
        df.loc[df[col] < 0, col] = 0

    # 2) 移除年龄异常
    df = df[df['Age'] >= 17]

    return df
