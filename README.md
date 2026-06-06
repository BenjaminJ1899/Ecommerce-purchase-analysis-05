# 客户购买行为分析 · 电子商务

> 基于 50 万条客户数据的购买驱动因素分析，识别影响购买决策的关键行为信号，评估客户分层和忠诚度计划的效果。

## 业务背景

某电商平台需要回答三个问题：**什么因素最能预测客户是否会购买？** 现有的 VIP/Premium/Regular 分层是否有效？忠诚度计划到底值不值？

本分析通过相关性分析和逻辑回归，量化各因素的预测能力，并发现了一个反直觉的事实：**预置的客户分层标签与实际购买行为几乎无关。**

## 数据说明

- 数据来源：Kaggle（CC-BY-NC 许可）
- 数据量：500,000 条客户记录（清洗后 499,381 条）
- 购买率：41.8%（正负样本均衡）

### 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| Age | int | 年龄（17-81） |
| AnnualIncome | float | 年收入（$11,966-$204,178） |
| NumberOfPurchases | int | 历史购买次数 |
| TimeSpentOnWebsite | float | 网站停留时间 |
| CustomerTenureYears | float | 客户年限 |
| LastPurchaseDaysAgo | int | 最近购买距今天数 |
| CustomerSatisfaction | int | 满意度（1-5） |
| CustomerSegment | str | 客户分层标签（VIP/Premium/Regular） |
| LoyaltyProgram | int | 是否忠诚度计划会员（0/1） |
| PurchaseStatus | int | 是否购买（目标变量，0/1） |

完整字段含 Gender、ProductCategory、PreferredDevice、Region、ReferralSource、DiscountsAvailed、SessionCount。

## 分析框架

| 步骤 | Notebook | 分析方法 | 回答的问题 |
|------|----------|---------|-----------|
| 1 | [数据清洗](notebooks/01_data_cleaning.ipynb) | 负值修复、年龄过滤、分类分布 | 数据可信吗？ |
| 2 | [探索性分析](notebooks/02_exploratory_analysis.ipynb) | 按分层/渠道/品类/设备拆解购买率 | 哪些维度有差异？差异多大？ |
| 3 | [购买驱动因素](notebooks/03_purchase_drivers.ipynb) | 相关性分析 + 逻辑回归 | 什么因素最能预测购买？ |

## 关键发现

### 1. Recency 是压倒性的最强预测因子

| 特征 | 与购买的相关性 |
|------|---------------|
| LastPurchaseDaysAgo | **-0.700** |
| CustomerSatisfaction | +0.125 |
| AnnualIncome | +0.088 |
| NumberOfPurchases | +0.066 |
| Age | -0.064 |

LastPurchaseDaysAgo 的预测能力是第二名的 5 倍以上。**最近买过的人极有可能再买，长久未买的人几乎不会买。** 逻辑回归模型 Test Accuracy 与 ROC-AUC 均验证了行为信号的有效性（详见 Notebook 03）。

### 2. 预置客户分层标签不反映购买行为

| 标签 | 用户占比 | 购买率 | 平均购买次数 | 平均收入 | 平均满意度 |
|------|---------|--------|------------|---------|----------|
| Regular | 22.7% | **44.2%** | 11.4 | $84,976 | 3.2 |
| Premium | 47.5% | 43.2% | 11.4 | $85,114 | 3.2 |
| VIP | 29.8% | 37.8% | 11.4 | $85,074 | 3.2 |

三个分层的所有行为指标几乎完全相同。**VIP 标签不等于更高的购买率，也不等于更高的消费或满意度。** 这个发现展示了"先验证标签再使用"的分析纪律——盲目相信数据标签会得出错误结论。

### 3. 忠诚度计划是最大的业务杠杆（+8.4pp）

- 忠诚度计划会员购买率：**46.0%**
- 非会员购买率：37.6%
- 提升幅度：**+8.4pp**

在所有可干预的因素中，忠诚度计划的提升效果最大。品类、渠道、设备的差异均不到 2pp。

### 4. 行为信号远强于人口画像

年龄、性别、地区、品类的购买率差异全部在 2pp 以内。真正区分买与不买的是行为信号——多久前买过、满意度多少。

## 数据局限性

- **循环论证风险（最重要）**：LastPurchaseDaysAgo（最近购买距今天数）与 PurchaseStatus（是否购买）可能共享同一个参考时间点。如果 PurchaseStatus 定义为"最近 N 天内是否购买"，则 r=-0.700 至少部分是定义性的——购买者必然 LastPurchaseDaysAgo 较小。在确认两者独立定义之前，"Recency 是最强预测因子"的结论需要谨慎解读
- **数据可能为合成数据**：三个 CustomerSegment 的行为指标（购买次数 11.4、收入 ~$85k、满意度 3.2）精确到多位小数完全一致，所有分类维度购买率差异 < 2pp，且 50 万条数据零缺失零重复——真实世界的电商数据极少如此"干净"。结论的外部有效性需在真实数据上验证
- CustomerSegment 标签的来源和定义未在数据集中说明，其与行为指标无相关性可能是数据构造方式导致的
- 无时间序列信息，无法构建纵向的购买行为轨迹；无法用历史窗口特征预测未来窗口的购买行为

## 业务建议

1. **把资源从"客群画像运营"转向"行为触发运营"**：与其按"25-35岁女性"推优惠，不如按"最近 30 天没买且满意度<3 分"触发召回
2. **扩大忠诚度计划覆盖**：当前会员与非会员各占一半，扩大会员覆盖可能带来显著的总体购买率提升
3. **放弃按现有 Segment 做差异化运营**：标签不反映实际行为差异，建议基于 RFM 或行为聚类重新构建分层

## 工具

Python (pandas, matplotlib, seaborn, scikit-learn) · Jupyter Notebook

## 快速复现

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

*2026-06-03*
