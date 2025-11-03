"""
COLA Data 使用示例

展示如何使用新的 COLAData 接口
"""

import pandas as pd
import numpy as np
from xai_cola.data import COLAData

# ========== 示例 1: Pandas DataFrame 输入（推荐） ==========

print("=" * 60)
print("示例 1: 使用 Pandas DataFrame")
print("=" * 60)

# 创建示例数据
factual_df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [2, 3, 4, 5, 6],
    'feature3': [3, 4, 5, 6, 7],
    'Risk': [0, 1, 0, 1, 0]  # label column
})

print("\nFactual 数据:")
print(factual_df)

# 初始化（自动验证）
data = COLAData(
    factual_data=factual_df,
    label_column='Risk'
)

print(f"\n数据摘要: {data.summary()}")
print(f"所有列: {data.get_all_columns()}")
print(f"特征列: {data.get_feature_columns()}")

# 获取各种数据
print("\n获取完整数据（包含 label）:")
print(data.get_factual_all())

print("\n获取特征数据（不含 label）:")
print(data.get_factual_features())

print("\n获取标签:")
print(data.get_factual_labels())


# ========== 示例 2: 添加 counterfactual ==========

print("\n" + "=" * 60)
print("示例 2: 添加 Counterfactual")
print("=" * 60)

# 创建 counterfactual 数据（列必须一致）
counterfactual_df = pd.DataFrame({
    'feature1': [10, 20, 30, 40, 50],
    'feature2': [20, 30, 40, 50, 60],
    'feature3': [30, 40, 50, 60, 70],
    'Risk': [1, 0, 1, 0, 1]  # label column
})

print("\nCounterfactual 数据:")
print(counterfactual_df)

# 添加 counterfactual
data.add_counterfactuals(counterfactual_df)

print(f"\n是否包含 counterfactual: {data.has_counterfactual()}")
print(f"Counterfactual 数据:")
print(data.get_counterfactual_all())


# ========== 示例 3: NumPy 输入 ==========

print("\n" + "=" * 60)
print("示例 3: 使用 NumPy Array")
print("=" * 60)

# 创建 NumPy 数据
factual_array = np.array([
    [1, 2, 3, 0],  # features + label
    [2, 3, 4, 1],
    [3, 4, 5, 0],
    [4, 5, 6, 1],
])

# 提供列名（包含 label column）
column_names = ['feature1', 'feature2', 'feature3', 'Risk']

# 初始化
data_numpy = COLAData(
    factual_data=factual_array,
    label_column='Risk',
    column_names=column_names
)

print("\nNumPy 输入的数据:")
print(data_numpy.get_factual_all())


# ========== 示例 4: NumPy counterfactual ==========

print("\n" + "=" * 60)
print("示例 4: NumPy counterfactual")
print("=" * 60)

# 创建 NumPy counterfactual（列数必须一致）
counterfactual_array = np.array([
    [10, 20, 30, 1],
    [20, 30, 40, 0],
    [30, 40, 50, 1],
])

# 添加 counterfactual（自动使用 factual 的列名）
data_numpy.add_counterfactuals(counterfactual_array)

print("\nCounterfactual (NumPy):")
print(data_numpy.get_counterfactual_all())


# ========== 示例 5: 错误处理 ==========

print("\n" + "=" * 60)
print("示例 5: 错误处理示例")
print("=" * 60)

try:
    # 尝试使用错误的 label column
    wrong_data = COLAData(
        factual_data=factual_df,
        label_column='WrongColumn'  # 不存在的列
    )
except ValueError as e:
    print(f"✅ 正确捕获错误: {e}")

try:
    # 尝试添加列不一致的 counterfactual
    wrong_cf = pd.DataFrame({
        'wrong_col': [1, 2, 3],
        'wrong_col2': [4, 5, 6]
    })
    data.add_counterfactuals(wrong_cf)
except ValueError as e:
    print(f"✅ 正确捕获错误: {e}")

print("\n" + "=" * 60)
print("所有示例完成！")
print("=" * 60)

