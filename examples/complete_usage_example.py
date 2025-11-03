"""
COLA 完整使用示例

展示如何使用 COLAData, Model, 和 COLA 类
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from xai_cola import COLA
from xai_cola.data import COLAData
from xai_cola.models import Model


# ========== 步骤 1: 准备数据 ==========
print("=" * 60)
print("步骤 1: 准备数据")
print("=" * 60)

# 创建示例数据
np.random.seed(42)
n_samples = 100
n_features = 5

X = np.random.randn(n_samples, n_features)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 转换为 DataFrame（推荐）
feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['Risk'] = y  # label column

print("\nFactual 数据:")
print(df.head())

# 初始化 COLAData
factual_data = COLAData(
    factual_data=df,
    label_column='Risk'
)

print(f"\n数据摘要: {factual_data.summary()}")


# ========== 步骤 2: 训练模型 ==========
print("\n" + "=" * 60)
print("步骤 2: 训练模型")
print("=" * 60)

# 分离特征和标签
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 训练简单模型
ml_model = RandomForestClassifier(n_estimators=10, random_state=42)
ml_model.fit(X_train, y_train)

print(f"模型准确率: {ml_model.score(X_test, y_test):.3f}")


# ========== 步骤 3: 生成 counterfactual ==========
print("\n" + "=" * 60)
print("步骤 3: 生成 Counterfactual")
print("=" * 60)

# 使用 DICE 生成 counterfactual（假设已安装）
try:
    from counterfactual_explainer.dice import DiCE
    
    # 取一个实例
    factual_instance = X_test[0:1]
    factual_label = y_test[0]
    
    print(f"原始实例: {factual_instance}")
    print(f"原始标签: {factual_label}")
    
    # 初始化 DiCE
    dice = DiCE(
        factual_data=factual_instance,
        ml_model=ml_model
    )
    
    # 生成 counterfactual
    counterfactual_instance = dice.generate()
    
    print(f"\nCounterfactual 实例: {counterfactual_instance}")
    
    # 转换为 DataFrame
    cf_df = pd.DataFrame(counterfactual_instance, columns=feature_names)
    cf_df['Risk'] = ml_model.predict(counterfactual_instance)
    
    print(f"Counterfactual 标签: {cf_df['Risk'].values}")
    
    # 添加 counterfactual
    factual_data.add_counterfactuals(cf_df)
    
    print(f"\n✅ Counterfactual 已添加")
    
except Exception as e:
    print(f"⚠️  DICE 不可用: {e}")
    print("使用模拟 counterfactual 数据...")
    
    # 使用模拟数据
    cf_instance = factual_instance.copy()
    cf_instance[0, 0] += 1  # 简单修改
    cf_df = pd.DataFrame(cf_instance, columns=feature_names)
    cf_df['Risk'] = ml_model.predict(cf_instance)
    factual_data.add_counterfactuals(cf_df)


# ========== 步骤 4: 使用 COLA 优化 ==========
print("\n" + "=" * 60)
print("步骤 4: 使用 COLA 优化")
print("=" * 60)

# 创建 Model 包装器
model = Model(ml_model, backend='sklearn')

# 初始化 COLA
cola = COLA(
    data=factual_data,
    ml_model=model
)

# 设置策略
print("\n设置 COLA 策略...")
cola.set_policy(
    matcher='ot',         # Optimal Transport
    attributor='pshap',   # PSHAP attribution
    Avalues_method='max'  # Maximum value method
)

# 生成 action-limited counterfactual
print("\n生成 action-limited counterfactual...")
limited_actions = 3  # 最多改变 3 个特征

factual_df, cf_df, ace_df = cola.get_refined_counterfactual(limited_actions)

print("\n原始 Factual:")
print(factual_df)

print("\n原始 Counterfactual:")
print(cf_df)

print("\nAction-Limited Counterfactual:")
print(ace_df)


# ========== 步骤 5: 可视化 ==========
print("\n" + "=" * 60)
print("步骤 5: 可视化")
print("=" * 60)

try:
    print("\n生成高亮显示...")
    _, cce_style, ace_style = cola.highlight_changes()
    
    print("Counterfactual vs Factual 变化:")
    # display(cce_style)  # 在 Jupyter 中显示
    
    print("\nAction-Limited vs Factual 变化:")
    # display(ace_style)  # 在 Jupyter 中显示
    
except Exception as e:
    print(f"⚠️  可视化失败: {e}")

try:
    print("\n生成热力图...")
    plot1, plot2 = cola.heatmap()
    print("✅ 热力图已生成")
    
except Exception as e:
    print(f"⚠️  热力图生成失败: {e}")


# ========== 步骤 6: 查询最小 actions ==========
print("\n" + "=" * 60)
print("步骤 6: 查询最小 actions")
print("=" * 60)

try:
    min_actions = cola.query_minimum_actions()
    print(f"\n✅ 最小 actions: {min_actions}")
    
except Exception as e:
    print(f"⚠️  查询失败: {e}")


print("\n" + "=" * 60)
print("完整示例完成！")
print("=" * 60)

