# 新的 COLA Data 接口使用指南

## 🎯 概述

新的 `COLAData` 类提供了统一、清晰、易用的数据接口，支持：
- ✅ Pandas DataFrame 和 NumPy array 输入
- ✅ 自动验证数据一致性
- ✅ 同时管理 factual 和 counterfactual
- ✅ 灵活的初始化方式

## 📦 安装和使用

```python
from xai_cola.data import COLAData
```

## 🚀 快速开始

### 基本用法

```python
import pandas as pd
from xai_cola.data import COLAData

# 1. 准备数据（包含 label column）
factual_df = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': [2, 3, 4],
    'Risk': [0, 1, 0]  # label column
})

# 2. 初始化
data = COLAData(
    factual_data=factual_df,
    label_column='Risk'
)

# 3. 获取数据
print(data.get_factual_features())  # 特征数据
print(data.get_factual_labels())    # 标签数据
print(data.get_all_columns())       # 所有列名
```

## 📚 详细 API

### 初始化

```python
COLAData(
    factual_data: Union[pd.DataFrame, np.ndarray],  # 必须
    label_column: str,                              # 必须
    counterfactual_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,  # 可选
    column_names: Optional[List[str]] = None         # 仅 numpy 需要
)
```

### 方法概览

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `get_all_columns()` | `List[str]` | 所有列名（含 label） |
| `get_feature_columns()` | `List[str]` | 特征列名（不含 label） |
| `get_factual_all()` | `pd.DataFrame` | 完整 factual（含 label） |
| `get_factual_features()` | `pd.DataFrame` | Factual 特征（不含 label） |
| `get_factual_labels()` | `pd.Series` | Factual 标签 |
| `get_counterfactual_all()` | `pd.DataFrame` | 完整 counterfactual（含 label） |
| `get_counterfactual_features()` | `pd.DataFrame` | Counterfactual 特征（不含 label） |
| `get_counterfactual_labels()` | `pd.Series` | Counterfactual 标签 |
| `add_counterfactuals()` | `None` | 添加/更新 counterfactual |
| `has_counterfactual()` | `bool` | 是否设置了 counterfactual |
| `summary()` | `dict` | 数据摘要信息 |

## 💡 使用场景

### 场景 1: Pandas DataFrame（最简单）

```python
import pandas as pd
from xai_cola.data import COLAData

# Factual 数据
factual = pd.DataFrame({
    'Age': [25, 30, 35],
    'Income': [50000, 60000, 70000],
    'Risk': [0, 1, 0]  # label
})

# 初始化
data = COLAData(
    factual_data=factual,
    label_column='Risk'
)

# 使用
features = data.get_factual_features()  # 只有 Age, Income
labels = data.get_factual_labels()      # 只有 Risk
```

### 场景 2: NumPy Array

```python
import numpy as np
from xai_cola.data import COLAData

# NumPy array（必须包含 label column）
factual_array = np.array([
    [25, 50000, 0],
    [30, 60000, 1],
    [35, 70000, 0]
])

# 提供列名
column_names = ['Age', 'Income', 'Risk']

# 初始化
data = COLAData(
    factual_data=factual_array,
    label_column='Risk',
    column_names=column_names
)
```

### 场景 3: 添加 Counterfactual

```python
# 方式 1: 初始化时添加
cf_df = pd.DataFrame({
    'Age': [30, 35, 40],
    'Income': [55000, 65000, 75000],
    'Risk': [1, 0, 1]
})

data = COLAData(
    factual_data=factual,
    label_column='Risk',
    counterfactual_data=cf_df  # 初始化时添加
)

# 方式 2: 稍后添加
data = COLAData(factual_data=factual, label_column='Risk')
# ... 其他操作 ...
data.add_counterfactuals(cf_df)  # 添加 counterfactual
```

### 场景 4: NumPy counterfactual

```python
# Factual 是 DataFrame
data = COLAData(factual_data=factual, label_column='Risk')

# Counterfactual 是 NumPy array（自动使用 factual 的列名）
cf_array = np.array([
    [30, 55000, 1],
    [35, 65000, 0]
])

data.add_counterfactuals(cf_array)  # 自动使用 factual 的列名
```

## ⚠️ 验证规则

### Factual 验证

1. **Pandas DataFrame**:
   - 必须包含 `label_column`
   - 如果不存在会抛出 `ValueError`

2. **NumPy Array**:
   - 必须提供 `column_names`
   - `column_names` 必须包含 `label_column`
   - 维度必须匹配

### Counterfactual 验证

1. **Pandas DataFrame**:
   - 列必须与 factual 完全一致（名称和顺序）

2. **NumPy Array**:
   - 列数必须与 factual 一致
   - 自动使用 factual 的列名

## 🎯 最佳实践

### 1. 推荐：始终使用 Pandas DataFrame

```python
# ✅ 推荐
data = COLAData(factual_data=df, label_column='Risk')

# ⚠️ 可行但不推荐
data = COLAData(
    factual_data=np_array,
    label_column='Risk',
    column_names=columns
)
```

### 2. Label Column 放在最后一列

```python
# ✅ 推荐
factual = pd.DataFrame({
    'feature1': [...],
    'feature2': [...],
    'Risk': [...]  # 最后一列
})

# ✅ 也可接受
factual = pd.DataFrame({
    'Risk': [...],  # 不是最后一列
    'feature1': [...],
    'feature2': [...]
})
```

### 3. Counterfactual 列顺序要一致

```python
# ✅ 正确
factual = pd.DataFrame({'A': [...], 'B': [...], 'Risk': [...]})
cf = pd.DataFrame({'A': [...], 'B': [...], 'Risk': [...]})

# ❌ 错误（顺序不一致）
cf = pd.DataFrame({'B': [...], 'A': [...], 'Risk': [...]})  # 会报错
```

## 📊 完整示例

```python
import pandas as pd
import numpy as np
from xai_cola.data import COLAData
from xai_cola import COLA
from xai_cola.models import Model
from counterfactual_explainer import DiCE

# 1. 准备数据
factual_df = pd.DataFrame({
    'Age': [25, 30, 35],
    'Income': [50000, 60000, 70000],
    'Risk': [0, 1, 0]
})

# 2. 初始化 COLAData
data = COLAData(
    factual_data=factual_df,
    label_column='Risk'
)

# 3. 生成反事实（使用其他工具）
explainer = DiCE(ml_model=model)
factual_features = data.get_factual_features()
cf_features = explainer.generate_counterfactuals(factual_features)

# 4. 添加 counterfactual
cf_df = factual_df.copy()
cf_df.iloc[:, :-1] = cf_features  # 替换特征
data.add_counterfactuals(cf_df)

# 5. 使用 COLA
cola = COLA(
    data=data,
    ml_model=model,
    x_factual=factual_features.values,
    x_counterfactual=cf_features
)
```

## 🔄 迁移指南

### 从旧接口迁移

```python
# 旧方式
from xai_cola.data import PandasData
data = PandasData(df, target_name='Risk')

# 新方式（推荐）
from xai_cola.data import COLAData
data = COLAData(factual_data=df, label_column='Risk')

# 兼容性：旧接口仍然可用
from xai_cola.data import PandasData  # 仍然可用
```

## 🎉 优势总结

1. ✅ **统一的接口** - 一个类处理所有情况
2. ✅ **自动验证** - 减少错误
3. ✅ **灵活初始化** - 支持延迟添加 counterfactual
4. ✅ **清晰的命名** - 不使用 target_name，使用 label_column
5. ✅ **完整的 API** - 获取各种形式的数据
6. ✅ **向后兼容** - 旧接口仍然可用

