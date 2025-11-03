# COLA Data 接口快速参考

## 🚀 快速开始

```python
from xai_cola.data import COLAData
```

## 📝 基本用法

### 初始化

```python
# 方式 1: Pandas DataFrame（推荐）
data = COLAData(
    factual_data=df,           # 包含所有列，包括 label
    label_column='Risk'         # label 列名
)

# 方式 2: 带数据预处理（transform）
data = COLAData(
    factual_data=df,
    label_column='Risk',
    transform='ohe-zscore',     # 或 'ohe-min-max', None
    numerical_features=['Age', 'Credit amount', 'Duration']  # 指定数值特征
)

# 方式 3: 带 counterfactual
data = COLAData(
    factual_data=factual_df,
    label_column='Risk',
    counterfactual_data=cf_df  # 可选
)

# 方式 4: NumPy Array
data = COLAData(
    factual_data=np_array,
    label_column='Risk',
    column_names=['col1', 'col2', 'Risk']  # 必须提供列名
)
```

### 添加 Counterfactual（稍后）

```python
# 初始化时不带 counterfactual
data = COLAData(factual_data=df, label_column='Risk')

# 稍后添加
data.add_counterfactuals(cf_df)

# 或使用 numpy
data.add_counterfactuals(cf_array)
```

## 📊 获取数据

```python
# 获取列名
columns = data.get_all_columns()           # 所有列（含 label）
features = data.get_feature_columns()     # 特征列（不含 label）
label_name = data.get_label_column()      # label 列名

# 获取 Factual 数据
df_all = data.get_factual_all()            # 完整（含 label）
df_features = data.get_factual_features() # 特征（不含 label）
labels = data.get_factual_labels()        # 标签

# 获取 Counterfactual 数据
cf_all = data.get_counterfactual_all()
cf_features = data.get_counterfactual_features()
cf_labels = data.get_counterfactual_labels()

# NumPy 转换
np_features = data.to_numpy_factual_features()
np_cf_features = data.to_numpy_counterfactual_features()
```

## 📋 信息方法

```python
# 检查
has_cf = data.has_counterfactual()  # bool

# 统计
n_features = data.get_feature_count()  # int
n_samples = data.get_sample_count()    # int

# 摘要
info = data.summary()  # dict
```

## ⚠️ 注意事项

1. **列名验证**
   - Pandas: 自动检查 label_column 是否存在
   - NumPy: 必须提供 column_names

2. **Counterfactual 验证**
   - 列必须与 factual 完全一致
   - NumPy 会自动使用 factual 的列名

3. **Label Column 位置**
   - 推荐放在最后一列
   - 不强制，只要是合法列名即可

## 🎯 常见用法

### 完整工作流

```python
# 1. 初始化
data = COLAData(factual_data=df, label_column='Risk')

# 2. 生成 counterfactual（使用外部工具）
explainer = DiCE(model)
cf_features = explainer.generate_counterfactuals(
    data.get_factual_features()
)

# 3. 构建 counterfactual DataFrame
cf_df = df.copy()
cf_df.iloc[:, :-1] = cf_features
data.add_counterfactuals(cf_df)

# 4. 使用
cola = COLA(
    data=data,
    ml_model=model,
    x_factual=data.to_numpy_factual_features(),
    x_counterfactual=data.to_numpy_counterfactual_features()
)
```

## 📚 API 参考

### 初始化参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `factual_data` | Union[DataFrame, ndarray] | ✅ | 事实数据 |
| `label_column` | str | ✅ | 标签列名 |
| `counterfactual_data` | Union[DataFrame, ndarray] | ❌ | 反事实数据 |
| `column_names` | List[str] | NumPy 必需 | 列名列表 |
| `transform` | str, optional | ❌ | 数据预处理方法: "ohe-zscore", "ohe-min-max", None |
| `numerical_features` | List[str], optional | ❌ | 数值特征列表，用于区分数值和分类特征 |

### 主要方法

| 方法 | 返回 | 说明 |
|------|------|------|
| `get_all_columns()` | List[str] | 所有列名 |
| `get_factual_features()` | DataFrame | 特征数据 |
| `get_factual_labels()` | Series | 标签数据 |
| `add_counterfactuals()` | None | 添加反事实 |
| `has_counterfactual()` | bool | 是否已设置 |
| `summary()` | dict | 数据摘要 |
| `_transform(data)` | DataFrame | 数据变换（内部方法） |
| `_inverse_transform(data)` | DataFrame | 逆变换（内部方法） |

## 🔗 相关文档

- 详细使用指南: `NEW_DATA_INTERFACE.md`
- WachterCF 使用指南: `WACHTERCF_USAGE.md`
- 示例代码: `examples/data_usage_example.py`

