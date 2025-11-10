# COLAData Transform Method 使用示例

`COLAData` 现在支持数据预处理转换，这对于使用 DisCount 等需要标准化数据的算法非常有用。

## 基本用法

### 1. 使用 StandardScaler

```python
from sklearn.preprocessing import StandardScaler
from xai_cola.ce_sparsifier.data import COLAData
from xai_cola.ce_generator import DisCount
from xai_cola.ce_sparsifier.models import Model

# 准备数据
X_train = ...  # 训练数据
X_test = ...   # 测试数据
df_test = ...  # 包含 label 列的测试 DataFrame

# 1. 训练预处理器
scaler = StandardScaler()
scaler.fit(X_train)

# 2. 创建 COLAData，传入 transform_method
data = COLAData(
    factual_data=df_test,
    label_column='Risk',
    numerical_features=['Age', 'Credit amount', 'Duration'],
    transform_method=scaler  # 关键：传入已训练的 scaler
)

# 3. 使用 DisCount（会自动使用 transform_method）
explainer = DisCount(ml_model=ml_model)
factual, counterfactual = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    lr=0.1,
    max_iter=15
)
```

### 2. 使用 ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 定义特征列
numerical_features = ['Age', 'Credit amount', 'Duration']
categorical_features = ['Sex', 'Job', 'Saving accounts']

# 创建 ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

# 训练预处理器
preprocessor.fit(X_train)

# 使用 preprocessor（transform_method 的别名）
data = COLAData(
    factual_data=df_test,
    label_column='Risk',
    numerical_features=numerical_features,
    preprocessor=preprocessor  # 可以使用 preprocessor 作为别名
)
```

### 3. 使用 Pipeline 中的预处理器

```python
from sklearn.pipeline import Pipeline

# 如果你有一个 Pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])
pipe.fit(X_train, y_train)

# 提取预处理器
data = COLAData(
    factual_data=df_test,
    label_column='Risk',
    transform_method=pipe.named_steps['preprocessor']
)
```

## 工作原理

### DisCount 内部流程

```python
# 1. DisCount 检查是否有 transform_method
if self.data.transform_method is not None:
    # 2. 转换事实数据
    x_transformed = self.data._transform(x_factual)

    # 3. 在转换后的空间中生成反事实
    counterfactual_transformed = discount_explainer.optimize(...)

    # 4. 逆转换回原始空间
    counterfactual = self.data._inverse_transform(counterfactual_transformed)
else:
    # 不使用转换，直接处理原始数据
    counterfactual = discount_explainer.optimize(x_factual)
```

### 手动使用转换方法

```python
# 你也可以手动调用转换方法
data = COLAData(
    factual_data=df,
    label_column='Risk',
    transform_method=scaler
)

# 转换数据
factual_features = data.get_factual_features()
transformed = data._transform(factual_features)

# 逆转换
original = data._inverse_transform(transformed)
```

## 完整示例

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from xai_cola.ce_sparsifier.data import COLAData
from xai_cola.ce_sparsifier.models import Model
from xai_cola.ce_generator import DisCount

# 1. 加载数据
df = pd.read_csv('german_credit.csv')
X = df.drop(columns=['Risk'])
y = df['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. 训练预处理器
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 训练 PyTorch 模型（在标准化数据上）
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

model = BinaryClassifier(X_train.shape[1])
# ... 训练模型 ...

# 4. 包装模型
ml_model = Model(model=model, backend="pytorch")

# 5. 准备测试数据（Risk=1 的样本）
df_test = pd.concat([X_test, y_test], axis=1)
df_risk_1 = df_test[df_test['Risk'] == 1].head(10)

# 6. 创建 COLAData（传入 transform_method）
data = COLAData(
    factual_data=df_risk_1,
    label_column='Risk',
    numerical_features=X.columns.tolist(),
    transform_method=scaler  # 关键！
)

# 7. 使用 DisCount
explainer = DisCount(ml_model=ml_model)
factual, counterfactual = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    lr=0.1,
    n_proj=10,
    U_1=0.4,
    U_2=0.3,
    max_iter=15,
    tau=100,
    silent=False
)

print("Factual samples:")
print(factual.head())
print("\nCounterfactual samples:")
print(counterfactual.head())
```

## 注意事项

1. **必须先训练预处理器**：`transform_method` 必须是已经在训练数据上 fit 过的
2. **必须有 transform 和 inverse_transform 方法**：预处理器必须支持双向转换
3. **模型训练与预处理一致**：如果使用 transform_method，确保模型是在转换后的数据上训练的
4. **transform_method 与 preprocessor 是别名**：两者选其一即可，不能同时指定

## 支持的预处理器

任何有 `transform()` 和 `inverse_transform()` 方法的对象都可以使用：

- ✅ `sklearn.preprocessing.StandardScaler`
- ✅ `sklearn.preprocessing.MinMaxScaler`
- ✅ `sklearn.preprocessing.RobustScaler`
- ✅ `sklearn.compose.ColumnTransformer`
- ✅ 自定义转换器（只要实现了这两个方法）

## 错误处理

```python
# 错误1：同时指定两个参数
data = COLAData(
    factual_data=df,
    label_column='Risk',
    transform_method=scaler,
    preprocessor=scaler  # ❌ 错误！
)
# ValueError: Cannot specify both 'transform_method' and 'preprocessor'

# 错误2：预处理器缺少必要方法
class BadTransformer:
    def transform(self, X):
        return X
    # 缺少 inverse_transform

data = COLAData(
    factual_data=df,
    label_column='Risk',
    transform_method=BadTransformer()  # ❌ 错误！
)
# ValueError: transform_method must have an 'inverse_transform()' method
```

## 检查是否使用转换

```python
data = COLAData(
    factual_data=df,
    label_column='Risk',
    transform_method=scaler
)

# 检查
print(data.summary())
# {'has_transform_method': True, ...}

# 或直接检查
if data.transform_method is not None:
    print("Using data transformation")
```
