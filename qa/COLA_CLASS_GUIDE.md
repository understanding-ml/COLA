# COLA 类指南

## 📍 COLA 类位置

**主文件**: `xai_cola/cola.py`

```python
# 导入方式
from xai_cola import COLA
```

## 🎯 COLA 类的作用

`COLA` 类是整个框架的**总控制器**，负责：

1. **数据管理** - 通过 `COLAData` 接口管理 factual 和 counterfactual 数据
2. **模型管理** - 通过 `Model` 接口管理机器学习模型
3. **策略控制** - 设置匹配策略（matching）和特征归因策略（attribution）
4. **结果生成** - 生成 action-limited counterfactual
5. **可视化** - 提供高亮显示和热力图

## 📝 使用流程

### 1. 初始化数据（COLAData）

```python
from xai_cola.data import COLAData

# 使用 DataFrame（推荐）
data = COLAData(
    factual_data=df,
    label_column='Risk'  # 目标列名称
)

# 添加 counterfactual
data.add_counterfactuals(cf_df)
```

### 2. 初始化模型（Model）

```python
from xai_cola.models import Model

# 包装你的 ML 模型
model = Model(ml_model, backend='sklearn')  # 或 'pytorch'
```

### 3. 使用 COLA

```python
from xai_cola import COLA

# 初始化 COLA
cola = COLA(
    data=factual_data,
    ml_model=model
)

# 设置策略
cola.set_policy(
    matcher='ot',         # 匹配策略: 'ot', 'nn', 'ect'
    attributor='pshap',   # 归因策略: 'pshap'
    Avalues_method='max'  # 计算方法: 'max'
)

# 生成优化后的 counterfactual
factual_df, cf_df, ace_df = cola.get_refined_counterfactual(limited_actions=3)

# 可视化
_, style1, style2 = cola.highlight_changes()
plot1, plot2 = cola.heatmap()

# 查询最小 actions
min_actions = cola.query_minimum_actions()
```

## 🔧 COLA 类方法

### 初始化方法

```python
__init__(
    self,
    data: COLAData,              # 数据容器
    ml_model: Model,             # 模型接口
    x_factual: np.ndarray = None,           # 可选：直接提供 factual 数组
    x_counterfactual: np.ndarray = None     # 可选：直接提供 counterfactual 数组
)
```

### 策略设置

```python
set_policy(
    self,
    matcher: str = "ot",         # 匹配策略
    attributor: str = "pshap",   # 归因策略
    Avalues_method: str = "max", # 计算方法
    **kwargs                      # 额外参数
)
```

**Matcher 选项**:
- `"ot"` - Optimal Transport (最优传输)
- `"nn"` - Nearest Neighbor (最近邻)
- `"ect"` - Exact Matching (精确匹配)
- `"cem"` - Coarsened Exact Matching (暂不可用)

### 核心方法

```python
# 生成 action-limited counterfactual
get_refined_counterfactual(self, limited_actions: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

# 高亮显示变化
highlight_changes(self) -> Tuple[pd.DataFrame, Styler, Styler]

# 生成热力图
heatmap(self) -> Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]

# 查询最小 actions
query_minimum_actions(self) -> int
```

## 📊 项目结构

```
xai_cola/
├── __init__.py              # 导出: COLA, data, models
├── cola.py                  # ⭐ COLA 主类（这里！）
│
├── data/
│   ├── __init__.py
│   └── coladata.py          # COLAData 类
│
├── models/
│   ├── __init__.py
│   ├── base.py              # Model 基类
│   ├── sklearn.py           # Scikit-learn 实现
│   └── pytorch.py           # PyTorch 实现
│
├── policies/
│   ├── matching/            # 匹配策略
│   ├── feature_attributor/  # 特征归因
│   └── data_composer/       # 数据组合
│
└── visualization/           # 可视化工具
```

## 🔗 相关导入

```python
# 主类
from xai_cola import COLA

# 数据
from xai_cola.data import COLAData

# 模型
from xai_cola.models import Model

# 如果需要直接访问策略
from xai_cola.policies.matching import CounterfactualOptimalTransportPolicy
from xai_cola.policies.feature_attributor import PSHAP
```

## 💡 完整示例

参见:
- `examples/data_usage_example.py` - COLAData 使用
- `examples/complete_usage_example.py` - 完整 COLA 流程
- `demo.ipynb` - Jupyter 演示

## 🆘 常见问题

**Q: COLA 类在哪里？**
A: `xai_cola/cola.py`

**Q: 如何导入？**
A: `from xai_cola import COLA`

**Q: 需要先初始化什么？**
A: 先初始化 `COLAData` 和 `Model`，然后传递给 `COLA`

**Q: COLA 类职责太重吗？**
A: 目前实现是合理的，详见 `COLA_CLASS_RECOMMENDATION.md`

