# COLA 软件架构分析

## 📐 架构概览

COLA 采用**模块化分层架构**，通过清晰的职责划分实现了**高内聚、低耦合**的设计。

```
┌─────────────────────────────────────────────────────────┐
│                      应用层 (User Layer)                   │
│                   from xai_cola import COLA                │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                     核心层 (Core Layer)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  COLA 类 - 协调器/编排器                              │  │
│  │  职责：协调各模块完成反事实精炼任务                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        │             │             │             │
┌───────▼──────┐ ┌───▼──────┐ ┌─────▼──────┐ ┌───▼──────────┐
│  数据层      │ │  模型层  │ │  策略层    │ │  可视化层    │
│ Data Layer   │ │ Model    │ │ Policies   │ │ Visualization│
└──────────────┘ └──────────┘ └────────────┘ └──────────────┘
     │                │             │
     └────────────────┴─────────────┘
              依赖关系
```

## 🧩 模块构成

### 1. **核心模块 (Core Module)**

#### `cola.py` - COLA 主类（编排器）

**职责：** 作为顶层协调器，整合所有模块完成反事实精炼任务

**特点：**
- 🎯 **高内聚**: 所有反事实精炼逻辑集中在一个类
- 🔗 **低耦合**: 通过接口（BaseData, Model）与其他模块交互
- 🎛️ **策略模式**: 支持多种匹配和归因策略的动态切换

**关键方法：**
```python
class COLA:
    def __init__(data, ml_model, x_factual, x_counterfactual)  # 初始化
    def set_policy(matcher, attributor, Avalues_method)         # 设置策略
    def get_refined_counterfactual(limited_actions)            # 精炼反事实
    def highlight_changes()                                     # 高亮变化
    def heatmap()                                              # 生成热力图
```

**协作方式：**
- 使用 `data` 获取数据特征名称
- 使用 `ml_model` 进行预测
- 使用 `policies` 执行匹配、归因和合成
- 使用 `visualization` 展示结果

---

### 2. **数据模块 (Data Module)**

#### 结构：
```
data/
├── base.py          # 抽象基类
├── pandas.py        # PandasData 实现
└── numpy.py         # NumpyData 实现
```

#### `BaseData` - 抽象接口

**职责：** 定义统一的数据访问接口

**设计模式：**
- ✅ **策略模式** + **接口隔离**
- 🎯 **高内聚**: 数据相关操作集中
- 🔗 **低耦合**: 只定义接口，不依赖具体实现

**核心方法：**
```python
class BaseData(ABC):
    def get_x()              # 获取特征数据
    def get_y()              # 获取目标变量
    def get_target_name()    # 获取目标名称
    def get_x_labels()       # 获取特征名称
    def get_dataframe()      # 获取 DataFrame
    def get_numpy()          # 获取 NumPy 数组
```

**独立工作：**
- 数据格式转换
- 特征名称管理
- 目标变量处理

**协同工作：**
- 为 COLA 提供统一的特征名称和目标信息
- 为 policies 提供数据访问接口

---

### 3. **模型模块 (Model Module)**

#### 结构：
```
models/
├── base.py          # 抽象基类
├── factory.py       # 工厂类
├── pytorch.py       # PyTorch 实现
└── sklearn.py       # Sklearn 实现
```

#### `BaseModel` - 抽象接口

**职责：** 定义统一的模型预测接口

**设计模式：**
- ✅ **工厂模式** + **适配器模式**
- 🎯 **高内聚**: 模型相关操作集中
- 🔗 **低耦合**: 通过接口抽象不同框架

**核心方法：**
```python
class BaseModel(ABC):
    def predict(x_factual)      # 预测
    def predict_proba(X)        # 预测概率
```

#### `Model` - 工厂类

**职责：** 根据 backend 自动选择合适的实现

**独立工作：**
- 模型类型判断
- 自动选择实现类

**协同工作：**
- 为 COLA 提供统一预测接口
- 为 policies 提供模型访问

---

### 4. **策略模块 (Policy Module)**

#### 结构：
```
policies/
├── matching/           # 匹配策略
│   ├── base_matcher.py
│   ├── ot_matcher.py   # 最优传输
│   ├── ect_matcher.py  # 精确匹配
│   ├── nn_matcher.py   # 最近邻
│   └── cem_matcher.py  # 粗化精确匹配
├── feature_attributor/ # 特征归因
│   ├── base_attributor.py
│   └── pshap.py        # PSHAP
└── data_composer/      # 数据合成
    └── data_composer.py
```

#### 设计原则：

**🎯 高内聚：**
- 每个子模块专注于一种策略（匹配/归因/合成）
- 相关算法放在同一目录

**🔗 低耦合：**
- 通过抽象基类定义接口
- 策略可以独立替换
- 模块间通过数据流连接

#### Matching - 匹配策略

**职责：** 建立事实和反事实之间的对应关系

**独立工作：**
- 计算联合概率矩阵
- 实现不同的匹配算法

**协同工作：**
- 为 attributor 提供匹配结果
- 为 data_composer 提供概率矩阵

#### Feature Attributor - 特征归因

**职责：** 计算特征重要性/归因分数

**独立工作：**
- 计算 φ 矩阵（特征重要性分布）
- 不同的归因算法（PSHAP, RandomShap）

**协同工作：**
- 使用 matcher 的结果作为输入
- 为 COLA 提供特征重要性

#### Data Composer - 数据合成

**职责：** 根据匹配结果合成新的反事实

**独立工作：**
- 根据概率分布合成数据
- 不同的合成方法（max, mean 等）

**协同工作：**
- 使用 matcher 的概率矩阵
- 为 COLA 提供精炼的反事实

---

### 5. **可视化模块 (Visualization Module)**

#### 结构：
```
visualization/
├── heatmap.py              # 热力图
└── highlight_dataframe.py  # 高亮显示
```

**职责：** 展示数据变化

**独立工作：**
- 生成热力图
- 高亮显示差异

**协同工作：**
- 接收 COLA 处理后的数据
- 提供可视化展示

---

## 🎯 高内聚、低耦合分析

### ✅ 高内聚 (High Cohesion)

每个模块内部职责单一、紧密相关：

1. **数据模块** - 所有数据操作集中
2. **模型模块** - 所有预测操作集中
3. **策略模块** - 策略按功能细分（匹配、归因、合成）
4. **可视化模块** - 展示功能集中

### ✅ 低耦合 (Low Coupling)

模块间通过清晰的接口交互：

1. **依赖方向**（单向）：
   ```
   应用层 → 核心层 → 数据层/模型层/策略层
   ```

2. **接口抽象**：
   - `BaseData` - 数据层抽象
   - `BaseModel` - 模型层抽象
   - `Attributor` - 归因器抽象
   - `BaseMatcher` - 匹配器抽象

3. **依赖倒置**：
   - 上层依赖抽象接口，不依赖具体实现
   - 具体实现可以在不影响上层的情况下替换

## 🔄 模块独立工作方式

### 数据模块独立工作

```python
# 可以独立使用数据接口
from xai_cola.data import PandasData

data = PandasData(df, target_name='target')
feature_names = data.get_x_labels()
X = data.get_x()
```

### 模型模块独立工作

```python
# 可以独立使用模型接口
from xai_cola.models import Model

model = Model(your_model, backend="sklearn")
predictions = model.predict(X)
probs = model.predict_proba(X)
```

### 策略模块独立工作

```python
# 可以独立使用策略
from xai_cola.policies.matching import CounterfactualOptimalTransportPolicy

matcher = CounterfactualOptimalTransportPolicy(x_factual, x_counterfactual)
prob_matrix = matcher.compute_prob_matrix_of_factual_and_counterfactual()
```

## 🤝 模块协同工作方式

### 完整流程（数据流）

```
1. 输入数据
   ↓
2. Data 模块：处理数据，提供特征名称
   ↓
3. COLA 核心：接收数据接口
   ↓
4. Policy.Matching：匹配事实和反事实
   ↓
5. Policy.Attributor：计算特征重要性（使用匹配结果）
   ↓
6. Policy.DataComposer：合成精炼的反事实（使用匹配结果）
   ↓
7. COLA 核心：整合结果，调用模型预测
   ↓
8. Visualization：展示最终结果
   ↓
9. 输出精炼的反事实
```

### 代码示例

```python
# 1. 初始化数据模块
data_obj = PandasData(df, target_name='target')

# 2. 初始化模型模块
ml_model = Model(model, backend="sklearn")

# 3. COLA 协调所有模块
refiner = COLA(data=data_obj, ml_model=ml_model, 
               x_factual=factual, x_counterfactual=counterfactual)

# 4. 设置策略（自动选择匹配器、归因器、合成器）
refiner.set_policy(matcher="ect", attributor="pshap", Avalues_method="max")

# 5. 执行（内部协调所有策略模块）
factual_df, ce_df, ace_df = refiner.get_refined_counterfactual(limited_actions=10)

# 6. 可视化
refiner.heatmap()
```

## 📊 设计模式总结

| 模块 | 使用模式 | 目的 |
|------|---------|------|
| BaseData/BaseModel | 策略模式 | 支持多种数据/模型类型 |
| Model | 工厂模式 | 自动选择合适的实现 |
| Policies | 策略模式 | 可替换的算法实现 |
| Attributor/Matcher | 模板方法 | 定义算法骨架 |
| COLA | 门面模式 | 简化复杂子系统 |

## 🎨 架构优势

### 1. **可扩展性** ✅
- 添加新的数据源：实现 `BaseData`
- 添加新的模型：实现 `BaseModel`
- 添加新的策略：实现相应的基类

### 2. **可测试性** ✅
- 每个模块可以独立测试
- 通过 mock 接口可以隔离测试

### 3. **可维护性** ✅
- 清晰的职责划分
- 模块内部修改不影响其他模块

### 4. **可替换性** ✅
- 不同实现可以无缝替换
- 通过接口保证兼容性

## 📈 改进建议

### 当前架构评分：

- **内聚性**: ⭐⭐⭐⭐⭐ (9/10) - 模块内部高度相关
- **耦合度**: ⭐⭐⭐⭐ (8/10) - 通过接口解耦，但 COLA 直接依赖具体策略类

### 可优化点：

1. **进一步解耦 COLA 和策略**：
   - 可以考虑使用策略注册机制
   - 避免直接导入具体策略类

2. **增强错误处理**：
   - 添加统一的异常处理机制
   - 提供更友好的错误提示

3. **性能优化**：
   - 添加缓存机制
   - 支持并行计算

## 🎯 总结

COLA 的架构设计**很好地实现了高内聚、低耦合的原则**：

✅ **高内聚**：
- 每个模块职责单一
- 相关功能集中管理

✅ **低耦合**：
- 通过接口和抽象基类解耦
- 模块可以独立开发和测试
- 支持灵活的替换和扩展

这是一个**优秀的面向对象设计案例**，符合 SOLID 原则，特别是：
- **S**ingle Responsibility（单一职责）
- **O**pen/Closed（开闭原则）
- **D**ependency Inversion（依赖倒置）

