# COLA 包命名优化建议

## 📋 当前问题分析

### 1. ❌ **命名问题**

#### `counterfactual_limited_actions.py` 
- **问题**: 文件名过长，不够直观，不符合 Python 常见命名习惯
- **当前作用**: 包含主要的 COLA 类
- **建议**: 重命名为 `cola.py` 或 `refiner.py`

#### `ml_model_interface/`
- **问题**: 文件夹名有点冗余（ml_ 前缀不必要）
- **当前作用**: 模型接口模块
- **建议**: 重命名为 `models/` 或 `model_interface/`

#### `ares_dataset_info.py`
- **问题**: 特定于某个算法的文件，放在 data_interface 中不合适
- **建议**: 移到 `utils/` 或删除（如果目前未使用）

### 2. ⚠️ **命名不一致**

#### `model.py` vs 其他模型文件
- `model.py` - 主模型类
- `base_model.py` - 基类
- `pytorch_model.py` - PyTorch 实现
- `sklearn_model.py` - Sklearn 实现

**建议**: 统一命名
- `model.py` → `colamodel.py` 或保持原样（因为这是工厂类）
- 其他保持：`base_model.py`, `pytorch_model.py`, `sklearn_model.py`

## ✅ 推荐的重命名方案

### 方案 1：最小改动（推荐）
保持向后兼容，只改最关键的文件名：

```
xai_cola/
├── cola.py                        # ← 重命名 from counterfactual_limited_actions.py
├── models/                        # ← 重命名 from ml_model_interface/
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py                  # ← 重命名 from model.py  
│   ├── pytorch_model.py
│   └── sklearn_model.py
├── data/                          # ← 重命名 from data_interface/
│   ├── __init__.py
│   ├── base.py
│   ├── pandas_data.py
│   ├── numpy_data.py
│   └── ares_dataset_info.py      # ← 移到 utils/ 或删除
├── cola_policy/
├── utils/
├── plot/
└── version.py
```

### 方案 2：全面优化
彻底的命名优化：

```
xai_cola/
├── cola.py                        # 主 COLA 类
├── models/
│   ├── factory.py                 # 模型工厂
│   ├── base.py                    # 基类
│   ├── pytorch.py                 # PyTorch 实现
│   └── sklearn.py                    # Sklearn 实现
├── data/
│   ├── base.py
│   ├── pandas.py
│   └── numpy.py
├── policies/                       # ← 重命名 from cola_policy/
│   ├── matching/
│   ├── attributor/
│   └── composer/
├── visualization/                  # ← 重命名 from plot/
│   ├── heatmap.py
│   └── highlight.py
└── utils/
    └── logger.py
```

## 📊 命名规范总结

### Python 包命名最佳实践：
1. ✅ **简短明了**: `cola.py` 而不是 `counterfactual_limited_actions.py`
2. ✅ **功能明确**: `models/` 而不是 `ml_model_interface/`
3. ✅ **一致性**: 所有接口类统一用 `base.py`
4. ✅ **避免缩写**: `factory.py` 而不是 `fac.py`
5. ✅ **全小写**: 文件名全小写，用下划线分隔

## 🔄 具体改动列表

### 必须改动的文件：

1. **counterfactual_limited_actions.py → cola.py**
   - 这是最核心的文件，需要重命名
   
2. **ml_model_interface/ → models/**
   - 更简洁，更符合 Python 包命名习惯

3. **data_interface/ → data/**
   - 同样更简洁

4. **plot/ → visualization/**
   - 可选，但如果改了 `data` 就保持一致

5. **ares_dataset_info.py → 移到 utils/ 或删除**
   - 如果目前没有使用

### 需要同步更新的地方：

```python
# 需要更新的导入语句
from xai_cola import COLA                        # 新
from xai_cola.models import Model                # 新
from xai_cola.data import PandasData, NumpyData  # 新
```

## 💡 实施建议

### 优先级排序：

1. **高优先级**（立即改动）:
   - `counterfactual_limited_actions.py` → `cola.py`

2. **中优先级**（建议改动）:
   - `ml_model_interface/` → `models/`
   - `data_interface/` → `data/`

3. **低优先级**（可选）:
   - `plot/` → `visualization/`
   - `model.py` → `factory.py`
   - `colamodel.py`

## ⚠️ 迁移注意事项

如果进行重命名，需要更新：
1. ✅ 所有 `__init__.py` 文件中的导入
2. ✅ `counterfactual_limited_actions.py` 中的所有导入
3. ✅ 测试文件中的导入
4. ✅ 文档中的所有示例代码
5. ✅ `setup.py` 和 `pyproject.toml`（通常不需要）

## 🎯 最终推荐

**最简洁的方案**（推荐采用）:
```python
from xai_cola import COLA
from xai_cola.models import Model
from xai_cola.data import PandasData
```

而不是当前的：
```python
from xai_cola.counterfactual_limited_actions import COLA
from xai_cola.ml_model_interface import Model
from xai_cola.data_interface import PandasData
```

