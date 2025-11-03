# COLA 包打包说明

## 📦 打包内容

当用户执行 `pip install xai-cola` 后，会安装以下内容：

### ✅ 会包含的模块

```
site-packages/
├── xai_cola/              ← 主包
│   ├── __init__.py
│   ├── data/
│   ├── models/
│   ├── policies/
│   ├── visualization/
│   └── utils/
├── counterfactual_explainer/  ← 反事实解释器
│   ├── __init__.py
│   ├── base_explainer.py
│   ├── dice.py
│   ├── discount.py
│   ├── alibi_cfi.py
│   ├── knn.py
│   ├── ares.py
│   └── auxiliary.py
└── datasets/               ← 数据集
    ├── __init__.py
    ├── german_credit.py
    ├── compas.py
    ├── heloc.py
    ├── hotel_bookings.py
    └── rawdata/
```

### ❌ 不会包含的内容

- `tests/` - 测试文件
- `examples/` - 示例代码
- `docs/` - 文档文件
- `scripts/` - 脚本文件
- `*.md` 文档（MANIFEST.in 中指定的除外）

## 📝 使用方式

### 安装后可以这样使用：

```python
# 导入主包
from xai_cola import COLA
from xai_cola.data import PandasData
from xai_cola.models import Model

# 导入反事实解释器
from counterfactual_explainer import DiCE
from counterfactual_explainer import DisCount
from counterfactual_explainer import AlibiCounterfactualInstances

# 使用
explainer = DiCE(ml_model=model)
factual, counterfactual = explainer.generate_counterfactuals(data)

cola = COLA(data, model, factual, counterfactual)
results = cola.get_refined_counterfactual(limited_actions=10)
```

## 🔧 当前配置

### setup.py
```python
packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "examples", "scripts", "docs"])
```

这会包含：
- ✅ `xai_cola` 包
- ✅ `counterfactual_explainer` 包
- ✅ `datasets` 包
- ❌ `tests` 包
- ❌ `examples` 包
- ❌ `scripts` 包

### pyproject.toml
```toml
[tool.setuptools]
packages = ["xai_cola", "counterfactual_explainer", "datasets"]
```

显式指定了要包含的包。

## ⚠️ 注意事项

### counterfactual_explainer 的导入路径

由于 `counterfactual_explainer` 是顶级包，导入时需要：

```python
# ✅ 正确
from counterfactual_explainer import DiCE

# ❌ 错误
from xai_cola.counterfactual_explainer import DiCE  # 这个路径不存在
```

### 文档更新建议

如果你的文档中使用了：
```python
from xai_cola.counterfactual_limited_actions import COLA
```

这需要更新，因为 COLA 文件已经不在了（被删除了）。

## 🎯 建议

如果你希望 `counterfactual_explainer` 作为 `xai_cola` 的子模块，需要：

1. 将 `counterfactual_explainer` 移动到 `xai_cola/counterfactual_explainer/`
2. 更新所有导入路径
3. 或者保持现状，作为独立包

当前的状态：`counterfactual_explainer` 是**独立的顶级包**，用户可以直接从 `counterfactual_explainer` 导入，就像从 `xai_cola` 导入一样。

