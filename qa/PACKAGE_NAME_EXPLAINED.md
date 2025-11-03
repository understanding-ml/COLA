# 包名和导入名的关系 - 详细解释

## 🔑 核心概念

**PyPI 包名 ≠ Python 模块名**

### 关键点：

1. **PyPI 包名**（pypi.org 上的名称）- 用于安装
2. **Python 模块名**（代码中的导入名）- 用于使用

## 📦 你的包的具体情况

### PyPI 包名（安装时使用）
```bash
pip install xai-cola
```
- ✅ **名称**: `xai-cola`（带连字符）
- ✅ **用途**: 在 PyPI 上注册和安装

### Python 模块名（导入时使用）
```python
from xai_cola import COLA
from counterfactual_explainer import DiCE
from datasets import GermanCreditDataset
```
- ✅ **名称**: `xai_cola`, `counterfactual_explainer`, `datasets`（下划线）
- ✅ **用途**: 在 Python 代码中导入

## ❓ 常见问题解答

### Q1: 为什么 `pip install xai-cola` 而不是 `pip install xai_cola`？

**答案**: 
- `xai-cola` 是 PyPI 上的包名（可以用连字符）
- `xai_cola` 是 Python 模块名（不能用连字符）

当你执行：
```bash
pip install xai-cola  # 使用连字符安装
```

pip 会：
1. 从 PyPI 下载名为 `xai-cola` 的包
2. 安装到 `site-packages/` 目录
3. 创建名为 `xai_cola` 的文件夹（下划线版本）

### Q2: 为什么安装了 `xai-cola` 就能用 `counterfactual_explainer`？

**答案**: 
因为 `counterfactual_explainer` 被包含在 `xai-cola` 包中！

当你构建包时，`setup.py` 中的配置：

```python
packages=find_packages(exclude=[...])  # 会自动找到所有顶级包
```

这会包含：
- ✅ `xai_cola/`
- ✅ `counterfactual_explainer/`
- ✅ `datasets/`

所以安装 `xai-cola` 后，所有这些包都可以直接导入。

## 📐 完整的安装和使用流程

### 步骤 1: 安装

```bash
# 用户在终端执行
pip install xai-cola

# pip 会：
# 1. 从 PyPI 下载包
# 2. 解压并安装到 site-packages/
# 3. 创建以下目录结构：
```

```
site-packages/
├── xai_cola-0.1.0.dist-info/  # 包元数据
└── 
    ├── xai_cola/               # 主包（用下划线）
    │   ├── __init__.py
    │   ├── data/
    │   ├── models/
    │   └── policies/
    ├── counterfactual_explainer/  # 反事实解释器包（用下划线）
    │   ├── __init__.py
    │   ├── dice.py
    │   └── ...
    └── datasets/               # 数据集包（用下划线）
        ├── __init__.py
        └── ...
```

### 步骤 2: 使用

```python
# 现在用户可以这样导入

# 从主包导入
from xai_cola import COLA
from xai_cola.data import PandasData
from xai_cola.models import Model

# 从反事实解释器包导入
from counterfactual_explainer import DiCE
from counterfactual_explainer import DisCount

# 从数据集包导入
from datasets import GermanCreditDataset
```

## 🎯 setup.py 中的关键配置

```python
setup(
    name="xai-cola",  # ← PyPI 包名（连字符）
    packages=find_packages(exclude=[...])  # 自动包含所有包
)
```

`find_packages()` 会找到：
- `xai_cola/` → 成为可导入模块
- `counterfactual_explainer/` → 成为可导入模块
- `datasets/` → 成为可导入模块

## 💡 为什么这样设计？

### 包名（PyPI）- 便于记忆和搜索
- 人类友好
- 搜索友好：`xai-cola` 在 Google/PyPI 上更容易找到
- 标准命名：Python 社区推荐用连字符

### 模块名（代码）- 符合 Python 规范
- Python 不允许包名带连字符（需要下划线）
- 导入时只能用下划线：`import xai_cola`

## 📊 对比表

| 阶段 | 名称 | 格式 | 示例 | 用途 |
|------|------|------|------|------|
| PyPI 注册 | 包名 | 连字符 | `xai-cola` | 供 pip 搜索和安装 |
| 本地安装 | 目录名 | 下划线 | `xai_cola/` | 在 site-packages 中 |
| 代码导入 | 模块名 | 下划线 | `import xai_cola` | 在代码中使用 |

## 🎨 完整的命令对比

### 安装阶段
```bash
# PyPI 上的包名（带连字符）
pip install xai-cola

# 或者指定版本
pip install xai-cola==0.1.0
```

### 使用阶段
```python
# Python 代码中的导入（下划线）
from xai_cola import COLA
from counterfactual_explainer import DiCE
```

## ⚠️ 常见混淆

### ❌ 错误理解
```bash
pip install xai_cola  # 找不到！因为 PyPI 上注册的是 xai-cola
```

### ✅ 正确理解
```bash
pip install xai-cola   # 正确！PyPI 包名
```

然后使用：
```python
import xai_cola        # 正确！Python 模块名
```

## 🔍 验证方法

安装后可以这样验证：

```python
# 1. 查看已安装的包
import pkg_resources
installed = [p.project_name for p in pkg_resources.working_set]
print('xai-cola' in installed)  # True

# 2. 查看可用的模块
import sys
print('xai_cola' in sys.modules or 'xai_cola' in sys.path)  # True

# 3. 尝试导入
from xai_cola import COLA
from counterfactual_explainer import DiCE
print("成功！")
```

## 🎯 总结

| 问题 | 答案 |
|------|------|
| **pip install 时用什么？** | `xai-cola`（连字符）|
| **import 时用什么？** | `xai_cola`（下划线）|
| **为什么能 import counterfactual_explainer？** | 因为它是 xai-cola 包的一部分 |
| **包名和模块名必须一样吗？** | 不！通常不一样 |

**关键记忆点**：
- 🏪 **商店名称**（PyPI）: `xai-cola`（连字符，便于搜索）
- 🏠 **房间名称**（代码）: `xai_cola`（下划线，Python 规范）

## 🚀 实际例子

当前流行项目：

1. **requests**
   ```bash
   pip install requests        # PyPI
   import requests             # Python
   ```

2. **scikit-learn**
   ```bash
   pip install scikit-learn    # PyPI（连字符）
   import sklearn              # Python（不同名称！）
   ```

3. **tensorflow**
   ```bash
   pip install tensorflow      # PyPI
   import tensorflow           # Python
   ```

你的 `xai-cola` 遵循相同的模式！✅

