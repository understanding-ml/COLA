# PyPI 发布检查清单

## 当前状态分析

### ✅ 已完成项目

1. **基础配置文件**
   - ✅ `pyproject.toml` - 现代化的包配置
   - ✅ `setup.py` - 向后兼容的安装脚本
   - ✅ `requirements.txt` - 依赖声明
   - ✅ `MANIFEST.in` - 文件包含规则
   - ✅ `LICENSE` - MIT 许可证
   - ✅ `README.rst` - 项目说明（RST格式）
   - ✅ 包结构 - 正确的 `__init__.py` 文件

2. **元数据配置**
   - ✅ 包名称: `xai-cola`
   - ✅ 版本号: `0.1.0`
   - ✅ 作者信息
   - ✅ Python版本要求: `>=3.8`
   - ✅ 分类标签 (classifiers)
   - ✅ 关键词 (keywords)

---

## ⚠️ 需要修复的问题

### 1. **PyProject.toml 缺少依赖声明** (严重 - 必须修复)

**问题:** `pyproject.toml` 的 `[project]` 部分缺少 `dependencies` 字段

**修复:** 在 `pyproject.toml` 中添加:
```toml
[project]
name = "xai-cola"
version = "0.1.0"
# ... 其他字段 ...
dependencies = [
    "numpy>=1.26.4,<2.0",
    "pandas>=2.2.0,<3.0",
    "scikit-learn>=1.4.0,<2.0",
    "scipy>=1.13.0,<2.0",
    "lightgbm>=4.0.0,<5.0",
    "joblib>=1.4.0,<2.0",
    "dice-ml>=0.12,<1.0",
    "alibi>=0.9.6,<1.0",
    "matplotlib>=3.8.0,<4.0",
    "seaborn>=0.13.0,<1.0",
    "shap>=0.42.0,<1.0",
    "POT>=0.9.0,<1.0",
]

[project.optional-dependencies]
torch = [
    "torch>=2.0.0,<3.0; sys_platform != 'darwin' or platform_machine != 'arm64'",
]
jupyter = [
    "ipython>=8.0.0",
    "jupyter>=1.0.0",
    "notebook>=6.0.0",
]
all = [
    "xai-cola[torch,jupyter]",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "flake8>=6.0",
    "mypy>=1.0",
]
docs = [
    "sphinx>=5.0",
    "sphinx-rtd-theme>=1.0",
    "furo>=2023.3.27",
    "sphinx-copybutton>=0.5.0",
    "myst-parser>=1.0.0",
]
```

### 2. **依赖版本不一致** (中等)

**问题:** `requirements.txt` 中的版本过于严格,不利于兼容性

**当前 requirements.txt 问题:**
- 使用精确版本号 (如 `pandas==2.2.3`) 过于严格
- 不同文件中版本不一致

**建议:** 使用版本范围而不是精确版本
```
# 推荐的 requirements.txt
numpy>=1.26.4,<2.0
pandas>=2.2.0,<3.0
scikit-learn>=1.4.0,<2.0
scipy>=1.13.0,<2.0
lightgbm>=4.0.0,<5.0
joblib>=1.4.0,<2.0
dice-ml>=0.12,<1.0
alibi>=0.9.6,<1.0
matplotlib>=3.8.0,<4.0
seaborn>=0.13.0,<1.0
shap>=0.42.0,<1.0
POT>=0.9.0,<1.0
```

### 3. **MANIFEST.in 问题** (中等)

**问题:**
- 使用了错误的语法 `recursive-include demo.ipynb`
- 引用了已删除的文件 `demo.ipynb`

**修复:**
```manifest
# Include the README file
include README.rst
include LICENSE
include CHANGELOG.md
include requirements.txt

# Include example notebooks (updated)
include examples/*.ipynb
include examples/*.py

# Include data files
recursive-include datasets *.csv
recursive-include datasets *.data

# Include trained models
include *.pkl

# Include documentation
recursive-include docs *.png
recursive-include docs *.jpg
recursive-include docs *.md

# Exclude unnecessary files
global-exclude __pycache__
global-exclude *.py[cod]
global-exclude *.so
global-exclude *.egg
global-exclude .DS_Store
global-exclude .git*
global-exclude test_*.py
global-exclude debug_*.py
global-exclude notebook_test_*.py
```

### 4. **GitHub URL 占位符** (中等)

**问题:** `setup.py` 和 `pyproject.toml` 中使用占位符 URL
```
https://github.com/your-repo/COLA
```

**修复:** 更新为实际的 GitHub 仓库地址

### 5. **Python 版本兼容性声明不一致** (低)

**问题:**
- `pyproject.toml`: `>=3.8`
- `README.md`: `3.7+`

**建议:** 统一使用 `>=3.8` (因为 pandas 2.2+ 不支持 Python 3.7)

---

## 📋 发布前必做清单

### 步骤 1: 修复配置文件

```bash
# 1. 更新 pyproject.toml (添加 dependencies)
# 2. 更新 requirements.txt (使用版本范围)
# 3. 修复 MANIFEST.in
# 4. 更新 GitHub URL
# 5. 统一 Python 版本声明
```

### 步骤 2: 本地构建测试

```bash
# 1. 清理旧的构建文件
rm -rf build/ dist/ *.egg-info

# 2. 安装构建工具
pip install build twine

# 3. 构建包
python -m build

# 4. 检查构建结果
twine check dist/*
```

**预期输出:**
```
Checking dist/xai_cola-0.1.0-py3-none-any.whl: PASSED
Checking dist/xai-cola-0.1.0.tar.gz: PASSED
```

### 步骤 3: 测试本地安装

```bash
# 1. 创建虚拟环境测试
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate

# 2. 从构建的 wheel 安装
pip install dist/xai_cola-0.1.0-py3-none-any.whl

# 3. 测试导入
python -c "from xai_cola import COLA; print('Success!')"
python -c "from xai_cola.data import COLAData; print('Success!')"
python -c "from xai_cola.models import Model; print('Success!')"

# 4. 清理
deactivate
rm -rf test_env
```

### 步骤 4: Python 版本兼容性测试

```bash
# 测试 Python 3.8
python3.8 -m venv test_py38
source test_py38/bin/activate
pip install dist/xai_cola-0.1.0-py3-none-any.whl
python -c "from xai_cola import COLA; print('Python 3.8 OK')"
deactivate

# 测试 Python 3.9
python3.9 -m venv test_py39
source test_py39/bin/activate
pip install dist/xai_cola-0.1.0-py3-none-any.whl
python -c "from xai_cola import COLA; print('Python 3.9 OK')"
deactivate

# 测试 Python 3.10
python3.10 -m venv test_py310
source test_py310/bin/activate
pip install dist/xai_cola-0.1.0-py3-none-any.whl
python -c "from xai_cola import COLA; print('Python 3.10 OK')"
deactivate

# 测试 Python 3.11
python3.11 -m venv test_py311
source test_py311/bin/activate
pip install dist/xai_cola-0.1.0-py3-none-any.whl
python -c "from xai_cola import COLA; print('Python 3.11 OK')"
deactivate
```

### 步骤 5: TestPyPI 测试发布

```bash
# 1. 注册 TestPyPI 账号
# 访问: https://test.pypi.org/account/register/

# 2. 配置 API token (推荐)
# 访问: https://test.pypi.org/manage/account/token/
# 创建 ~/.pypirc:
cat > ~/.pypirc << EOF
[testpypi]
username = __token__
password = pypi-your-test-token-here
EOF

# 3. 上传到 TestPyPI
twine upload --repository testpypi dist/*

# 4. 从 TestPyPI 安装测试
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ xai-cola

# 注意: 使用 --extra-index-url 是因为依赖包在 PyPI 而不是 TestPyPI
```

### 步骤 6: 正式发布到 PyPI

```bash
# 1. 确认版本号正确
grep version pyproject.toml
grep __version__ xai_cola/version.py

# 2. 创建 git tag
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0

# 3. 配置 PyPI API token
# 访问: https://pypi.org/manage/account/token/
# 更新 ~/.pypirc:
cat >> ~/.pypirc << EOF
[pypi]
username = __token__
password = pypi-your-production-token-here
EOF

# 4. 上传到 PyPI
twine upload dist/*

# 5. 验证安装
pip install xai-cola
python -c "from xai_cola import COLA; print('Production install successful!')"
```

---

## 🔧 推荐的依赖版本策略

### 为什么使用版本范围?

1. **过于严格** (`==`)
   ```
   pandas==2.2.3  # ❌ 只允许这个精确版本
   ```
   - 问题: 用户无法使用 pandas 2.2.4 或更高版本
   - 导致: 依赖冲突、安装失败

2. **推荐做法** (`>=x.y.z,<major+1`)
   ```
   pandas>=2.2.0,<3.0  # ✅ 允许 2.2.x 和 2.x 的所有版本
   ```
   - 优点: 灵活性高、减少冲突
   - 保证: 主版本内的向后兼容

### 核心依赖版本建议

```toml
dependencies = [
    # 数值计算
    "numpy>=1.26.4,<2.0",       # 避免 numpy 2.0 的破坏性变更
    "pandas>=2.2.0,<3.0",       # 允许 pandas 2.x
    "scipy>=1.13.0,<2.0",       # 需要 1.13+ 的特性

    # 机器学习
    "scikit-learn>=1.4.0,<2.0", # sklearn 1.4+
    "lightgbm>=4.0.0,<5.0",     # LightGBM 4.x
    "joblib>=1.4.0,<2.0",       # 模型序列化

    # 反事实解释器
    "dice-ml>=0.12,<1.0",       # DiCE 0.12+
    "alibi>=0.9.6,<1.0",        # Alibi 0.9.6+

    # 可视化
    "matplotlib>=3.8.0,<4.0",   # Matplotlib 3.x
    "seaborn>=0.13.0,<1.0",     # Seaborn 0.13+

    # 特征归因和最优传输
    "shap>=0.42.0,<1.0",        # SHAP 0.42+
    "POT>=0.9.0,<1.0",          # Python Optimal Transport
]
```

---

## 📊 兼容性测试脚本

创建 `test_compatibility.py`:

```python
"""测试不同 Python 版本的兼容性"""
import sys

def test_basic_import():
    """测试基本导入"""
    try:
        from xai_cola import COLA
        from xai_cola.data import COLAData
        from xai_cola.models import Model
        print("✅ 基本导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_dice_integration():
    """测试 DiCE 集成"""
    try:
        from xai_cola.ce_generator import DiCE
        print("✅ DiCE 集成成功")
        return True
    except Exception as e:
        print(f"❌ DiCE 集成失败: {e}")
        return False

def test_discount_integration():
    """测试 DiSCOUNT 集成"""
    try:
        from xai_cola.ce_generator import DisCount
        print("✅ DiSCOUNT 集成成功")
        return True
    except Exception as e:
        print(f"❌ DiSCOUNT 集成失败: {e}")
        return False

def test_version():
    """测试版本信息"""
    try:
        from xai_cola import __version__
        print(f"✅ 版本: {__version__}")
        return True
    except Exception as e:
        print(f"❌ 版本信息失败: {e}")
        return False

if __name__ == "__main__":
    print(f"Python 版本: {sys.version}")
    print(f"平台: {sys.platform}")
    print("\n" + "="*60)

    results = [
        test_basic_import(),
        test_dice_integration(),
        test_discount_integration(),
        test_version(),
    ]

    print("="*60)
    if all(results):
        print("\n✅ 所有测试通过!")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败!")
        sys.exit(1)
```

运行测试:
```bash
python test_compatibility.py
```

---

## 🚀 发布后的维护

### 1. 版本号管理 (语义化版本)

```
主版本.次版本.修订号 (Major.Minor.Patch)

0.1.0 → 0.1.1  # Bug 修复
0.1.0 → 0.2.0  # 新功能 (向后兼容)
0.1.0 → 1.0.0  # 破坏性变更
```

### 2. 发布新版本流程

```bash
# 1. 更新版本号
# 修改 xai_cola/version.py
echo '__version__ = "0.1.1"' > xai_cola/version.py

# 修改 pyproject.toml
# version = "0.1.1"

# 2. 更新 CHANGELOG.md
cat >> CHANGELOG.md << EOF
## [0.1.1] - 2024-XX-XX
### Fixed
- 修复了 XXX bug
- 改进了 YYY 性能
EOF

# 3. 提交更改
git add .
git commit -m "chore: bump version to 0.1.1"

# 4. 创建标签
git tag -a v0.1.1 -m "Release version 0.1.1"
git push origin main --tags

# 5. 构建和发布
rm -rf dist/
python -m build
twine upload dist/*
```

### 3. 用户安装方式

发布后用户可以通过以下方式安装:

```bash
# 基础安装
pip install xai-cola

# 包含 PyTorch 支持
pip install xai-cola[torch]

# 包含 Jupyter 支持
pip install xai-cola[jupyter]

# 完整安装
pip install xai-cola[all]

# 开发模式安装
pip install xai-cola[dev]
```

---

## 📝 发布检查清单总结

### 必须完成 (阻塞发布):
- [ ] 在 `pyproject.toml` 中添加 `dependencies` 字段
- [ ] 修复 `MANIFEST.in` 语法错误
- [ ] 更新 GitHub URL (替换占位符)
- [ ] 确保 `README.rst` 存在且格式正确
- [ ] 统一 Python 版本声明 (`>=3.8`)
- [ ] 通过 `twine check dist/*` 检查

### 强烈建议:
- [ ] 使用 TestPyPI 预发布测试
- [ ] 测试至少 2 个 Python 版本 (3.8, 3.11)
- [ ] 创建 GitHub Release
- [ ] 更新 CHANGELOG.md

### 可选优化:
- [ ] 设置 GitHub Actions CI/CD
- [ ] 添加单元测试
- [ ] 提高代码覆盖率
- [ ] 添加在线文档 (Read the Docs)

---

## 🔗 参考资源

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Publishing Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Semantic Versioning](https://semver.org/)
- [TestPyPI](https://test.pypi.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
