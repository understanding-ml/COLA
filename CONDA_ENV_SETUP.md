# 使用 Conda 创建 COLA 环境

根据 `requirements.txt` 创建名为 `cola` 的 conda 环境。

## 🚀 快速开始

### 方法 1: 使用批处理脚本（推荐，Windows）

```bash
# 运行批处理脚本
conda_install.bat
```

### 方法 2: 手动安装

#### 步骤 1: 创建环境

```bash
conda create -n cola python=3.10 -y
```

#### 步骤 2: 安装核心依赖

```bash
conda activate cola
conda install numpy>=1.26.4 pandas=2.2.3 scikit-learn=1.4.2 scipy>=1.13.0 -y
conda install lightgbm=4.6.0 joblib=1.4.0 -y
conda install matplotlib=3.10.1 seaborn=0.13.2 -y
conda install ipython=8.20.0 jupyter>=1.0.0 -y
```

#### 步骤 3: 使用 pip 安装其他包

```bash
pip install dice-ml==0.12.1
pip install alibi==0.16.2
pip install shap==0.45.0
pip install POT==0.11.0
```

#### 步骤 4: 安装 PyTorch（可选）

```bash
pip install torch==2.3.0
```

### 方法 3: 使用环境文件

```bash
# 使用精简版环境文件
conda env create -f environment_minimal.yml

# 或使用完整版
conda env create -f environment.yml
```

## 📋 依赖列表

### 核心依赖（通过 conda 安装）

- numpy>=1.26.4
- pandas==2.2.3
- scikit-learn==1.4.2
- scipy>=1.13.0
- lightgbm==4.6.0
- joblib==1.4.0
- matplotlib==3.10.1
- seaborn==0.13.2
- ipython==8.20.0
- jupyter>=1.0.0

### 其他依赖（通过 pip 安装）

- dice-ml==0.12.1
- alibi==0.16.2
- shap==0.45.0
- POT==0.11.0
- torch==2.3.0（可选）

## ✅ 验证安装

```bash
# 激活环境
conda activate cola

# 验证 Python 版本
python --version

# 查看已安装的包
conda list

# 或使用 pip
pip list
```

## 🔧 使用环境

```bash
# 激活环境
conda activate cola

# 退出环境
conda deactivate

# 删除环境（如果需要）
conda env remove -n cola
```

## 📝 备注

1. **环境名称**: `cola`
2. **Python 版本**: 3.10
3. **安装时间**: 约 5-10 分钟（取决于网络速度）
4. **磁盘空间**: 约 2-3 GB

## 🆘 常见问题

### Q: conda 命令找不到？
A: 需要先安装 Anaconda 或 Miniconda。

### Q: pip 安装失败？
A: 尝试使用 `conda run -n cola pip install <package>`。

### Q: PyTorch 安装失败？
A: 根据你的系统（Windows/Linux/Mac）选择合适的 PyTorch 版本。

### Q: 想要更新环境？
A: 运行 `conda_install.bat` 会重新安装所有依赖。

## 📂 相关文件

- `requirements.txt` - pip 依赖列表
- `environment.yml` - 完整 conda 环境文件
- `environment_minimal.yml` - 精简 conda 环境文件
- `conda_install.bat` - Windows 批处理安装脚本
- `create_conda_env.py` - Python 安装脚本

