"""
创建和配置 cola conda 环境
"""
import subprocess
import sys

def run_command(cmd):
    """执行命令"""
    print(f"执行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return False
    print(result.stdout)
    return True

# 1. 安装核心依赖
print("=" * 60)
print("步骤 1: 安装核心依赖")
print("=" * 60)

conda_packages = [
    ("numpy", ">=1.26.4"),
    ("pandas", "2.2.3"),
    ("scikit-learn", "1.4.2"),
    ("scipy", ">=1.13.0"),
    ("lightgbm", "4.6.0"),
    ("joblib", "1.4.0"),
    ("matplotlib", "3.10.1"),
    ("seaborn", "0.13.2"),
    ("ipython", "8.20.0"),
    ("jupyter", ">=1.0.0"),
]

for pkg, version in conda_packages:
    cmd = f'conda install -n cola {pkg}{version} -y'
    run_command(cmd)

# 2. 使用 pip 安装其他包
print("\n" + "=" * 60)
print("步骤 2: 安装 pip 依赖")
print("=" * 60)

pip_packages = [
    "dice-ml==0.12.1",
    "alibi==0.16.2",
    "shap==0.45.0",
    "POT==0.11.0",
]

for pkg in pip_packages:
    cmd = f'conda run -n cola pip install {pkg}'
    run_command(cmd)

# 3. 安装 PyTorch（可选）
print("\n" + "=" * 60)
print("步骤 3: 安装 PyTorch（可选）")
print("=" * 60)

# 根据平台安装 PyTorch
import platform
if platform.system() == "Windows":
    cmd = 'conda run -n cola pip install torch==2.3.0'
else:
    # Linux/Mac
    cmd = 'conda run -n cola pip install torch==2.3.0; sys_platform != "darwin" or platform_machine != "arm64"'
    
run_command(cmd)

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
print("\n激活环境: conda activate cola")
print("验证安装: conda list -n cola")

