@echo off
echo ============================================================
echo 创建 cola conda 环境
echo ============================================================

echo.
echo 步骤 1: 创建环境
conda create -n cola python=3.10 -y

echo.
echo 步骤 2: 安装核心依赖
call conda install -n cola numpy>=1.26.4 pandas=2.2.3 scikit-learn=1.4.2 scipy>=1.13.0 -y
call conda install -n cola lightgbm=4.6.0 joblib=1.4.0 -y
call conda install -n cola matplotlib=3.10.1 seaborn=0.13.2 -y
call conda install -n cola ipython=8.20.0 jupyter>=1.0.0 -y

echo.
echo 步骤 3: 使用 pip 安装其他包
call conda run -n cola pip install dice-ml==0.12.1
call conda run -n cola pip install alibi==0.16.2
call conda run -n cola pip install shap==0.45.0
call conda run -n cola pip install POT==0.11.0

echo.
echo 步骤 4: 安装 PyTorch
call conda run -n cola pip install torch==2.3.0

echo.
echo ============================================================
echo 完成！
echo ============================================================
echo.
echo 激活环境: conda activate cola
echo 验证安装: conda list -n cola

pause

