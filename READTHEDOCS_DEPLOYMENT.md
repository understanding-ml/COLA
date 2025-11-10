# 📖 部署文档到 Read the Docs

## ✅ 准备工作检查清单

在部署之前，确认以下文件都已准备好：

- ✅ `.readthedocs.yaml` - Read the Docs 配置文件
- ✅ `docs/conf.py` - Sphinx 配置文件
- ✅ `docs/index.rst` - 文档主页
- ✅ `docs/requirements-docs.txt` - 文档依赖
- ✅ 所有文档文件（User Guide、API Reference、FAQ等）

**状态：所有文件已准备就绪！** ✅

---

## 🚀 部署步骤

### 方法 1：通过 Read the Docs 网站（推荐）

#### 第 1 步：推送代码到 GitHub

确保所有文档文件都已提交并推送到 GitHub：

```bash
# 检查当前状态
git status

# 添加所有文档文件
git add docs/
git add .readthedocs.yaml
git add README.rst
git add READTHEDOCS_DEPLOYMENT.md

# 提交
git commit -m "docs: add complete documentation structure

- Add User Guide (data, models, explainers, policies, visualization)
- Add API Reference for all modules
- Add FAQ with 50+ Q&A
- Add installation and quickstart guides
- Add contributing and changelog docs
- Configure Read the Docs integration
"

# 推送到 GitHub
git push origin main
```

#### 第 2 步：登录 Read the Docs

1. 访问 https://readthedocs.org/
2. 点击右上角 **"Sign Up"** 或 **"Log In"**
3. 使用 GitHub 账号登录（推荐）

#### 第 3 步：导入项目

1. 登录后，点击右上角的用户名，选择 **"My Projects"**
2. 点击 **"Import a Project"** 按钮
3. 如果是第一次使用，需要授权 Read the Docs 访问你的 GitHub 仓库
4. 从列表中找到 **COLA** 项目
5. 点击右侧的 **"+"** 按钮

#### 第 4 步：配置项目

导入后，Read the Docs 会自动检测到 `.readthedocs.yaml` 配置文件。

**基本信息：**
- **Name**: xai-cola
- **Repository URL**: https://github.com/understanding-ml/COLA
- **Repository type**: Git
- **Default branch**: main
- **Default version**: latest

**高级设置（可选）：**
- **Language**: English
- **Programming Language**: Python
- **Project homepage**: https://github.com/understanding-ml/COLA

点击 **"Next"** 或 **"Finish"** 完成配置。

#### 第 5 步：触发构建

1. 项目导入后，Read the Docs 会自动触发第一次构建
2. 点击 **"Builds"** 标签查看构建进度
3. 等待构建完成（通常需要 2-5 分钟）

构建日志会显示：
```
Running Sphinx v5.x.x
building [html]: targets for 20 source files...
...
build succeeded
```

#### 第 6 步：查看文档

构建成功后：

1. 点击 **"View Docs"** 按钮
2. 或访问：`https://xai-cola.readthedocs.io/en/latest/`

**🎉 恭喜！您的文档已成功部署！**

---

## 📱 查看文档的方式

### 主文档 URL

部署成功后，您的文档将在以下地址可用：

- **最新版本**: https://xai-cola.readthedocs.io/en/latest/
- **稳定版本**: https://xai-cola.readthedocs.io/en/stable/
- **特定版本**: https://xai-cola.readthedocs.io/en/v0.1.0/

### 具体页面 URL

根据您创建的文档结构：

**入门指南：**
- 安装: https://xai-cola.readthedocs.io/en/latest/installation.html
- 快速开始: https://xai-cola.readthedocs.io/en/latest/quickstart.html
- 教程: https://xai-cola.readthedocs.io/en/latest/tutorials/01_basic_tutorial.html

**用户指南：**
- 数据接口: https://xai-cola.readthedocs.io/en/latest/user_guide/data_interface.html
- 模型接口: https://xai-cola.readthedocs.io/en/latest/user_guide/models.html
- 反事实生成器: https://xai-cola.readthedocs.io/en/latest/user_guide/explainers.html
- 匹配策略: https://xai-cola.readthedocs.io/en/latest/user_guide/matching_policies.html
- 可视化: https://xai-cola.readthedocs.io/en/latest/user_guide/visualization.html

**API 参考：**
- COLA: https://xai-cola.readthedocs.io/en/latest/api/cola.html
- Data: https://xai-cola.readthedocs.io/en/latest/api/data.html
- Models: https://xai-cola.readthedocs.io/en/latest/api/models.html
- CE Generator: https://xai-cola.readthedocs.io/en/latest/api/ce_generator.html
- Policies: https://xai-cola.readthedocs.io/en/latest/api/policies.html
- Visualization: https://xai-cola.readthedocs.io/en/latest/api/visualization.html

**其他资源：**
- FAQ: https://xai-cola.readthedocs.io/en/latest/faq.html
- 贡献指南: https://xai-cola.readthedocs.io/en/latest/contributing.html
- 更新日志: https://xai-cola.readthedocs.io/en/latest/changelog.html

---

## 🔧 本地预览（构建前测试）

在推送到 GitHub 之前，您可以本地构建文档预览效果：

### 安装依赖

```bash
pip install -r docs/requirements-docs.txt
```

### 构建 HTML 文档

```bash
cd docs
make html
```

### 查看生成的文档

**Windows:**
```bash
start _build/html/index.html
```

**macOS:**
```bash
open _build/html/index.html
```

**Linux:**
```bash
xdg-open _build/html/index.html
```

### 实时预览（推荐）

使用 sphinx-autobuild 实时查看更改：

```bash
# 安装 sphinx-autobuild
pip install sphinx-autobuild

# 启动实时预览服务器
cd docs
sphinx-autobuild . _build/html

# 在浏览器中访问
# http://127.0.0.1:8000
```

每次保存文件，浏览器会自动刷新！

---

## 🔄 自动更新机制

配置完成后，Read the Docs 会自动：

1. **监听 GitHub 推送**
   - 每次推送到 `main` 分支时自动重新构建
   - 无需手动触发

2. **构建所有分支**
   - 可以为不同分支构建不同版本的文档
   - 例如：`main` → latest, `v0.1.0` → v0.1.0

3. **生成多种格式**
   - HTML（网页版）
   - PDF（可下载）
   - ePub（电子书）

---

## ⚙️ 高级配置

### 1. 设置自定义域名（可选）

在 Read the Docs 项目设置中：

1. 进入 **Admin** → **Domains**
2. 添加自定义域名：`docs.your-domain.com`
3. 配置 DNS CNAME 记录：
   ```
   docs.your-domain.com CNAME xai-cola.readthedocs.io
   ```

### 2. 配置版本管理

在 Read the Docs 项目设置中：

1. 进入 **Admin** → **Versions**
2. 激活需要构建的版本（分支或标签）
3. 设置默认版本（stable 或 latest）

推荐设置：
- `latest`: 跟踪 `main` 分支（最新开发版）
- `stable`: 跟踪最新的 release tag（稳定版）

### 3. 启用 Pull Request 预览

在 Read the Docs 项目设置中：

1. 进入 **Admin** → **Advanced Settings**
2. 勾选 **"Build pull requests for this project"**
3. 每个 PR 都会生成预览链接

### 4. 添加徽章到 README

在 README.rst 中添加：

```rst
.. image:: https://readthedocs.org/projects/xai-cola/badge/?version=latest
    :target: https://xai-cola.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
```

---

## 🐛 故障排除

### 问题 1：构建失败

**错误信息：**
```
Command 'python setup.py egg_info' failed
```

**解决方案：**
检查 `docs/requirements-docs.txt` 是否包含所有依赖。

### 问题 2：找不到模块

**错误信息：**
```
WARNING: autodoc: failed to import module 'xai_cola'
```

**解决方案：**
在 `docs/conf.py` 中添加：
```python
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
```

### 问题 3：图片不显示

**错误信息：**
```
WARNING: image file not readable: images/problem.png
```

**解决方案：**
- 确保图片文件存在于 `docs/images/` 目录
- 或使用 GitHub raw URL

### 问题 4：构建很慢

**原因：** 安装了太多不必要的依赖

**解决方案：**
在 `.readthedocs.yaml` 中只安装文档构建需要的包：
```yaml
python:
  install:
    - requirements: docs/requirements-docs.txt
```

---

## 📊 构建状态检查

### 查看构建日志

1. 进入 Read the Docs 项目页面
2. 点击 **"Builds"** 标签
3. 点击具体的构建查看详细日志

### 构建成功的标志

日志中应该看到：
```
[rtd-command-info] start-time: 2024-xx-xx...
[rtd-command-info] building [html]...
build succeeded, 0 warnings.
[rtd-command-info] Build finished successfully.
```

### 下载构建产物

构建成功后可以下载：
- HTML 压缩包
- PDF 文件
- ePub 文件

---

## 📝 维护文档

### 更新文档内容

1. 修改 `docs/` 目录下的 `.rst` 文件
2. 本地预览确认无误：
   ```bash
   cd docs
   make html
   ```
3. 提交并推送到 GitHub
4. Read the Docs 自动重新构建

### 发布新版本

当发布新版本时：

1. 更新 `VERSION` 文件
2. 更新 `docs/changelog.rst`
3. 创建 Git tag：
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```
4. Read the Docs 会自动构建新版本文档

---

## 🎯 SEO 优化（可选）

在 `docs/conf.py` 中添加：

```python
# HTML meta tags
html_meta = {
    'description': 'COLA - Counterfactual Explanations with Limited Actions',
    'keywords': 'machine learning, XAI, counterfactual, explainability',
    'author': 'Lei You, Lin Zhu'
}
```

---

## 📞 获取帮助

如果遇到问题：

1. 查看 [Read the Docs 官方文档](https://docs.readthedocs.io/)
2. 搜索 [Read the Docs 社区论坛](https://community.readthedocs.org/)
3. 检查 [Sphinx 文档](https://www.sphinx-doc.org/)
4. 查看构建日志获取具体错误信息

---

## ✅ 部署后检查清单

部署完成后，确认以下内容：

- [ ] 主页正常显示（https://xai-cola.readthedocs.io/）
- [ ] 所有导航链接可以点击
- [ ] 用户指南所有页面正常
- [ ] API 参考所有页面正常
- [ ] 代码高亮显示正确
- [ ] 图片正常加载
- [ ] 搜索功能可用
- [ ] PDF/ePub 下载链接有效
- [ ] 移动端显示正常

---

## 🎉 总结

完成这些步骤后：

1. ✅ 文档在 Read the Docs 上公开可访问
2. ✅ 每次推送代码自动更新文档
3. ✅ 支持多版本文档
4. ✅ 提供 HTML、PDF、ePub 多种格式
5. ✅ 内置搜索功能
6. ✅ 响应式设计，支持移动设备

**您的 COLA 项目现在拥有专业级的在线文档！** 🎊

---

## 📚 相关链接

- **Read the Docs**: https://readthedocs.org/
- **Sphinx 文档**: https://www.sphinx-doc.org/
- **reStructuredText 指南**: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
- **Furo 主题文档**: https://pradyunsg.me/furo/
