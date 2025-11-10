# 📖 COLA 文档快速参考

## 🚀 快速开始

### 立即查看本地文档

```bash
# 1. 构建文档
cd docs
make html

# 2. 打开文档（选择你的操作系统）
start _build/html/index.html     # Windows
open _build/html/index.html      # macOS
xdg-open _build/html/index.html  # Linux
```

### 部署到 Read the Docs

详细步骤见：**[READTHEDOCS_DEPLOYMENT.md](READTHEDOCS_DEPLOYMENT.md)**

简要步骤：
1. 推送代码到 GitHub
2. 在 https://readthedocs.org/ 登录
3. 导入项目 `COLA`
4. 自动构建完成
5. 访问 `https://xai-cola.readthedocs.io/`

---

## 📂 文档结构速查

### 所有文档文件位置

```
docs/
├── index.rst                          # 主页 ⭐
├── installation.rst                   # 安装指南
├── quickstart.rst                     # 快速开始
├── faq.rst                           # FAQ（50+问答）⭐
├── contributing.rst                   # 贡献指南
├── changelog.rst                      # 更新日志
│
├── user_guide/                       # 用户指南 📖
│   ├── data_interface.rst            # 数据接口
│   ├── models.rst                    # 模型接口
│   ├── explainers.rst                # 反事实生成器
│   ├── matching_policies.rst         # 匹配策略
│   └── visualization.rst             # 可视化 ⭐
│
├── api/                              # API 参考 🔍
│   ├── cola.rst                      # COLA主类
│   ├── data.rst                      # 数据API
│   ├── models.rst                    # 模型API
│   ├── ce_generator.rst              # 生成器API
│   ├── policies.rst                  # 策略API
│   └── visualization.rst             # 可视化API
│
└── tutorials/                        # 教程
    └── 01_basic_tutorial.md          # 基础教程
```

---

## 🔗 部署后的 URL

部署到 Read the Docs 后，文档将在以下地址可访问：

### 主入口

- **文档主页**: https://xai-cola.readthedocs.io/
- **最新版本**: https://xai-cola.readthedocs.io/en/latest/
- **稳定版本**: https://xai-cola.readthedocs.io/en/stable/

### 入门指南

| 页面 | URL |
|------|-----|
| 安装 | https://xai-cola.readthedocs.io/en/latest/installation.html |
| 快速开始 | https://xai-cola.readthedocs.io/en/latest/quickstart.html |
| 基础教程 | https://xai-cola.readthedocs.io/en/latest/tutorials/01_basic_tutorial.html |

### 用户指南

| 页面 | URL |
|------|-----|
| 数据接口 | https://xai-cola.readthedocs.io/en/latest/user_guide/data_interface.html |
| 模型接口 | https://xai-cola.readthedocs.io/en/latest/user_guide/models.html |
| 反事实生成器 | https://xai-cola.readthedocs.io/en/latest/user_guide/explainers.html |
| 匹配策略 | https://xai-cola.readthedocs.io/en/latest/user_guide/matching_policies.html |
| **可视化** ⭐ | https://xai-cola.readthedocs.io/en/latest/user_guide/visualization.html |

### API 参考

| 页面 | URL |
|------|-----|
| COLA类 | https://xai-cola.readthedocs.io/en/latest/api/cola.html |
| Data | https://xai-cola.readthedocs.io/en/latest/api/data.html |
| Models | https://xai-cola.readthedocs.io/en/latest/api/models.html |
| CE Generator | https://xai-cola.readthedocs.io/en/latest/api/ce_generator.html |
| Policies | https://xai-cola.readthedocs.io/en/latest/api/policies.html |
| Visualization | https://xai-cola.readthedocs.io/en/latest/api/visualization.html |

### 其他资源

| 页面 | URL |
|------|-----|
| **FAQ** ⭐ | https://xai-cola.readthedocs.io/en/latest/faq.html |
| 贡献指南 | https://xai-cola.readthedocs.io/en/latest/contributing.html |
| 更新日志 | https://xai-cola.readthedocs.io/en/latest/changelog.html |

---

## 🎯 常用文档位置

### 你最常需要查看的文档

1. **新用户？** → [quickstart.rst](docs/quickstart.rst)
   - 5分钟快速上手

2. **可视化问题？** → [user_guide/visualization.rst](docs/user_guide/visualization.rst) ⭐
   - 完整的可视化指南
   - 所有可视化方法
   - 参数说明
   - 示例代码

3. **遇到错误？** → [faq.rst](docs/faq.rst) ⭐
   - 50+ 常见问题
   - 故障排除
   - 最佳实践

4. **查找函数参数？** → [api/](docs/api/) 📂
   - 完整的 API 参考
   - 所有参数说明

5. **如何贡献？** → [contributing.rst](docs/contributing.rst)
   - 开发指南
   - 代码规范

---

## 📝 更新 README.rst 链接

**已完成！** ✅

原来的链接：
```rst
For more details, please refer to the `visualization documentation
<https://xai-cola.readthedocs.io/en/latest/visualization.html>`_.
```

已更新为：
```rst
For complete visualization options, see the full documentation.
```

**原因：**
- 文档尚未部署，旧链接会404
- 部署后正确的URL是：`user_guide/visualization.html`（不是根目录的 `visualization.html`）

**部署后可以更新为：**
```rst
For complete visualization options, see the
`visualization guide <https://xai-cola.readthedocs.io/en/latest/user_guide/visualization.html>`_.
```

---

## 🔍 如何在文档中搜索

### 本地搜索

在生成的 HTML 文档中：
1. 打开 `_build/html/index.html`
2. 使用左侧搜索框
3. 输入关键词即可搜索

### Read the Docs 搜索

部署后：
1. 访问任意文档页面
2. 使用页面左侧的搜索框
3. 支持全文搜索

---

## 📊 文档覆盖范围

### 用户指南（5个文件）

- ✅ **数据接口**（2,883 tokens）
  - DataFrame 和 NumPy 使用
  - 添加反事实
  - 预处理器集成
  - 最佳实践

- ✅ **模型接口**（3,432 tokens）
  - Sklearn、PyTorch、TensorFlow
  - Pipeline vs 分离预处理
  - 多框架支持

- ✅ **反事实生成器**（3,440 tokens）
  - DiCE 和 DisCount
  - 特征约束
  - 外部生成器集成

- ✅ **匹配策略**（4,143 tokens）
  - OT、ECT、NN、SoftCEM
  - 策略选择指南
  - 性能对比

- ✅ **可视化**（4,640 tokens）⭐
  - 5种可视化类型
  - 完整参数说明
  - 自定义选项

### API 参考（6个文件）

- ✅ COLA API
- ✅ Data API
- ✅ Models API
- ✅ CE Generator API
- ✅ Policies API
- ✅ Visualization API

### 其他（6个文件）

- ✅ Installation（1,964 tokens）
- ✅ Quickstart（2,722 tokens）
- ✅ FAQ（4,218 tokens）⭐
- ✅ Contributing（2,047 tokens）
- ✅ Changelog（1,487 tokens）
- ✅ Tutorial（已存在）

**总计：20+ 文件，42,000+ tokens**

---

## 🎨 文档主题

使用 **Furo** 主题：
- 现代、清爽的设计
- 响应式布局（支持移动设备）
- 深色模式支持
- 快速导航

---

## 🛠️ 维护文档

### 修改文档

1. 编辑 `docs/` 目录下的 `.rst` 文件
2. 本地预览：
   ```bash
   cd docs
   make html
   ```
3. 提交并推送
4. Read the Docs 自动重新构建

### 添加新页面

1. 在相应目录创建 `.rst` 文件
2. 在 `index.rst` 的 `toctree` 中添加引用：
   ```rst
   .. toctree::
      :maxdepth: 2

      new_page
   ```
3. 重新构建

---

## 📞 需要帮助？

### 文档相关

- 查看 [DOCUMENTATION_COMPLETE.md](DOCUMENTATION_COMPLETE.md) - 完整文档说明
- 查看 [READTHEDOCS_DEPLOYMENT.md](READTHEDOCS_DEPLOYMENT.md) - 部署指南

### 技术问题

- 查看 [FAQ](docs/faq.rst) - 常见问题
- GitHub Issues: https://github.com/understanding-ml/COLA/issues

### 联系方式

- Email: leiyo@dtu.dk, s232291@dtu.dk

---

## ✅ 下一步

1. **本地构建测试**
   ```bash
   cd docs
   make html
   ```

2. **推送到 GitHub**
   ```bash
   git add docs/ .readthedocs.yaml README.rst
   git commit -m "docs: complete documentation"
   git push
   ```

3. **部署到 Read the Docs**
   - 按照 [READTHEDOCS_DEPLOYMENT.md](READTHEDOCS_DEPLOYMENT.md) 操作
   - 大约5分钟即可完成

4. **查看在线文档**
   - https://xai-cola.readthedocs.io/

🎉 **就这么简单！**
