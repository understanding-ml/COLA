# `highlight_changes()` 返回值说明和使用指南

## 两个方法

COLA 提供了两种高亮显示方法：

### 1. `highlight_changes_comparison()` - 对比格式（推荐用于分析）

显示 **"factual_value -> counterfactual_value"** 格式，方便对比前后变化。

```python
factual_df, ce_style, ace_style = refiner.highlight_changes_comparison()
```

### 2. `highlight_changes_final()` - 简洁格式（推荐用于展示）

只显示最终的 counterfactual 值，更简洁美观。

```python
factual_df, ce_style, ace_style = refiner.highlight_changes_final()
```

## 返回类型

两种方法都返回一个三元组：

1. **`factual_df`**: `pandas.DataFrame` - 原始 factual 数据
2. **`ce_style`**: `pandas.io.formats.style.Styler` - 样式化的 DataFrame
3. **`ace_style`**: `pandas.io.formats.style.Styler` - 样式化的 DataFrame

## 使用方式

### 1. 在 Jupyter Notebook 中显示（推荐）

```python
from IPython.display import display

factual_df, ce_style, ace_style = refiner.highlight_changes_comparison()

# 显示 factual DataFrame（普通表格）
display(factual_df)

# 显示带高亮的 counterfactual 对比（样式化表格）
print("Factual → Full Counterfactual:")
display(ce_style)

print("Factual → Action-Limited Counterfactual:")
display(ace_style)
```

**效果**：会在 notebook 中显示带颜色高亮的 HTML 表格

### 2. 保存为 HTML 文件

```python
factual_df, ce_style, ace_style = refiner.highlight_changes_comparison()

# 保存为 HTML 文件
with open('counterfactual_highlight.html', 'w', encoding='utf-8') as f:
    f.write(ce_style.to_html())

# 或者保存 action-limited 版本
with open('action_limited_highlight.html', 'w', encoding='utf-8') as f:
    f.write(ace_style.to_html())

# 保存完整的对比页面（包含所有三个表格）
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Counterfactual Highlights</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h2 {{ color: #333; }}
        table {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Counterfactual Explanation Highlights</h1>
    
    <h2>Factual Data</h2>
    {factual_df.to_html()}
    
    <h2>Factual → Full Counterfactual</h2>
    {ce_style.to_html()}
    
    <h2>Factual → Action-Limited Counterfactual</h2>
    {ace_style.to_html()}
</body>
</html>
"""

with open('complete_highlight.html', 'w', encoding='utf-8') as f:
    f.write(html_content)
```

### 3. 在普通 Python 脚本中打印（无样式）

```python
factual_df, ce_style, ace_style = refiner.highlight_changes_comparison()

# 打印原始 DataFrame（普通文本表格）
print("Factual DataFrame:")
print(factual_df)

# Styler 对象不能直接打印，需要转换为 DataFrame 或 HTML
print("\nCounterfactual Changes (without styling):")
print(ce_style.data)  # 获取底层 DataFrame（无样式）

# 或者打印 HTML（可以在浏览器中查看）
print("\nHTML Output:")
print(ce_style.to_html())
```

### 4. 保存为 Excel/CSV（无样式，只保存数据）

```python
factual_df, ce_style, ace_style = refiner.highlight_changes_comparison()

# 保存为 Excel（包含多个 sheet）
with pd.ExcelWriter('counterfactual_results.xlsx', engine='openpyxl') as writer:
    factual_df.to_excel(writer, sheet_name='Factual', index=False)
    ce_style.data.to_excel(writer, sheet_name='Full_CF', index=False)  # .data 获取底层 DataFrame
    ace_style.data.to_excel(writer, sheet_name='Limited_CF', index=False)

# 保存为 CSV（分别保存）
factual_df.to_csv('factual.csv', index=False)
ce_style.data.to_csv('full_counterfactual.csv', index=False)
ace_style.data.to_csv('limited_counterfactual.csv', index=False)
```

### 5. HTML 转图片（需要额外库）

如果需要保存为图片，可以使用以下方法：

```python
# 方法 1: 使用 imgkit（需要安装 wkhtmltopdf）
# pip install imgkit
import imgkit

factual_df, ce_style, ace_style = refiner.highlight_changes_comparison()

html = ce_style.to_html()
imgkit.from_string(html, 'highlight.png', options={'width': 1200})

# 方法 2: 使用 weasyprint（推荐，纯 Python）
# pip install weasyprint
from weasyprint import HTML

html = ce_style.to_html()
HTML(string=html).write_png('highlight.png')
```

## 完整示例

### 示例 1: 使用对比格式（highlight_changes_comparison）

```python
from xai_cola import COLA
from xai_cola.data import COLAData
from xai_cola.models import Model
from IPython.display import display

# ... 初始化 COLA ...

# 生成结果
factual_df, cf_df, ace_df = refiner.get_all_results(limited_actions=8)

# 获取高亮显示（对比格式：old -> new）
factual_df, ce_style, ace_style = refiner.highlight_changes_comparison()

# 在 notebook 中显示
print("=" * 60)
print("Factual Data")
print("=" * 60)
display(factual_df)

print("\n" + "=" * 60)
print("Changes: Factual → Full Counterfactual (对比格式)")
print("=" * 60)
display(ce_style)  # 显示 "1553 -> 1103" 这样的格式

print("\n" + "=" * 60)
print("Changes: Factual → Action-Limited Counterfactual (对比格式)")
print("=" * 60)
display(ace_style)  # 显示 "1553 -> 1103" 这样的格式

# 保存为 HTML
ce_style.to_html('cf_highlight_comparison.html')
ace_style.to_html('ace_highlight_comparison.html')

print("\n✅ HTML files saved!")
```

### 示例 2: 使用简洁格式（highlight_changes_final）

```python
# 获取高亮显示（简洁格式：只显示最终值）
factual_df, ce_style, ace_style = refiner.highlight_changes_final()

# 在 notebook 中显示
print("=" * 60)
print("Factual Data")
print("=" * 60)
display(factual_df)

print("\n" + "=" * 60)
print("Counterfactual (简洁格式)")
print("=" * 60)
display(ce_style)  # 只显示 "1103" 这样的最终值

print("\n" + "=" * 60)
print("Action-Limited Counterfactual (简洁格式)")
print("=" * 60)
display(ace_style)  # 只显示 "1103" 这样的最终值

# 保存为 HTML
ce_style.to_html('cf_highlight_final.html')
ace_style.to_html('ace_highlight_final.html')

print("\n✅ HTML files saved!")
```

### 对比

| 特性 | `highlight_changes_comparison()` | `highlight_changes_final()` |
|------|----------------------|----------------------------|
| **格式** | 显示 "old -> new" | 只显示 "new" |
| **用途** | 分析对比变化 | 简洁展示结果 |
| **可读性** | 可以看到前后值对比 | 更简洁清晰 |
| **示例** | `1553 -> 1103` | `1103` |

## 注意事项

1. **Styler 对象**是 pandas 专门为 Jupyter notebook 设计的，在普通 Python 脚本中直接打印只会显示对象信息，不会显示样式
2. **保存 HTML**是最简单的方式，可以在任何浏览器中查看带样式的表格
3. **`.data` 属性**可以获取 Styler 对象底层的 DataFrame（不含样式）
4. **保存为图片**需要额外的库（如 imgkit、weasyprint），这些库可能需要系统依赖

## 推荐工作流

- **开发/探索**：在 Jupyter Notebook 中使用 `display()` 直接查看
- **报告/演示**：保存为 HTML 文件，在浏览器中查看或嵌入报告
- **数据分析**：使用 `.data` 获取底层 DataFrame，保存为 Excel/CSV 进行进一步分析

