# highlight_changes 方法更新总结

## 修改内容

添加了一个新的方法 `highlight_changes_final()`，同时保留并改进了原有的 `highlight_changes_comparison()` 方法。

## 两个方法的区别

### 1. `highlight_changes_comparison()` - 对比格式

**显示格式**: `factual_value -> counterfactual_value`  
**示例**: `1553 -> 1103`

**适用场景**:
- 需要详细对比分析时
- 想看清楚每个特征从什么值变成什么值
- 做数据分析或调试时

**代码示例**:
```python
factual_df, ce_style, ace_style = refiner.highlight_changes_comparison()

# 在 Jupyter 中显示
display(ce_style)  # 显示 "1553 -> 1103"

# 保存为 HTML
ce_style.to_html('comparison.html')
```

### 2. `highlight_changes_final()` - 简洁格式

**显示格式**: 只显示 `counterfactual_value`  
**示例**: `1103`

**适用场景**:
- 用于报告或演示
- 希望展示更简洁美观
- 只关心最终的 counterfactual 结果

**代码示例**:
```python
factual_df, ce_style, ace_style = refiner.highlight_changes_final()

# 在 Jupyter 中显示
display(ce_style)  # 只显示 "1103"

# 保存为 HTML
ce_style.to_html('final.html')
```

## 返回值

两种方法返回相同的格式：

```python
tuple: (factual_df, ce_style, ace_style)
```

- `factual_df`: `pandas.DataFrame` - 原始数据
- `ce_style`: `pandas.io.formats.style.Styler` - full counterfactual 的高亮显示
- `ace_style`: `pandas.io.formats.style.Styler` - action-limited counterfactual 的高亮显示

## 使用方式

### 在 Jupyter Notebook 中

```python
from IPython.display import display

# 方法 1: 对比格式
factual_df, ce_style, ace_style = refiner.highlight_changes_comparison()
display(ce_style)  # 显示 "1553 -> 1103"

# 方法 2: 简洁格式
factual_df, ce_style, ace_style = refiner.highlight_changes_final()
display(ce_style)  # 只显示 "1103"
```

### 保存为 HTML

```python
# 方法 1
ce_style.to_html('comparison.html')
ace_style.to_html('ace_comparison.html')

# 方法 2
ce_style.to_html('final.html')
ace_style.to_html('ace_final.html')
```

### 保存为 Excel/CSV（无样式）

```python
# 获取原始数据
ce_style.data.to_excel('data.xlsx', index=False)
ce_style.data.to_csv('data.csv', index=False)
```

## 向后兼容性

⚠️ **注意** - 方法名已更改：
- 旧的 `highlight_changes()` → 新的 `highlight_changes_comparison()`
- 新增 `highlight_changes_final()`

如果需要向后兼容，可以添加别名方法。

## 高亮效果

两种方法的高亮效果相同：
- **特征变化**: 黄色背景 + 黑色边框
- **目标列变化**: 浅灰色背景 + 黑色边框

区别仅在于单元格中显示的文本内容格式。

## 文档

完整的使用指南请参考：`HIGHLIGHT_CHANGES_USAGE.md`

