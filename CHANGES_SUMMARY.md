# COLA Data 接口重构总结

## 🎯 选择的方案

采用**选择 1 的优化版本**：

✅ **保留文件结构，但改为包装器**
- `pandas.py` → 成为 COLAData 的包装器
- `numpy.py` → 成为 COLAData 的包装器
- `base.py` → 保留用于兼容

## 📁 当前文件结构

```
xai_cola/data/
├── __init__.py         # 导出接口
├── coladata.py         # 核心实现（新接口）⭐ 推荐使用
├── base.py             # 抽象基类（保留）
├── pandas.py           # COLAData 包装器（兼容）
└── numpy.py            # COLAData 包装器（兼容）
```

## ✅ 优点

1. **向后兼容** - 旧代码无需修改
2. **代码简洁** - 只有一套实现（COLAData）
3. **易于维护** - 旧接口只是薄包装层
4. **渐进迁移** - 可以慢慢迁移到新接口

## 📝 使用方式

### 推荐方式（新代码）

```python
from xai_cola.data import COLAData

data = COLAData(
    factual_data=df,
    label_column='Risk'
)
```

### 兼容方式（旧代码仍然工作）

```python
from xai_cola.data import PandasData

data = PandasData(df, target_name='Risk')
# 内部使用 COLAData 实现
```

## 🎯 总结

**最佳实践**：
- ✅ 新项目使用 `COLAData`
- ✅ 旧代码无需修改（自动兼容）
- ✅ 维护成本低（只有一套实现）

