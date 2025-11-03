# COLA 类改进建议

## 🔍 当前结构分析

### 存在的问题

#### 1. ❌ **职责过多（违反单一职责原则）**

COLA 类当前承担了太多职责：
```python
class COLA:
    - 策略管理
    - 数据转换
    - 可视化
    - 算法执行
    - 最小动作数查询
    - 结果格式化
```

#### 2. ❌ **硬编码的策略选择逻辑**

```python
def _get_matcher(self):
    if self.matcher == "ot":
        joint_prob = CounterfactualOptimalTransportPolicy(...)
    elif self.matcher == "cem":
        joint_prob = CounterfactualCoarsenedExactMatchingOTPolicy(...)
    # ...
```

**问题：**
- 添加新策略需要修改核心类
- 违反了开闭原则
- 可扩展性差

#### 3. ❌ **直接依赖具体类**

```python
from .policies.matching import CounterfactualOptimalTransportPolicy, ...
```

**问题：**
- 违反了依赖倒置原则
- 增加了耦合度

#### 4. ❌ **缺少状态管理**

```python
self.row_indices = None
self.col_indices = None
# 状态分散，管理混乱
```

#### 5. ❌ **方法过长，可读性差**

`get_refined_counterfactual` 方法包含了太多逻辑（60+ 行）

## 🎯 改进方案

### 方案 1：策略注册机制（推荐）

#### 核心改进

```python
# xai_cola/cola.py

from typing import Dict, Callable, Optional
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """策略基类"""
    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

class COLAStrategyFactory:
    """策略工厂 - 管理所有策略"""
    
    def __init__(self):
        self._matchers: Dict[str, Callable] = {}
        self._attributors: Dict[str, Callable] = {}
        self._composers: Dict[str, Callable] = {}
    
    def register_matcher(self, name: str, matcher_class: Callable):
        self._matchers[name] = matcher_class
    
    def register_attributor(self, name: str, attributor_class: Callable):
        self._attributors[name] = attributor_class
    
    def register_composer(self, name: str, composer_class: Callable):
        self._composers[name] = composer_class
    
    def create_matcher(self, name: str, *args, **kwargs):
        if name not in self._matchers:
            raise ValueError(f"Matcher '{name}' not registered")
        return self._matchers[name](*args, **kwargs)
    
    def create_attributor(self, name: str, *args, **kwargs):
        if name not in self._attributors:
            raise ValueError(f"Attributor '{name}' not registered")
        return self._attributors[name](*args, **kwargs)
    
    def create_composer(self, name: str, *args, **kwargs):
        if name not in self._composers:
            raise ValueError(f"Composer '{name}' not registered")
        return self._composers[name](*args, **kwargs)


class COLAPolicy:
    """策略配置对象"""
    
    def __init__(self):
        self.matcher_name: str = "ot"
        self.attributor_name: str = "pshap"
        self.composer_name: str = "max"
        self.matcher_params: dict = {}
        self.attributor_params: dict = {}
        self.composer_params: dict = {}
    
    def to_dict(self) -> dict:
        return {
            'matcher': self.matcher_name,
            'attributor': self.attributor_name,
            'composer': self.composer_name,
            'matcher_params': self.matcher_params,
            'attributor_params': self.attributor_params,
            'composer_params': self.composer_params,
        }


class COLA:
    """改进后的 COLA 类"""
    
    def __init__(
        self,
        data: BaseData,
        ml_model: Model,
        x_factual: np.ndarray,
        x_counterfactual: np.ndarray,
        policy: Optional[COLAPolicy] = None
    ):
        # 验证输入
        self._validate_inputs(data, ml_model, x_factual, x_counterfactual)
        
        self.data = data
        self.ml_model = ml_model
        self.x_factual = x_factual
        self.x_counterfactual = x_counterfactual
        self.policy = policy or COLAPolicy()
        
        # 策略工厂
        self.strategy_factory = self._create_strategy_factory()
        
        # 状态管理
        self._state = COLAResultState()
    
    def _create_strategy_factory(self) -> COLAStrategyFactory:
        """注册所有可用的策略"""
        factory = COLAStrategyFactory()
        
        # 注册匹配器
        factory.register_matcher('ot', CounterfactualOptimalTransportPolicy)
        factory.register_matcher('ect', CounterfactualExactMatchingPolicy)
        factory.register_matcher('nn', CounterfactualNearestNeighborMatchingPolicy)
        factory.register_matcher('cem', CounterfactualCoarsenedExactMatchingOTPolicy)
        
        # 注册归因器
        factory.register_attributor('pshap', PSHAP)
        # factory.register_attributor('randomshap', RandomShap)  # 未来添加
        
        # 注册合成器
        factory.register_composer('max', DataComposer)
        
        return factory
    
    def set_policy(self, matcher: str, attributor: str, **kwargs):
        """设置策略 - 更简洁的 API"""
        self.policy.matcher_name = matcher
        self.policy.attributor_name = attributor
        self.policy.matcher_params.update(kwargs)
    
    def get_refined_counterfactual(self, limited_actions: int):
        """精炼反事实 - 重构后的版本"""
        # 1. 获取匹配结果
        joint_prob = self._compute_matching()
        
        # 2. 计算特征归因
        varphi = self._compute_attribution(joint_prob)
        
        # 3. 合成数据
        q = self._compose_data(joint_prob)
        
        # 4. 应用限制动作
        result = self._apply_action_limit(varphi, q, limited_actions)
        
        # 5. 更新状态
        self._state.update(result)
        
        return self._state.get_results()
    
    def _compute_matching(self) -> np.ndarray:
        """计算匹配"""
        matcher = self.strategy_factory.create_matcher(
            self.policy.matcher_name,
            self.x_factual,
            self.x_counterfactual,
            **self.policy.matcher_params
        )
        return matcher.compute_prob_matrix_of_factual_and_counterfactual()
    
    def _compute_attribution(self, joint_prob: np.ndarray) -> np.ndarray:
        """计算归因"""
        attributor = self.strategy_factory.create_attributor(
            self.policy.attributor_name,
            self.ml_model,
            self.x_factual,
            self.x_counterfactual,
            joint_prob,
            **self.policy.attributor_params
        )
        return attributor.calculate_varphi()
    
    def _compose_data(self, joint_prob: np.ndarray) -> np.ndarray:
        """合成数据"""
        composer = self.strategy_factory.create_composer(
            self.policy.composer_name,
            self.x_counterfactual,
            joint_prob,
            method=self.policy.composer_name,
            **self.policy.composer_params
        )
        return composer.calculate_q()
    
    def _apply_action_limit(
        self, varphi: np.ndarray, q: np.ndarray, limited_actions: int
    ):
        """应用动作限制"""
        action_indices = self._select_actions(varphi, limited_actions)
        x_action_constrained = self._apply_actions(action_indices, q)
        
        return {
            'action_indices': action_indices,
            'x_action_constrained': x_action_constrained,
            'predictions': self.ml_model.predict(x_action_constrained)
        }
    
    def _select_actions(self, varphi: np.ndarray, limited_actions: int):
        """选择动作"""
        action_indice = np.random.choice(
            a=varphi.size,
            size=limited_actions,
            p=varphi.flatten(),
            replace=False,
        )
        return np.unravel_index(np.unique(action_indice), varphi.shape)
    
    def _apply_actions(self, action_indices, q: np.ndarray) -> np.ndarray:
        """应用动作到数据"""
        x_action_constrained = self.x_factual.copy()
        row_indices, col_indices = action_indices
        q_values = q[row_indices, col_indices]
        
        for row_idx, col_idx, q_val in zip(row_indices, col_indices, q_values):
            x_action_constrained[row_idx, col_idx] = q_val
        
        return x_action_constrained
    
    def highlight_changes(self):
        """高亮变化 - 委托给独立的视图层"""
        return COLADisplay(self._state).highlight_changes()
    
    def heatmap(self):
        """热力图 - 委托给独立的视图层"""
        return COLADisplay(self._state).heatmap()


class COLAResultState:
    """状态管理类"""
    
    def __init__(self):
        self.factual_df = None
        self.ce_df = None
        self.ace_df = None
        self.corresponding_counterfactual_df = None
    
    def update(self, result: dict):
        """更新状态"""
        # 更新数据...
        pass
    
    def get_results(self):
        """获取结果"""
        return self.factual_df, self.ce_df, self.ace_df


class COLADisplay:
    """显示层 - 负责可视化"""
    
    def __init__(self, state: COLAResultState):
        self.state = state
    
    def highlight_changes(self):
        """高亮显示变化"""
        # 实现...
        pass
    
    def heatmap(self):
        """生成热力图"""
        # 实现...
        pass


# 使用示例
cola = COLA(data, model, factual, counterfactual)
cola.set_policy(matcher='ect', attributor='pshap')
results = cola.get_refined_counterfactual(limited_actions=10)
```

---

### 方案 2：建造者模式

对于复杂配置的情况：

```python
class COLABuilder:
    """COLA 构建器"""
    
    def __init__(self, data: BaseData, ml_model: Model):
        self.data = data
        self.ml_model = ml_model
        self.x_factual = None
        self.x_counterfactual = None
        self.policy = COLAPolicy()
    
    def with_counterfactuals(self, x_factual, x_counterfactual):
        self.x_factual = x_factual
        self.x_counterfactual = x_counterfactual
        return self
    
    def with_matcher(self, name: str, **params):
        self.policy.matcher_name = name
        self.policy.matcher_params.update(params)
        return self
    
    def with_attributor(self, name: str, **params):
        self.policy.attributor_name = name
        self.policy.attributor_params.update(params)
        return self
    
    def build(self) -> COLA:
        if not all([self.x_factual, self.x_counterfactual]):
            raise ValueError("Must provide factual and counterfactual data")
        return COLA(self.data, self.ml_model, self.x_factual, 
                    self.x_counterfactual, self.policy)

# 使用
cola = (COLABuilder(data, model)
        .with_counterfactuals(factual, counterfactual)
        .with_matcher('ect')
        .with_attributor('pshap')
        .build())

results = cola.get_refined_counterfactual(limited_actions=10)
```

---

### 方案 3：责任链模式（高级）

将算法流程分解为多个处理器：

```python
class COLAPipeline:
    """流水线 - 责任链模式"""
    
    def __init__(self):
        self.handlers = []
    
    def add_handler(self, handler):
        self.handlers.append(handler)
        return self
    
    def execute(self, context):
        """执行流水线"""
        for handler in self.handlers:
            context = handler.process(context)
        return context

class MatchingHandler:
    """匹配处理器"""
    def process(self, context):
        context['joint_prob'] = self._compute_matching(context)
        return context

class AttributionHandler:
    """归因处理器"""
    def process(self, context):
        context['varphi'] = self._compute_attribution(context)
        return context

# 使用
pipeline = (COLAPipeline()
            .add_handler(MatchingHandler())
            .add_handler(AttributionHandler())
            .add_handler(CompositionHandler())
            .add_handler(ActionLimitHandler()))

result = pipeline.execute(initial_context)
```

---

## 📊 改进方案对比

| 特性 | 当前实现 | 方案1: 策略注册 | 方案2: 建造者 | 方案3: 责任链 |
|------|---------|--------------|-------------|-------------|
| 可扩展性 | ❌ 差 | ✅ 优秀 | ✅ 好 | ✅ 非常好 |
| 可维护性 | ❌ 差 | ✅ 好 | ✅ 很好 | ✅ 优秀 |
| 可测试性 | ⚠️ 一般 | ✅ 好 | ✅ 很好 | ✅ 优秀 |
| 代码简洁性 | ⚠️ 一般 | ✅ 很好 | ✅ 好 | ⚠️ 一般 |
| 学习曲线 | ✅ 简单 | ⚠️ 中等 | ⚠️ 中等 | ❌ 复杂 |

## 🎯 推荐方案

### 短期改进（方案 1 的部分）

**立即改进：**

1. **添加策略工厂类** - 解耦策略选择
2. **提取配置对象** - COLAPolicy 类
3. **分离状态管理** - COLAResultState 类
4. **方法拆分** - 将长方法拆分为小方法

### 长期改进（方案 3）

逐步引入责任链模式，实现完全的解耦和可扩展性。

---

## ✅ 具体实施建议

**优先级 1（立即实施）：**

1. 提取策略工厂
2. 添加配置对象
3. 方法拆分（每个方法 <= 20 行）

**优先级 2（中期）：**

4. 状态管理类
5. 显示层分离
6. 完善错误处理

**优先级 3（长期）：**

7. 引入责任链模式
8. 添加缓存机制
9. 性能优化

## 🎓 核心原则

改进后的设计遵循：
- ✅ **单一职责原则** (SRP)
- ✅ **开闭原则** (OCP)
- ✅ **依赖倒置原则** (DIP)
- ✅ **接口隔离原则** (ISP)

