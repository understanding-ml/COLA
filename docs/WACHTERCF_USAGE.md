# WachterCF Counterfactual Explainer - Usage Guide

## Overview

WachterCF is a gradient-based counterfactual explanation method that generates counterfactuals by optimizing a loss function that balances:
1. **Prediction loss**: How well the counterfactual reaches the target class probability
2. **Distance loss**: How close the counterfactual is to the original instance

This implementation supports both **PyTorch models** (with automatic gradients) and **sklearn models** (with numerical gradients via finite differences).

## Key Features

- ✅ **PyTorch models**: Fast automatic gradient computation
- ✅ **sklearn models**: Numerical gradient support via finite differences
- ✅ **Feature constraints**: Specify which features can vary
- ✅ **Customizable optimization**: Adjustable learning rate, optimizer, and iterations
- ✅ **Data transformation**: Automatic handling of data preprocessing and inverse transformation

## Installation

No additional installation required. WachterCF is included in the COLA package.

```python
from counterfactual_explainer import WachterCF
from xai_cola.models import Model
from xai_cola.data import COLAData
```

## Basic Usage

### With PyTorch Models (Recommended)

PyTorch models use automatic gradients, making optimization faster and more accurate:

```python
import torch.nn as nn
from xai_cola.models import Model
from xai_cola.data import COLAData
from counterfactual_explainer import WachterCF

# Your PyTorch model
pytorch_model = nn.Sequential(
    nn.Linear(num_features, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

# Initialize model wrapper
model = Model(pytorch_model, backend='pytorch')

# Initialize data
data = COLAData(
    factual_data=df,
    label_column='Risk',
    transform='ohe-min-max',  # Optional: normalize data
    numerical_features=['Age', 'Credit amount', 'Duration']
)

# Create explainer
explainer = WachterCF(ml_model=model, data=data)

# Generate counterfactuals
factual_df, counterfactual_df = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    features_to_vary=['Age', 'Credit amount', 'Duration'],
    target_proba=0.7,
    _lambda=10.0,
    max_iter=100
)
```

### With sklearn Models

sklearn models use numerical gradients (slower but compatible):

```python
from sklearn.ensemble import RandomForestClassifier
from xai_cola.models import Model
from xai_cola.data import COLAData
from counterfactual_explainer import WachterCF

# Your sklearn model
sklearn_model = RandomForestClassifier()
sklearn_model.fit(X_train, y_train)

# Initialize model wrapper
model = Model(sklearn_model, backend='sklearn')

# Initialize data
data = COLAData(
    factual_data=df,
    label_column='Risk'
)

# Create explainer
explainer = WachterCF(ml_model=model, data=data)

# Generate counterfactuals
factual_df, counterfactual_df = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    target_proba=0.7,
    max_iter=100  # May need more iterations for sklearn models
)
```

## Parameters

### `generate_counterfactuals()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | COLAData | None | Factual data wrapper |
| `factual_class` | int | 1 | Original class of factual data |
| `features_to_vary` | List[str] | None | Features allowed to change (None = all) |
| `target_proba` | float | 0.7 | Target probability for counterfactual class |
| `feature_weights` | List[float] | None | Weights for distance computation (None = equal) |
| `_lambda` | float | 10.0 | Weight for prediction loss (higher = prioritize reaching target) |
| `optimizer` | str | "adam" | Optimizer: "adam" or "rmsprop" |
| `lr` | float | 0.01 | Learning rate |
| `max_iter` | int | 100 | Maximum optimization iterations |

### Parameter Tuning Guide

1. **`_lambda`**: 
   - Higher values (10-50): Emphasize reaching target probability
   - Lower values (1-5): Emphasize staying close to original
   - Default: 10.0

2. **`target_proba`**: 
   - Range: [0, 1]
   - Higher values (0.8-0.9): More confident counterfactuals
   - Default: 0.7

3. **`lr`**: 
   - Higher (0.1): Faster convergence, may overshoot
   - Lower (0.001): Slower but more stable
   - Default: 0.01

4. **`max_iter`**: 
   - PyTorch: 50-100 iterations usually sufficient
   - sklearn: May need 100-200 iterations (slower convergence)
   - Default: 100

## Advanced Usage

### Constraining Specific Features

```python
# Only allow certain features to vary
factual_df, counterfactual_df = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    features_to_vary=['Age', 'Credit amount'],  # Other features stay fixed
    target_proba=0.7
)
```

### Custom Feature Weights

```python
# Weight features differently in distance computation
feature_weights = [1.0, 2.0, 0.5, ...]  # One weight per feature
factual_df, counterfactual_df = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    feature_weights=feature_weights,
    target_proba=0.7
)
```

### Using Data Transformation

```python
# With automatic data preprocessing
data = COLAData(
    factual_data=df,
    label_column='Risk',
    transform='ohe-zscore',  # One-hot encode categorical, standardize numerical
    numerical_features=['Age', 'Credit amount', 'Duration']
)

explainer = WachterCF(ml_model=model, data=data)
# Transformation is automatically handled
factual_df, counterfactual_df = explainer.generate_counterfactuals(data=data)
# Counterfactuals are automatically inverse-transformed to original format
```

## Performance Comparison

| Model Type | Gradient Method | Speed | Accuracy |
|------------|----------------|-------|----------|
| PyTorch | Automatic | ⚡ Fast | ⭐⭐⭐ High |
| sklearn | Numerical | 🐢 Slower | ⭐⭐ Medium |

**Recommendation**: Use PyTorch models when possible for better performance.

## Complete Example

```python
import pandas as pd
import torch.nn as nn
from xai_cola.data import COLAData
from xai_cola.models import Model
from counterfactual_explainer import WachterCF

# 1. Prepare data
df = pd.read_csv('your_data.csv')
df_to_explain = df[df['Risk'] == 1].head(5)

data = COLAData(
    factual_data=df_to_explain,
    label_column='Risk',
    transform='ohe-min-max',
    numerical_features=['Age', 'Credit amount', 'Duration']
)

# 2. Prepare model
# Option A: PyTorch
pytorch_model = nn.Sequential(...)
model = Model(pytorch_model, backend='pytorch')

# Option B: sklearn
# from sklearn.ensemble import RandomForestClassifier
# sklearn_model = RandomForestClassifier()
# sklearn_model.fit(X_train, y_train)
# model = Model(sklearn_model, backend='sklearn')

# 3. Generate counterfactuals
explainer = WachterCF(ml_model=model, data=data)
factual_df, counterfactual_df = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    features_to_vary=['Age', 'Credit amount'],
    target_proba=0.7,
    _lambda=10.0,
    optimizer='adam',
    lr=0.01,
    max_iter=100
)

print("Factual:")
print(factual_df)
print("\nCounterfactual:")
print(counterfactual_df)

# 4. Use with COLA for refinement
data.add_counterfactuals(counterfactual_df, with_target_column=True)
from xai_cola import COLA

cola = COLA(data=data, ml_model=model)
cola.set_policy(matcher='ect', attributor='pshap')
refined_cf = cola.get_refined_counterfactual(limited_actions=5)
```

## Troubleshooting

### Issue: Slow convergence with sklearn models
**Solution**: Increase `max_iter` to 200-300, or use a PyTorch model instead.

### Issue: Counterfactuals don't reach target probability
**Solution**: 
- Increase `_lambda` (try 20-50)
- Increase `target_proba` (try 0.8-0.9)
- Check if `features_to_vary` allows enough features to change

### Issue: Counterfactuals are too different from original
**Solution**: 
- Decrease `_lambda` (try 1-5)
- Add more features to `features_to_vary`
- Use custom `feature_weights` to penalize certain feature changes

### Issue: ValueError about backend
**Solution**: Ensure your model backend is 'pytorch' or 'sklearn':
```python
model = Model(your_model, backend='pytorch')  # or 'sklearn'
```

## References

- Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations without opening the black box: automated decisions and the GDPR. *Harv. JL & Tech., 31*, 841.

## See Also

- [DiCE Usage Guide](dice_usage.md) - For DiCE explainer documentation
- [COLA Main Documentation](../README.md) - For COLA refinement pipeline
- [API Reference](../API_REFERENCE.md) - Complete API documentation

