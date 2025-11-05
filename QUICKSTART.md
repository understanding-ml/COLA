# COLA Quick Start Guide

This guide will help you get started with COLA in minutes.

## Installation

```bash
pip install xai-cola
```

## Basic Usage

### Step 1: Import Necessary Modules

```python
from xai_cola import COLA
from xai_cola.data import COLAData
from xai_cola.models import Model
from counterfactual_explainer import DiCE, WachterCF
import joblib
```

### Step 2: Load Your Data

```python
# Load your dataset
from datasets.german_credit import GermanCreditDataset
dataset = GermanCreditDataset()
df = dataset.get_dataframe()

# Select instances to explain
df_to_explain = df[df['Risk'] == 1].sample(5)

# Initialize data interface (with optional transformation)
data = COLAData(
    factual_data=df_to_explain,
    label_column='Risk',
    transform=None,  # or 'ohe-zscore', 'ohe-min-max'
    numerical_features=['Age', 'Credit amount', 'Duration']  # if using transform
)
```

### Step 3: Load Your Model

```python
# Load your trained model
lgbmcClassifier = joblib.load('lgbm_GremanCredit.pkl')

# Initialize model interface
ml_model = Model(model=lgbmcClassifier, backend="sklearn")
```

### Step 4: Generate Counterfactuals

You can choose from different counterfactual explainers:

**Option A: Using DiCE (recommended for most cases)**

```python
explainer = DiCE(ml_model=ml_model)
factual, counterfactual = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    total_cfs=1,
    features_to_keep=['Age', 'Sex'],
    continuous_features=['Age', 'Credit amount', 'Duration']  # optional
)
```

**Option B: Using WachterCF (gradient-based, supports both sklearn and PyTorch)**

```python
# With sklearn model
explainer = WachterCF(ml_model=ml_model)
factual, counterfactual = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    features_to_vary=['Age', 'Credit amount'],
    target_proba=0.7,
    max_iter=100
)

# With PyTorch model (faster, uses automatic gradients)
import torch.nn as nn
pytorch_model = nn.Sequential(...)  # Your PyTorch model
pytorch_ml_model = Model(pytorch_model, backend='pytorch')
explainer = WachterCF(ml_model=pytorch_ml_model)
factual, counterfactual = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    target_proba=0.7
)
```

### Step 5: Add Counterfactuals and Refine with COLA

```python
# Add counterfactuals to data
data.add_counterfactuals(counterfactual, with_target_column=True)

# Initialize COLA
refiner = COLA(
    data=data,
    ml_model=ml_model
)

# Set refinement policy
refiner.set_policy(
    matcher="ect",      # Exact Matching
    attributor="pshap",  # PSHAP attribution
    Avalues_method="max" # Max method for A-values
)

# Get refined counterfactuals with limited actions
# Option 1: All features can be modified
factual, ce, ace = refiner.get_all_results(limited_actions=10)

# Option 2: Only modify specific features
factual, ce, ace = refiner.get_all_results(
    limited_actions=10,
    features_to_vary=['Age', 'Credit amount', 'Duration']  # Only these features will be modified
)
```

### Step 6: Visualize Results

COLA provides multiple visualization methods to help you understand the changes:

#### Option 1: Highlight Changes (Best for Small Datasets)

```python
# Comparison format (shows "old -> new")
factual_df, ce_style, ace_style = refiner.highlight_changes_comparison()
display(factual_df)
display(ce_style)   # Shows factual → full counterfactual
display(ace_style)  # Shows factual → action-limited counterfactual

# Or final format (shows only final values)
factual_df, ce_style, ace_style = refiner.highlight_changes_final()
display(ce_style)
display(ace_style)
```

#### Option 2: Binary Heatmap (Shows Changed/Unchanged)

```python
# Generate binary change heatmap
plot1, plot2 = refiner.heatmap_binary(
    save_path='./results',     # Optional: save to file
    save_mode='combined',      # 'combined' or 'separate'
    show_axis_labels=True      # Show column names and row indices
)

# Color scheme:
# - Red: Changed features
# - Light grey: Unchanged cells
# - Dark blue (#000080): Target column (changed)
```

#### Option 3: Directional Heatmap (Shows Increase/Decrease)

```python
# Generate directional change heatmap
plot1, plot2 = refiner.heatmap_direction(
    save_path='./results',
    save_mode='combined',
    show_axis_labels=True
)

# Color scheme:
# - Teal-green (#009E73): Increased values
# - Red-orange (#D55E00): Decreased values
# - Light grey: Unchanged cells
# - Dark blue (#000080): Target column (changed)
```

**Visualization Tips:**
- **Small datasets (< 20 instances)**: Use `highlight_changes_comparison()` or `highlight_changes_final()`
- **Large datasets**: Use `heatmap_binary()` or `heatmap_direction()`
- **Combined mode**: Creates two heatmaps stacked vertically (top: full counterfactual, bottom: action-limited counterfactual)
- **Separate mode**: Saves two separate image files
- **show_axis_labels=False**: Hides column names and row indices for cleaner visualization

## Available Explainers

- **DiCE**: Diverse Counterfactual Explanations (supports sklearn and PyTorch models)
- **DisCount**: Distributional Counterfactual with Optimal Transport (PyTorch only)
- **Alibi-CFI**: Alibi Counterfactual Instances
- **ARecourseS**: Actionable Recourse
- **WachterCF**: Gradient-based counterfactual generator (supports sklearn and PyTorch models)

## Available Matching Policies

- **ECT**: Exact Counterfactual Transport (default)
- **OT**: Optimal Transport
- **NN**: Nearest Neighbor Matching
- **CEM**: Coarsened Exact Matching

## Available Attribution Methods

- **PSHAP**: Shapley values with joint probability (default)
- **RandomShap**: Random Shapley values

## Tips

1. **Small datasets (< 20 instances)**: Use `highlight_changes()` for detailed comparison
2. **Large datasets**: Use `heatmap()` for visualization
3. **Choose matching policy**: 
   - ECT: Best for 1-to-1 matching
   - OT: Best for distributional matching
   - NN: Fast alternative to OT
4. **Limited actions**: Start with 5-10 and adjust based on your needs

## Next Steps

- Check out the [full documentation](README.md)
- See [examples](demo.ipynb) for more use cases
- Read the [paper](https://arxiv.org/pdf/2410.05419) for theoretical details

