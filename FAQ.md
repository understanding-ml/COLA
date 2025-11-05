# Frequently Asked Questions (FAQ)

## General Questions

### What is COLA?

COLA (COunterfactual explanations with Limited Actions) is a Python framework that refines counterfactual explanations from any CE algorithm to require fewer feature changes while maintaining similar outcomes. It's based on joint-distribution-informed Shapley values (PSHAP) and works with any machine learning model.

### How is COLA different from other counterfactual explainers?

COLA is not a counterfactual explainer itself—it's a **refinement framework** that works on top of existing explainers (DiCE, DisCount, Alibi, etc.). While traditional CE algorithms might suggest changing 10 features, COLA can refine this to just 3-4 features while achieving the same or similar outcome.

### Which counterfactual explainers does COLA support?

COLA is explainer-agnostic and can work with any CE algorithm. We provide built-in support for:
- **DiCE** (Diverse Counterfactual Explanations)
- **DisCount** (Distributional Counterfactuals)
- **Alibi-CFI** (Counterfactual Instances)
- **ARecourseS** (Actionable Recourse)
- **WachterCF** (Gradient-based optimization)
- **Growing Sphere**
- Custom explainers (by providing factual and counterfactual arrays)

### What machine learning frameworks does COLA support?

COLA supports:
- **scikit-learn** models (any classifier/regressor)
- **PyTorch** models
- **TensorFlow** (1.x and 2.x)
- **LightGBM**, **XGBoost**, **CatBoost** (via sklearn interface)

## Installation & Setup

### How do I install COLA?

```bash
pip install xai-cola
```

For detailed instructions, see the [Installation Guide](INSTALLATION.md).

### What are the minimum requirements?

- Python 3.7 or higher
- NumPy, Pandas, scikit-learn
- For specific features: PyTorch (for PyTorch models), TensorFlow (for TF models)

### I get an import error after installation. What should I do?

1. Verify installation: `pip list | grep xai-cola`
2. Check Python environment: `which python`
3. Try reinstalling: `pip install --force-reinstall xai-cola`
4. See [Troubleshooting](INSTALLATION.md#troubleshooting)

## Usage Questions

### How do I use COLA with my own dataset?

```python
from xai_cola.data import COLAData
import pandas as pd

# Your dataset as DataFrame
df = pd.read_csv('your_data.csv')

# Create data interface
data = COLAData(
    factual_data=df,
    label_column='target',
    transform='ohe-zscore'  # or None for no preprocessing
)
```

See [Data Interface Guide](docs/DATA_INTERFACE_QUICKREF.md) for details.

### How do I use COLA with my custom ML model?

```python
from xai_cola.models import Model

# Wrap your model
ml_model = Model(model=your_trained_model, backend="sklearn")
# backend can be "sklearn", "pytorch", "tensorflow1", "tensorflow2"
```

### What is the `features_to_vary` parameter?

This parameter lets you specify which features COLA can modify. For example:

```python
factual, ce, ace = refiner.get_all_results(
    limited_actions=5,
    features_to_vary=['Age', 'Income', 'Credit_Score']
)
```

Only these three features will be modified; all others remain unchanged. This is useful for:
- **Immutable features**: Don't modify gender, race, etc.
- **Domain constraints**: Only modify actionable features
- **User preferences**: Focus on specific features

### What do "limited_actions" mean?

The `limited_actions` parameter specifies the maximum number of features that can be changed. For example:

```python
# Original counterfactual might change 10 features
# This will refine it to change at most 5 features
factual, ce, ace = refiner.get_all_results(limited_actions=5)
```

### Which matching policy should I use?

Different policies suit different scenarios:

- **`"ect"` (Exact Matching)**: Best for DiCE and when you have one-to-one factual-counterfactual pairs. **Recommended default**.
- **`"ot"` (Optimal Transport)**: Best for group-based counterfactuals (DisCount) or when you have many counterfactuals per factual.
- **`"nn"` (Nearest Neighbor)**: Simple and fast, good for quick experimentation.
- **`"cem"` (Coarsened Exact Matching)**: Good for high-dimensional data.

### What is PSHAP?

PSHAP (Probability-weighted Shapley values) is a feature attribution method that computes Shapley values weighted by joint probability distribution. It helps COLA identify which features are most important to change, considering not just individual feature importance but also feature interactions.

### How do I visualize the results?

COLA provides several visualization methods:

```python
# For small datasets - highlighted DataFrames
factual, ce, ace = refiner.highlight_changes_final()
display(ce)  # Shows color-coded changes

# For any dataset - heatmaps
refiner.heatmap_binary(save_path='./results', save_mode='combined')
refiner.heatmap_direction(save_path='./results', save_mode='combined')

# Comparison charts
refiner.stacked_bar_chart(save_path='./results')

# Diversity analysis - minimal feature combinations
factual_df, diversity_styles = refiner.diversity()
```

## Performance Questions

### How long does COLA take to run?

Performance depends on:
- Dataset size (10 instances: ~1 second, 1000 instances: ~30 seconds)
- Number of features
- Matching policy (NN is fastest, OT is slowest)
- Model complexity

For typical use cases (100 instances, 20 features), expect 5-10 seconds.

### Can COLA handle large datasets?

Yes, but consider:
- **Use batch processing** for >10,000 instances
- **Use simpler matching policies** (NN instead of OT)
- **Sample your data** for exploration, then apply to full dataset

### Does COLA support GPU acceleration?

COLA itself doesn't use GPU, but if your ML model uses GPU (PyTorch/TensorFlow), COLA will benefit from faster model predictions.

## Troubleshooting

### Why do I get "No valid counterfactuals found"?

This happens when:
1. **The CE generator failed** - Check that your CE algorithm works independently
2. **All counterfactuals have same class as factuals** - The CE generator didn't flip the prediction
3. **Invalid data format** - Ensure data is correctly formatted

Solution: Verify your CE generator works before passing to COLA.

### The refined counterfactual still changes many features. Why?

Possible reasons:
1. **`limited_actions` is set too high** - Reduce this number
2. **Few feasible changes** - Your model might require many changes
3. **Restrictive `features_to_vary`** - Too few features available to modify
4. **CE quality** - Original counterfactuals are far from factuals

Try: Generate better quality counterfactuals first, or relax constraints.

### COLA results differ from paper results. Why?

Possible reasons:
1. **Different random seeds** - Set seed for reproducibility
2. **Different datasets** - Results vary by dataset characteristics
3. **Different CE generators** - Each generator produces different counterfactuals
4. **Different hyperparameters** - Matching policy, attribution method, etc.

### Can I use COLA without COLAData?

Yes! You can pass raw numpy arrays directly:

```python
refiner = COLA(
    data=None,
    ml_model=ml_model,
    x_factual=factual_array,
    x_counterfactual=counterfactual_array,
    feature_names=feature_names,
    target_name='target'
)
```

However, `COLAData` is recommended for automatic preprocessing and better integration.

## Advanced Questions

### How do I add a custom matching policy?

Subclass `BaseMatcher`:

```python
from xai_cola.ce_sparsifier.policies.matching.base_matcher import BaseMatcher

class MyMatcher(BaseMatcher):
    def match(self, factual, counterfactual):
        # Your matching logic
        # Return joint probability matrix
        pass
```

### How do I add a custom feature attributor?

Subclass `BaseAttributor`:

```python
from xai_cola.ce_sparsifier.policies.feature_attributor.base_attributor import BaseAttributor

class MyAttributor(BaseAttributor):
    def attribute(self, data, ml_model):
        # Your attribution logic
        # Return feature importance scores
        pass
```

### Can I use COLA for regression problems?

Currently, COLA is primarily designed for classification. Regression support is planned for future versions.

### How can I contribute to COLA?

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Areas where we need help:
- Adding new matching policies
- Adding new feature attribution methods
- Improving documentation
- Adding examples and tutorials
- Bug reports and fixes

## Citation & Academic Use

### How do I cite COLA?

```bibtex
@article{you2024refining,
  title={Refining Counterfactual Explanations With Joint-Distribution-Informed Shapley Towards Actionable Minimality},
  author={You, Lei and Bian, Yijun and Cao, Lele},
  journal={arXiv preprint arXiv:2410.05419},
  year={2024}
}
```

### Where can I find the paper?

The paper is available on [arXiv](https://arxiv.org/pdf/2410.05419).

## Getting Help

### I have a question not covered here. Where can I ask?

1. **Check the documentation**: [API Reference](API_REFERENCE.md), [Quick Start](QUICKSTART.md)
2. **Search GitHub Issues**: Someone may have asked before
3. **Open a new issue**: [GitHub Issues](https://github.com/your-repo/COLA/issues)
4. **Contact maintainers**:
   - Lei You (leiyo@dtu.dk)
   - Lin Zhu (s232291@dtu.dk)

### How do I report a bug?

Open an issue on GitHub with:
- Python version
- COLA version
- Operating system
- Minimal reproducible example
- Full error traceback

### Can I request a new feature?

Yes! Open an issue labeled "feature request" and describe:
- What feature you want
- Why it's useful
- Proposed API (if applicable)
- Willingness to contribute implementation

## License & Legal

### What license is COLA under?

COLA is licensed under the MIT License. You're free to use it in commercial and academic projects.

### Can I use COLA in production?

Yes, but COLA is research software primarily for explainability purposes. Ensure thorough testing for your specific use case.

### Can I modify COLA for my needs?

Yes! Under the MIT License, you can modify and distribute COLA. We appreciate attribution and welcome contributions back to the main project.
