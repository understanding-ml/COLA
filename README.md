# COLA - COunterfactual explanations with Limited Actions

[![arXiv](https://img.shields.io/badge/arXiv-2410.05419-B31B1B.svg)](https://arxiv.org/pdf/2410.05419)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/xai-cola.svg)](https://pypi.org/project/xai-cola/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

<p align="center">
  <img src="docs/images/problem.png" alt="COLA Framework" width="700"/>
</p>

## Overview

Explainable Artificial Intelligence (XAI) is essential for making artificial intelligence systems transparent and trustworthy. Within XAI, **counterfactual explanations (CE)** show how small changes in input features can lead to different outcomes. However, existing CE algorithms often generate explanations that require changing too many features, making them impractical for real-world applications.

**COLA** is a Python framework that refines counterfactual explanations by generating **action-limited plans** that require significantly fewer feature changes while maintaining similar or equal outcomes. COLA adapts to various CE methods and ML models, with near-optimal performance under certain circumstances.

### Key Advantages

- **Fewer Actions**: Reduces the number of feature changes needed by 30-70% compared to raw counterfactuals
- **Model Agnostic**: Works with any ML model (sklearn, PyTorch, TensorFlow)
- **CE Method Agnostic**: Compatible with various counterfactual explainers (DiCE, DisCount, Alibi, etc.)
- **Theoretically Grounded**: Based on joint-distribution-informed Shapley values (PSHAP)
- **Easy to Use**: Simple API with sensible defaults

## Installation

```bash
pip install xai-cola
```

For detailed installation instructions, troubleshooting, and alternative methods, see the [Installation Guide](INSTALLATION.md).

## Quick Start

```python
from xai_cola import COLA
from xai_cola.data import COLAData
from xai_cola.models import Model
from counterfactual_explainer import DiCE
import joblib

# 1. Initialize data interface
from datasets.german_credit import GermanCreditDataset
dataset = GermanCreditDataset()
df = dataset.get_dataframe()

# Pick samples you want to explain
df_Risk_1 = df[df['Risk'] == 1].sample(5)

data = COLAData(
    factual_data=df_Risk_1,
    label_column='Risk',
    transform=None  # or 'ohe-zscore', 'ohe-min-max' for automatic preprocessing
)

# 2. Initialize model interface
lgbmcClassifier = joblib.load('lgbm_GremanCredit.pkl')
ml_model = Model(model=lgbmcClassifier, backend="sklearn")

# 3. Generate counterfactuals
explainer = DiCE(ml_model=ml_model)
factual, counterfactual = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    total_cfs=1,
    features_to_keep=['Age','Sex']
)

# 4. Add counterfactuals and refine with COLA
data.add_counterfactuals(counterfactual, with_target_column=True)

refiner = COLA(
    data=data,
    ml_model=ml_model
)

refiner.set_policy(
    matcher="ect",
    attributor="pshap",
    Avalues_method="max"
)

# Get refined counterfactuals (all features can be modified)
factual, ce, ace = refiner.get_all_results(limited_actions=10)

# Or restrict to specific features only
factual, ce, ace = refiner.get_all_results(
    limited_actions=10,
    features_to_vary=['Age', 'Credit amount', 'Duration']  # Only modify these features
)

# 5. Visualize results
refine_factual, refine_ce, refine_ace = refiner.highlight_changes_final()

# Binary heatmap (shows which features changed)
refiner.heatmap_binary(save_path='./results', save_mode='combined')

# Directional heatmap (shows if features increased or decreased)
refiner.heatmap_direction(save_path='./results', save_mode='combined')

# Stacked bar chart (shows percentage of modifications)
refiner.stacked_bar_chart(save_path='./results')

# Diversity analysis (shows minimal feature combinations)
factual_df, diversity_styles = refiner.diversity()
for i, style in enumerate(diversity_styles):
    print(f"Instance {i+1} diversity:")
    display(style)
```

## Features

- **Multiple Counterfactual Explainers**: Support for DiCE, DisCount, Alibi-CFI, ARecourseS, and WachterCF
- **Flexible Matching**: Optimal Transport (OT), Exact Matching (ECT), Nearest Neighbor (NN), and Coarsened Exact Matching (CEM)
- **Feature Attribution**: PSHAP for Shapley values with joint probability
- **Feature Selection**: Control which features can be modified using `features_to_vary` parameter
- **Rich Visualizations**:
  - Highlighted DataFrames showing changes (comparison and final formats)
  - Binary heatmaps showing which features changed
  - Directional heatmaps showing increase/decrease patterns
  - Stacked bar charts comparing modification efficiency
  - Diversity analysis showing minimal feature combinations
- **Data Interfaces**: Support for Pandas and NumPy data formats with automatic transformation support
- **Model Support**: Works with scikit-learn and PyTorch models
- **Built-in Datasets**: GermanCredit, Compas, HELOC, HotelBookings

## Documentation

### Core Documentation

- **[Installation Guide](INSTALLATION.md)** - Detailed installation instructions and troubleshooting
- **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[FAQ](FAQ.md)** - Frequently asked questions

### Specialized Guides

- **[Data Interface Guide](docs/DATA_INTERFACE_QUICKREF.md)** - COLAData usage with transformation support
- **[WachterCF Usage](docs/WACHTERCF_USAGE.md)** - Detailed guide for WachterCF explainer
- **[Architecture Overview](ARCHITECTURE.md)** - System design and implementation details (中文)

### For Contributors

- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to COLA
- **[Project Structure](PROJECT_STRUCTURE.md)** - Codebase organization
- **[Release Guide](RELEASE_GUIDE.md)** - Release process for maintainers
- **[Changelog](CHANGELOG.md)** - Version history and updates

## Citation

If you use COLA in your research, please cite:

```bibtex
@article{you2024refining,
  title={Refining Counterfactual Explanations With Joint-Distribution-Informed Shapley Towards Actionable Minimality},
  author={You, Lei and Bian, Yijun and Cao, Lele},
  journal={arXiv preprint arXiv:2410.05419},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

- Lin Zhu (s232291@dtu.dk)
- Lei You (leiyo@dtu.dk)
