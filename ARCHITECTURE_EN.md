# COLA Software Architecture

## 📐 Architecture Overview

COLA adopts a **modular layered architecture** with clear separation of concerns, achieving **high cohesion and low coupling** design.

```
┌─────────────────────────────────────────────────────────┐
│                  User Layer / API Surface                │
│              from xai_cola import COLA                   │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   Core Layer (Orchestration)             │
│  ┌──────────────────────────────────────────────────┐   │
│  │  COLA Class - Main Orchestrator                   │   │
│  │  Responsibility: Coordinate all modules for        │   │
│  │  counterfactual refinement                        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        │             │             │             │
┌───────▼──────┐ ┌───▼──────┐ ┌────▼──────┐ ┌────▼─────────┐
│  Data Layer  │ │  Model   │ │  Policy   │ │Visualization │
│              │ │  Layer   │ │  Layer    │ │    Layer     │
└──────────────┘ └──────────┘ └───────────┘ └──────────────┘
     │                │             │
     └────────────────┴─────────────┘
          Dependency Flow
```

## 🧩 Module Components

### 1. **Core Module**

#### `cola.py` - Main COLA Class (Orchestrator)

**Responsibility:** Top-level coordinator that integrates all modules to accomplish counterfactual refinement

**Characteristics:**
- 🎯 **High Cohesion**: All refinement logic centralized in one class
- 🔗 **Low Coupling**: Interacts with other modules through interfaces (BaseData, Model)
- 🎛️ **Strategy Pattern**: Supports dynamic switching between matching and attribution strategies

**Key Methods:**
```python
class COLA:
    def __init__(data, ml_model, x_factual, x_counterfactual)  # Initialize
    def set_policy(matcher, attributor, Avalues_method)         # Configure policy
    def get_all_results(limited_actions, features_to_vary)     # Get refined CEs
    def highlight_changes_final()                               # Highlight changes
    def heatmap_binary()                                        # Binary heatmap
    def heatmap_direction()                                     # Directional heatmap
    def stacked_bar_chart()                                     # Comparison chart
    def diversity()                                             # Diversity analysis
```

**Workflow:**
1. Accept data through `COLAData` interface
2. Use `ml_model` for predictions
3. Execute `policies` for matching, attribution, and composition
4. Use `visualization` to display results

---

### 2. **Data Module**

#### Structure:
```
data/
└── coladata.py      # COLAData unified interface
```

#### `COLAData` - Unified Data Interface

**Responsibility:** Unified data access and preprocessing for counterfactual refinement

**Design Patterns:**
- ✅ **Adapter Pattern** - Unifies Pandas and NumPy interfaces
- ✅ **Facade Pattern** - Simplifies data preprocessing
- 🎯 **High Cohesion**: Data operations centralized
- 🔗 **Low Coupling**: Independent of model and policy layers

**Core Methods:**
```python
class COLAData:
    def __init__(factual_data, label_column, transform)
    def add_counterfactuals(cf_data, with_target_column)
    def get_transformed_data()
    def get_inverse_transformed_data(data)
    def get_feature_names()
    def get_factual_df()
    def get_counterfactual_df()
```

**Key Features:**
- **Automatic Preprocessing**: One-hot encoding, z-score normalization, min-max scaling
- **Inverse Transformation**: Convert results back to original scale
- **Flexible Input**: Accepts both Pandas DataFrames and NumPy arrays
- **Metadata Management**: Tracks feature names, types, and transformations

**Works Independently:**
- Data format conversion (Pandas ↔ NumPy)
- Feature name management
- Preprocessing and inverse transformation
- Data validation

**Collaborates With:**
- Provides uniform feature names and target info to COLA
- Supplies data access interface to policies
- Handles transformation for model predictions

---

### 3. **Model Module**

#### Structure:
```
models/
├── base.py          # BaseModel abstract interface
├── factory.py       # Model factory for auto-detection
├── sklearn.py       # Scikit-learn adapter
├── pytorch.py       # PyTorch adapter
├── tensorflow1.py   # TensorFlow 1.x adapter
└── tensorflow2.py   # TensorFlow 2.x adapter
```

#### `BaseModel` - Abstract Interface

**Responsibility:** Define unified model interaction interface

**Design Patterns:**
- ✅ **Factory Pattern** - Automatic adapter selection
- ✅ **Adapter Pattern** - Uniform interface for different frameworks
- 🎯 **High Cohesion**: Model operations encapsulated
- 🔗 **Low Coupling**: Framework-agnostic interface

**Core Methods:**
```python
class BaseModel(ABC):
    def predict(X)            # Get predictions
    def predict_proba(X)      # Get probability distributions
    def get_backend()         # Get framework type
```

**Supported Frameworks:**
- **scikit-learn**: Standard sklearn interface
- **PyTorch**: Handles `torch.Tensor` and device management
- **TensorFlow 1.x**: Session-based execution
- **TensorFlow 2.x**: Eager execution

**Factory Method:**
```python
# Automatically detects model type
model = Model(model=trained_model, backend="sklearn")
```

**Works Independently:**
- Model prediction
- Probability distribution computation
- Backend-specific handling

**Collaborates With:**
- Provides predictions to COLA for validation
- Supplies predictions to attributors (PSHAP) for importance computation

---

### 4. **Policy Module (Strategy Pattern)**

#### Structure:
```
policies/
├── matching/              # Matching strategies
│   ├── base_matcher.py
│   ├── ot_matcher.py      # Optimal Transport
│   ├── ect_matcher.py     # Exact Matching
│   ├── nn_matcher.py      # Nearest Neighbor
│   └── cem_matcher.py     # Coarsened Exact Matching
│
├── feature_attributor/    # Feature attribution
│   ├── base_attributor.py
│   └── pshap.py           # PSHAP implementation
│
└── data_composer/         # Data composition
    └── data_composer.py
```

#### 4.1 **Matching Policies**

**Responsibility:** Establish correspondence between factual and counterfactual instances

**Design Pattern:** Strategy Pattern - Pluggable matching algorithms

**Available Matchers:**

1. **Optimal Transport (OT)**
   - Uses Wasserstein distance
   - Best for group-based counterfactuals
   - Finds optimal matching minimizing transport cost

2. **Exact Matching (ECT)**
   - One-to-one direct matching
   - Best for DiCE and individual counterfactuals
   - **Recommended default**

3. **Nearest Neighbor (NN)**
   - Distance-based closest matching
   - Fast and simple
   - Good for quick experimentation

4. **Coarsened Exact Matching (CEM)**
   - Binning-based matching
   - Good for high-dimensional data
   - Reduces dimensionality through coarsening

**Interface:**
```python
class BaseMatcher(ABC):
    def match(factual, counterfactual) -> JointProbMatrix
```

#### 4.2 **Feature Attribution Policies**

**Responsibility:** Compute feature importance for refinement

**Available Attributors:**

1. **PSHAP** (Probability-weighted Shapley values)
   - Computes Shapley values with joint probability distribution
   - Considers feature interactions
   - Theoretically grounded in cooperative game theory

**Interface:**
```python
class BaseAttributor(ABC):
    def attribute(data, ml_model) -> FeatureImportanceScores
```

**PSHAP Workflow:**
1. Compute joint probability distribution
2. Calculate Shapley values for each feature
3. Weight by probability distribution
4. Return importance scores

#### 4.3 **Data Composition**

**Responsibility:** Synthesize refined counterfactuals from matched pairs and importance scores

**Methods:**
- `max`: Take maximum importance
- `mean`: Average importance
- `weighted`: Custom weighting

---

### 5. **Counterfactual Generator Module**

#### Structure:
```
ce_generator/
├── base_explainer.py      # Base class for all explainers
├── dice.py                # DiCE implementation
├── discount.py            # DisCount implementation
├── wachtercf.py           # WachterCF implementation
├── alibi_cfi.py           # Alibi-CFI wrapper
├── ares.py                # ARecourseS implementation
├── knn.py                 # KNN-based explainer
├── growingsphere.py       # Growing Sphere
└── auxiliary.py           # Helper functions
```

**Responsibility:** Generate initial counterfactual explanations

**Supported Explainers:**

1. **DiCE** - Diverse Counterfactual Explanations
   - Generates diverse set of counterfactuals
   - Uses genetic algorithm
   - Best for finding multiple alternatives

2. **DisCount** - Distributional Counterfactuals
   - Treats data as distributions
   - Finds counterfactual distributions
   - Best for group-level explanations

3. **WachterCF** - Gradient-based Optimization
   - Uses gradient descent
   - Requires differentiable models
   - Fast for neural networks

4. **Alibi-CFI** - Counterfactual Instances (Library Wrapper)
   - Wrapper for Alibi library
   - Prototyping and prototypical explanations

5. **ARecourseS** - Actionable Recourse
   - Focuses on actionable changes
   - Considers cost constraints

**Interface:**
```python
class CounterFactualExplainer(ABC):
    def generate_counterfactuals(
        data,
        factual_class,
        total_cfs,
        features_to_keep
    ) -> (factual, counterfactual)
```

---

### 6. **Visualization Module**

#### Structure:
```
visualization/
├── heatmap.py             # Heatmap visualizations
├── diversity.py           # Diversity analysis
└── highlight_dataframe.py # DataFrame highlighting
```

**Responsibility:** Visual representation of counterfactual refinement results

**Visualization Types:**

1. **Binary Heatmap**
   - Shows which features changed (0 = no change, 1 = changed)
   - Compare CE vs ACE side-by-side
   - Good for any dataset size

2. **Directional Heatmap**
   - Shows if features increased (+1), decreased (-1), or stayed same (0)
   - Reveals change patterns
   - Useful for understanding action directions

3. **Highlighted DataFrames**
   - Color-coded tables showing changes
   - Best for small datasets (<20 instances)
   - Two formats: comparison and final

4. **Stacked Bar Charts**
   - Percentage of features modified
   - Compare CE vs ACE efficiency
   - Aggregate view across all instances

5. **Diversity Analysis**
   - Shows minimal feature combinations
   - Highlights different ways to achieve same outcome
   - Helps understand action flexibility

---

## 🎨 Design Patterns Used

### 1. **Strategy Pattern** (Policies)
- **Location**: Matching and attribution policies
- **Purpose**: Pluggable algorithms
- **Benefit**: Easy to add new matchers/attributors without modifying COLA

### 2. **Factory Pattern** (Model Factory)
- **Location**: Model adapter creation
- **Purpose**: Automatic backend detection
- **Benefit**: Framework-agnostic user interface

### 3. **Adapter Pattern** (Model Adapters)
- **Location**: Model interface
- **Purpose**: Unified interface for different ML frameworks
- **Benefit**: Works with sklearn, PyTorch, TensorFlow seamlessly

### 4. **Facade Pattern** (COLA Class, COLAData)
- **Location**: Main COLA class, data interface
- **Purpose**: Simplify complex subsystems
- **Benefit**: Easy-to-use API hiding implementation complexity

### 5. **Template Method** (Base Classes)
- **Location**: BaseModel, BaseMatcher, BaseAttributor
- **Purpose**: Define algorithm skeleton
- **Benefit**: Consistent interface, extensible implementations

---

## 🔄 Data Flow

### Complete Workflow:

```
1. User Input
   ├─ Factual data → COLAData
   └─ ML model → Model adapter

2. CE Generation
   └─ CE Generator (DiCE/DisCount/etc.) → Initial counterfactuals

3. COLA Refinement
   ├─ set_policy() → Configure matching + attribution
   └─ get_all_results(limited_actions, features_to_vary)
       │
       ├─ Matcher → Create joint probability matrix
       ├─ Attributor (PSHAP) → Compute feature importance
       ├─ Composer → Synthesize refined counterfactual
       └─ Action Limiter → Apply constraints

4. Output
   ├─ factual: Original instances
   ├─ ce: Original counterfactuals
   └─ ace: Action-limited counterfactuals (refined)

5. Visualization
   ├─ highlight_changes_final() → Color-coded DataFrames
   ├─ heatmap_binary() → Binary change heatmap
   ├─ heatmap_direction() → Directional change heatmap
   ├─ stacked_bar_chart() → Efficiency comparison
   └─ diversity() → Minimal feature combinations
```

---

## 🏗️ Design Principles

### 1. **High Cohesion**
- Each module has focused responsibility
- Related functionality grouped together
- Minimal scattering of related code

### 2. **Low Coupling**
- Modules interact through well-defined interfaces
- Changes in one module don't ripple to others
- Easy to test modules independently

### 3. **Open/Closed Principle**
- Open for extension (add new matchers, attributors)
- Closed for modification (no need to change existing code)

### 4. **Dependency Inversion**
- Depend on abstractions (BaseModel, BaseMatcher)
- Not on concrete implementations
- Enables framework flexibility

### 5. **Single Responsibility**
- Each class has one reason to change
- Clear, focused purpose
- Easier to maintain and test

### 6. **Interface Segregation**
- Clients depend only on methods they use
- No fat interfaces forcing unnecessary dependencies

---

## 📦 Package Structure

```
xai_cola/
├── __init__.py                 # Main exports
├── version.py                  # Version info
│
├── ce_generator/               # CE Generators (10 files)
│   ├── __init__.py
│   ├── base_explainer.py       # Base class
│   ├── dice.py                 # DiCE
│   ├── discount.py             # DisCount
│   ├── wachtercf.py            # WachterCF
│   ├── alibi_cfi.py            # Alibi wrapper
│   ├── ares.py                 # ARecourseS
│   ├── knn.py                  # KNN explainer
│   ├── growingsphere.py        # Growing Sphere
│   └── auxiliary.py            # Helpers
│
└── ce_sparsifier/              # COLA Refinement (34 files)
    ├── __init__.py
    ├── cola.py                 # Main COLA class ⭐
    │
    ├── data/                   # Data interfaces
    │   ├── __init__.py
    │   └── coladata.py         # COLAData class
    │
    ├── models/                 # ML model adapters
    │   ├── __init__.py
    │   ├── base.py             # BaseModel
    │   ├── factory.py          # Model factory
    │   ├── sklearn.py          # sklearn adapter
    │   ├── pytorch.py          # PyTorch adapter
    │   ├── tensorflow1.py      # TF 1.x adapter
    │   └── tensorflow2.py      # TF 2.x adapter
    │
    ├── policies/               # Refinement policies
    │   ├── matching/           # Matching strategies
    │   │   ├── __init__.py
    │   │   ├── base_matcher.py
    │   │   ├── ot_matcher.py   # Optimal Transport
    │   │   ├── ect_matcher.py  # Exact Matching
    │   │   ├── nn_matcher.py   # Nearest Neighbor
    │   │   └── cem_matcher.py  # Coarsened Exact
    │   │
    │   ├── feature_attributor/ # Feature attribution
    │   │   ├── __init__.py
    │   │   ├── base_attributor.py
    │   │   └── pshap.py        # PSHAP implementation
    │   │
    │   └── data_composer/      # Data composition
    │       ├── __init__.py
    │       └── data_composer.py
    │
    ├── utils/                  # Utilities
    │   ├── __init__.py
    │   └── pipeline_utils.py
    │
    └── visualization/          # Visualizations
        ├── __init__.py
        ├── heatmap.py          # Heatmaps
        ├── diversity.py        # Diversity analysis
        └── highlight_dataframe.py  # DataFrame styling
```

---

## 🔧 Extensibility

### Adding a New Matcher

```python
from xai_cola.ce_sparsifier.policies.matching.base_matcher import BaseMatcher

class MyMatcher(BaseMatcher):
    def match(self, factual, counterfactual):
        # Your matching logic here
        joint_prob_matrix = compute_matching(factual, counterfactual)
        return joint_prob_matrix

# Usage
refiner.set_policy(matcher=MyMatcher(), attributor="pshap")
```

### Adding a New Attributor

```python
from xai_cola.ce_sparsifier.policies.feature_attributor.base_attributor import BaseAttributor

class MyAttributor(BaseAttributor):
    def attribute(self, data, ml_model):
        # Your attribution logic here
        importance_scores = compute_importance(data, ml_model)
        return importance_scores

# Usage
refiner.set_policy(matcher="ect", attributor=MyAttributor())
```

### Adding a New CE Generator

```python
from xai_cola.ce_generator.base_explainer import CounterFactualExplainer

class MyExplainer(CounterFactualExplainer):
    def generate_counterfactuals(self, data, factual_class, total_cfs, **kwargs):
        # Your CE generation logic
        factual, counterfactual = generate_ces(data, factual_class, total_cfs)
        return factual, counterfactual

# Usage
explainer = MyExplainer(ml_model=ml_model)
factual, cf = explainer.generate_counterfactuals(data, factual_class=1, total_cfs=1)
```

---

## 🎯 Architecture Strengths

1. **Modularity**: Clear module boundaries, easy to understand and maintain
2. **Extensibility**: Easy to add new matchers, attributors, explainers
3. **Flexibility**: Works with any ML framework, any CE method
4. **Testability**: Modules can be tested independently
5. **Reusability**: Components can be used in different contexts
6. **Maintainability**: Changes localized to specific modules
7. **Scalability**: Can handle small and large datasets with policy selection

---

## 📚 Key Abstractions

1. **COLAData**: Unified data representation
2. **Model**: Framework-agnostic model interface
3. **BaseMatcher**: Pluggable matching strategies
4. **BaseAttributor**: Pluggable attribution methods
5. **COLA**: Orchestrator coordinating all components

These abstractions enable COLA to be:
- **Model-agnostic**: Works with any ML model
- **CE-method-agnostic**: Works with any CE generator
- **Framework-agnostic**: Works with sklearn, PyTorch, TensorFlow
- **Flexible**: Configurable policies for different use cases

---

## 🔗 Related Documentation

- [API Reference](API_REFERENCE.md) - Detailed API documentation
- [Quick Start](QUICKSTART.md) - Getting started guide
- [Tutorials](docs/tutorials/) - Step-by-step tutorials
- [Contributing](CONTRIBUTING.md) - How to extend COLA

---

## Summary

COLA's architecture exemplifies modern software engineering principles:

- **Layered architecture** for clear separation of concerns
- **Design patterns** for extensibility and maintainability
- **Interface-based design** for flexibility and testability
- **Modular structure** for independent component development
- **Strategy pattern** for algorithm flexibility
- **Factory pattern** for automatic framework detection

This architecture makes COLA both powerful for researchers and easy to use for practitioners.
