# COLA API Reference

Complete API documentation for COLA package.

**Note:** All code examples use the new module structure:
- `from xai_cola.ce_sparsifier import COLA`
- `from xai_cola.ce_generator import DiCE, DisCount, WachterCF`

## Core Modules

### COLA Class

Main class for refining counterfactual explanations.

```python
class COLA:
    def __init__(
        self, 
        data: BaseData,
        ml_model: Model,
        x_factual: np.ndarray, 
        x_counterfactual: np.ndarray,
    ):
        """
        Initialize COLA refiner.
        
        Parameters:
        -----------
        data : BaseData
            Data interface wrapper
        ml_model : Model
            Trained ML model interface
        x_factual : np.ndarray
            Factual instances
        x_counterfactual : np.ndarray
            Counterfactual instances
        """
    
    def set_policy(
        self,
        matcher: str = "ot", 
        attributor: str = "pshap",
        Avalues_method: str = "max",
        prob_matrix = None
    ):
        """
        Set the refinement policy.
        
        Parameters:
        -----------
        matcher : str
            Matching strategy: "ot", "ect", "nn", "cem"
        attributor : str
            Attribution method: "pshap", "randomshap"
        Avalues_method : str
            Method for computing A-values: "max"
        """
    
    def get_refined_counterfactual(self, limited_actions: int, features_to_vary: Optional[List[str]] = None):
        """
        Get counterfactuals with limited actions.
        
        Parameters:
        -----------
        limited_actions : int
            Number of feature changes to limit
        features_to_vary : List[str], optional
            List of feature names that are allowed to be modified.
            If None, all features can be modified (default).
            If specified, only these features will be considered for modification.
        
        Returns:
        --------
        pd.DataFrame
            Refined counterfactual DataFrame with target column
        
        Raises:
        -------
        ValueError
            If any feature name in features_to_vary is not a valid feature column name.
        """
    
    def get_all_results(self, limited_actions: int, features_to_vary: Optional[List[str]] = None):
        """
        Get all results: factual, counterfactual, and refined counterfactual.
        
        Parameters:
        -----------
        limited_actions : int
            Number of feature changes to limit
        features_to_vary : List[str], optional
            List of feature names that are allowed to be modified.
            If None, all features can be modified (default).
            If specified, only these features will be considered for modification.
        
        Returns:
        --------
        tuple
            (factual_dataframe, ce_dataframe, ace_dataframe)
            All are pd.DataFrame with target column
        
        Raises:
        -------
        ValueError
            If any feature name in features_to_vary is not a valid feature column name.
        """
    
    def highlight_changes_comparison(self):
        """
        Highlight changes with comparison format (old -> new).
        
        This method displays changes in the format "factual_value -> counterfactual_value"
        to show both the original and modified values side by side.
        
        Returns:
        --------
        tuple
            (factual_df, ce_style, ace_style)
            - factual_df: pandas.DataFrame - Original factual data
            - ce_style: pandas.io.formats.style.Styler - Styled DataFrame showing factual → full counterfactual
            - ace_style: pandas.io.formats.style.Styler - Styled DataFrame showing factual → action-limited counterfactual
        
        Raises:
        -------
        ValueError
            If refined counterfactuals have not been generated yet. 
            Must call get_refined_counterfactual() or get_all_results() method first.
        """
    
    def highlight_changes_final(self):
        """
        Highlight changes showing only the final values.
        
        This method displays only the final counterfactual values without showing
        the "factual -> counterfactual" format, making it cleaner for presentation.
        
        Returns:
        --------
        tuple
            (factual_df, ce_style, ace_style)
            - factual_df: pandas.DataFrame - Original factual data
            - ce_style: pandas.io.formats.style.Styler - Styled DataFrame showing only full counterfactual values
            - ace_style: pandas.io.formats.style.Styler - Styled DataFrame showing only action-limited counterfactual values
        
        Raises:
        -------
        ValueError
            If refined counterfactuals have not been generated yet. 
            Must call get_refined_counterfactual() or get_all_results() method first.
        """
    
    def heatmap_binary(self, save_path: Optional[str] = None, save_mode: str = "combined", show_axis_labels: bool = True):
        """
        Generate binary change heatmap visualizations (shows if value changed or not).
        
        This method generates heatmaps showing binary changes: whether a value 
        changed (red) or remained unchanged (lightgrey). For the target column, 
        changed values are shown in dark blue (#000080).
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the heatmap images. 
            If None, plots are automatically displayed in Jupyter (default).
            If provided, saves to the specified path and closes the plots.
            Can be a directory path or file path.
            
        save_mode : str, default="combined"
            How to save the heatmaps when save_path is provided:
            - "combined": Save both heatmaps in a single combined image (top and bottom)
            - "separate": Save two separate image files
            - Ignored if save_path is None
            
        show_axis_labels : bool, default=True
            Whether to show x and y axis labels (column names and row indices).
            If True, displays column names and row indices. If False, hides them.
            
        Returns:
        --------
        tuple
            (plot1, plot2) - Heatmap plots (matplotlib Figure objects)
        
        Color Scheme:
        -------------
        - Changed features: Red
        - Unchanged cells: Light grey
        - Target column (changed): Dark blue (#000080)
        - Target column (unchanged): Light grey
        
        Raises:
        -------
        ValueError
            If refined counterfactuals have not been generated yet. 
            Must call get_refined_counterfactual() or get_all_results() method first.
        """
    
    def heatmap_direction(self, save_path: Optional[str] = None, save_mode: str = "combined", show_axis_labels: bool = True):
        """
        Generate directional change heatmap visualizations (shows if value increased, decreased, or unchanged).
        
        This method generates heatmaps showing the direction of changes:
        - Value increased: Teal-green (#009E73)
        - Value decreased: Red-orange (#D55E00)
        - Value unchanged: Light grey
        - Target column (changed): Dark blue (#000080)
        - Target column (unchanged): Light grey
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the heatmap images. 
            If None, plots are automatically displayed in Jupyter (default).
            If provided, saves to the specified path and closes the plots.
            Can be a directory path or file path.
            
        save_mode : str, default="combined"
            How to save the heatmaps when save_path is provided:
            - "combined": Save both heatmaps in a single combined image (top and bottom)
            - "separate": Save two separate image files
            - Ignored if save_path is None
            
        show_axis_labels : bool, default=True
            Whether to show x and y axis labels (column names and row indices).
            If True, displays column names and row indices. If False, hides them.
            
        Returns:
        --------
        tuple
            (plot1, plot2) - Heatmap plots (matplotlib Figure objects)
        
        Color Scheme:
        -------------
        - Increased values: Teal-green (#009E73)
        - Decreased values: Red-orange (#D55E00)
        - Unchanged cells: Light grey
        - Target column (changed): Dark blue (#000080)
        - Target column (unchanged): Light grey
        
        Notes:
        ------
        - Directional comparison is only applied to numeric features
        - Non-numeric features that changed will appear as unchanged (grey)
        
        Raises:
        -------
        ValueError
            If refined counterfactuals have not been generated yet. 
            Must call get_refined_counterfactual() or get_all_results() method first.
        """
    
    def query_minimum_actions(self, features_to_vary: Optional[List[str]] = None):
        """
        Find minimum actions needed for desired outcome.

        Parameters:
        -----------
        features_to_vary : List[str], optional
            List of feature names that are allowed to be modified.
            If None, all features can be modified (default).
            If specified, only these features will be considered for modification.

        Returns:
        --------
        int
            Minimum number of actions required

        Raises:
        -------
        ValueError
            If any feature name in features_to_vary is not a valid feature column name.
        """

    def stacked_bar_chart(self, save_path: Optional[str] = None, refined_color: str = '#D9F2D0', counterfactual_color: str = '#FBE3D6', instance_labels: Optional[List[str]] = None):
        """
        Generate a horizontal stacked percentage bar chart comparing modification positions.

        This method creates a percentage-based stacked bar chart where each bar represents
        an instance (100% total), showing the proportion of modified positions in refined
        counterfactual vs. original counterfactual relative to factual data.

        Each bar shows:
        - Green segment (#D9F2D0): percentage of positions modified by refined_counterfactual
        - Orange segment (#FBE3D6): percentage of additional positions modified only by counterfactual
        - Total bar length: 100% (representing all counterfactual modifications)

        Labels on bars show both percentage and actual count (e.g., "60.0% (3)")

        Parameters:
        -----------
        save_path : str, optional
            Path to save the chart image. If None, chart is not saved.
            Can be a directory path or file path.
        refined_color : str, default='#D9F2D0'
            Color for refined counterfactual modified positions (light green)
        counterfactual_color : str, default='#FBE3D6'
            Color for counterfactual modified positions (light pink/orange)
        instance_labels : list, optional
            Custom labels for instances. If None, uses "instance 1", "instance 2", etc.
            Length must match the number of instances.

        Returns:
        --------
        matplotlib.figure.Figure
            The stacked bar chart figure

        Raises:
        -------
        ValueError
            If refined counterfactuals have not been generated yet.
            Must call get_refined_counterfactual() or get_all_results() method first.
        """

    def diversity(self):
        """
        Generate diversity analysis showing minimal feature combinations that can flip the target.

        This method finds all minimal feature combinations that can change the prediction
        from the factual's target value to the refined counterfactual's target value (e.g., from 1 to 0).
        For each instance, it returns styled DataFrames showing the different minimal combinations.

        The algorithm:
        1. For each instance, identify features that differ between factual and refined counterfactual
        2. Test combinations of increasing size (1 feature, 2 features, etc.)
        3. Find minimal sets that flip the prediction from factual target to refined counterfactual target
        4. Skip larger combinations that contain successful smaller combinations

        Visualization:
        - First row: Shows the complete refined counterfactual (all changes) with yellow background
        - Subsequent rows: Each row shows one minimal combination with #FFFFCC background
        - All changed features and target column have 1px black borders

        Returns:
        --------
        Tuple[pd.DataFrame, List[Styler]]
            - factual_df: Original factual data (copy)
            - diversity_styles: List of styled DataFrames (one per instance),
              each showing all minimal feature combinations for that instance

        Raises:
        -------
        ValueError
            If refined counterfactuals have not been generated yet.
            Must call get_refined_counterfactual() or get_all_results() first.
        """
```

**Example Usage:**

```python
from xai_cola import COLA
from xai_cola.ce_sparsifier.data import COLAData
from xai_cola.ce_sparsifier.models import Model

# Initialize COLA
cola = COLA(data=data, ml_model=model)
cola.set_policy(matcher='ect', attributor='pshap')

# Get refined counterfactuals (all features can be modified)
refined_cf = cola.get_refined_counterfactual(limited_actions=10)

# Or restrict to specific features only
refined_cf = cola.get_refined_counterfactual(
    limited_actions=10,
    features_to_vary=['Age', 'Credit amount', 'Duration']  # Only modify these features
)

# Get all results with feature restriction
factual_df, ce_df, ace_df = cola.get_all_results(
    limited_actions=10,
    features_to_vary=['Age', 'Credit amount']  # Only these features will be modified
)

# Visualize results
# Option 1: Highlight changes (for small datasets)
factual_df, ce_style, ace_style = cola.highlight_changes_comparison()
display(ce_style)  # Show in Jupyter

# Option 2: Binary heatmap (shows changed/unchanged)
plot1, plot2 = cola.heatmap_binary(
    save_path='./results',  # Optional: save to file
    save_mode='combined',   # 'combined' or 'separate'
    show_axis_labels=True   # Show column names and row indices
)

# Option 3: Directional heatmap (shows increase/decrease)
plot1, plot2 = cola.heatmap_direction(
    save_path='./results',
    save_mode='combined',
    show_axis_labels=True
)

# Option 4: Stacked bar chart (shows percentage of modifications)
fig = cola.stacked_bar_chart(
    save_path='./results',              # Optional: save to file
    refined_color='#D9F2D0',            # Light green for refined CF
    counterfactual_color='#FBE3D6',     # Light orange for original CF
    instance_labels=['Sample 1', 'Sample 2']  # Optional custom labels
)

# Option 5: Diversity analysis (shows minimal feature combinations)
factual_df, diversity_styles = cola.diversity()
# Display results for each instance
for i, style in enumerate(diversity_styles):
    print(f"Instance {i+1} diversity:")
    display(style)
```

## Data Interfaces

### COLAData

```python
class COLAData:
    def __init__(
        self,
        factual_data: Union[pd.DataFrame, np.ndarray],
        label_column: str,
        counterfactual_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        column_names: Optional[List[str]] = None,
        transform: Optional[str] = None,
        numerical_features: Optional[List[str]] = None
    ):
        """
        Initialize COLAData wrapper.
        
        Parameters:
        -----------
        factual_data : Union[pd.DataFrame, np.ndarray]
            Factual data, must contain label column
        label_column : str
            Label column name
        counterfactual_data : Optional[Union[pd.DataFrame, np.ndarray]]
            Counterfactual data (optional)
        column_names : Optional[List[str]]
            Column names (required if factual_data is numpy array)
        transform : Optional[str], default=None
            Data preprocessing method:
            - "ohe-zscore": categorical features one-hot encoded, numerical features standardized (z-score)
            - "ohe-min-max": categorical features one-hot encoded, numerical features normalized (min-max)
            - None: no preprocessing (data is already preprocessed)
        numerical_features : Optional[List[str]], default=None
            List of numerical feature names. If None, all features are treated as numerical.
            Other features are automatically inferred as categorical.
        
        Methods:
        --------
        - get_factual_all() -> pd.DataFrame: Get complete factual data with target column
        - get_factual_features() -> pd.DataFrame: Get factual features only
        - add_counterfactuals(counterfactual_data, with_target_column=True): Add counterfactual data
        - get_counterfactual_all() -> pd.DataFrame: Get complete counterfactual data
        - _transform(data) -> pd.DataFrame: Transform data using configured transformers
        - _inverse_transform(data) -> pd.DataFrame: Inverse transform data back to original format
        """
```

**Example Usage:**

```python
from xai_cola.data import COLAData
import pandas as pd

# With transformation
data = COLAData(
    factual_data=df,
    label_column='Risk',
    transform='ohe-zscore',
    numerical_features=['Age', 'Credit amount', 'Duration']
)

# Without transformation (data already preprocessed)
data = COLAData(
    factual_data=df,
    label_column='Risk',
    transform=None
)

# Add counterfactuals
factual_df, counterfactual_df = explainer.generate_counterfactuals(data=data)
data.add_counterfactuals(counterfactual_df, with_target_column=True)
```

### PandasData (Legacy)

```python
class PandasData(BaseData):
    def __init__(self, df: pd.DataFrame, target_name: str):
        """
        Initialize with pandas DataFrame (legacy interface).
        
        Note: COLAData is the recommended interface for new code.
        """
```

### NumpyData (Legacy)

```python
class NumpyData(BaseData):
    def __init__(
        self, 
        X: np.ndarray, 
        feature_names: List[str], 
        target_name: str
    ):
        """
        Initialize with numpy array (legacy interface).
        
        Note: COLAData is the recommended interface for new code.
        """
```

## Model Interfaces

### Model

```python
class Model(BaseModel):
    def __init__(self, model, backend: str = "sklearn"):
        """
        Initialize model interface.
        
        Parameters:
        -----------
        model : Any
            Trained model (sklearn, pytorch, etc.)
        backend : str
            Backend type: "sklearn" or "pytorch"
        """
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
```

## Counterfactual Explainers

### DiCE

```python
class DiCE(CounterFactualExplainer):
    def generate_counterfactuals(
        self,
        data: COLAData = None,
        factual_class: int = 1,
        total_cfs: int = 1,
        features_to_keep: List[str] = None,
        continuous_features: List[str] = None
    ) -> tuple:
        """
        Generate counterfactuals using DiCE.
        
        Parameters:
        -----------
        data : COLAData, optional
            Factual data wrapper
        factual_class : int, default=1
            The class of the factual data. Counterfactuals will target (1 - factual_class)
        total_cfs : int, default=1
            Total number of counterfactuals required (of each query_instance)
        features_to_keep : list, optional
            List of features to keep unchanged in the counterfactuals
        continuous_features : list, optional
            List of continuous feature names for dice_ml.Data
            If None, will use all features as continuous features
            Categorical features are automatically inferred as all features minus continuous_features
        
        Returns:
        --------
        tuple: (factual_df, counterfactual_df)
            factual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
            counterfactual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
                Target column values for counterfactual are set to (1 - factual_class)
        
        Notes:
        ------
        - Supports data transformation via COLAData.transform parameter
        - Automatically handles inverse transformation of counterfactuals
        """
```

### DisCount

```python
class DisCount(CounterFactualExplainer):
    def __init__(
        self,
        ml_model: Model,
        data: COLAData = None
    ):
        """
        Initialize DisCount explainer.
        
        Parameters:
        -----------
        ml_model : Model
            Pre-trained model wrapped in Model interface
            Supports both PyTorch (backend='pytorch') and sklearn (backend='sklearn') models
        data : COLAData, optional
            Data wrapper containing factual data
        """
    
    def generate_counterfactuals(
        self,
        data: COLAData = None,
        factual_class: int = 1,
        lr: float = 1e-1,
        n_proj: int = 10,
        delta: float = 0.05,
        U_1: float = 0.4,
        U_2: float = 0.3,
        l: float = 0.2,
        r: float = 1,
        max_iter: int = 15,
        tau: float = 1e2,
        silent: bool = False,
    ) -> tuple:
        """
        Generate counterfactuals using DisCount algorithm.
        
        Parameters:
        -----------
        data : COLAData, optional
            Factual data wrapper
        factual_class : int, default=1
            The class of the factual data. Counterfactuals will target (1 - factual_class)
        lr : float, default=1e-1
            Learning rate
        n_proj : int, default=10
            Number of projections
        delta : float, default=0.05
            Trimming constant
        U_1 : float, default=0.4
            Upper bound for the Wasserstein distance
        U_2 : float, default=0.3
            Upper bound for the sliced Wasserstein distance
        l : float, default=0.2
            Lower bound for the interval narrowing
        r : float, default=1
            Upper bound for the interval narrowing
        max_iter : int, default=15
            Maximum number of iterations
        tau : float, default=1e2
            Step size (can't be too large or too small)
        silent : bool, default=False
            Whether to print the log information
        
        Returns:
        --------
        tuple: (factual_df, counterfactual_df)
            factual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
            counterfactual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
                Target column values for counterfactual are set to (1 - factual_class)
        
        Notes:
        ------
        - Supports both PyTorch (backend='pytorch') and sklearn (backend='sklearn') models
        - For PyTorch models, automatic gradients are used
        - For sklearn models, numerical gradients (finite differences) are used
        - Supports data transformation via COLAData.transform parameter
        - Automatically handles inverse transformation of counterfactuals
        """
```

**Example Usage:**

```python
from counterfactual_explainer import DisCount
from xai_cola.models import Model
from xai_cola.data import COLAData
import torch.nn as nn

# PyTorch model
pytorch_model = nn.Sequential(...)  # Your PyTorch model
model = Model(pytorch_model, backend='pytorch')
# Or sklearn model:
# from sklearn.ensemble import RandomForestClassifier
# sklearn_model = RandomForestClassifier()
# model = Model(sklearn_model, backend='sklearn')

data = COLAData(factual_data=df, label_column='Risk')

explainer = DisCount(ml_model=model, data=data)
factual_df, counterfactual_df = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    max_iter=15,
    U_1=0.4,
    U_2=0.3
)
```

### AlibiCounterfactualInstances

```python
class AlibiCounterfactualInstances(CounterFactualExplainer):
    def generate_counterfactuals(
        self,
        data: BaseData,
        feature_range: tuple = (-1e10, 1e10),
        max_iter: int = 8000
    ) -> np.ndarray:
        """Generate counterfactuals using Alibi-CFI."""
```

### WachterCF

```python
class WachterCF(CounterFactualExplainer):
    def __init__(
        self,
        ml_model: Model,
        data: COLAData = None
    ):
        """
        Initialize WachterCF explainer.
        
        Parameters:
        -----------
        ml_model : Model
            Pre-trained model wrapped in Model interface
            Supports 'pytorch' (with automatic gradients) and 'sklearn' (with numerical gradients)
        data : COLAData, optional
            Data wrapper containing factual data
        """
    
    def generate_counterfactuals(
        self,
        data: COLAData = None,
        factual_class: int = 1,
        features_to_vary: List[str] = None,
        target_proba: float = 0.7,
        feature_weights: List[float] = None,
        _lambda: float = 10.0,
        optimizer: str = "adam",
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> tuple:
        """
        Generate counterfactuals using Wachter's gradient-based method.
        
        Parameters:
        -----------
        data : COLAData, optional
            Factual data wrapper
        factual_class : int, default=1
            The class of the factual data. Counterfactuals will target (1 - factual_class)
        features_to_vary : list, optional
            List of feature names to vary. If None, all features can vary
        target_proba : float, default=0.7
            Target probability for the counterfactual class
        feature_weights : list, optional
            Weights for features in distance computation. If None, equal weights
        _lambda : float, default=10.0
            Weight for the prediction loss term (higher = more emphasis on reaching target)
        optimizer : str, default="adam"
            Optimizer to use: "adam" or "rmsprop"
        lr : float, default=0.01
            Learning rate for optimization
        max_iter : int, default=100
            Maximum number of optimization iterations
        
        Returns:
        --------
        tuple: (factual_df, counterfactual_df)
            factual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
            counterfactual_df : pd.DataFrame
                DataFrame with shape (n_samples, n_features + 1), includes target column
                Target column values for counterfactual are set to (1 - factual_class)
        
        Notes:
        ------
        - For PyTorch models: Uses automatic gradients (faster, more accurate)
        - For sklearn models: Uses numerical gradients via finite differences (slower but compatible)
        - Data normalization: Assumes input data is normalized to [0, 1] range
        """
```

**Example Usage:**

```python
from counterfactual_explainer import WachterCF
from xai_cola.models import Model
from xai_cola.data import COLAData

# Using PyTorch model (recommended for faster optimization)
import torch.nn as nn
pytorch_model = nn.Sequential(...)  # Your PyTorch model
model = Model(pytorch_model, backend='pytorch')
data = COLAData(factual_data=df, label_column='Risk')

explainer = WachterCF(ml_model=model, data=data)
factual_df, counterfactual_df = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    features_to_vary=['Age', 'Credit amount'],
    target_proba=0.7,
    _lambda=10.0,
    max_iter=100
)

# Using sklearn model (uses numerical gradients)
from sklearn.ensemble import RandomForestClassifier
sklearn_model = RandomForestClassifier()  # Your sklearn model
model = Model(sklearn_model, backend='sklearn')
data = COLAData(factual_data=df, label_column='Risk')

explainer = WachterCF(ml_model=model, data=data)
factual_df, counterfactual_df = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    target_proba=0.7,
    max_iter=100
)
```

## Policy Modules

### Matching Policies

Located in `xai_cola.cola_policy.matching`:

- `CounterfactualOptimalTransportPolicy`
- `CounterfactualExactMatchingPolicy`
- `CounterfactualNearestNeighborMatchingPolicy`
- `CounterfactualCoarsenedExactMatchingOTPolicy`

### Feature Attributors

Located in `xai_cola.cola_policy.feature_attributor`:

- `PSHAP`: Shapley values with joint probability
- `Attributor`: Base attributor class

### Data Composers

Located in `xai_cola.cola_policy.data_composer`:

- `DataComposer`: Compose counterfactuals from matched instances

## Visualization Functions

Located in `xai_cola.ce_sparsifier.visualization`:

### Pure Visualization Functions

The visualization module provides pure functions (no class state) for creating heatmaps:

#### Binary Change Heatmap

```python
from xai_cola.ce_sparsifier.visualization import generate_binary_heatmap

plot1, plot2 = generate_binary_heatmap(
    factual_df=factual_df,
    counterfactual_df=counterfactual_df,
    refined_counterfactual_df=refined_counterfactual_df,
    label_column='Risk',
    save_path='./results',      # Optional
    save_mode='combined',       # 'combined' or 'separate'
    show_axis_labels=True       # Show axis labels
)
```

**Color Scheme:**
- Changed features: Red
- Unchanged cells: Light grey
- Target column (changed): Dark blue (#000080)

#### Directional Change Heatmap

```python
from xai_cola.ce_sparsifier.visualization import generate_direction_heatmap

plot1, plot2 = generate_direction_heatmap(
    factual_df=factual_df,
    counterfactual_df=counterfactual_df,
    refined_counterfactual_df=refined_counterfactual_df,
    label_column='Risk',
    save_path='./results',
    save_mode='combined',
    show_axis_labels=True
)
```

**Color Scheme:**
- Increased values: Teal-green (#009E73)
- Decreased values: Red-orange (#D55E00)
- Unchanged cells: Light grey
- Target column (changed): Dark blue (#000080)

**Notes:**
- Combined mode creates two heatmaps stacked vertically (not side by side)
- Directional comparison is only applied to numeric features
- Both heatmap types use the same target column color (#000080) with alpha=0.6 for consistency

#### Stacked Bar Chart

```python
from xai_cola.ce_sparsifier.visualization import generate_stacked_bar_chart

fig = generate_stacked_bar_chart(
    factual_df=factual_df,
    counterfactual_df=counterfactual_df,
    refined_counterfactual_df=refined_counterfactual_df,
    label_column='Risk',
    save_path='./results',              # Optional
    refined_color='#D9F2D0',            # Light green
    counterfactual_color='#FBE3D6',     # Light orange
    instance_labels=['Sample 1', 'Sample 2']  # Optional
)
```

**Visualization:**
- Each horizontal bar represents 100% of all counterfactual modifications for one instance
- Green segment: percentage modified by refined counterfactual
- Orange segment: percentage of additional modifications only in original counterfactual
- Labels show both percentage and count (e.g., "60.0% (3)")

**Use Cases:**
- Compare efficiency of refined vs. original counterfactuals across multiple instances
- Visualize the sparsification achieved by COLA
- Identify which instances achieved the most reduction in modifications

#### Diversity Analysis

```python
from xai_cola.ce_sparsifier.visualization import generate_diversity_for_all_instances

factual_df, diversity_styles = generate_diversity_for_all_instances(
    factual_df=factual_df,
    refined_counterfactual_df=refined_counterfactual_df,
    ml_model=ml_model,
    label_column='Risk'
)

# Display results for each instance
for i, style in enumerate(diversity_styles):
    print(f"Instance {i+1} diversity:")
    display(style)
```

**Algorithm:**
1. For each instance, finds features that differ between factual and refined counterfactual
2. Enumerates all combinations of changed features in increasing size order (1, 2, 3 features...)
3. Tests each combination to see if it flips the prediction from factual to refined counterfactual target
4. Returns only minimal combinations (skips larger sets that contain successful smaller sets)

**Visualization:**
- First row: Complete refined counterfactual with yellow background (#FFFF00)
- Subsequent rows: Each minimal combination with lighter yellow background (#FFFFCC)
- Changed features: 1px black border
- Target column: Lightgray background with 1px black border

**Use Cases:**
- Understand which features are critical for achieving the desired outcome
- Provide users with multiple actionable options (diversity)
- Identify redundant feature changes
- Support decision-making with alternative paths to the same goal

**Performance:**
- Time complexity: O(k × 2^n × T_predict) where k = instances, n = changed features, T_predict ≈ 10ms
- For typical cases (n ≤ 10): Completes in seconds
- Pruning optimization reduces actual combinations tested

## Complete Example

See `QUICKSTART.md` for complete working examples.

## Additional Documentation

- [WachterCF Usage Guide](docs/WACHTERCF_USAGE.md) - Detailed guide for WachterCF explainer
- [Data Interface Quick Reference](docs/DATA_INTERFACE_QUICKREF.md) - COLAData usage guide
- [Data Transformation Guide](docs/NEW_DATA_INTERFACE.md) - Data preprocessing and transformation

