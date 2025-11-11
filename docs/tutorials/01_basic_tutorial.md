# Tutorial 1: Basic COLA Workflow

## Learning Objectives

By the end of this tutorial, you will:
- Understand the complete COLA workflow
- Generate and refine counterfactual explanations
- Visualize the results
- Interpret the output

## Prerequisites

- COLA installed (`pip install xai-cola`)
- Basic Python knowledge
- Understanding of machine learning concepts

## The Problem

Imagine you're building a loan approval system. A customer was denied a loan, and you need to explain what they should change to get approved. Traditional counterfactual explainers might suggest changing 2-6 different things. COLA helps you identify the 1-2 most important changes.

## Step 1: Prepare Your Data

```python
import pandas as pd
from xai_cola.data import COLAData
from datasets.german_credit import GermanCreditDataset

# Load the German Credit dataset
dataset = GermanCreditDataset()
df = dataset.get_dataframe()

# Select instances that were denied (Risk = 1)
# In this dataset, Risk=1 means bad credit
denied_customers = df[df['Risk'] == 1].sample(5, random_state=42)

# Create the data interface
data = COLAData(
    factual_data=denied_customers,
    label_column='Risk',
    transform='ohe-zscore'  # One-hot encode categoricals and z-score normalize
)

print(f"Selected {len(denied_customers)} customers to explain")
print(denied_customers.columns)
```

**What's happening:**
- We load the German Credit dataset (built-in with COLA)
- Select 5 customers who were denied loans
- Wrap the data in `COLAData` for automatic preprocessing

## Step 2: Load Your ML Model

```python
import joblib
from xai_cola.models import Model

# Load pre-trained model (you can use your own model here)
classifier = joblib.load('lgbm_GremanCredit.pkl')

# Wrap it in COLA's model interface
ml_model = Model(model=classifier, backend="sklearn")

# Verify the model works
predictions = ml_model.predict(data.get_transformed_data())
print(f"Model predictions: {predictions}")
```

**What's happening:**
- Load a pre-trained LightGBM classifier
- Wrap it in COLA's `Model` interface so COLA can interact with it
- The model interface handles backend-specific details

## Step 3: Generate Initial Counterfactuals

```python
from xai_cola.ce_generator import DiCE

# Initialize the DiCE explainer
explainer = DiCE(ml_model=ml_model)

# Generate counterfactuals
# We want to flip predictions from 1 (bad) to 0 (good)
factual, counterfactual = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,           # Current class: denied
    total_cfs=1,               # Generate 1 counterfactual per factual
    features_to_keep=['Age', 'Sex']  # Don't change these features
)

print(f"Generated {len(counterfactual)} counterfactuals")
print(f"Factual shape: {factual.shape}")
print(f"Counterfactual shape: {counterfactual.shape}")
```

**What's happening:**
- DiCE generates counterfactual explanations
- Each denied customer gets a "what-if" scenario that would lead to approval
- We keep Age and Sex fixed (immutable features)

**Output example:**
```
Generated 5 counterfactuals
Factual shape: (5, 20)
Counterfactual shape: (5, 20)
```

## Step 4: Initialize COLA and Set Policy

```python
from xai_cola import COLA

# Add counterfactuals to the data
data.add_counterfactuals(counterfactual, with_target_column=True)

# Initialize COLA refiner
refiner = COLA(
    data=data,
    ml_model=ml_model
)

# Configure the refinement policy
refiner.set_policy(
    matcher="ect",           # Exact matching (best for DiCE)
    attributor="pshap",      # Use PSHAP for feature importance
    Avalues_method="max"     # How to aggregate importance scores
)

print("COLA initialized and ready to refine")
```

**What's happening:**
- We tell COLA about the counterfactuals
- Set up the refinement policy:
  - `matcher="ect"`: Match each factual to its counterfactual
  - `attributor="pshap"`: Use Shapley values with joint probability
  - `Avalues_method="max"`: Take maximum importance when aggregating

## Step 5: Refine Counterfactuals

```python
# Refine to use at most 3 feature changes
factual_refined, ce_refined, ace_refined = refiner.get_all_results(
    limited_actions=3
)

print(f"Original counterfactuals: {ce_refined.shape}")
print(f"Action-limited counterfactuals: {ace_refined.shape}")

# Check how many features actually changed
original_changes = (factual_refined != ce_refined).sum(axis=1)
refined_changes = (factual_refined != ace_refined).sum(axis=1)

print(f"\nOriginal CE - features changed: {original_changes.tolist()}")
print(f"Refined ACE - features changed: {refined_changes.tolist()}")
```

**What's happening:**
- COLA refines the counterfactuals to change at most 3 features
- We compare the original vs refined versions
- Refined counterfactuals require fewer changes

**Example output:**
```
Original CE - features changed: [8, 10, 7, 9, 11]
Refined ACE - features changed: [3, 3, 3, 3, 3]
```

## Step 6: Visualize Results

### Method 1: Highlighted DataFrames (Best for Small Datasets)

```python
# Get highlighted versions showing what changed
refine_factual, refine_ce, refine_ace = refiner.highlight_changes_final()

print("\n=== FACTUAL (Original Customer) ===")
display(refine_factual)

print("\n=== COUNTERFACTUAL (DiCE Suggestion) ===")
display(refine_ce)

print("\n=== ACTION-LIMITED COUNTERFACTUAL (COLA Refinement) ===")
display(refine_ace)
```

**What you'll see:**
- Color-coded DataFrames where:
  - 🟢 Green: No change
  - 🟡 Yellow: Feature changed
- Easy to see which features were modified

### Method 2: Heatmaps (Best for Any Dataset)

```python
# Binary heatmap: Shows which features changed (0 = no change, 1 = changed)
refiner.heatmap_binary(save_path='./results', save_mode='combined')

# Directional heatmap: Shows if features increased (+1), decreased (-1), or stayed same (0)
refiner.heatmap_direction(save_path='./results', save_mode='combined')
```

**What you'll see:**
- Visual comparison of CE vs ACE
- Clear patterns of which features change most often

### Method 3: Stacked Bar Chart

```python
# Compare efficiency: What % of features changed?
refiner.stacked_bar_chart(save_path='./results')
```

**What you'll see:**
- Bar chart showing percentage of features modified
- Clear comparison: ACE requires fewer changes than CE

### Method 4: Diversity Analysis

```python
# Find minimal feature combinations
factual_df, diversity_styles = refiner.diversity()

print("\n=== DIVERSITY ANALYSIS ===")
for i, style in enumerate(diversity_styles):
    print(f"\nCustomer {i+1} - Minimal changes needed:")
    display(style)
```

**What you'll see:**
- For each customer, shows the minimal set of features that need to change
- Multiple valid combinations highlighted

## Step 7: Interpret the Results

```python
# Let's look at a specific customer
customer_idx = 0

# Original customer data
print(f"Customer {customer_idx} was denied because:")
print(denied_customers.iloc[customer_idx])

# What DiCE suggests (many changes)
print(f"\nDiCE suggests changing {original_changes[customer_idx]} features")

# What COLA suggests (fewer changes)
print(f"COLA suggests changing only {refined_changes[customer_idx]} features")

# Which features to change
changed_features = factual_refined.columns[
    (factual_refined.iloc[customer_idx] != ace_refined.iloc[customer_idx])
]
print(f"\nCOLA recommends changing: {changed_features.tolist()}")
```

## Complete Example

Here's the complete code in one place:

```python
import pandas as pd
import joblib
from xai_cola import COLA
from xai_cola.data import COLAData
from xai_cola.models import Model
from xai_cola.ce_generator import DiCE
from datasets.german_credit import GermanCreditDataset

# 1. Load data
dataset = GermanCreditDataset()
df = dataset.get_dataframe()
denied_customers = df[df['Risk'] == 1].sample(5, random_state=42)

# 2. Create data interface
data = COLAData(
    factual_data=denied_customers,
    label_column='Risk',
    transform='ohe-zscore'
)

# 3. Load model
classifier = joblib.load('lgbm_GremanCredit.pkl')
ml_model = Model(model=classifier, backend="sklearn")

# 4. Generate counterfactuals
explainer = DiCE(ml_model=ml_model)
factual, counterfactual = explainer.generate_counterfactuals(
    data=data,
    factual_class=1,
    total_cfs=1,
    features_to_keep=['Age', 'Sex']
)

# 5. Refine with COLA
data.add_counterfactuals(counterfactual, with_target_column=True)
refiner = COLA(data=data, ml_model=ml_model)
refiner.set_policy(matcher="ect", attributor="pshap", Avalues_method="max")

factual_refined, ce_refined, ace_refined = refiner.get_all_results(limited_actions=3)

# 6. Visualize
refine_factual, refine_ce, refine_ace = refiner.highlight_changes_final()
display(refine_ace)

refiner.heatmap_binary(save_path='./results', save_mode='combined')
refiner.stacked_bar_chart(save_path='./results')
```

## Exercises

### Exercise 1: Vary the Number of Actions
Try different values for `limited_actions` (1, 3, 5, 10). How does this affect the results?

### Exercise 2: Try Different Matchers
Change `matcher="ect"` to `"ot"`, `"nn"`, or `"cem"`. Which works best for this dataset?

### Exercise 3: Feature Selection
Use `features_to_vary` to restrict which features can be changed:

```python
factual_refined, ce_refined, ace_refined = refiner.get_all_results(
    limited_actions=3,
    features_to_vary=['Credit amount', 'Duration', 'Age']
)
```

## Solutions

<details>
<summary>Click to see solutions</summary>

### Exercise 1 Solution
```python
for actions in [1, 3, 5, 10]:
    f, c, a = refiner.get_all_results(limited_actions=actions)
    changes = (f != a).sum(axis=1)
    print(f"Limited to {actions} actions: {changes.tolist()}")
```

### Exercise 2 Solution
```python
for matcher in ["ect", "ot", "nn", "cem"]:
    refiner.set_policy(matcher=matcher, attributor="pshap", Avalues_method="max")
    f, c, a = refiner.get_all_results(limited_actions=3)
    changes = (f != a).sum(axis=1)
    print(f"Matcher {matcher}: {changes.tolist()}")
```

### Exercise 3 Solution
```python
f, c, a = refiner.get_all_results(
    limited_actions=3,
    features_to_vary=['Credit amount', 'Duration', 'Age']
)
# Check which features actually changed
for col in f.columns:
    if (f[col] != a[col]).any():
        print(f"Changed: {col}")
```

</details>

## Next Steps

<!-- - [Tutorial 2: Working with Different Explainers](02_explainers.md)
- [Tutorial 3: Data Interface Deep Dive](03_data_interface.md) -->
- [API Reference](../../API_REFERENCE.md)

## Summary

In this tutorial, you learned:
- ✅ The complete COLA workflow
- ✅ How to generate and refine counterfactuals
- ✅ Multiple visualization techniques
- ✅ How to interpret COLA's output

COLA reduces the number of feature changes needed while maintaining the same outcome, making counterfactual explanations more actionable and practical.
