import pandas as pd
import numpy as np
from typing import List, Tuple, Set, TYPE_CHECKING
from itertools import combinations

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler


def find_minimal_feature_combinations(
    factual_row: pd.Series,
    refined_counterfactual_row: pd.Series,
    ml_model,
    label_column: str
) -> List[Set[str]]:
    """
    Find all minimal feature combinations that can flip the target from factual to refined counterfactual.

    This function finds minimal sets of features that, when changed from factual to refined counterfactual
    values, will cause the model prediction to change from factual's target value to refined counterfactual's
    target value (e.g., from 1 to 0).

    Parameters:
    -----------
    factual_row : pd.Series
        Single factual instance (with label column)
    refined_counterfactual_row : pd.Series
        Single refined counterfactual instance (with label column)
    ml_model : Model
        ML model for prediction
    label_column : str
        Name of target column

    Returns:
    --------
    List[Set[str]]
        List of minimal feature combinations (each is a set of feature names)
    """
    # Get feature columns (exclude label)
    feature_columns = [col for col in factual_row.index if col != label_column]

    # Get target values from factual and refined counterfactual
    target_value = refined_counterfactual_row[label_column]  # We want to achieve refined counterfactual's target value

    # Find features that differ between factual and refined counterfactual
    changed_features = []
    for col in feature_columns:
        if factual_row[col] != refined_counterfactual_row[col]:
            changed_features.append(col)

    if not changed_features:
        return []

    # Store successful combinations
    successful_combinations = []

    # Try combinations of increasing size
    for size in range(1, len(changed_features) + 1):
        # Generate all combinations of current size
        for combo in combinations(changed_features, size):
            combo_set = set(combo)

            # Skip if this combination contains features already covered by a smaller successful combination
            if any(covered <= combo_set for covered in successful_combinations):
                continue

            # Create test instance: start with factual, change only features in combo
            test_instance = factual_row.copy()
            for feature in combo:
                test_instance[feature] = refined_counterfactual_row[feature]

            # Predict using model (convert to DataFrame with feature columns only)
            test_df = pd.DataFrame([test_instance[feature_columns]])
            prediction = ml_model.predict(test_df.values)[0]

            # If prediction matches refined counterfactual's target value, this is a successful combination
            if prediction == target_value:
                successful_combinations.append(combo_set)

    return successful_combinations


def generate_diversity_dataframe(
    factual_row: pd.Series,
    refined_counterfactual_row: pd.Series,
    minimal_combinations: List[Set[str]],
    label_column: str
) -> pd.DataFrame:
    """
    Generate diversity DataFrame with all minimal combinations.

    The first row shows the refined counterfactual (all changes from refined CF), followed by
    rows showing each minimal combination.

    Parameters:
    -----------
    factual_row : pd.Series
        Single factual instance
    refined_counterfactual_row : pd.Series
        Single refined counterfactual instance
    minimal_combinations : List[Set[str]]
        List of minimal feature combinations
    label_column : str
        Name of target column

    Returns:
    --------
    pd.DataFrame
        DataFrame with first row as refined counterfactual,
        followed by one row per minimal combination
    """
    rows = []

    # First row: refined counterfactual (all changes)
    complete_cf = refined_counterfactual_row.copy()
    rows.append(complete_cf)

    # Following rows: minimal combinations
    if minimal_combinations:
        for i, combo in enumerate(minimal_combinations):
            # Start with factual values
            row = factual_row.copy()
            # Change only features in this combination
            for feature in combo:
                row[feature] = refined_counterfactual_row[feature]
            # Update label column to refined counterfactual's target value
            row[label_column] = refined_counterfactual_row[label_column]
            rows.append(row)

    # Create DataFrame with unique index to avoid Styler errors
    df = pd.DataFrame(rows)
    df.index = range(len(df))  # Ensure unique index
    return df


def highlight_diversity_changes(
    factual_row: pd.Series,
    diversity_df: pd.DataFrame,
    label_column: str
) -> 'Styler':
    """
    Highlight changes in diversity DataFrame.

    The first row shows the refined counterfactual with all changes highlighted,
    with yellow background to distinguish it from minimal combinations.
    Following rows show minimal combinations with only their specific changes highlighted.

    Parameters:
    -----------
    factual_row : pd.Series
        Single factual instance
    diversity_df : pd.DataFrame
        DataFrame with diversity combinations (first row is complete counterfactual)
    label_column : str
        Name of target column

    Returns:
    --------
    Styler
        Styled DataFrame with highlighted changes
    """
    def highlight_differences(data):
        """Apply highlighting to each cell."""
        df_style = pd.DataFrame('', index=data.index, columns=data.columns)

        for row_idx in range(len(data)):
            for col_idx, col in enumerate(data.columns):
                val_factual = factual_row[col]
                val_diversity = data.iat[row_idx, col_idx]

                # First row (complete counterfactual) gets yellow highlight with borders
                if row_idx == 0:
                    if val_factual != val_diversity:
                        # Highlight changed values with yellow background and border
                        if col == label_column:
                            df_style.iat[row_idx, col_idx] = 'background-color: lightgray; border: 1px solid black'
                        else:
                            df_style.iat[row_idx, col_idx] = 'background-color: yellow; border: 1px solid black'
                else:
                    # Other rows (minimal combinations) get #FFFFCC highlight for changes
                    if val_factual != val_diversity:
                        if col == label_column:
                            df_style.iat[row_idx, col_idx] = 'background-color: lightgray; border: 1px solid black'
                        else:
                            df_style.iat[row_idx, col_idx] = 'background-color: #FFFFCC; border: 1px solid black'

        return df_style

    styled = diversity_df.style.apply(lambda x: highlight_differences(diversity_df), axis=None)
    styled = styled.set_properties(**{'text-align': 'center'})

    return styled


def generate_diversity_for_all_instances(
    factual_df: pd.DataFrame,
    refined_counterfactual_df: pd.DataFrame,
    ml_model,
    label_column: str
) -> Tuple[pd.DataFrame, List['Styler']]:
    """
    Generate diversity analysis for all instances.

    This is the main pure function that processes all instances. For each instance,
    it finds minimal feature combinations that can flip the prediction from factual's
    target value to refined counterfactual's target value (e.g., from 1 to 0).

    Parameters:
    -----------
    factual_df : pd.DataFrame
        Factual data (with label column)
    refined_counterfactual_df : pd.DataFrame
        Refined counterfactual data (with label column)
    ml_model : Model
        ML model for prediction
    label_column : str
        Name of target column

    Returns:
    --------
    Tuple[pd.DataFrame, List[Styler]]
        - factual_df: Original factual data (copy)
        - diversity_styles: List of styled DataFrames (one per instance),
          each showing all minimal feature combinations for that instance
    """
    factual_copy = factual_df.copy()
    diversity_styles = []

    for idx in factual_df.index:
        factual_row = factual_df.loc[idx]
        refined_counterfactual_row = refined_counterfactual_df.loc[idx]

        # Find minimal combinations that flip from factual target to refined counterfactual target
        minimal_combos = find_minimal_feature_combinations(
            factual_row=factual_row,
            refined_counterfactual_row=refined_counterfactual_row,
            ml_model=ml_model,
            label_column=label_column
        )

        # Generate diversity DataFrame
        diversity_df = generate_diversity_dataframe(
            factual_row=factual_row,
            refined_counterfactual_row=refined_counterfactual_row,
            minimal_combinations=minimal_combos,
            label_column=label_column
        )

        # Highlight changes
        diversity_style = highlight_diversity_changes(
            factual_row=factual_row,
            diversity_df=diversity_df,
            label_column=label_column
        )

        diversity_styles.append(diversity_style)

    return factual_copy, diversity_styles
