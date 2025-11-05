import pandas as pd
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler


def highlight_differences(data, df_a, df_b, target_name):
    """
    Pure function to highlight differences between two DataFrames.
    
    Creates a style DataFrame with background colors and borders to highlight
    differences between the two input DataFrames.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame to style (should match df_a and df_b structure)
    df_a : pd.DataFrame
        First DataFrame for comparison (typically factual)
    df_b : pd.DataFrame
        Second DataFrame for comparison (typically counterfactual)
    target_name : str
        Name of the target/label column
    
    Returns:
    --------
    pd.DataFrame
        Style DataFrame with CSS styling strings for highlighting differences
    """
    # Create an empty style DataFrame to store background colors and border styles
    df_style = pd.DataFrame('', index=data.index, columns=data.columns)

    # Iterate through each element in both DataFrames to find the differences
    for row in range(df_b.shape[0]):
        for col in range(df_b.shape[1]):
            val_a = df_a.iat[row, col]
            val_b = df_b.iat[row, col]
            column_name = df_a.columns[col]  # Get the column name of the current column
            
            if val_a != val_b:
                # If the current column is the target column, set a light gray background and black border
                if column_name == target_name:
                    df_style.iat[row, col] = 'background-color: lightgray; border: 1px solid black'
                else:
                    # For other columns, set a yellow background and black border
                    df_style.iat[row, col] = 'background-color: yellow; border: 1px solid black'
    
    return df_style


def _change_df_value(factual: pd.DataFrame, ce: pd.DataFrame) -> pd.DataFrame:
    """
    Pure function to format DataFrame values showing changes as "old -> new".
    
    Parameters:
    -----------
    factual : pd.DataFrame
        Factual DataFrame
    ce : pd.DataFrame
        Counterfactual DataFrame
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with changed values formatted as "factual_value -> counterfactual_value"
    """
    ce_formatted = ce.copy().astype(object)
    for row in range(ce_formatted.shape[0]):
        for col in range(ce_formatted.shape[1]):
            val_factual = factual.iat[row, col]
            val_ce = ce_formatted.iat[row, col]
            if val_factual != val_ce:
                ce_formatted.iat[row, col] = f'{val_factual} -> {val_ce}'
    return ce_formatted


def highlight_changes_comparison(
    factual_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    refined_counterfactual_df: pd.DataFrame,
    label_column: str
) -> Tuple[pd.DataFrame, 'Styler', 'Styler']:
    """
    Pure function to highlight changes with comparison format (old -> new).
    
    This function displays changes in the format "factual_value -> counterfactual_value"
    to show both the original and modified values side by side.
    
    Parameters:
    -----------
    factual_df : pd.DataFrame
        Original factual data
    counterfactual_df : pd.DataFrame
        Full counterfactual data (corresponding counterfactual)
    refined_counterfactual_df : pd.DataFrame
        Action-limited counterfactual data
    label_column : str
        Name of the target/label column
    
    Returns:
    --------
    tuple
        (factual_df, ce_style, ace_style)
        - factual_df: pandas.DataFrame - Original factual data (copy)
        - ce_style: pandas.io.formats.style.Styler - Styled DataFrame showing factual → full counterfactual
        - ace_style: pandas.io.formats.style.Styler - Styled DataFrame showing factual → action-limited counterfactual
    """
    # Prepare copies for highlighting (avoid modifying original data)
    factual_df_copy = factual_df.copy().astype(object)
    ace_df = refined_counterfactual_df.copy().astype(object)
    corresponding_cf_df = counterfactual_df.copy().astype(object)
    
    # Apply highlighting with "old -> new" format
    cce_df = _change_df_value(factual_df_copy, corresponding_cf_df)
    ace_df_formatted = _change_df_value(factual_df_copy, ace_df)
    
    cce_style = cce_df.style.apply(
        lambda x: highlight_differences(x, factual_df_copy, cce_df, label_column),
        axis=None
    ).set_properties(**{'text-align': 'center'})
    
    ace_style = ace_df_formatted.style.apply(
        lambda x: highlight_differences(x, factual_df_copy, ace_df_formatted, label_column),
        axis=None
    ).set_properties(**{'text-align': 'center'})
    
    return factual_df_copy, cce_style, ace_style


def highlight_changes_final(
    factual_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    refined_counterfactual_df: pd.DataFrame,
    label_column: str
) -> Tuple[pd.DataFrame, 'Styler', 'Styler']:
    """
    Pure function to highlight changes showing only the final values.
    
    This function displays only the final counterfactual values without showing
    the "factual -> counterfactual" format, making it cleaner for presentation.
    
    Parameters:
    -----------
    factual_df : pd.DataFrame
        Original factual data
    counterfactual_df : pd.DataFrame
        Full counterfactual data (corresponding counterfactual)
    refined_counterfactual_df : pd.DataFrame
        Action-limited counterfactual data
    label_column : str
        Name of the target/label column
    
    Returns:
    --------
    tuple
        (factual_df, ce_style, ace_style)
        - factual_df: pandas.DataFrame - Original factual data (copy)
        - ce_style: pandas.io.formats.style.Styler - Styled DataFrame showing only full counterfactual values
        - ace_style: pandas.io.formats.style.Styler - Styled DataFrame showing only action-limited counterfactual values
    """
    # Prepare copies for highlighting (avoid modifying original data)
    factual_df_copy = factual_df.copy().astype(object)
    ace_df = refined_counterfactual_df.copy().astype(object)
    corresponding_cf_df = counterfactual_df.copy().astype(object)
    
    # Use original counterfactual DataFrames without modification
    cce_df = corresponding_cf_df
    ace_df_final = ace_df
    
    cce_style = cce_df.style.apply(
        lambda x: highlight_differences(x, factual_df_copy, cce_df, label_column),
        axis=None
    ).set_properties(**{'text-align': 'center'})
    
    ace_style = ace_df_final.style.apply(
        lambda x: highlight_differences(x, factual_df_copy, ace_df_final, label_column),
        axis=None
    ).set_properties(**{'text-align': 'center'})
    
    return factual_df_copy, cce_style, ace_style
