import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, Tuple
import os


def heatmap_binary_changes(
        factual: pd.DataFrame = None, 
        counterfactual: pd.DataFrame = None, 
        target_name: str = None, 
        background_color: str = 'lightgrey',
        changed_feature_color: str = 'red',
        show_axis_labels: bool = False,
        ) -> plt.Figure:
    """
    Pure function to generate binary change heatmap (shows if value changed or not).
    
    This function generates a heatmap that shows binary changes: whether a value 
    changed (red) or remained unchanged (lightgrey). For the target column, changed 
    values are shown in dark blue.

    Args:
        factual (pd.DataFrame, optional): Factual DataFrame. Defaults to None.
        counterfactual (pd.DataFrame, optional): Counterfactual DataFrame. Defaults to None.
        target_name (str, optional): Name of the target/label column. Defaults to None.
        background_color (str, optional): Background color for unchanged cells. Defaults to 'lightgrey'.
        changed_feature_color (str, optional): Color for changed features. Defaults to 'red'.
        show_axis_labels (bool, optional): Whether to show x and y axis labels. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The heatmap figure for changes from factual to counterfactual
    """
    # Convert the boolean values (True, False) to integers (1, 0)
    # changes_df is a DataFrame with 'target_name' as one of the columns
    changes_df = (factual != counterfactual).astype(int)

    # Get the dataframes of features(without target_column)
    features_lists = changes_df.drop(columns=[target_name])
    # Get the dataframes with features and target_column, set the values of features to 0. Make it as the top layer
    top_layer_df = changes_df.copy()
    top_layer_df.loc[:, top_layer_df.columns != target_name] = 0

    # Create the colormaps
    cmap_bottom = ListedColormap([background_color, changed_feature_color])
    cmap_top = ListedColormap([background_color, '#000080'])  # Dark blue for target column

    # Create a figure and get the current figure
    fig, ax = plt.subplots(figsize=(max(10, len(factual.columns) * 0.8), max(6, len(factual) * 0.5)))

    # Plot the features_lists with one colormap (base heatmap)
    sns.heatmap(features_lists, cmap=cmap_bottom, annot=False, cbar=False, square=True, linewidths=0.5, ax=ax)

    # Overlay the target_list with a different colormap and transparent alpha layer
    sns.heatmap(top_layer_df, cmap=cmap_top, annot=False, cbar=False, square=True, linewidths=0.5, alpha=0.6, ax=ax)

    # Adjust the ticks based on show_axis_labels
    if show_axis_labels:
        ax.set_xticks(range(len(factual.columns)))
        ax.set_xticklabels(factual.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(factual.index)))
        ax.set_yticklabels(factual.index, fontsize=9)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Return the figure object
    return fig


def generate_binary_heatmap(
    factual_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    refined_counterfactual_df: pd.DataFrame,
    label_column: str,
    save_path: Optional[str] = None,
    save_mode: str = "combined",
    show_axis_labels: bool = True
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Pure function to generate binary change heatmap visualizations.
    
    This function generates heatmaps that show binary changes: whether a value 
    changed (red) or remained unchanged (lightgrey).
    
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
    save_path : str, optional
        Path to save the heatmap images. 
        If None, plots are automatically displayed in Jupyter (default).
        If provided, saves to the specified path and closes the plots.
        Can be a directory path or file path.
        
    save_mode : str, default="combined"
        How to save the heatmaps when save_path is provided:
        - "combined": Save both heatmaps in a single combined image (top and bottom)
        - "separate": Save two separate image files (heatmap_ce.png and heatmap_ace.png)
        - Ignored if save_path is None
    show_axis_labels : bool, default=True
        Whether to show x and y axis labels (column names and row indices).
        If True, displays column names and row indices. If False, hides them.
        
    Returns:
    --------
    tuple
        (plot1, plot2) - Heatmap plots (matplotlib Figure objects)
    """
    print("Binary changes from factual to counterfactual:")
    plot1 = heatmap_binary_changes(
        factual_df,
        counterfactual_df,
        label_column
    )
    
    print("Binary changes from factual to action-limited counterfactual:")
    plot2 = heatmap_binary_changes(
        factual_df,
        refined_counterfactual_df,
        label_column
    )
    
    # Handle saving if path is provided
    if save_path is not None:
        # Normalize the path (remove trailing slashes if present)
        if save_path.endswith('/') or save_path.endswith('\\'):
            save_dir = save_path.rstrip('/\\')
        else:
            save_dir = save_path
        
        os.makedirs(save_dir, exist_ok=True)
        
        if save_mode == "separate":
            # Save as two separate files
            plot1.savefig(os.path.join(save_dir, 'heatmap_counterfactual.png'), bbox_inches='tight', dpi=300)
            plot2.savefig(os.path.join(save_dir, 'heatmap_counterfactual_with_limited_actions.png'), bbox_inches='tight', dpi=300)
            print(f"✅ Heatmaps saved to: {os.path.join(save_dir, 'heatmap_counterfactual.png')} and {os.path.join(save_dir, 'heatmap_counterfactual_with_limited_actions.png')}")
            plt.close(plot1)
            plt.close(plot2)
        else:  # "combined" mode (default)
            # Save as combined image
            # Create a figure with two subplots (vertically arranged)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(factual_df.columns) * 0.8), max(12, len(factual_df) * 1.0)))
            
            # Get the data from plot1 and plot2
            plot1_data_ce = (factual_df != counterfactual_df).astype(int)
            plot2_data_ace = (factual_df != refined_counterfactual_df).astype(int)
            
            features_ce = plot1_data_ce.drop(columns=[label_column])
            top_layer_ce = plot1_data_ce.copy()
            top_layer_ce.loc[:, top_layer_ce.columns != label_column] = 0
            
            features_ace = plot2_data_ace.drop(columns=[label_column])
            top_layer_ace = plot2_data_ace.copy()
            top_layer_ace.loc[:, top_layer_ace.columns != label_column] = 0
            
            cmap_bottom = ListedColormap(['lightgrey', 'red'])
            cmap_top = ListedColormap(['lightgrey', '#000080'])  # Dark blue for target column
            
            # Plot first heatmap (top)
            sns.heatmap(features_ce, cmap=cmap_bottom, annot=False, cbar=False, square=True, linewidths=0.5, ax=ax1)
            sns.heatmap(top_layer_ce, cmap=cmap_top, annot=False, cbar=False, square=True, linewidths=0.5, alpha=0.6, ax=ax1)
            if show_axis_labels:
                ax1.set_xticks(range(len(factual_df.columns)))
                ax1.set_xticklabels(factual_df.columns, rotation=45, ha='right', fontsize=9)
                ax1.set_yticks(range(len(factual_df.index)))
                ax1.set_yticklabels(factual_df.index, fontsize=9)
            else:
                ax1.set_xticks([])
                ax1.set_yticks([])
            ax1.set_title('Original Counterfactual', fontsize=12)
            
            # Plot second heatmap (bottom)
            sns.heatmap(features_ace, cmap=cmap_bottom, annot=False, cbar=False, square=True, linewidths=0.5, ax=ax2)
            sns.heatmap(top_layer_ace, cmap=cmap_top, annot=False, cbar=False, square=True, linewidths=0.5, alpha=0.6, ax=ax2)
            if show_axis_labels:
                ax2.set_xticks(range(len(factual_df.columns)))
                ax2.set_xticklabels(factual_df.columns, rotation=45, ha='right', fontsize=9)
                ax2.set_yticks(range(len(factual_df.index)))
                ax2.set_yticklabels(factual_df.index, fontsize=9)
            else:
                ax2.set_xticks([])
                ax2.set_yticks([])
            ax2.set_title('Counterfactual with Limited Actions', fontsize=12)
            
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, 'combined_heatmap.png'), bbox_inches='tight', dpi=300)
            print(f"✅ Combined heatmap saved to: {os.path.join(save_dir, 'combined_heatmap.png')}")
            plt.close(fig)
            plt.close(plot1)
            plt.close(plot2)
    else:
        # Just display plots in Jupyter (plots will be automatically displayed)
        pass
    
    return plot1, plot2
