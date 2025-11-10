import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, Tuple
import os


def heatmap_direction_changes(
        factual: pd.DataFrame = None,
        counterfactual: pd.DataFrame = None,
        target_name: str = None,
        numerical_features: list = None,
        unchanged_color: str = 'lightgrey',
        increased_color: str = '#56B4E9',  # Light blue for increased values
        decreased_color: str = '#7FCDCD',  # Light cyan for decreased values
        categorical_changed_color: str = '#CD853F',  # Peru (soft warm tone) for categorical changes
        show_axis_labels: bool = False,
        ) -> plt.Figure:
    """
    Pure function to generate directional change heatmap with distinction between numerical and categorical features.

    This function generates a heatmap that shows the direction of changes:
    - Numerical features (increased): light blue (#56B4E9)
    - Numerical features (decreased): light cyan (#7FCDCD)
    - Categorical features (changed): peru (#CD853F)
    - Value unchanged: lightgrey
    - Target column: changed values shown in black (unchanged remain lightgrey)

    Args:
        factual (pd.DataFrame, optional): Factual DataFrame. Defaults to None.
        counterfactual (pd.DataFrame, optional): Counterfactual DataFrame. Defaults to None.
        target_name (str, optional): Name of the target/label column. Defaults to None.
        numerical_features (list, optional): List of numerical feature names. If None, all features are treated as numerical. Defaults to None.
        unchanged_color (str, optional): Background color for unchanged cells. Defaults to 'lightgrey'.
        increased_color (str, optional): Color for increased values (numerical only). Defaults to '#56B4E9' (light blue).
        decreased_color (str, optional): Color for decreased values (numerical only). Defaults to '#7FCDCD' (light cyan).
        categorical_changed_color (str, optional): Color for categorical feature changes. Defaults to '#CD853F' (peru).
        show_axis_labels (bool, optional): Whether to show x and y axis labels. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The heatmap figure showing direction of changes from factual to counterfactual
    """
    # Create direction matrix:
    # -1 for decreased (numerical), 0 for unchanged, 1 for increased (numerical), 2 for categorical changed
    # Keep all columns including target column to maintain same structure as input DataFrame
    direction_df = pd.DataFrame(0, index=factual.index, columns=factual.columns)

    # Create a separate matrix to track which cells should use target color (for target column changes)
    target_change_mask = pd.DataFrame(False, index=factual.index, columns=factual.columns)

    # If numerical_features is None, treat all features as numerical (backward compatibility)
    if numerical_features is None:
        numerical_features = [col for col in factual.columns if col != target_name]

    # Identify categorical features (all non-numerical, non-target features)
    categorical_features = [col for col in factual.columns if col not in numerical_features and col != target_name]

    for col in factual.columns:
        if col == target_name:
            # For target column, mark changed cells but keep direction as 0 (will use special color)
            target_change_mask[col] = (factual[col] != counterfactual[col])
            # Set direction to 0 (unchanged) - will be colored using target_color later
            direction_df[col] = 0
        elif col in categorical_features:
            # For categorical features, use value 2 to indicate "changed" (will be colored peru)
            # 0 for unchanged
            changed_mask = (factual[col] != counterfactual[col])
            direction_df[col] = np.where(changed_mask, 2, 0)
        else:
            # For numerical feature columns, use -1, 0, 1 for decreased, unchanged, increased
            try:
                # Try numeric comparison
                diff = pd.to_numeric(counterfactual[col], errors='coerce') - pd.to_numeric(factual[col], errors='coerce')
                direction_df[col] = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
                # Fill NaN values (non-numeric) with 0 (unchanged)
                direction_df[col] = direction_df[col].fillna(0).astype(int)
            except (TypeError, ValueError):
                # If numeric conversion fails, treat as unchanged
                direction_df[col] = 0
    
    # Create colormap with 4 colors: decreased (-1), unchanged (0), increased (1), categorical changed (2)
    # Map -1, 0, 1, 2 to 0, 1, 2, 3 for the colormap
    cmap = ListedColormap([decreased_color, unchanged_color, increased_color, categorical_changed_color])
    direction_mapped = direction_df + 1  # Maps -1,0,1,2 to 0,1,2,3

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(max(10, len(factual.columns) * 0.8), max(6, len(factual) * 0.5)))

    # Plot the base heatmap with directional colors
    sns.heatmap(direction_mapped, cmap=cmap, annot=False, cbar=False,
                square=True, linewidths=0.5, vmin=0, vmax=3, ax=ax, fmt='d')
    
    # Overlay target column changes with dark blue color
    # Create a mask: only show target column changed cells, hide everything else
    overlay_mask = ~target_change_mask  # Hide unchanged cells and non-target columns
    target_change_values = target_change_mask.astype(int) * 1  # Convert to 0/1 for colormap
    target_cmap = ListedColormap([unchanged_color, 'black'])  # Black for target column
    sns.heatmap(target_change_values, cmap=target_cmap, annot=False, cbar=False,
                square=True, linewidths=0.5, vmin=0, vmax=1, ax=ax, alpha=0.6,
                mask=overlay_mask)  # Only show target column changed cells
    
    # Set column names and row indices based on show_axis_labels
    if show_axis_labels:
        ax.set_xticks(range(len(factual.columns)))
        ax.set_xticklabels(factual.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(factual.index)))
        ax.set_yticklabels(factual.index, fontsize=9)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=increased_color, edgecolor='black', label='Numerical (Increased)'),
        Patch(facecolor=decreased_color, edgecolor='black', label='Numerical (Decreased)'),
        Patch(facecolor=categorical_changed_color, edgecolor='black', label='Categorical (Changed)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=True, fontsize=9)

    plt.tight_layout()

    # Return the figure object
    return fig


def generate_direction_heatmap(
    factual_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    refined_counterfactual_df: pd.DataFrame,
    label_column: str,
    numerical_features: Optional[list] = None,
    save_path: Optional[str] = None,
    save_mode: str = "combined",
    show_axis_labels: bool = True
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Pure function to generate directional change heatmap visualizations with distinction between numerical and categorical features.

    This function generates heatmaps that show the direction of changes:
    - Numerical features (increased): light blue (#56B4E9)
    - Numerical features (decreased): light cyan (#7FCDCD)
    - Categorical features (changed): peru (#CD853F)
    - Value unchanged: lightgrey
    - Target column: changed values shown in black

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
    numerical_features : list, optional
        List of numerical feature names. If None, all features (except label) are treated as numerical.
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
    print("Directional changes from factual to counterfactual:")
    plot1 = heatmap_direction_changes(
        factual_df,
        counterfactual_df,
        label_column,
        numerical_features=numerical_features,
        show_axis_labels=show_axis_labels
    )

    print("Directional changes from factual to action-limited counterfactual:")
    plot2 = heatmap_direction_changes(
        factual_df,
        refined_counterfactual_df,
        label_column,
        numerical_features=numerical_features,
        show_axis_labels=show_axis_labels
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
            plot1.savefig(os.path.join(save_dir, 'heatmap_direction_counterfactual.png'), bbox_inches='tight', dpi=300)
            plot2.savefig(os.path.join(save_dir, 'heatmap_direction_counterfactual_with_limited_actions.png'), bbox_inches='tight', dpi=300)
            print(f"✅ Directional heatmaps saved to: {os.path.join(save_dir, 'heatmap_direction_counterfactual.png')} and {os.path.join(save_dir, 'heatmap_direction_counterfactual_with_limited_actions.png')}")
            plt.close(plot1)
            plt.close(plot2)
        else:  # "combined" mode (default)
            # Save as combined image
            # Create a figure with two subplots (vertically arranged) with increased spacing
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(factual_df.columns) * 0.8), max(12, len(factual_df) * 1.0)),
                                          gridspec_kw={'hspace': 0.4})  # Increase vertical spacing between subplots

            # If numerical_features is None, treat all features as numerical (backward compatibility)
            if numerical_features is None:
                numerical_features_list = [col for col in factual_df.columns if col != label_column]
            else:
                numerical_features_list = numerical_features

            # Identify categorical features (all non-numerical, non-target features)
            categorical_features_list = [col for col in factual_df.columns if col not in numerical_features_list and col != label_column]

            # Create direction matrices for CE (counterfactual)
            direction_df_ce = pd.DataFrame(0, index=factual_df.index, columns=factual_df.columns)
            target_change_mask_ce = pd.DataFrame(False, index=factual_df.index, columns=factual_df.columns)

            # Create direction matrices for ACE (action-limited counterfactual)
            direction_df_ace = pd.DataFrame(0, index=factual_df.index, columns=factual_df.columns)
            target_change_mask_ace = pd.DataFrame(False, index=factual_df.index, columns=factual_df.columns)

            for col in factual_df.columns:
                if col == label_column:
                    # For target column, mark changed cells
                    target_change_mask_ce[col] = (factual_df[col] != counterfactual_df[col])
                    target_change_mask_ace[col] = (factual_df[col] != refined_counterfactual_df[col])
                    direction_df_ce[col] = 0
                    direction_df_ace[col] = 0
                elif col in categorical_features_list:
                    # For categorical features, use value 2 to indicate "changed" (will be colored peru)
                    changed_mask_ce = (factual_df[col] != counterfactual_df[col])
                    changed_mask_ace = (factual_df[col] != refined_counterfactual_df[col])
                    direction_df_ce[col] = np.where(changed_mask_ce, 2, 0)
                    direction_df_ace[col] = np.where(changed_mask_ace, 2, 0)
                else:
                    # For numerical feature columns, use -1, 0, 1 for decreased, unchanged, increased
                    try:
                        diff_ce = pd.to_numeric(counterfactual_df[col], errors='coerce') - pd.to_numeric(factual_df[col], errors='coerce')
                        diff_ace = pd.to_numeric(refined_counterfactual_df[col], errors='coerce') - pd.to_numeric(factual_df[col], errors='coerce')
                        direction_df_ce[col] = np.where(diff_ce > 0, 1, np.where(diff_ce < 0, -1, 0))
                        direction_df_ace[col] = np.where(diff_ace > 0, 1, np.where(diff_ace < 0, -1, 0))
                        direction_df_ce[col] = direction_df_ce[col].fillna(0).astype(int)
                        direction_df_ace[col] = direction_df_ace[col].fillna(0).astype(int)
                    except (TypeError, ValueError):
                        direction_df_ce[col] = 0
                        direction_df_ace[col] = 0

            # Create colormaps with 4 colors: decreased (-1), unchanged (0), increased (1), categorical changed (2)
            cmap = ListedColormap(['#7FCDCD', 'lightgrey', '#56B4E9', '#CD853F'])  # Light cyan (decreased), grey (unchanged), light blue (increased), peru (categorical)
            target_cmap = ListedColormap(['lightgrey', 'black'])  # Grey, black for target column

            # Map directions for CE
            direction_mapped_ce = direction_df_ce + 1  # Maps -1,0,1,2 to 0,1,2,3
            target_change_values_ce = target_change_mask_ce.astype(int)

            # Map directions for ACE
            direction_mapped_ace = direction_df_ace + 1  # Maps -1,0,1,2 to 0,1,2,3
            target_change_values_ace = target_change_mask_ace.astype(int)
            
            # Plot first heatmap (CE)
            sns.heatmap(direction_mapped_ce, cmap=cmap, annot=False, cbar=False,
                       square=True, linewidths=0.5, vmin=0, vmax=3, ax=ax1, fmt='d')
            overlay_mask_ce = ~target_change_mask_ce
            sns.heatmap(target_change_values_ce, cmap=target_cmap, annot=False, cbar=False,
                       square=True, linewidths=0.5, vmin=0, vmax=1, ax=ax1, alpha=0.6,
                       mask=overlay_mask_ce)
            if show_axis_labels:
                ax1.set_xticks(range(len(factual_df.columns)))
                ax1.set_xticklabels(factual_df.columns, rotation=45, ha='right', fontsize=9)
                ax1.set_yticks(range(len(factual_df.index)))
                ax1.set_yticklabels(factual_df.index, fontsize=9)
            else:
                ax1.set_xticks([])
                ax1.set_yticks([])
            ax1.set_title('Original Counterfactual (Direction)', fontsize=12)

            # Plot second heatmap (ACE)
            sns.heatmap(direction_mapped_ace, cmap=cmap, annot=False, cbar=False,
                       square=True, linewidths=0.5, vmin=0, vmax=3, ax=ax2, fmt='d')
            overlay_mask_ace = ~target_change_mask_ace
            sns.heatmap(target_change_values_ace, cmap=target_cmap, annot=False, cbar=False,
                       square=True, linewidths=0.5, vmin=0, vmax=1, ax=ax2, alpha=0.6,
                       mask=overlay_mask_ace)
            if show_axis_labels:
                ax2.set_xticks(range(len(factual_df.columns)))
                ax2.set_xticklabels(factual_df.columns, rotation=45, ha='right', fontsize=9)
                ax2.set_yticks(range(len(factual_df.index)))
                ax2.set_yticklabels(factual_df.index, fontsize=9)
            else:
                ax2.set_xticks([])
                ax2.set_yticks([])
            ax2.set_title('Counterfactual with Limited Actions (Direction)', fontsize=12)

            # Add legend to the combined figure
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#56B4E9', edgecolor='black', label='Numerical (Increased)'),
                Patch(facecolor='#7FCDCD', edgecolor='black', label='Numerical (Decreased)'),
                Patch(facecolor='#CD853F', edgecolor='black', label='Categorical (Changed)')
            ]
            # Place legend between the two subplots (centered horizontally, positioned in the gap)
            fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.5),
                      ncol=3, frameon=True, fontsize=11, edgecolor='black', fancybox=False)

            # Adjust layout with rect to prevent legend overlap
            plt.tight_layout(rect=[0, 0, 1, 1])
            fig.savefig(os.path.join(save_dir, 'combined_direction_heatmap.png'), bbox_inches='tight', dpi=300)
            print(f"✅ Combined directional heatmap saved to: {os.path.join(save_dir, 'combined_direction_heatmap.png')}")
            plt.close(fig)
            plt.close(plot1)
            plt.close(plot2)
    else:
        # Just display plots in Jupyter (plots will be automatically displayed)
        pass
    
    return plot1, plot2

