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
        unchanged_color: str = 'lightgrey',
        increased_color: str = '#009E73',  # Teal-green for increased values
        decreased_color: str = '#D55E00',  # Red-orange for decreased values
        show_axis_labels: bool = False,
        ) -> plt.Figure:
    """
    Pure function to generate directional change heatmap (shows if value increased, decreased, or unchanged).
    
    This function generates a heatmap that shows the direction of changes:
    - Value increased: teal-green (#009E73)
    - Value decreased: red-orange (#D55E00)
    - Value unchanged: lightgrey
    - Target column: changed values shown in dark blue (unchanged remain lightgrey)

    Args:
        factual (pd.DataFrame, optional): Factual DataFrame. Defaults to None.
        counterfactual (pd.DataFrame, optional): Counterfactual DataFrame. Defaults to None.
        target_name (str, optional): Name of the target/label column. Defaults to None.
        unchanged_color (str, optional): Background color for unchanged cells. Defaults to 'lightgrey'.
        increased_color (str, optional): Color for increased values. Defaults to '#009E73' (teal-green).
        decreased_color (str, optional): Color for decreased values. Defaults to '#D55E00' (red-orange).
        show_axis_labels (bool, optional): Whether to show x and y axis labels. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The heatmap figure showing direction of changes from factual to counterfactual
    """
    # Create direction matrix: -1 for decreased, 0 for unchanged, 1 for increased
    # Keep all columns including target column to maintain same structure as input DataFrame
    direction_df = pd.DataFrame(0, index=factual.index, columns=factual.columns)
    
    # Create a separate matrix to track which cells should use target color (for target column changes)
    target_change_mask = pd.DataFrame(False, index=factual.index, columns=factual.columns)
    
    for col in factual.columns:
        if col == target_name:
            # For target column, mark changed cells but keep direction as 0 (will use special color)
            target_change_mask[col] = (factual[col] != counterfactual[col])
            # Set direction to 0 (unchanged) - will be colored using target_color later
            direction_df[col] = 0
        else:
            # For feature columns, use -1, 0, 1 for decreased, unchanged, increased
            # Only apply directional comparison for numeric columns
            try:
                # Try numeric comparison
                diff = pd.to_numeric(counterfactual[col], errors='coerce') - pd.to_numeric(factual[col], errors='coerce')
                direction_df[col] = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
                # Fill NaN values (non-numeric) with 0 (unchanged)
                direction_df[col] = direction_df[col].fillna(0).astype(int)
            except (TypeError, ValueError):
                # For non-numeric columns, we can't determine direction (increase/decrease)
                # So we'll treat them as unchanged (0) - they will appear grey
                # If they actually changed, they'll still be visible as grey (unchanged)
                direction_df[col] = 0
    
    # Create colormap: light orange (-1), lightgrey (0), light green (1)
    # Map -1, 0, 1 to 0, 1, 2 for the colormap
    cmap = ListedColormap([decreased_color, unchanged_color, increased_color])
    direction_mapped = direction_df + 1  # Maps -1,0,1 to 0,1,2
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(max(10, len(factual.columns) * 0.8), max(6, len(factual) * 0.5)))
    
    # Plot the base heatmap with directional colors
    sns.heatmap(direction_mapped, cmap=cmap, annot=False, cbar=False, 
                square=True, linewidths=0.5, vmin=0, vmax=2, ax=ax, fmt='d')
    
    # Overlay target column changes with dark blue color
    # Create a mask: only show target column changed cells, hide everything else
    overlay_mask = ~target_change_mask  # Hide unchanged cells and non-target columns
    target_change_values = target_change_mask.astype(int) * 1  # Convert to 0/1 for colormap
    target_cmap = ListedColormap([unchanged_color, '#000080'])  # Dark blue for target column
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
    
    plt.tight_layout()
    
    # Return the figure object
    return fig


def generate_direction_heatmap(
    factual_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    refined_counterfactual_df: pd.DataFrame,
    label_column: str,
    save_path: Optional[str] = None,
    save_mode: str = "combined",
    show_axis_labels: bool = True
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Pure function to generate directional change heatmap visualizations.
    
    This function generates heatmaps that show the direction of changes:
    - Value increased: teal-green (#009E73)
    - Value decreased: red-orange (#D55E00)
    - Value unchanged: lightgrey
    - Target column: changed values shown in dark blue
    
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
    print("Directional changes from factual to counterfactual:")
    plot1 = heatmap_direction_changes(
        factual_df,
        counterfactual_df,
        label_column,
        show_axis_labels=show_axis_labels
    )
    
    print("Directional changes from factual to action-limited counterfactual:")
    plot2 = heatmap_direction_changes(
        factual_df,
        refined_counterfactual_df,
        label_column,
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
            # Create a figure with two subplots (vertically arranged)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(factual_df.columns) * 0.8), max(12, len(factual_df) * 1.0)))
            
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
                else:
                    # For feature columns, use -1, 0, 1 for decreased, unchanged, increased
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
            
            # Create colormaps
            cmap = ListedColormap(['#D55E00', 'lightgrey', '#009E73'])  # Red-orange (decreased), grey (unchanged), teal-green (increased)
            target_cmap = ListedColormap(['lightgrey', '#000080'])  # Grey, dark blue for target column
            
            # Map directions for CE
            direction_mapped_ce = direction_df_ce + 1  # Maps -1,0,1 to 0,1,2
            target_change_values_ce = target_change_mask_ce.astype(int)
            
            # Map directions for ACE
            direction_mapped_ace = direction_df_ace + 1  # Maps -1,0,1 to 0,1,2
            target_change_values_ace = target_change_mask_ace.astype(int)
            
            # Plot first heatmap (CE)
            sns.heatmap(direction_mapped_ce, cmap=cmap, annot=False, cbar=False, 
                       square=True, linewidths=0.5, vmin=0, vmax=2, ax=ax1, fmt='d')
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
                       square=True, linewidths=0.5, vmin=0, vmax=2, ax=ax2, fmt='d')
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
            
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, 'combined_direction_heatmap.png'), bbox_inches='tight', dpi=300)
            print(f"✅ Combined directional heatmap saved to: {os.path.join(save_dir, 'combined_direction_heatmap.png')}")
            plt.close(fig)
            plt.close(plot1)
            plt.close(plot2)
    else:
        # Just display plots in Jupyter (plots will be automatically displayed)
        pass
    
    return plot1, plot2

