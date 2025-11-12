import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import os


def create_stacked_bar_chart(
    factual_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    refined_counterfactual_df: pd.DataFrame,
    label_column: str,
    save_path: Optional[str] = None,
    refined_color: str = '#D9F2D0',
    counterfactual_color: str = '#FBE3D6',
    instance_labels: Optional[list] = None
) -> plt.Figure:
    """
    Pure function to create a horizontal stacked percentage bar chart comparing modification positions.

    This function creates a percentage-based stacked bar chart where each bar represents
    an instance (100% total), showing the proportion of modified positions in refined
    counterfactual vs. original counterfactual relative to factual data.

    Each bar shows:
    - Green segment (#D9F2D0): percentage of positions modified by refined_counterfactual
    - Orange segment (#FBE3D6): percentage of additional positions modified only by counterfactual
    - Total bar length: 100% (representing all counterfactual modifications)

    Labels on bars show both percentage and actual count (e.g., "60.0% (3)")

    Parameters:
    -----------
    factual_df : pd.DataFrame
        Original factual data
    counterfactual_df : pd.DataFrame
        Full counterfactual data (corresponding counterfactual)
    refined_counterfactual_df : pd.DataFrame
        Action-limited counterfactual data
    label_column : str
        Name of the target/label column to exclude from comparison
    save_path : str, optional
        Path to save the chart image. If None, chart is not saved.
        Can be a directory path or file path.
    refined_color : str, default='#D9F2D0'
        Color for refined counterfactual modified positions (light green)
    counterfactual_color : str, default='#FBE3D6'
        Color for counterfactual modified positions (light pink/orange)
    instance_labels : list, optional
        Custom labels for instances. If None, uses "instance 1", "instance 2", etc.

    Returns:
    --------
    matplotlib.figure.Figure
        The stacked bar chart figure
    """
    # Calculate number of modified positions for each instance
    # Use same logic as highlight_dataframe.py: iterate through each element
    n_instances = len(factual_df)
    n_columns = len(factual_df.columns)

    # Build data for pivot table
    data_rows = []
    for row in range(n_instances):
        refined_count = 0
        counterfactual_count = 0

        for col in range(n_columns):
            column_name = factual_df.columns[col]

            # Skip label column
            if column_name == label_column:
                continue

            val_factual = factual_df.iat[row, col]
            val_refined = refined_counterfactual_df.iat[row, col]
            val_counterfactual = counterfactual_df.iat[row, col]

            # Count modifications in refined counterfactual
            if val_factual != val_refined:
                refined_count += 1

            # Count modifications in counterfactual
            if val_factual != val_counterfactual:
                counterfactual_count += 1

        # Create instance label
        if instance_labels is None:
            instance_label = f'instance {row + 1}'
        else:
            instance_label = instance_labels[row]

        # Add rows for this instance
        data_rows.append({
            'group': instance_label,
            'sub': 'refined',
            'value': refined_count
        })
        data_rows.append({
            'group': instance_label,
            'sub': 'counterfactual',
            'value': counterfactual_count
        })

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # 1) Aggregate → row-normalize to percentages
    tab = df.pivot_table(index='group', columns='sub', values='value',
                         aggfunc='sum', fill_value=0)

    # Ensure columns order is exactly ["refined", "counterfactual"]
    tab = tab.reindex(columns=['refined', 'counterfactual'], fill_value=0)

    # Sort index naturally (to handle "instance 1", "instance 2", ..., "instance 10" correctly)
    # Extract numeric parts if instance labels follow "instance N" pattern
    if all('instance' in str(idx).lower() for idx in tab.index):
        # Natural sort for "instance N" format
        def extract_number(s):
            import re
            match = re.search(r'\d+', str(s))
            return int(match.group()) if match else 0
        tab = tab.loc[sorted(tab.index, key=extract_number)]
    # Otherwise keep the original order from data_rows (which preserves input order)

    # Calculate percentages
    pct = tab.div(tab.sum(axis=1), axis=0) * 100

    # 2) Horizontal 100% stacked bars
    fig, ax = plt.subplots(figsize=(10, max(4, len(pct) * 0.4)))
    left_accumulate = np.zeros(len(pct))
    y = np.arange(len(pct.index))

    # Plot refined counterfactual (green) - with thinner bars
    ax.barh(y, pct['refined'].values, left=left_accumulate, height=0.6,
            color=refined_color, label='Refined Counterfactual',
            edgecolor='black', linewidth=0.5)
    left_accumulate += pct['refined'].values

    # Plot counterfactual (orange/pink) - with thinner bars
    ax.barh(y, pct['counterfactual'].values, left=left_accumulate, height=0.6,
            color=counterfactual_color, label='Original Counterfactual (additional)',
            edgecolor='black', linewidth=0.5)

    # 3) Cosmetics
    ax.set_yticks(y)
    ax.set_yticklabels(pct.index)
    ax.invert_yaxis()  # Top to bottom: instance 1, instance 2, etc.
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add percentage labels on bars (with actual counts in parentheses)
    left_accumulate = np.zeros(len(pct))
    for i in range(len(pct)):
        # Label for refined counterfactual segment
        refined_pct = pct['refined'].iloc[i]
        refined_count = int(tab['refined'].iloc[i])
        if refined_pct > 5:  # Only show label if segment is large enough
            label_text = f'{refined_pct:.1f}%\n({refined_count})'
            ax.text(left_accumulate[i] + refined_pct / 2, i, label_text,
                   ha='center', va='center', fontsize=9, fontweight='bold', color='black')
        left_accumulate[i] += refined_pct

        # Label for counterfactual segment
        counterfactual_pct = pct['counterfactual'].iloc[i]
        counterfactual_count = int(tab['counterfactual'].iloc[i])
        if counterfactual_pct > 5:
            label_text = f'{counterfactual_pct:.1f}%\n({counterfactual_count})'
            ax.text(left_accumulate[i] + counterfactual_pct / 2, i, label_text,
                   ha='center', va='center', fontsize=9, fontweight='bold', color='black')

    plt.tight_layout()

    # Save if path is provided
    if save_path is not None:
        # Normalize the path
        if save_path.endswith('/') or save_path.endswith('\\'):
            save_dir = save_path.rstrip('/\\')
            save_file = os.path.join(save_dir, 'stacked_bar_chart.png')
        else:
            # Check if it's a directory or file
            if os.path.isdir(save_path) or not save_path.endswith(('.png', '.jpg', '.pdf', '.svg')):
                os.makedirs(save_path, exist_ok=True)
                save_file = os.path.join(save_path, 'stacked_bar_chart.png')
            else:
                save_file = save_path
                os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)

        fig.savefig(save_file, bbox_inches='tight', dpi=300)
        print(f"✅ Stacked bar chart saved to: {save_file}")

    return fig


def generate_stacked_bar_chart(
    factual_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    refined_counterfactual_df: pd.DataFrame,
    label_column: str,
    save_path: Optional[str] = None,
    refined_color: str = '#D9F2D0',
    counterfactual_color: str = '#FBE3D6',
    instance_labels: Optional[list] = None
) -> plt.Figure:
    """
    Pure function to generate and optionally save a stacked percentage bar chart.

    This is a wrapper around create_stacked_bar_chart() for consistency with other
    visualization functions.

    Creates a percentage-based stacked bar chart where each bar shows:
    - Green segment (#D9F2D0): percentage of positions modified by refined_counterfactual
    - Orange segment (#FBE3D6): percentage of additional positions modified only by counterfactual
    - Total bar length: 100% (representing all counterfactual modifications)

    Labels show both percentage and actual count (e.g., "60.0% (3)")

    Parameters:
    -----------
    factual_df : pd.DataFrame
        Original factual data
    counterfactual_df : pd.DataFrame
        Full counterfactual data (corresponding counterfactual)
    refined_counterfactual_df : pd.DataFrame
        Action-limited counterfactual data
    label_column : str
        Name of the target/label column to exclude from comparison
    save_path : str, optional
        Path to save the chart image. If None, chart is not saved.
    refined_color : str, default='#D9F2D0'
        Color for refined counterfactual modified positions (light green)
    counterfactual_color : str, default='#FBE3D6'
        Color for counterfactual modified positions (light pink/orange)
    instance_labels : list, optional
        Custom labels for instances

    Returns:
    --------
    matplotlib.figure.Figure
        The stacked bar chart figure
    """
    return create_stacked_bar_chart(
        factual_df=factual_df,
        counterfactual_df=counterfactual_df,
        refined_counterfactual_df=refined_counterfactual_df,
        label_column=label_column,
        save_path=save_path,
        refined_color=refined_color,
        counterfactual_color=counterfactual_color,
        instance_labels=instance_labels
    )


