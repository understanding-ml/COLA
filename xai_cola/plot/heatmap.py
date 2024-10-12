import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from io import BytesIO

def heatmap_massivedata(
        factual:pd.DataFrame=None, 
        counterfactual:pd.DataFrame=None, 
        target_name:str=None, 
        background_color:str='lightgrey',
        changed_feature_color:str='red',
        target_color:str='#000080',
        ):
    """Generate heatmap for massivedata (with too many rows or columns)

    Args:
        factual (pd.DataFrame, optional): [generated from counterfactual_limited_actions.py]. Defaults to None.
        counterfactual (pd.DataFrame, optional): [generated from counterfactual_limited_actions.py]. Defaults to None.

    Returns:
        The heatmap plot for changes from factual to counterfactual
    """
    # Convert the boolean values (True, False) to integers (1, 0)
    # changes_df is a DataFrame with 'target_name' as one of the columns
    changes_df = (factual != counterfactual).astype(int)

    # Get the dataframes of features(without target_column)
    features_lists = changes_df.drop(columns=[target_name])
    # Get the dataframes with features and target_column, set the values of features to 0. Make it as the top layer
    top_layer_df = changes_df.copy()
    top_layer_df.loc[:, top_layer_df.columns != target_name]=0

    # Create the colormaps
    cmap_bottom = ListedColormap([background_color, changed_feature_color])
    cmap_top = ListedColormap([background_color, target_color])

    # Create a figure
    # plt.figure(figsize=(8, 8))

    # Plot the features_lists with one colormap (base heatmap)
    sns.heatmap(features_lists, cmap=cmap_bottom, annot=False, cbar=False, square=True, linewidths=0.5)

    # Overlay the target_list with a different colormap and transparent alpha layer
    sns.heatmap(top_layer_df, cmap=cmap_top, annot=False, cbar=False, square=True, linewidths=0.5, alpha=0.6)

    # Adjust the ticks
    plt.xticks([])  # Remove x ticks if necessary
    plt.yticks([])  # Remove y ticks if necessary
    
    # # Save the plot to a buffer
    # buf = BytesIO()
    # plt.savefig(buf, format='png', bbox_inches='tight')
    # plt.close()  # Close the figure to prevent display

    # # Move the buffer cursor to the beginning
    # buf.seek(0)

    # Return the in-memory image file
    return plt.show()