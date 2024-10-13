import pandas as pd

def highlight_differences(data, df_a, df_b, target_name):

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
                
                # # Modify the contents of df_b to display 'val_a -> val_b'
                # df_b.iat[row, col] = f'{val_a} -> {val_b}'
    
    return df_style
