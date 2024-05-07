from sklearn.model_selection import train_test_split as tts

def find_constant_columns(dataframe):
    constant_columns=[]
    for column in dataframe.columns:
        unique_values=dataframe[column].unique()
        if len(unique_values)==1:
            constant_columns.append(column)
    return constant_columns 

def delete_constant_columns(dataframe,columns_to_delete):
    dataframe=dataframe.drop(columns_to_delete,axis=1)
    return dataframe

def find_columns_with_few_values(dataframe,threshold):
        few_values_columns=[]
        for column in dataframe.columns:
             unique_values_count=len(dataframe[column].unique())
             if unique_values_count<threshold:
                  few_values_columns.append(column)
        return few_values_columns


def find_duplicate_rows(dataframe):
     duplicate_rows=dataframe[dataframe.duplicated()]
     return duplicate_rows
def delete_duplicate_rows(dataframe):
     dataframe=dataframe.drop_duplicates(keep="first")
     return dataframe

def drop_and_fill(dataframe):
     cols_to_drop=dataframe.columns[dataframe.isnull().mean() > 0.5]
     dataframe=dataframe.drop(cols_to_drop,axis=1)
     dataframe=dataframe.fillna(dataframe.mean())
     return dataframe

from sklearn.model_selection import train_test_split

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataframe into training and testing sets.

    Parameters:
    - df (DataFrame): Input dataframe.
    - target_column (str): Name of the target column.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Seed for random number generator.

    Returns:
    - x_train (DataFrame): Training features.
    - x_test (DataFrame): Testing features.
    - y_train (Series): Training target.
    - y_test (Series): Testing target.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return x_train, y_train, y_test
                  