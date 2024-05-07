from data_ingest import IngestData

from data_processing import(
    find_constant_columns,
    find_columns_with_few_values,
    drop_and_fill,  
)
from feature_engineering import bin_to_num,cat_to_col,one_hot_encoding


ingest_data = IngestData()  # Create an instance
df = ingest_data.get_data("data/cancer_reg.csv")  # Call the get_data method with a data_path argument


constant_columns=find_constant_columns(df)
print("Columns that contain single value:",constant_columns)
columns_with_few_values=find_columns_with_few_values(df,10)

df["binnedInc"][0]
df=bin_to_num(df) 
df=cat_to_col(df)
df=one_hot_encoding(df)
df=drop_and_fill(df)
print(df.shape)
df.to_csv("data/cancer_reg_processed.csv",index=False)