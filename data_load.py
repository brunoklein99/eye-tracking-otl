import pandas as pd
from sklearn.model_selection import train_test_split


def load_mpii_dataframes():
    df = pd.read_csv(r'F:\MPIIFaceGaze\MPIIFaceGaze_prepared\labels.csv')
    df = df.sample(frac=1, random_state=1)
    df_train_valid, df_test = train_test_split(df, random_state=1, test_size=0.15)
    df_train, df_valid = train_test_split(df_train_valid, random_state=1, test_size=0.1765)
    return df_train, df_valid, df_test
