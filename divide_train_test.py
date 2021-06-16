"""

    Collaborative Filtering: Modeling Methods

"""

import pandas as pd


from sklearn.model_selection import train_test_split


def read_table(file_input, sep=':'):
    """
        Read table with data

        return a list of lists
    """

    return pd.read_table(file_input, sep=sep, engine='python')

if __name__ == '__main__':
    

    df = read_table("Data/ratings.csv", '[,:]').sample(100000)


    train, test, _, _  = train_test_split(df, df['Prediction'], test_size=0.3)

    train.to_csv("Data/train.csv", sep=';', index=False)

    test.to_csv("Data/test.csv", sep=';', index=False)

