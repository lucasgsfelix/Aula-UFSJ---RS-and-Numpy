"""

    Collaborative Filtering: Modeling Methods

"""

import random
import math
import time

import numpy as np
import pandas as pd

from operator import itemgetter


from sklearn.metrics import mean_squared_error


def measure_cosine_similarity(tokens_one, tokens_two):
    """
        Calculating the cosine similarity between two arrays

        return a float value between -1 and 1
    """

    return (np.dot(tokens_one, tokens_two)/(np.linalg.norm(tokens_one) * np.linalg.norm(tokens_two)))


def get_rating_based_on_closest_items(similarity, adj_matrix, user, amount_neighbors=10):

    similarity_sum, similarity_rating, index = 0, 0, 0

    # the index order that sort that array
    user_historic = np.where(adj_matrix[user] != 0)

    # getting the biggest indexes where the rating is not 0
    indexes = np.intersect1d(np.where(similarity != 0), user_historic)

    sorted_indexes = [x for _, x in sorted(zip(similarity[indexes], indexes), key=lambda pair: pair[0], reverse=True)]

    return (np.sum(adj_matrix[user][sorted_indexes] * similarity[sorted_indexes]))/np.sum(similarity[sorted_indexes])


def retrieve_values_by_list(attributes_dict, list_keys):
    """
        Get the attributes of a dictionary given a list of keys

    """

    return list(itemgetter(*list_keys)(attributes_dict))


def measure_items_similarity(data, adj_matrix, index_info):
    """

        Mount a similarity matrix between all tokens in the 

    """

    historic_items = data['Historic Data']['ItemId'].unique()
    prediction_items = data['Prediction Data']['ItemId'].unique()

    # retangular matrix
    similarities_matrix = np.zeros((len(prediction_items), len(historic_items)))

    print("Measure the distance matrix !")

    #distance_matrix = np.matmul(items_matrix, items_matrix.T)

    for item in prediction_items:

        # users that want a rating in that specific item
        users = data['Prediction Data'][data['Prediction Data']['ItemId'] == item]['UserId'].unique()

        # getting the index of the users index
        historic_index = data['Historic Data'][data['Historic Data']['UserId'].isin(users)]['ItemId'].unique()

        #historic_index = unique_items

        # return the index of the elements that are zero, i.e, where not filled yet
        zero_indexes = np.argwhere(similarities_matrix[item] == 0).flatten()

        # intersection == common values
        # intersection between the indexes with zero
        # and the historic indexes that we want to measure
        historic_index = np.intersect1d(zero_indexes, historic_index)

        # in this case all values are covered
        if len(historic_index) == 0:

            continue

        # measuring the cosine distance between one item and all other
        similarities = np.array(list(map(lambda index: measure_cosine_similarity(adj_matrix[index], adj_matrix[item]), historic_index)))

        similarities_matrix[item, historic_index] = similarities

        # now the indexes have inverted but our matrix is not quadratic
        # we have to identify where is the data
        
        '''
        historic_index = retrieve_values_by_list(index_info['test items'], historic_index)

        if item in index_info['train item']:

            item = index_info['train item'][item]

            similarities_matrix[historic_index, item] = similarities

        '''


    return similarities_matrix


def generate_historic_data_matrix(df, modeling, fill_zero=0):
    """
        Modeling the matrix of historical data

        Modeling:

            Define if the matrix generate will be a item x item or user x user modeling


    """

    places_info = {}

    for token in set(df['ItemId']):


        model_df = df[df['ItemId'] == token]

        # Place Name: All Visitors of that profile and the rating
        places_info[token] = dict(zip(model_df['UserId'], model_df['Prediction']))

    df = pd.DataFrame(places_info).fillna(fill_zero)

    if modeling == 'items':

        return df.values

    elif modeling == 'user':

        return df.transpose().values

    return df


def measure_ratings_by_nearest_neighbors(data, index_info, modeling='items'):

    # a matrix users x items
    historic_rating_matrix = generate_historic_data_matrix(data['Historic Data'], modeling)

    means = {}

    # making a dict of token: rating - user mean and item mean
    means['user'] = data['Historic Data'].groupby("UserId").mean('Prediction')['Prediction'].to_dict()
    means['item'] = data['Historic Data'].groupby("ItemId").mean('Prediction')['Prediction'].to_dict()

    similarities = measure_items_similarity(data, historic_rating_matrix, index_info)

    predictions = []

    for item, user in data['Prediction Data'][['ItemId', 'UserId']].values:

        similarity = similarities[item]

        if user not in index_info['train users']: # cold start user

            predicted_rating = means['item'][item]

        elif item not in index_info['train items']: # cold start item

            predicted_rating = means['user'][user]

        else:

            predicted_rating = get_rating_based_on_closest_items(similarity, historic_rating_matrix, user)

            if np.isnan(predicted_rating):

                predicted_rating = (means['item'][item] + means['user'][user])/2

        predictions.append(predicted_rating)

    return predictions


def read_table(file_input, sep=':'):
    """
        Read table with data

        return a list of lists
    """

    return pd.read_table(file_input, sep=sep, engine='python')



def index_data(data):
    """

        Indexing all the data

        Transforming all the values to number

    """

    index_info = {}

    index_keys = {
                    'train items': 'ItemId',
                    'train users': 'UserId',
                    'test items': 'ItemId',
                    'test users': 'UserId'
                 }


    for key, value in index_keys.items():

        data_token = 'Prediction'

        if 'train' in key:

            data_token = 'Historic'


        # identifying the unique tokens
        unique_values = data[data_token + " Data"][value].unique()

        # making a index for each one of them
        replace_dict = {value: index for index, value in enumerate(unique_values)}

        # we will need this later
        index_info[key] = {index: value for index, value in enumerate(unique_values)}

        # replacing the value so it can be indexed
        data[data_token + ' Data'][value] = data[data_token + ' Data'][value].apply(lambda token_id: replace_dict[token_id])


    return data, index_info



if __name__ == '__main__':


    input_arguments = {"Historic Data": read_table("Data/train.csv", ';'),
                       "Prediction Data": read_table("Data/test.csv", ';')}


    input_arguments, index_info = index_data(input_arguments)

    output_file = "predictions.txt"

    start = time.time()

    with open("Data/time_reports.csv", "a+") as time_report:

        input_arguments['Prediction Data']['Y Predicted'] = measure_ratings_by_nearest_neighbors(input_arguments, index_info, modeling='items')

        print("The Final RMSE is: ", mean_squared_error(input_arguments['Prediction Data']['Prediction'],
                                                        input_arguments['Prediction Data']['Y Predicted'], squared=False))

        time_report.write('\t'.join([str(time.time() - start)]) + '\n')


    end = time.time()

    print(end - start)
