"""

    Collaborative Filtering: Modeling Methods

"""

import random
import math
import time


def select_similarity_metric(similarity_metric):
    """
        Selecting the similarity metric that is more suited to the problem

        return a function of the choosen similarity metric
    """

    if similarity_metric == 'cosine':

       return measure_cosine_similarity

    assert "Unknow Similarity Measure"


def measure_cosine_similarity(tokens_one, tokens_two):
    """
        Calculating the cosine similarity between two arrays

        return a float value between -1 and 1
    """

    # measuring the numerator step
    numerator = sum(list(map(lambda a, b: a * b, tokens_one, tokens_two)))

    # denominator from the first token
    denominator_one = math.sqrt(sum(list(map(lambda a: a**2, tokens_one))))

    # denominator from the second token
    denominator_two = math.sqrt(sum(list(map(lambda a: a**2, tokens_two))))

    if denominator_one == 0 or denominator_two == 0:

        return 0

    return numerator/(denominator_one * denominator_two)


def measure_similarity(tokens_one, tokens_two, similarity_metric='cosine'):
    """
        Receives as input two arrays with tokens (items or users) ratings

        Example:

            If the modeling is by user (The rating that a user has given to itens): 

                User -> 3, 4, 1

            If the modeling is by item (The Rating that a item has receive from differnent users):
                
                item -> 9, 4, 3, 4, 0

        return a float value
    """

    similarity_method = select_similarity_metric(similarity_metric)

    value = similarity_method(tokens_one, tokens_two)

    return value


def retrieve_neighbors(matrix, token_index, other_tokens, similarity_metric='cosine'):
    """

        Given a item or user, retrieve the closest neighbors

    """

    similarities = {}

    for token, index in other_tokens.items():

        similarities[token] = measure_similarity(matrix[token_index], matrix[index], similarity_metric)

    # sorting a dictionary by values, we want reverse 
    return {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)}


def get_rating_based_on_closest_items(similarities, user_historic, user, mean, amount_neighbors=10):

    similarity_sum, similarity_rating, index = 0, 0, 0

    for item in similarities.keys(): # a dictionary user - items - rating

        # what is the rating that my user has given to that similar item ?

        if item not in user_historic[user].keys():

            # if the item is not in the user historic, there is no way to measure it rating

            continue

        rating = user_historic[user][item]

        similarity_rating += similarities[item] * rating 

        similarity_sum += similarities[item]

        if index == amount_neighbors:

            break

        index += 1


    if similarity_sum == 0:

        return mean

    return similarity_rating/similarity_sum


def retrieve_unique_tokens(data):
    """
        Retrieve the unique tokens to model the matrix

        return a dictionary of token and its unique items

    """

    tokens = {"users": list(set(list(map(lambda row: row[0], data)))),
              "items": list(set(list(map(lambda row: row[1], data))))}

    return tokens


def define_prediction_features(prediction_data, modeling):

    tokens = retrieve_unique_tokens(prediction_data)

    # a dictionary of the modeling tokens a empty lists
    tokens_info = dict(zip(tokens[modeling], [[]] * len(tokens[modeling])))

    if modeling == 'items':

        keys_index, values_index = 1, 0

    else:

        keys_index, values_index = 0, 1

    for row in prediction_data:

        tokens_info[row[keys_index]].append(row[values_index])

    return tokens_info


def measure_average_rating(data):


    ratings_sum = 0

    for row in data:

        ratings_sum += float(row[2])

    return ratings_sum/len(data)


def generate_historic_data_matrix(historic_data, modeling, users, items, fill_zero=0):
    """
        Modeling the matrix of historical data

        Modeling:

            Define if the matrix generate will be a item x item or user x user modeling


    """

    # making a matrix of zeros
    if modeling == 'items':

        matrix = [[0] * len(users) for row in range(0, len(items))]

    else:

        matrix = [[0] * len(items) for row in range(0, len(users))]


    for row in historic_data:

        user = users[row[0]]
        item = items[row[1]]

        # rating given by the user
        rating = int(row[2])

        if rating == 0:

            rating = fill_zero

        if modeling == 'items':

            matrix[item][user] = rating

        else:

            matrix[user][item] = rating

    return matrix


def retrieve_guide_features(historic_data):

    tokens = retrieve_unique_tokens(historic_data)

    users_items = {user: [] for user in tokens['users']}

    # ratings, amount [0], [0]
    items_ratings = dict(zip(tokens['items'], [[0, 0]] * len(tokens['items'])))
    users_ratings = dict(zip(tokens['users'], [[0, 0]] * len(tokens['users'])))

    for row in historic_data:

        users_items[row[0]].append(row[1])

        items_ratings[row[1]][0] += float(row[2])

        users_ratings[row[0]][0] += float(row[2]) # summing the raitings

        items_ratings[row[1]][1] += 1

        users_ratings[row[0]][1] += 1


    for user in users_ratings.keys():

        users_ratings[user] = users_ratings[user][0]/users_ratings[user][1]

    for item in items_ratings.keys():

        items_ratings[item] = items_ratings[item][0]/items_ratings[item][1]


    return users_items, define_index(tokens['users']), define_index(tokens['items']), users_ratings, items_ratings



def define_user_item_rating(historic_data):

    tokens = retrieve_unique_tokens(historic_data)

    users_ratings = dict(zip(tokens['users'], [{} for _ in range(len(tokens['users']))]))

    for row in historic_data:

        users_ratings[row[0]][row[1]] = float(row[2])

    return users_ratings


def define_index(tokens):

    return {token: index for index, token in enumerate(tokens)}


def measure_ratings_by_nearest_neighbors(data, modeling='items'):

    users_items, users, items, users_ratings, items_ratings = retrieve_guide_features(data['Historic Data'])

    # a matrix users x items
    historic_rating_matrix = generate_historic_data_matrix(data['Historic Data'], modeling, users, items)

    # prediction data
    modeling_tokens = define_prediction_features(data['Prediction Data'], modeling)

    ratings_mean = measure_average_rating(data['Historic Data'])

    users_ratings = define_user_item_rating(data['Historic Data'])

    prediction_users = []

    for user in modeling_tokens.values():

        prediction_users.extend(user)

    predictions = dict(zip(prediction_users, [{} for _ in prediction_users]))

    for token, token_values in modeling_tokens.items():

        if modeling == 'items':

            if token in items:

                similarities = retrieve_neighbors(historic_rating_matrix, items[token], items)

            else:

                for user in token_values:

                    predictions[user][token] = ratings_mean

            for user in token_values:

                if user not in users.keys() or token not in items.keys():

                    predicted_rating = ratings_mean

                else:

                    predicted_rating = get_rating_based_on_closest_items(similarities, users_ratings, user, ratings_mean)

                predictions[user][token] = predicted_rating

        elif modeling == 'users':

            predictions = retrieve_neighbors(historic_rating_matrix, users[token], users)


    return user_historic


def read_table(file_input, sep=':', replace_char=None):
    """
        Read table with data

        return a list of lists
    """

    with open(file_input, 'r') as read_input:

        data = read_input.read()

        if replace_char:

            data = data.replace(replace_char, sep)

        data = data.split('\n')

        # removing the header
        data.pop(0)

    data = list(map(lambda row: row.split(sep), data))

    return random.sample(list(filter(lambda row: row[0] != '', data)), 10000)
    #return list(filter(lambda row: row[0] != '', data))


def mean_squared_error(real, predicted):
    """
        Measure the root mean square between the real and predicted values
        Params:
            two arrays of lenght n, where the predicted value is given by a algorithm
            and the real value is retrieved from the historic dataset
        return a float value
    """

    if type(predicted) in [float, int] and type(real) in [float, int]:

        return math.sqrt((predicted - real) ** 2)

    if len(predicted) != len(real):

        assert "Predicted and Real arrays most have the same lenght !"


    return math.sqrt(sum(list(map(lambda y_pred, y_real: (y_pred - y_real)** 2, predicted, real)))/len(predicted))

if __name__ == '__main__':


    input_arguments = {"Historic Data": read_table("Data/train.csv", ';'),
                       "Prediction Data": read_table("Data/test.csv", ';')}


    output_file = "predictions.txt"

    start = time.time()

    with open("Data/time_reports.csv", "a+") as time_report:

        input_arguments['Prediction Data']['Y Predicted'] = measure_ratings_by_nearest_neighbors(input_arguments, modeling='items')

        print("The Final RMSE is: ", mean_squared_error(input_arguments['Prediction Data']['Prediction'],
                                                        input_arguments['Prediction Data']['Y Predicted']))

        time_report.write('\t'.join([time.time() - start]) + '\n')


    end = time.time()

    print(end - start)
