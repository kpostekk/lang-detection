import pandas as pd
import numpy as np

class_codes = {
    'Iris-versicolor': 0,
    'Iris-virginica': 1
}


def convert_result(result: int | str):
    if type(result) == str:
        return class_codes[result]

    if type(result) == int:
        for key, value in class_codes.items():
            if value == result:
                return key

    raise ValueError(f'Invalid result type: {type(result)}')


def load_dataframes():
    # Load the dataframes
    raw_df_train = pd.read_csv('data/train_set.csv', header=None)
    raw_df_test = pd.read_csv('data/test_set.csv', header=None)

    train_data = pd.DataFrame(columns=['classname', 'vector'])
    train_data['vector'] = [col for col in raw_df_train.iloc[:, :-1].values]
    train_data['classname'] = [col for col in raw_df_train.iloc[:, -1].values]

    test_data = pd.DataFrame(columns=['classname', 'vector'])
    test_data['vector'] = [col for col in raw_df_test.iloc[:, :-1].values]
    test_data['classname'] = [col for col in raw_df_test.iloc[:, -1].values]

    vector_length = len(test_data['vector'][0])

    return test_data, train_data, vector_length


class Perceptron:
    def __init__(self, weights, threshold, learning_rate):
        self.weights = np.array(weights)
        self.threshold = threshold
        self.learning_rate = learning_rate

    def predict(self, vector):
        prediction_sum = sum(
            [self.weights[i] * vector[i] for i in range(len(self.weights))]
        )
        return prediction_sum

    def train(self, vector, expected: int):
        prediction = self.predict(vector) >= self.threshold

        if prediction == expected:
            return

        w = np.append(self.weights, self.threshold)
        x = np.append(vector, -1)
        d = expected
        y = prediction
        w_prime = w + self.learning_rate * (d - y) * x
        self.weights = w_prime[:-1]
        self.threshold = w_prime[-1]

    def __str__(self):
        return f'Perceptron(\n\t{self.weights=} \n\t{self.threshold=} \n\t{self.learning_rate=}\n)'

    def __repr__(self):
        return f'<Perceptron {self.weights=} {self.threshold=} {self.learning_rate=}>'
