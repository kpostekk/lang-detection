import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from _perceptron import load_dataframes, Perceptron
from _utils import detect_degradation


def create_perceptron(lr=0.05):
    test_data, train_data, vl = load_dataframes()

    # Initialize the perceptron
    perceptron = Perceptron(
        weights=[np.random.random() for _ in range(vl)],
        threshold=0,
        learning_rate=lr
    )

    accuracy = 0
    learn_tracker = pd.DataFrame(
        {'accuracy': [0], 'weights': [perceptron.weights], 'threshold': [perceptron.threshold]})

    while not detect_degradation(learn_tracker['accuracy'].values):
        test_result = pd.DataFrame(columns=['expected', 'predicted'])

        for index, row in test_data.iterrows():
            predicted = perceptron.predict(row['vector'])
            test_result = pd.concat([
                test_result,
                pd.DataFrame({'expected': [row['classname']], 'predicted': [predicted]})
            ])

        accuracy = len(test_result[test_result['expected'] == test_result['predicted']]) / len(test_result)

        # Learn from the training data
        for index, row in train_data.sample(frac=1).iterrows():
            perceptron.train(row['vector'], row['classname'])
            # print(perceptron)

        learn_tracker = pd.concat([
            learn_tracker,
            pd.DataFrame({'accuracy': [accuracy], 'weights': [perceptron.weights], 'threshold': [perceptron.threshold]})
        ])
        learn_tracker.reset_index(drop=True, inplace=True)

    # get the best weights
    best_row = learn_tracker.iloc[learn_tracker['accuracy'].idxmax()]
    best_perceptron = Perceptron(
        weights=best_row['weights'],
        threshold=best_row['threshold'],
        learning_rate=lr
    )
    print(f'Best accuracy: {best_row["accuracy"]:.2f}')
    print(f'Best weights: {best_row["weights"]}')
    print(f'Best threshold: {best_row["threshold"]}')

    return best_perceptron, learn_tracker


@click.group()
def group():
    pass


@group.command(help='Run the perceptron with the default datasets and prints plot.')
@click.option('--lr', '-l', type=float, help='Learning rate.', required=True)
def stats(lr):
    perceptron, learn_tracker = create_perceptron(lr)
    print(perceptron)
    print(learn_tracker)
    learn_tracker.plot(
        y='accuracy',
        kind='line',
        use_index=True,
        title='Accuracy over generations',
        ylabel='Accuracy',
        xlabel='Generation'
    )
    plt.show()


@group.command(help='Creates perceptron and allows to test it.')
@click.option('--vector', '-v', type=str, help='Vector to test.', required=True)
@click.option('--lr', '-l', type=float, help='Learning rate.')
def test(vector, lr):
    perceptron, _ = create_perceptron(lr)
    vector = np.array(eval(vector))  # do not use this in production
    print(perceptron.predict(vector))


if __name__ == '__main__':
    group()
