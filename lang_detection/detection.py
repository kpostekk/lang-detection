from lang_detection.dataloader import load_texts, get_letter_freq, sanitize
from perceptron import Perceptron
import numpy as np
import json


class LangDetector:
    def __init__(self, lr=0.1, threshold=0.5):
        self.learning_rate = lr
        self.train_data = load_texts('dane/*.txt')
        self.test_data = load_texts('dane/test/*.txt')
        self.perceptrons = {
            lang: Perceptron([np.random.random() for _ in range(26)], lr, threshold) for lang, _ in
            self.train_data.groupby('lang')
        }

    def train(self):
        for lang, perceptron in self.perceptrons.items():
            for index, row in self.train_data.iterrows():
                perceptron.train(row['vector'], int(row['lang'] == lang))

    # def train_multiprocess(self):
    #     from multiprocessing import Pool
    #
    #     # create process for each language
    #     with concurrent.futures.ProcessPoolExecutor(12) as executor:
    #         def target(lang, perceptron):
    #             for index, row in self.train_data.iterrows():
    #                 perceptron.train(row['vector'], int(row['lang'] == lang))
    #                 print(f'{lang}: {perceptron.weights})')
    #
    #         executor.map(target, self.perceptrons.items())

    def prompt(self, text):
        prompt_vector, _ = get_letter_freq(sanitize(text))
        print(f'Prompt: {text}')

        predictions = [(lang, perceptron.predict(prompt_vector)) for lang, perceptron in self.perceptrons.items()]

        for lang, v in predictions:
            print(f' {lang}: {v:.4f}')

        return max(predictions, key=lambda x: x[1])

    def as_dict(self):
        return {lang: [list(perceptron.weights), perceptron.threshold] for lang, perceptron in self.perceptrons.items()}

    @staticmethod
    def from_dump(dump):
        detector = LangDetector()
        detector.perceptrons = {lang: Perceptron(weights, threshold, detector.learning_rate) for
                                lang, (weights, threshold) in dump.items()}
        return detector


def load_detector(model_name):
    import json

    with open(model_name, 'r') as f:
        dump = json.load(f)

    return LangDetector.from_dump(dump)


def pretrain(e=10):
    detector = LangDetector()

    for i in range(e):
        detector.train()

    with open(f'lang_detect_e{e}.json', 'w') as f:
        json.dump(detector.as_dict(), f, indent=2)

    return detector


if __name__ == '__main__':
    pretrain(5)
    pretrain(10)
    pretrain(20)
    pretrain(50)
    pretrain(100)
    pretrain(200)
    pretrain(300)
    # x = load_detector('lang_detect_e125.json')
    # x.prompt('This sentence should be a little longer!')
    # x.prompt('To zdanie nie powinno być krótkie!')
