import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from Logging.Logging import output_log
import math


class NaiveBayes:
    def __init__(self):
        self.train_mean = None
        self.train_sd = None
        self.class1_mean = None
        self.class1_sd = None
        self.class2_mean = None
        self.class2_sd = None

    def dnorm(self, x, mu, sd):
        return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

    def calculate_mean_sd(self, X):
        self.train_mean = np.mean(X, axis=0)
        self.train_sd = np.std(X, axis=0)

    def normalize(self, X):
        train_scaled = (X - self.train_mean) / self.train_sd
        return train_scaled

    def fit(self, X, y):
        self.calculate_mean_sd(X)
        train_scaled = self.normalize(X)

        X_class1 = train_scaled[np.where(y == 0)]
        X_class2 = train_scaled[np.where(y == 1)]

        self.class1_mean = np.mean(X_class1, axis=0)
        self.class1_sd = np.std(X_class1, axis=0)

        self.class2_mean = np.mean(X_class2, axis=0)
        self.class2_sd = np.std(X_class2, axis=0)

        self.class1_prior = X_class1.shape[0] / X.shape[0]
        self.class2_prior = X_class2.shape[0] / X.shape[0]

    def predict(self, X):
        test_scaled = self.normalize(X)

        len = test_scaled.shape[0]

        prediction = np.zeros([len])

        for row in range(len):

            log_sum_class1 = 0
            log_sum_class2 = 0

            for col in range(test_scaled.shape[1]):
                log_sum_class1 += math.log(self.dnorm(test_scaled[row, col], self.class1_mean[col], self.class1_sd[col]))
                log_sum_class2 += math.log(self.dnorm(test_scaled[row, col], self.class2_mean[col], self.class2_sd[col]))

            log_sum_class1 += math.log(self.class1_prior)
            log_sum_class2 += math.log(self.class2_prior)

            if log_sum_class1 < log_sum_class2:
                prediction[row] = 1

        return prediction

    def accuracy(self, y, prediction):
        accuracy = (prediction == y).mean()
        return accuracy * 100


if __name__ == '__main__':
    iris = sns.load_dataset("iris")
    iris = iris.loc[iris["species"] != "setosa"]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris["species"])
    X = iris.drop(["species"], axis=1).values

    train_accuracy = np.zeros([100])
    test_accuracy = np.zeros([100])

    for loop in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=loop)

        model = NaiveBayes()
        model.fit(X_train, y_train)
        prediction = model.predict(X_train)
        train_accuracy[loop] = model.accuracy(y_train, prediction)
        prediction = model.predict(X_test)
        test_accuracy[loop] = model.accuracy(y_test, prediction)

    output_log("Average Train Accuracy {}%".format(np.mean(train_accuracy)))
    output_log("Average Test Accuracy {}%".format(np.mean(train_accuracy)))
