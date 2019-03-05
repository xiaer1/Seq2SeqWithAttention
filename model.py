import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from process_data import dataLoader
from argparse import Namespace

linear_args = Namespace(
    seed = 1234,
    test_size = 0.2,
    degree = 2
)
seq2seq_args = Namespace(

)
class LinearModel(object):
    def __init__(self):

        data_loader = dataLoader()
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(data_loader.new_other,data_loader.new_temperature,
                                                      test_size=linear_args.test_size,random_state=linear_args.seed)
        self.y_std = data_loader.temperature_std

    def add_ploynomial(self,X, degree):
        tmp = X
        for i in range(2, degree + 1):
            X = np.concatenate([X, tmp ** i], axis=1)
        return X

    def train(self,x_train,y_train):
        self.lr = LinearRegression()
        x_train = self.add_ploynomial(x_train, degree=linear_args.degree)  # tune degree = 8 is best
        self.lr.fit(X=x_train,y=y_train)

    def test(self,x_train,x_test):
        x_train = self.add_ploynomial(x_train, degree=linear_args.degree)
        x_test = self.add_ploynomial(x_test, degree=linear_args.degree)
        predict_train = self.lr.predict(x_train)
        predict_test = self.lr.predict(x_test)
        return predict_train,predict_test

    def evaluation(self,predict_train,predict_test):
        train_mse = np.mean((self.y_train - predict_train)**2,axis=0) * self.y_std
        test_mse = np.mean((self.y_test - predict_test) ** 2, axis=0) * self.y_std

        plt.figure()
        plt.title('MSE of every temperature column(degree={0})'.format(linear_args.degree))
        plt.scatter(range(len(train_mse)),train_mse,color='red',label='train MSE',marker='+')
        plt.scatter(range(len(test_mse)), test_mse, color='blue', label='test MSE',marker='v')
        plt.xlabel('temperature column')
        plt.ylabel('MSE')
        plt.legend(loc='upper right')
        plt.show()


class Seq2SeqWithAttention(object):
    def __init__(self):
        pass


'''
LinearModel usage:
    linear_model = LinearModel()
    linear_model.train(linear_model.x_train,linear_model.y_train)
    predict_train,predict_test = linear_model.test(linear_model.x_train,linear_model.x_test)
    linear_model.evaluation(predict_train,predict_test)
'''
def main():
    pass
if __name__ == '__main__':
    main()
