import matplotlib.pyplot as plt

def plot_temperature(predict,true,mean,std):
    '''
    :param predict: shape:(y_len)
    :param true:
    :param mean:
    :param std:
    :return:
    '''
    true_value = (true * std + mean)[-100:]
    predict_value = predict * std + mean
    print(predict_value)
    len1 = len(true_value)
    len2 = len(predict_value)
    plt.plot(range(len1),true_value,c='blue')
    plt.plot(range(len1,len1+len2),predict_value,c='red')
    plt.show()


def main():
    pass
if __name__ == '__main__':
    main()