import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

class dataLoader(object):
    def __init__(self,seed):
        random.seed(seed)
        self.temperature,self.other = self.split_temperature_column()
        self.temperature_col,self.other_col = self.temperature.columns,self.other.columns

        self.new_temperature,self.temperature_mean,self.temperature_std = self.get_columns_mean_std(self.temperature)
        self.new_other,self.other_mean, self.other_std = self.get_columns_mean_std(self.other)


    def split_dataset(self,x_len,y_len,train_size,val_size):
        n = len(self.new_temperature)
        valid_len = n - y_len - x_len
        index_list = random.sample(range(valid_len), valid_len)
        new_X = np.zeros(shape=(len(index_list),x_len,len(self.temperature_col)+len(self.other_col)))
        new_y = np.zeros(shape=(len(index_list),y_len,len(self.temperature_col)))
        for i,index in enumerate(index_list):
            X = np.concatenate([self.new_temperature[index:index+x_len,:],self.new_other[index:index+x_len,:]],axis=1)
            y = self.new_temperature[index+x_len:index+x_len+y_len,:]

            new_X[i] = X
            new_y[i] = y
        n_train = int(train_size * valid_len)
        n_val = int(val_size * valid_len)
        n_test = valid_len- n_train - n_val
        train_set = {}
        val_set = {}
        test_set = {}
        infer_set = {}
        print(new_X.shape)
        train_set['X'] = new_X[:n_train]
        train_set['y'] = new_y[:n_train]

        val_set['X'] = new_X[n_train:n_train+n_val]
        val_set['y'] = new_y[n_train:n_train+n_val]

        test_set['X'] = new_X[-n_test:]
        test_set['y'] = new_y[-n_test:]

        infer_set['X'] = np.expand_dims(np.concatenate([self.new_temperature[-x_len:,:],
                                         self.new_other[-x_len:,:]],axis=1),axis=0)
        return train_set,val_set,test_set,infer_set

    def split_temperature_column(self):
        file_name = 'new_data2.xlsx'
        df = pd.read_excel(file_name)
        df = df[~pd.isnull(df).any(axis=1)]
        df = df.drop(['时间'], axis=1)

        temperature = [name for name in list(df.columns) if '温度' in name]
        other = [name for name in list(df.columns) if '温度' not in name]
        temperature_data = df[temperature].astype(np.float64)
        other_data = df[other].astype(np.float64)

        return temperature_data,other_data

    def get_columns_mean_std(self,df):

        fit_scaler = StandardScaler().fit(df)
        new_df = fit_scaler.transform(df) #numpy
        mean = fit_scaler.mean_
        std = np.sqrt(fit_scaler.var_)
        return new_df,mean,std

def main():
    data_loader = dataLoader()
    train_set,val_set,test_set,infer_set = data_loader.split_dataset(x_len=10,y_len=10,train_size=0.98,val_size=0.01)
    print(train_set['X'].shape,train_set['y'].shape)
    print(val_set['X'].shape,val_set['y'].shape)
    print(test_set['X'].shape,test_set['y'].shape)
    print(infer_set['X'].shape)
if __name__ =='__main__':
    main()