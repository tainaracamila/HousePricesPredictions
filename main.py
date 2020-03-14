from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data_visualization import Graphs
if __name__ == '__main__':

    # Loading boston database
    boston = load_boston()
    model = LinearRegression()
    gp = Graphs()

    """
        Dimension of the database loaded (rows, columns):
            print(boston.data.shape)
    
        Description:
            print(boston.DESCR)
    
        Attributes names:
            print(boston.feature_names)
    
        Target:
            print(boston.target)
    """

    # Transforming data
    df = pd.DataFrame(boston.data)
    # Add columns names
    df.columns = boston.feature_names
    # Add column prices
    df['PRICES'] = boston.target
    # print(df.head())

    # Data
    X = df.drop('PRICES', axis=1)
    Y = df.PRICES

    # Slicing between train and test
    # test_size = % data for tests
    x, x_test, y, y_test = train_test_split(X, Y, test_size=0.3, random_state=5)
    # print(x.shape, x_test.shape) # 70% train and 30% tests

    # Using only train test
    model.fit(x, y)

    # prediction train
    prd_train = model.predict(x)
    # prediction test
    prd_test = model.predict(x_test)

    gp.double_scatter(prd_train, prd_test, y, y_test)

    print('\nCoefficient: %f' % model.intercept_)
    # Multiple Linear Regression Model = each attribute is a coefficient
    print('Number of coefficients: %f' % len(model.coef_))

    # MSE
    mse = np.mean((y_test - model.predict(x_test)) ** 2)
    print('\nMSE %f' % mse)




