

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from Aux_Function.PCAM import *

def linear_regression(X, y, step, initial_guess=None, error_diff=20,max_iter=1000):
    """
    Performs linear regression using gradient descent to minimize the squared error.

    Parameters:
    ----------
    X : pd.DataFrame or np.ndarray
        The feature matrix where each row is a sample and each column is a feature.
    y : pd.Series or np.ndarray
        The target vector (true values) corresponding to each sample in X.
    step : float
        The step size (learning rate) for gradient descent, controlling the magnitude of updates.
    initial_guess : np.ndarray, optional
        The initial guess for the coefficients (weights). If not provided, it defaults to an array of zeros.
    error_diff : float, optional
        Convergence criterion. The algorithm stops when the change in error between iterations is less than this value. Defaults to 20.
    max_iter: int,  optional
        Iterations of algorithm. Set the max number of iterations.
    Returns:
    -------
    co : list of np.ndarray
        A list containing the history of the coefficients over the iterations.
    be : list of float
        A list containing the history of the intercepts over the iterations.

    Raises:
    ------
    ValueError: 
        If the provided step size is non-positive or other invalid inputs are encountered.

    Description:
    ------------
    This function implements linear regression using the gradient descent optimization technique. It attempts 
    to find the best-fitting linear model by minimizing the squared error between the predicted values and the 
    actual target values. The function stops when the error difference between consecutive iterations falls 
    below `error_diff` or when the algorithm completes 1000 iterations.

    The function also provides a real-time plot of the error over iterations, allowing users to visually monitor 
    the convergence process.

    Example:
    --------
    >>> X = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    >>> y = pd.Series([1, 2, 3])
    >>> co, be = linear_regression(X, y, step=0.001)

    This will perform gradient descent on the provided data and return the coefficient and intercept history.
    """

    X = X.to_numpy()  # Convert feature matrix to NumPy array if it's a DataFrame
    y = y.to_numpy()  # Convert target vector to NumPy array if it's a Series
    y = y.reshape(-1, 1)  # Ensure y is a column vector

    # Get the number of samples (m) and number of features (n)
    m, n = np.shape(X)

    # Initialize the intercept (bias) to 0
    b = 0

    # Initialize the coefficients (weights), either with zeros or the given initial guess
    if initial_guess is None:
        coeficients = np.zeros((n, 1))
    else:
        coeficients = initial_guess

    # Initial predictions and error calculation
    predictions = X @ coeficients + b
    error = np.sum((y - predictions) ** 2)
    stop = 1e-2  # Convergence threshold for error
    co = [coeficients]  # Store coefficient history
    be = [b]  # Store intercept history

    # Set up the interactive plot for error over iterations
    plt.ion()
    plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error Over Iterations')

    iterations = 0
    errors = []  # To store error values over iterations
    prev_error = error  # Store previous error for convergence checking

    # Gradient descent loop
    while error > stop:
        error_vector = y - predictions  # Calculate the error vector

        # Compute the gradients for coefficients and intercept
        diff_m = -2/m * (X.T @ error_vector)
        diff_b = -2/m * np.sum(error_vector)

        # Update the coefficients and intercept using the gradients
        coeficients -= step * diff_m
        b -= step * diff_b

        # Store updated values
        co.append(coeficients)
        be.append(b)

        # Make new predictions with the updated coefficients and intercept
        predictions = X @ coeficients + b
        error = np.sum((y - predictions) ** 2)
        errors.append(error)

        # Update the plot with new error
        plt.plot(errors, color='blue')
        plt.xlim(0, len(errors))  # Adjust x-limits
        plt.ylim(0, max(errors) * 1.1)  # Adjust y-limits
        plt.pause(0.01)  # Pause to update the plot

        # Convergence check: stop if the change in error is less than the threshold
        if abs(prev_error - error) < error_diff:
            print(f"Convergence reached after {iterations} iterations.")
            print(f'Coefficients: {coeficients}')
            print(f'Intercept: {b}')
            break

        # Update the previous error for the next iteration
        prev_error = error

        iterations += 1
        # Stop after 1000 iterations if convergence hasn't been reached
        if iterations == max_iter:
            print("Max iterations reached.")
            print(f'Coefficients: {coeficients}')
            print(f'Intercept: {b}')
            break

    return co, be

def mae(y,pred):
     y = y.to_numpy()
     y = y.reshape(-1, 1)
     pred = pred.to_numpy()
     pred = pred.reshape(-1, 1)
     return 1/len(y)*np.sum(np.abs(y-pred))

sns.set_theme(style="whitegrid")

df = pd.read_csv('Linear Regression/Ecommerce_Customers.csv')

features_num = ['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership', 'Yearly Amount Spent']

df_regresion = df[features_num]
df_regresion_stand = (df_regresion - df_regresion.mean())/df_regresion.std()

target = df_regresion.pop('Yearly Amount Spent')
target_stand = df_regresion_stand.pop('Yearly Amount Spent')

X_train, X_test, y_train, y_test = train_test_split(df_regresion,target,test_size=0.3,random_state=42)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(df_regresion_stand,target_stand,test_size=0.3,random_state=42)

lr = LinearRegression(fit_intercept=False)
lr.fit(X_train,y_train)

print('Sklearn function MAE',mean_absolute_error(y_test,lr.predict(X_test)))

coef,b = linear_regression(X_train,y_train,0.00001,error_diff=0.1,initial_guess=np.array([[11.9],[34.7],[-14.1],[60.5]]))

y_pred = X_test@coef[-1] + b[-1]
y_pred = pd.DataFrame(y_pred)

print('My function MAE',mae(y_test,y_pred))

plt.ioff() 
plt.figure()
sns.regplot(x=y_pred, y=y_test, data=df, color='blue')
plt.show()


### Data Analysis

max_spent = df['Yearly Amount Spent'].max()
min_spent = df['Yearly Amount Spent'].min()

corr_matrix = df[features_num].corr()

columns = df[features_num].columns

mean = df[features_num] - np.mean(df[features_num] , axis=0)

cov_matrix = (mean.T@mean)/(np.shape(mean)[0])

tab_1,data_pca,f_1 = pca(df[features_num],corr_matrix,columns)
tab_2,data_fa,f_2 = fa(df[features_num],corr_matrix,columns)
print('tabla:','\n',tab_2)
print(f_2)

print('Describe')
print(df.describe())
plt.figure(figsize=(8, 6))

ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
            center=0,  # Center the colormap at 0 for better visualization
            fmt='.2f',  # Format of the annotations
            linewidths=0.5,  # Line width of the grid lines
            linecolor='gray',  # Color of the grid lines
            square=True
            ) # Make cells square-shaped)
ax.tick_params(axis='both', which='major', labelsize=8)  # Ajusta el tamaño de las etiquetas de los ejes

plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()

df['Max_Spent'] = 0
df.loc[df['Yearly Amount Spent'] >= (2*((max_spent - min_spent) / 4)+min_spent), 'Max_Spent'] = 1

sns.scatterplot(x=data_fa['F_1'], y=data_fa['F_2'], hue=df['Max_Spent'])

#Attempt to improve the regression model with another features

df_regresion.loc[:, 'PC_1'] = data_pca['PC_1'].values
df_regresion.loc[:, 'PC_2'] = data_pca['PC_2'].values

print(df_regresion.head(5))

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(df_regresion,target,test_size=0.3,random_state=42)

lr_n = LinearRegression(fit_intercept=False)
lr_n.fit(X_train_new,y_train_new)

print('Sklearn function MAE',mean_absolute_error(y_test_new,lr_n.predict(X_test_new)))




