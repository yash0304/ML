import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set working dir to the script's folder
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

def plot_actual_vs_predicted(y_test, y_pred):
        '''
        This method can be called for visulizing actual vs. predicted graph.
        It takes y_test and y_pred as arguments. 
        it will have y_test (actual values) on x axis and y_pred (predicted values) on y axis.
        '''
        plt.figure(figsize=(10,8))
        sns.scatterplot(x=y_test,y=y_pred)
        plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'--r',label = 'Ideal pred line')
        plt.title("Actual vs. predictions")
        plt.xlabel("Actual values")
        plt.ylabel("predicted values")
        plt.tight_layout()
        plt.show()

def plot_residuals(y_test, y_pred):
    '''
    This method can be called for vizualizing residuals from y_test and y_pred.
    good model will have a more points scattered around 0.
    '''
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals vs. Actual Values")
    plt.tight_layout()
    plt.show()