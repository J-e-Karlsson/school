import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def get_regression(path,x_par,y_par):
    with open(path, "r") as file:
        df = pd.read_csv(path,sep=",", on_bad_lines="skip")
        sns.regplot(
                data=df, x=x_par, y=y_par,
                scatter_kws = {"s": 10},
                line_kws = {"color": "blue", "linewidth": 0.8})
        #regplot = line plot with regression, scatter = size of points
        #line_kws = color and width of line
        plt.xlabel(f"{x_par}")#name of x
        plt.ylabel(f"{y_par}")#name of y
        plt.savefig("scatterplot.png")#save as scatterplot.png
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_par], df[y_par])
        #assigns 5 float values from linregress to corresponding variable before =
        return {
            "Intercept": intercept,
            "Coefficient": slope,
            "R-squared": r_value**2,
            "P_value": p_value
        }

def main():
    run = get_regression(sys.argv[1], sys.argv[2], sys.argv[3]) #[1] = x(age), [2] = y(time)
    return run
    #saves the figure and return the values except for std_err which we don't need here

print(main())

