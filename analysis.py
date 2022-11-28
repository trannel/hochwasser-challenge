import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# As a first step find a meaningful graphical representation to get a first grasp of the data and to detect anomalies or peculiarities that are easily observable.
# E.g. create a boxplot of the sensors data given by the 6 different sensor available; Plot the values of each sensor against the time parameter to get an idea how the time series data looks like.
# Could you possibly identify a time point in the data, where there was the Wupper-flood incident in 2021?
# Choose a visualization method to show the distribution of the sensor output data like a boxplot diagram
def box_plotter(data: pd.DataFrame):
    """
    Creates a simple boxplot over all columns in the dataframe
    """
    fig, ax = plt.subplots()
    ax.set_title('Sensor readouts - Boxplot')
    ax.boxplot(data, flierprops=dict(marker='.', markerfacecolor='black', markersize=4,
                                     linestyle='none', markeredgecolor='black'))

    ax.set_xticklabels(data.keys(), rotation=45, ha='right')
    plt.show()

# Basic statistical data
# Try to get some simple statistical values to describe the sensor data (e.g. mean, St. dev, quantiles).
# Explore further: Are there 'impossible' or just extremely unlikely values/entries in your data frame (e.g. negative values, where only positives are allowed or NaN entries)?
def describe(data: pd.DataFrame):
    data.describe()


# Add a function that checks all columns in a datafram for missing or negative values as those would clearly be erroneous
# information in the dataset
def data_check(dataset: pd.DataFrame):
    """
    A function that checks the dataframe for negative or missing values and prints

    """

    for i in range(dataset.shape[1]):
        data = np.array(dataset.iloc[0:, i])
        print(f'{dataset.keys()[i]}')
        if np.isnan(data).any() == True:
            print('Data contains NaN at indices: ' + str(np.argwhere(np.isnan(data))))
        else:
            print('Data contains no NaN')
        neg_value = []
        neg_value = data[data < 0]
        if len(neg_value) == 0:
            print('Data contains no negative values \n')
        else:
            print('Data contains negative values at indices:' + str(np.argwhere(data < 0)) + '\n')


# In this section try to plot each sensor output over time to get an impression how the signals behave and if you can spot any curiosities like trends or extreme signals visible by the bare eye.
# Plot to exemplary show one sensor signal over time
# fig, ax = plt.subplots()
#
# ax.set_title = 'Discharge, Stausee Beyenburg'
# ax.plot(np.arange(raw_data.shape[0]), raw_data.iloc[0:, 0])
# ax.set_xlabel('Time [h/2]')
# ax.set_ylabel('Discharge, Stausee Beyenburg')
#
# plt.show()

# Write a function to plot the output of each sensor on its own against the time parameter.
def single_plot(data: pd.DataFrame, counter: int):
    """
    The function allows to plot the output of a single sensor over time; the counter keyword is an integer calling the
    different columns of the data frame.
    """

    fig, ax = plt.subplots()

    ax.set_title = f'{data.keys()[counter]}'
    ax.plot(np.arange(data.shape[0]), data.iloc[0:, counter])
    ax.set_xlabel('Time [h/2]')
    ax.set_ylabel(f'{data.keys()[counter]}')

    plt.show()

# Try to plot all outputs of the sensors simultaneously, sharing the same time axis to compare them
# Write a function to plot all sensor outputs in seperate plot windows simultaneously but with a shared time axis.
def multi_singleplot(data: pd.DataFrame):
    """
    This function creates a single plot for every sensor output over the same time scale; it iterates automatically over all
    columns in the dataset printing the corresponding data with its title.
    """

    fig, ax = plt.subplots(data.shape[1], figsize=(15, 20), gridspec_kw={'hspace': 1})

    for i in range(data.shape[1]):
        ax[i].set_title(f'{data.keys()[i]}')
        ax[i].plot(np.arange(data.shape[0]), data.iloc[0:, i])
        ax[i].set_xlabel('Time [h/2]')
        ax[i].set_ylabel(f'{data.keys()[i]}')

    plt.show()

