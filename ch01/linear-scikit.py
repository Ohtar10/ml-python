import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

# Load the data
oecd_bli = pd.read_csv("/Users/ferro/trainings/ml-python/datasets/lifesat/oecd_bli_2015.csv", thousands=',')

# Pay attention to the different properties we can set while reading the file
gdb_per_capita = pd.read_csv("/Users/ferro/trainings/ml-python/datasets/lifesat/gdp_per_capita.csv",
                             thousands=',',
                             delimiter='\t',
                             encoding='latin1',
                             na_values="n/a")


def prepare_country_stats(oecd_bli, gdb_per_capita):
    """
    just prepares and format the data as needed for the analysis

    :param oecd_bli:
    :param gdb_per_capita:
    :return:
    """
    # filter those who are different from TOT in the inequality field
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    # Create a pivot table grouping by country and indicator showing the value
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    # Just rename the column called 2015 to GDP per capita
    gdb_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdb_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdb_per_capita, left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", "Life satisfaction"]].iloc[keep_indices]


# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdb_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y="Life satisfaction")

model = linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]] # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]
plt.show()

