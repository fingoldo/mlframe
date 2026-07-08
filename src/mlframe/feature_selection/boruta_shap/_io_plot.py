"""BorutaShap reporting / IO / plotting helpers.

CSV export, the box-plot of historical feature importances, and the
feature->decision colour-mapping utilities. Bound onto ``BorutaShap`` from the
parent module bottom. These read only fitted ``self.*`` history state and call
sibling methods through ``self``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def results_to_csv(self, filename="feature_importance"):
    """
    Saves the historical feature importance scores to csv.

    Parameters
    ----------
    filename : string
        used as the name for the outputted file.

    Returns
    -------
    comma delimnated file

    """

    features = pd.DataFrame(
        data={
            "Features": self.history_x.iloc[1:].columns.values,
            "Average Feature Importance": self.history_x.iloc[1:].mean(axis=0).values,
            "Standard Deviation Importance": self.history_x.iloc[1:].std(axis=0).values,
        }
    )

    decision_mapper = self.create_mapping_of_features_to_attribute(maps=["Tentative", "Rejected", "Accepted", "Shadow"])
    features["Decision"] = features["Features"].map(decision_mapper)
    features = features.sort_values(by="Average Feature Importance", ascending=False)

    features.to_csv(filename + ".csv", index=False, encoding="utf-8")


def plot(self, X_rotation=90, X_size=8, figsize=(12, 8), y_scale="log", which_features="all", display=True):
    """
    creates a boxplot of the feature importances

    Parameters
    ----------
    X_rotation: int
        Controls the orientation angle of the tick labels on the X-axis

    X_size: int
        Controls the font size of the tick labels

    y_scale: string
        Log transform of the y axis scale as hard to see the plot as it is normally dominated by two or three
        features.

    which_features: string
        Despite efforts if the number of columns is large the plot becomes cluttered so this parameter allows you to
        select subsets of the features like the accepted, rejected or tentative features default is all.

    Display: Boolean
    controls if the output is displayed or not, set to false when running test scripts

    """
    # data from wide to long
    data = self.history_x.iloc[1:]
    data["index"] = data.index
    data = pd.melt(data, id_vars="index", var_name="Methods")

    decision_mapper = self.create_mapping_of_features_to_attribute(maps=["Tentative", "Rejected", "Accepted", "Shadow"])
    data["Decision"] = data["Methods"].map(decision_mapper)
    data.drop(["index"], axis=1, inplace=True)

    options = {
        "accepted": self.filter_data(data, "Decision", "Accepted"),
        "tentative": self.filter_data(data, "Decision", "Tentative"),
        "rejected": self.filter_data(data, "Decision", "Rejected"),
        "all": data,
    }

    self.check_if_which_features_is_correct(which_features)
    data = options[which_features.lower()]

    fig = self.box_plot(data=data, X_rotation=X_rotation, X_size=X_size, y_scale=y_scale, figsize=figsize)
    if display:
        plt.show()
    # Always close the figure handle so the box-plot figure is not leaked (matters under test runs / repeated calls); display path also closes after show.
    plt.close(fig)


def box_plot(self, data, X_rotation, X_size, y_scale, figsize):
    if y_scale == "log":
        minimum = data["value"].min()
        if minimum <= 0:
            data["value"] += abs(minimum) + 0.01

    order = data.groupby(by=["Methods"])["value"].mean().sort_values(ascending=False).index
    my_palette = self.create_mapping_of_features_to_attribute(maps=["yellow", "red", "green", "blue"])

    # Use a color palette
    fig = plt.figure(figsize=figsize)
    ax = sns.boxplot(x=data["Methods"], y=data["value"], order=order, palette=my_palette)

    if y_scale == "log":
        ax.set(yscale="log")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=X_rotation, size=X_size)
    ax.set_title("Feature Importance")
    ax.set_ylabel("Z-Score")
    ax.set_xlabel("Features")
    return fig


def create_mapping_of_features_to_attribute(self, maps=None):
    if maps is None:
        maps = []
    rejected = list(self.rejected)
    tentative = list(self.tentative)
    accepted = list(self.accepted)
    shadow = ["Max_Shadow", "Median_Shadow", "Min_Shadow", "Mean_Shadow"]

    tentative_map = self.create_list(tentative, maps[0])
    rejected_map = self.create_list(rejected, maps[1])
    accepted_map = self.create_list(accepted, maps[2])
    shadow_map = self.create_list(shadow, maps[3])

    values = tentative_map + rejected_map + accepted_map + shadow_map
    keys = tentative + rejected + accepted + shadow

    return self.to_dictionary(keys, values)


def create_list(array, color):
    colors = [color for x in range(len(array))]
    return colors


def filter_data(data, column, value):
    data = data.copy()
    return data.loc[(data[column] == value) | (data[column] == "Shadow")]


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def check_if_which_features_is_correct(my_string):
    my_string = str(my_string).lower()
    if my_string in ["tentative", "rejected", "accepted", "all"]:
        pass

    else:
        raise ValueError(my_string + " is not a valid value did you mean to type 'all', 'tentative', 'accepted' or 'rejected' ?")


def to_dictionary(list_one, list_two):
    return dict(zip(list_one, list_two))
