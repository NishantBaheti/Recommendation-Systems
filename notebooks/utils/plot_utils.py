import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_recomendations(df: pd.DataFrame):
    """
    Bar plot of recommended movies and distances from given movie.

    Args:
        df (pd.DataFrame): dataframe.
    """

    _, ax = plt.subplots(1, 1, figsize=(10, 3))
    sns.barplot(data=df, y='distances', x='title',
                orient='v', hue='genres', ax=ax)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xticks(rotation=90)
    ax.grid()
    plt.show()
