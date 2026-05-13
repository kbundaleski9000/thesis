import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(data, title="Heatmap", xlabel="X-axis", ylabel="Y-axis"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(data, annot=True, cmap="viridis", fmt=".2f")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()