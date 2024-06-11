from matplotlib import pyplot as plt

def make_plot(num_rows, num_columns):
    # fig-0427, axs
    return plt.subplots(num_rows, num_columns, figsize=(15, 20))  # 1 row, 2 columns

def fig_coords(index, num_columns):
    row_index = index // num_columns
    column_index = index % num_columns
    return (row_index, column_index)


def plot_sub_scatter(fig, axs, posx, posy, dataset, labels, sc, cents, dist, title):
    axs[posx][posy].set_title(title)
    axs[posx][posy].scatter(dataset[:, 0], dataset[:, 1], c=labels, s=50, cmap='viridis', alpha=0.5)
    axs[posx][posy].scatter(cents[:, 0], cents[:, 1], c='red', s=200, alpha=0.5)
    axs[posx][posy].text(0, 0, 's = ' + str(round(sc, 3)), fontsize=10, verticalalignment='top')

def show():
    plt.tight_layout()  # Adjust layout to not overlap
    plt.show()
