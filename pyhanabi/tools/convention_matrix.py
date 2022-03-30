import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colors
import numpy as np

# data = np.random.rand(10, 10) * 20

# # create discrete colormap
# cmap = colors.ListedColormap(['red', 'blue'])
# bounds = [0,10,40]
# norm = colors.BoundaryNorm(bounds, cmap.N)

# fig, ax = plt.subplots()
# ax.imshow(data, cmap=cmap, norm=norm)

# # draw gridlines
# ax.grid(which='major', axis='both', linestyle='', color='k', linewidth=2)
# ax.set_xticks(np.arange(-.5, 10, 1));
# ax.set_yticks(np.arange(-.5, 10, 1));

# plt.show()

# arr = np.arange(100).reshape((10, 10))
# print("arr")
# print(arr)
# norm = mcolors.Normalize(vmin=0., vmax=0.6)
# # see note above: this makes all pcolormesh calls consistent:
# pc_kwargs = {'rasterized': True, 'cmap': 'viridis', 'norm': norm}
# fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
# im = ax.pcolormesh(arr, **pc_kwargs)
# fig.colorbar(im, ax=ax, shrink=0.6)

# plt.show()

def convention_matrix(data, xticklabels, yticklabels, title, colour_max=0.7):
    norm = mcolors.Normalize(vmin=0., vmax=colour_max)
    # see note above: this makes all pcolormesh calls consistent:
    pc_kwargs = {'rasterized': True, 'cmap': 'cividis', 'norm': norm}
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(arr, **pc_kwargs)
    cb = fig.colorbar(im, ax=ax,fraction=0.024, pad=0.04)
    cb.ax.tick_params(length=1)

    plt.ylabel("signal t")
    plt.xlabel("response t+1")
    plt.xticks(range(len(xticklabels)));
    ax.set_xticklabels(xticklabels);
    ax.xaxis.tick_top()
    ax.tick_params('both', length=1, width=1, which='major')
    plt.yticks(range(len(yticklabels)));
    ax.set_yticklabels(yticklabels);
    plt.title(title)
    plt.show()

arr = np.arange(0, 1, 0.01).reshape((10, 10))
xticklabels = ["D0", "D1", "D2", "D3", "D4", "P0", "P1", "P2", "P3", "P4"]
yticklabels = ["CR", "CY", "CG", "CW", "CB", "R1", "R2", "R3", "R4", "R5"]
convention_matrix(arr, xticklabels, yticklabels, "IQL")
