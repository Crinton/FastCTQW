import matplotlib.pyplot as plt

def plot_1():
    fig, ax = plt.subplots(figsize=(10,8*0.9))
    ax.plot([1,2,3],[2,3,4])
    return fig, ax
fig, ax = plot_1();

fig.show()
# fig.savefig()
plt.show()