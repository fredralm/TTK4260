import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
def line_plot(df, title, ylabel="Cases", h=None, v=None, xlim=(None, None), ylim=(0, None), math_scale=True):
    """
    Show chlonological change of the data.
    """
    ax = df.plot()
    if math_scale:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_title(title)
    ax.set_xlabel(None)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)
    if h is not None:
        ax.axhline(y=h, color="black", linestyle="--")
    if v is not None:
        ax.axvline(x=v, color="black", linestyle="--")
    plt.tight_layout()
    plt.grid()
    plt.show()
