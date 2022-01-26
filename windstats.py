import sqlite3
from datetime import datetime
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker
import numpy as np
from windrose import WindroseAxes

log = logging.getLogger(__name__)

blue2red = ListedColormap(np.dstack((np.linspace(0,1,256), np.zeros(256), np.linspace(1,0,256)))[0] )

class BeaufortWindroseAxes(WindroseAxes):
    @staticmethod
    def from_ax(ax=None, fig=None, rmax=None, theta_labels=None, rect=None, *args, **kwargs):
        if ax is None:
            if fig is None:
                fig = plt.figure(
                    figsize=FIGSIZE_DEFAULT,
                    dpi=DPI_DEFAULT,
                    facecolor="w",
                    edgecolor="w",
                )
            if rect is None:
                rect = [0.1, 0.1, 0.8, 0.8]
            ax = BeaufortWindroseAxes(fig, rect, rmax=rmax, theta_labels=theta_labels, *args, **kwargs)
            fig.add_axes(ax)
            return ax
        else:
            return ax

    def legend(self, loc="lower left", decimal_places=1, units=None, **kwargs):
        def get_handles():
            handles = list()
            for p in self.patches_list:
                if isinstance(p, mpl.patches.Polygon) or isinstance(
                    p, mpl.patches.Rectangle
                ):
                    color = p.get_facecolor()
                elif isinstance(p, mpl.lines.Line2D):
                    color = p.get_color()
                else:
                    raise AttributeError("Can't handle patches")
                handles.append(
                    mpl.patches.Rectangle(
                        (0, 0), 0.2, 0.2, facecolor=color, edgecolor="black"
                    )
                )
            return handles

        def get_labels(decimal_places=1, units=None):
            labels = [i for i in range(len(self._info["bins"]) - 1)]
            return labels

        kwargs.pop("labels", None)
        kwargs.pop("handles", None)

        # decimal_places = kwargs.pop('decimal_places', 1)

        handles = get_handles()
        labels = get_labels(decimal_places, units)
        self.legend_ = mpl.legend.Legend(self, handles, labels, loc, **kwargs)
        return self.legend_

    def set_legend(self, **pyplot_arguments):
        if "borderaxespad" not in pyplot_arguments:
            pyplot_arguments["borderaxespad"] = -0.10
        legend = self.legend(**pyplot_arguments)
        plt.setp(legend.get_texts(), fontsize=8)
        return 

def render(img_file, t, wd, ws, graph_type='contour', colormap=blue2red, title='windrose', legend_loc=(-0.25,0.0)):
    notcalm = ws >= 1.0
    t = t[notcalm]
    wd = wd[notcalm]
    ws = ws[notcalm]

    # size in inches!
    fig = plt.figure(figsize=(5,5), dpi=100.0)

    # make sure bars have rounded sides
    # see: https://github.com/python-windrose/windrose/issues/137
    rect = [0.1, 0.1, 0.8, 0.8]
    hist_ax = plt.Axes(fig, rect)
    hist_ax.bar(np.array([1]), np.array([1]))

    ax = BeaufortWindroseAxes.from_ax(fig=fig)

    # make sure ticklabels and winding direction are ok
    # TODO make ticklables configurable
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    ax.set_title(title)

    beaufort = [0,1,6,12,20,29,39,50,62,75,89]
    if graph_type == 'bar':
        ax.bar(wd, ws, bins=beaufort, opening=0.8, cmap=colormap, edgecolor='white')
    elif graph_type == 'box':
        ax.box(wd, ws, bins=beaufort, cmap=colormap)
    elif graph_type == 'contour':
        ax.contourf(wd, ws, bins=beaufort, cmap=colormap)
        ax.contour(wd, ws, bins=beaufort, colors='white')

    ax.set_yticklabels([])
    ax.set_legend(decimal_places=0, loc=legend_loc)

    try:
        fig.savefig(img_file, bbox_inches='tight')
    except IOError as e:
        log.error("Unable to save to file '%s' %s:", img_file, e)

def getWindData(database='weewx.sdb', delta=24*60*60):
    con = sqlite3.connect(database)
    cur = con.cursor()
    now = datetime.now().timestamp()
    now -= delta
    now = int(now)
    data = np.array(cur.execute(f'SELECT dateTime,windDir,windSpeed FROM archive WHERE dateTime >= :now',{'now':now}).fetchall(), dtype=np.float32)
    return data[:,0], data[:,1], data[:,2] * 1.609344

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename","-f", default="test.png")
    parser.add_argument("--database","-d", default="/var/lib/weewx/weewx.sdb")
    parser.add_argument("--period","-p", choices=['day','week','month','year'], default="day")
    parser.add_argument("--type","-t", choices=['bar','box','contour'], default="contour")
    parser.add_argument("--colormap","-cm", default=blue2red)
    args = parser.parse_args()
    t, wd, ws = getWindData(database=args.database, delta={'day':1,'week':7,'month':30,'year':365}[args.period]*24*60*60)
    render(args.filename, t, wd, ws, graph_type=args.type, colormap=args.colormap, title=f"{args.period} ending at {datetime.fromtimestamp(int(t[-1]))}\n")
