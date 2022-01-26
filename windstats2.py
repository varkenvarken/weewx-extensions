import sqlite3
from datetime import datetime
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from matplotlib.ticker import PercentFormatter

import numpy as np
from windrose import WindroseAxes

log = logging.getLogger(__name__)

blue2red = ListedColormap(np.dstack((np.linspace(0,1,256), np.zeros(256), np.linspace(1,0,256)))[0] )

beaufort = np.array([0,1,6,12,20,29,39,50,62,75,89])

def getWindData(database='weewx.sdb', delta=24*60*60):
    con = sqlite3.connect(database)
    cur = con.cursor()
    now = datetime.now().timestamp()
    now -= delta
    now = int(now)
    data = np.array(cur.execute(f'SELECT dateTime,windDir,windSpeed FROM archive WHERE dateTime >= :now',{'now':now}).fetchall(), dtype=np.float32)
    return data[:,0], data[:,1], data[:,2] * 1.609344

def histogram(v, bins, title, color, font, fontcolor, **kwargs):
    fig = plt.figure(**kwargs)
    nans = np.isnan(v)
    n, bins = np.histogram(v[np.logical_not(np.isnan(v))], bins=bins)
    f = fm.findfont('Amaranth')
    print(f)
    plt.title(title, fontfamily=font, color=fontcolor)
    plt.xticks(np.arange(n.shape[0]), labels= list(map(str,[0,1,2,3,4,5,6,7,8,9])))
    plt.xlabel('Beaufort')
    plt.bar(np.arange(n.shape[0]), n / np.sum(n), color=color)
    ax = fig.axes[0]
    y = ax.get_yaxis()
    y.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))    
    return fig

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename","-f", default="test.png")
    parser.add_argument("--database","-d", default="/var/lib/weewx/weewx.sdb")
    parser.add_argument("--period","-p", choices=['day','week','month','year'], default="day")
    parser.add_argument("--width","-x", type=int, default=500)
    parser.add_argument("--height","-y", type=int, default=180)
    parser.add_argument("--color","-c", default='#d22c2b')
    parser.add_argument("--textcolor","-tc", default='#d22c2b')
    parser.add_argument("--backgroundcolor","-b", default='#fff7f7')
    parser.add_argument("--edgecolor","-e", default='#e7dee1')
    parser.add_argument("--font", default='Amaranth')
    args = parser.parse_args()

    t, wd, ws = getWindData(database=args.database, delta={'day':1,'week':7,'month':30,'year':365}[args.period]*24*60*60)
    start = datetime.fromtimestamp(int(t[0]))
    end = datetime.fromtimestamp(int(t[-1]))

    dpi = 72
    size = args.width/dpi, args.height/dpi
    kwargs = {
        'figsize': size,
        'dpi': dpi,
        'facecolor': args.backgroundcolor,
        'edgecolor': args.edgecolor,
    }

    fig = histogram(ws, beaufort, f"Windspeed distribution\n{args.period} end at {end}", args.color, args.font, args.textcolor,  **kwargs)
    try:
        fig.savefig(args.filename, bbox_inches='tight')
    except IOError as e:
        log.error("Unable to save to file '%s' %s:", img_file, e)
