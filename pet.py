import sqlite3
from datetime import datetime,date
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates

import numpy as np
log = logging.getLogger(__name__)

blue2red = ListedColormap(np.dstack((np.linspace(0,1,256), np.zeros(256), np.linspace(1,0,256)))[0] )


def makkink(T, Kin):
    """
    Returns evapotranspiration in kg/(m2 * s)
    """
    λ = np.float32(2.45E6)  # heat of evaporation in J/Kg of water at 20 degrees C
    γ = np.float32(0.66)    # psychrometer constant mbar/C
    C1= np.float32(0.65)

    a = np.float32(6.1078)
    b = np.float32(17.294)
    c = np.float32(237.73)
    s = ((a * b * c) / ((c + T) ** 2)) * np.exp((b * T)/(c+T))  # temperature derivative of sat
    return (C1 * (s / (s + γ)) * Kin ) / λ

def getData(database, start, period=24 * 60 * 60):
    con = sqlite3.connect(database)
    cur = con.cursor()
    start = int(start.timestamp())
    data = np.array(cur.execute(f'select dateTime,rain,outTemp,radiation from archive WHERE dateTime >= :start',{'start':start}).fetchall(), dtype=np.float32)
    datetime = data[:,0]
    periods = datetime//np.float32(period)
    rain = data[:,1] * np.float32(25.4)  # inches to mm
    temp = (data[:,2] - np.float32(32)) * np.float32(5/9)   # Fahrenheit to Celcius
    radiation = data[:,3]    # radiation is already in W/m2
    et = makkink(temp, radiation)
    bins = range(int(periods.min()), int(periods.max()+2))
    rainsum, rbins  = np.histogram(periods, bins=bins, weights=rain)
    etsum, ebins = np.histogram(periods, bins=bins, weights=et)
    counts, _ = np.histogram(periods, bins=bins)
    return rbins * period, rainsum, period * (etsum / counts)

def graph(time, rain, et, cumsum, title, color, font, fontcolor, **kwargs):
    fig = plt.figure(**kwargs)
    plt.title(title, fontfamily=font, color=fontcolor)
    ax = fig.axes[0]
    ax.set_ylabel('ET (mm)', color='red')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_facecolor(kwargs['facecolor'])
    ax.plot(time, cumsum, color='red')
    ax2 = ax.twinx()
    ax2.set_ylabel('rain (mm)', color='blue')
    ax2.bar(time, rain, color='blue')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    fig.autofmt_xdate()
    #y = ax.get_yaxis()
    #y.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))    
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

    today = date.today()
    jan1 = date(year=today.year, month=4, day=1)
    start = datetime.fromordinal(jan1.toordinal())
    t, rain, et = getData(database=args.database, start=start, period={'day':1,'week':7,'month':30,'year':365}[args.period]*24*60*60)

    #for tt,rr,e in zip(t, rain, et):
    #    print(datetime.fromtimestamp(tt),rr,e)

    dpi = 72
    size = args.width/dpi, args.height/dpi
    kwargs = {
        'figsize': size,
        'dpi': dpi,
        'facecolor': args.backgroundcolor,
        'edgecolor': args.edgecolor,
    }

    d = np.cumsum(et - rain)
    offset = np.minimum.accumulate(d)
    d0 = d[0]
    d -= offset
    d += d0
    fig = graph(list(map(datetime.fromtimestamp, t[:-1])), rain, et, d, f"cumulative evapotranspiration", args.color, args.font, args.textcolor,  **kwargs)
    try:
        fig.savefig(args.filename, bbox_inches='tight')
    except IOError as e:
        log.error("Unable to save to file '%s' %s:", img_file, e)
