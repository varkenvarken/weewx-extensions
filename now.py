import sqlite3
from datetime import datetime
from dateutil.tz import tzlocal
import logging
from json import dump, dumps

log = logging.getLogger(__name__)

def celcius(fahrenheit):
    return (fahrenheit-32)*5/9

def getData(database):
    start = int(datetime.now().timestamp()) - 15 * 60
    con = sqlite3.connect(database)
    cur = con.cursor()
    data = list(cur.execute(f'select dateTime, outTemp, outHumidity, inTemp, inHumidity from archive WHERE dateTime > :start ORDER BY dateTime DESC LIMIT 1',{'start':start}).fetchone())
    data[1] = celcius(data[1])
    data[3] = celcius(data[3])

    timestamp = datetime.fromtimestamp(data[0], tzlocal()).isoformat()    
    indoor = {"time": timestamp, "stationid": "weewx-indoor", "name":"slaapkamer", "temperature": round(data[3],1), "humidity": round(data[4],1)} 
    outdoor = {"time": timestamp, "stationid": "weewx-outdoor", "name":"buiten", "temperature": round(data[1],1), "humidity": round(data[2],1)} 
    return [indoor, outdoor]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename","-f", default="now.json")
    parser.add_argument("--database","-d", default="/var/lib/weewx/weewx.sdb")
    args = parser.parse_args()

    data = getData(database=args.database)
    if args.filename == "-":
        print(dumps(data))
    else:
        with open(args.filename, "w") as fp:
            dump(data, fp)
