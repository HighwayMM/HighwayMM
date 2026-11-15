"""Download Iowa ASOS (airport METAR) present weather + visibility from IEM, monthly chunks."""
import sys, time, urllib.request, datetime as dt
from pathlib import Path

OUT = Path("D:/highwaymm/external/iowa_rwis/asos")
U = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network=IA_ASOS&data=vsby&data=wxcodes&data=p01i"
     "&data=tmpf&tz=Etc/UTC&format=onlycomma&latlon=yes&missing=empty&trace=0.0001&report_type=3&report_type=4"
     "&sts={}T00:00Z&ets={}T00:00Z")
UA = {"User-Agent": "academic-dataset-fetch (road-weather research)"}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for y in [int(x) for x in sys.argv[1].split(",")]:
        for m in [int(x) for x in sys.argv[2].split(",")]:
            s, e = dt.date(y, m, 1), dt.date(y + (m == 12), m % 12 + 1, 1)
            f = OUT / f"asos_{y}{m:02d}.csv"
            if f.exists() and f.stat().st_size > 0:
                continue
            for k in range(3):
                try:
                    f.write_bytes(urllib.request.urlopen(urllib.request.Request(U.format(s, e), headers=UA),
                                                         timeout=900).read())
                    break
                except Exception as ex:
                    print("retry", y, m, ex, flush=True)
                    time.sleep(20)
            print(f, f.stat().st_size, flush=True)
            time.sleep(2)
