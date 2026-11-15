"""Download Iowa RWIS atmospheric/pavement sensor data (IEM) in weekly chunks for given months."""
import sys, time, urllib.request, datetime as dt
from pathlib import Path

OUT = Path("D:/highwaymm/external/iowa_rwis/sensors")
U = ("https://mesonet.agron.iastate.edu/cgi-bin/request/rwis.py?network=IA_RWIS&stations=_ALL&tz=UTC&what=txt"
     "&src=atmos&gis=true&vars=pcpn,vsby,tfs0_text,tfs1_text,tfs2_text,tfs3_text,tmpf,dwpf,sknt&sts={}&ets={}")
UA = {"User-Agent": "academic-dataset-fetch (road-weather research)"}


def main(years, months):
    OUT.mkdir(parents=True, exist_ok=True)
    for y in years:
        for m in months:
            d = dt.date(y, m, 1)
            end = dt.date(y + (m == 12), m % 12 + 1, 1)
            while d < end:
                e = min(d + dt.timedelta(days=7), end)
                f = OUT / f"atmos_{d:%Y%m%d}.csv"
                if not f.exists() or f.stat().st_size == 0:
                    t0 = time.time()
                    for k in range(3):
                        try:
                            req = urllib.request.Request(U.format(f"{d}T00:00", f"{e}T00:00"), headers=UA)
                            f.write_bytes(urllib.request.urlopen(req, timeout=900).read())
                            break
                        except Exception as ex:
                            print("retry", d, ex, flush=True)
                            time.sleep(20)
                    print(f, f.stat().st_size if f.exists() else 0, f"{time.time() - t0:.0f}s", flush=True)
                    time.sleep(2)
                d = e


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1].split(",")], [int(x) for x in sys.argv[2].split(",")])
