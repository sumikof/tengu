from datetime import datetime
import pytz

def strformatdate(dateiso):
    iso = dateiso
    date = None
    try:
        date = datetime.strptime(iso[:-4], '%Y-%m-%dT%H:%M:%S.%f')
        date = pytz.utc.localize(date).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        print("formaterror", iso[:-4])

    return date
