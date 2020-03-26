import os, sys, glob, csv
import dateparser

PATH = 'csse_covid_19_data/csse_covid_19_daily_reports/*.csv'

def read(filename):
    with open(filename, 'r', encoding='ascii', errors='ignore') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            yield row

writer = csv.DictWriter(sys.stdout, [ 'date', 'Deaths', 'Confirmed', 'Recovered', 'country', 'region'])
writer.writeheader()

fields = {
    'Last_Update': 'date',
    'Last Update': 'date',
    'Province/State': 'region',
    'Province_State': 'region',
    'Country/Region': 'country',
    'Country_Region': 'country',
    'Latitude': None,
    'Longitude': None,
    'Active': None,
    'Lat': None,
    'Long': None,
    'Combined_Key': None,
    'Admin2': None,
    'FIPS': None,
    'Long_': None
}
for f in glob.glob(PATH):
    filename = os.path.basename(f).split(".")[0]
    d = str(dateparser.parse(filename))
    for row in read(f):
        for k, v in fields.items():
            if k in row:
                x = row.pop(k)
                if v is not None:
                    row[v] = x

        state = row['region']
        if state.find('Princess') != -1:
            continue

        parts = state.split(",")
        if len(parts) > 1:
            continue

        row['date'] = d#dateparser.parse(row['date']).date()
        writer.writerow(row)