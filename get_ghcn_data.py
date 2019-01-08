""" Find nearest GHCN stations to sample site and download_data """

import numpy as np
import geopandas as gpd
import pandas as pd
import requests

from datetime import datetime


def main():
    stations = ['HO000078714', 'CSM00078762']
    for station_id in stations:
        print('Processing ' + station_id + '...')
        df = download_data_by_station(station_id)
        #print_statistics(df)
        df.to_csv('data/ghcn_' + station_id  + '.csv')


def dist(lon0, lat0, lon1, lat1):
    return np.sqrt((lon0 - lon1) ** 2 + (lat0 - lat1) ** 2)


def download_and_merge_data(stations, indices):
    df = pd.DataFrame()
    for i, row in stations.iloc[indices].iterrows():
        s = row['id']
        x = row['dist']
        url = download_url(s)
        this_df = parse_data(url)
        this_df['dist'] = x
        df = df.append(this_df)
    return df


def download_data_by_station(station_id):
    url = download_url(station_id)
    df = parse_data(url)
    return df


def download_url(station_id):
    s = 'https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/all/{}.dly'
    return s.format(station_id)


def get_nearest_stations(lon, lat, code, stations, tol=1.0):
    stations['dist'] = np.nan
    indices = []
    for i, row in stations.iterrows():
        lon1 = row['lon']
        lat1 = row['lat']
        x = dist(lon, lat, lon1, lat1)
        if x < tol and row['id'][0:2] == code:
            indices.append(i)
            stations.dist.iloc[i] = x
    return stations, indices

#def get_stations_by_country(stations, code):
#    indices = stations.

def parse_data(url):
    stations = []
    dates = []
    precip = []

    r = requests.get(url)
    lines = r.text.split('\n')[:-1]
    lines = [x for x in lines if 'PRCP' in x]

    for x in lines:
        data = x.split()
        identifier = data[0]
        year = int(identifier[11:15])
        month = int(identifier[15:17])
        start = 21
        dx = 8
        positions = np.arange(start, len(x) + 1, dx)[:-1]
        for i, pos in enumerate(positions):
            value = float(x[pos:pos+5])
            day = i + 1

            if value != -9999:
                date = datetime(year, month, day)
                stations.append(identifier[0:11])
                dates.append(date)
                precip.append(value)

    df = pd.DataFrame({'station': stations, 'date': dates, 'precip': precip})

    return df


def print_statistics(df):
    print('{} - {}'.format(df.date.min().year, df.date.max().year))
    print('n:\t{}'.format(len(df)))

    mean = df.precip.mean()
    median = df.precip.median()
    p90 = np.nanpercentile(df.precip.values, 90)

    print('mean:\t\t{:.2f}'.format(mean))
    print('median:\t\t{:.2f}'.format(median))
    print('P90:\t\t{:.2f}'.format(p90))
    print('Peak to mean:\t{:.2f}'.format(p90 / mean))


def read_sample_list(filename='data/samples.csv'):
    df = pd.read_csv(filename)
    df.columns = ['id', 'lon', 'lat', 'country']
    return df


def read_station_list(filename='data/ghcnd-stations.txt'):
    stations = []
    lons = []
    lats = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data = line.split()
            stations.append(data[0])
            lats.append(float(data[1]))
            lons.append(float(data[2]))

    df = pd.DataFrame(data={'id': stations,
                            'lon': lons,
                            'lat': lats})
    return df


if __name__ == "__main__":
    main()
