import jdutil
import astropy
import astropy.time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import urllib
import zipfile
import gzip
from io import StringIO
import datetime
import os

### USEFUL CONST. VARIABLES

# this is from reference 1980's GPS time
J2000_gps_weeks = 1042
J2000_gps_residual_seconds = 561548.816
J2000_gps_seconds = 604800*J2000_gps_weeks + 561548.816    

ngl_full_list = "http://geodesy.unr.edu/NGLStationPages/llh.out"
ngl_24h_2w = "http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
ngl_24h_24h = "http://geodesy.unr.edu/NGLStationPages/DataHoldingsRapid24hr.txt"
ngl_5min_24h = "http://geodesy.unr.edu/NGLStationPages/DataHoldingsRapid5min.txt"
ngl_5min_1h30min = "http://geodesy.unr.edu/NGLStationPages/DataHoldingsUltra5min.txt"
east_ref_name = "___e-ref(m)"
north_ref_name = "___n-ref(m)"
vertical_ref_name = "___v-ref(m)"
east_mean_name = "___e-mean(m)"
north_mean_name = "___n-mean(m)"
vertical_mean_name = "___v-mean(m)"
east_sigma_name = "sig_e(m)"
north_sigma_name = "sig_n(m)"
vertical_sigma_name = "sig_v(m)"


''' 
Convenient method to convert GPS seconds from the 1st January 2000 to UTC Datetime

Keyword arguments:
secs: seconds from 01/01/2000 in GPS time
'''
def J2000_to_utc(secs):
    t_gps = astropy.time.Time(secs+J2000_gps_seconds, format="gps")
    t_utc = astropy.time.Time(t_gps, format="iso", scale="utc")
    return t_utc.datetime

def ngl_process_list(url):
    ''' 
        Download and pre-process the station list from NGL provided in the url
        The possible urls are given by the internal variables:
        ngl_24h_2w, ngl_24h_24h, ngl_5min_24h, ngl_5min_1h30min

        Keyword arguments:
        url (string): station list url

        Returns:
        station_list (DataFrame)
    '''
    station_list = pd.read_csv(url, sep=r"\s+", on_bad_lines='skip', parse_dates=['Dtbeg', 'Dtend'])
    station_list.rename(columns={'Sta' : 'name', 'Lat(deg)' : 'lat', 'Long(deg)' : 'lon', 'Hgt(m)' : 'height', 'X(m)' : 'x', 'Y(m)' : 'y', 'Z(m)' : 'z', 'Dtbeg' : 'begin', 'Dtend' : 'end'   }, inplace=True)
    station_list['lon'] =  (station_list['lon'] + 180)%360-180 # ensures values in [-180, 180]
    return station_list

def get_all_stations_box(station_list, min_lat, max_lat, min_lon, max_lon):
    '''
    By passing a 'station_list' dowloaded from 'ngl_process_list',
    returns the names, latitude, and longitude of
    all stations within the box described by ( min_lat, max_lat, min_lon, max_lon)

    Returns:
        names, lats, lons (Tuple)
    '''
    mask = station_list.lat.values >= min_lat
    mask *= station_list.lat.values <= max_lat
    mask *= station_list.lon.values >= min_lon
    mask *= station_list.lon.values <= max_lon
    return station_list.name.values[mask].astype('str'), station_list.lat.values[mask].astype('float'), station_list.lon.values[mask].astype('float')
def get_all_stations_within_radius(station_list, lat_origin, lon_origin, maximal_radius, r0=6371):
    ''' 
        Given a station list (Pandas), extract all the station within a radius
        
        Keyword arguments:
            station_list: Pandas DataFrame
            lat_origin: center latitude
            lon_origin: center longitude
            maximal_radius: circle radius
        Returns:
            names (numpy.ndarray): a NumPy array of string, containing the names of the station in the given circle.
    '''
    dists = haversine_distances(np.radians(station_list[['lat','lon']].values), np.radians(np.array([lat_origin, lon_origin]))[None,:])[:,0]
    mask = dists <= maximal_radius/r0
    return station_list.name.values[mask].astype('str'), station_list.lat.values[mask].astype('float'), station_list.lon.values[mask].astype('float')

def ngl_retrieve_24h(rootpath, station_name,force_download=False):
    ''' 
        Load if existing or download a station's data in folder 'rootpath'. 
        Only post-processing is a rename of columns to be more readable.

        Keyword arguments:
            rootpath (string): path (ending with "/")
            station_name (string): the name of the station

        Returns:
            data (DataFrame)
    '''
    # site YYMMMDD yyyy.yyyy __MJD week d reflon
    #  _e0(m) __east(m) ____n0(m) _north(m) u0(m) ____up(m) 
    # _ant(m) sig_e(m) sig_n(m) sig_u(m) 
    # __corr_en __corr_eu __corr_nu 
    # _latitude(deg) _longitude(deg) __height(m)
    data_type="tenv3"
    
    base_url = "http://geodesy.unr.edu/gps_timeseries/" + data_type
    filename = rootpath + station_name + ".csv"
    if(not force_download):
        if(os.path.exists(filename)):
            data = pd.read_csv(filename, sep =" ", parse_dates=['date'])
            data.lat = data.lat % 90
            data.lon = data.lon % 180
            return data, "loaded"
    download_url = base_url + "/IGS14/" + station_name + "." + data_type
    data =  pd.read_csv(download_url, sep=r"\s+")
    data['date'] = [str_to_datetime_2000(str(s)) for s in data['YYMMMDD']]
    data['date'] = data['date'].values.astype('datetime64[D]')
    #assert "_latitude(deg)" in data.columns
    labels_to_rename = {"_e0(m)" : "e0",
                         "__east(m)" : "east",
                         "____n0(m)" : "n0",
                         "_north(m)" : "north",
                         "u0(m)" : "u0",
                         "____up(m)" : "up",
                         "_ant(m)" : "antenna",
                         "sig_e(m)" : "sigma_e",
                         "sig_n(m)" : "sigma_n",
                         "sig_u(m)" : "sigma_u",
                         "__corr_en" : "corr_en",
                         "__corr_eu" : "corr_eu",
                         "__corr_nu" : "corr_nu", 
                          "_latitude(deg)" : "lat" ,
                          "_longitude(deg)" : "lon", 
                          "__height(m)": "height"}
    data.rename(labels_to_rename,axis=1, inplace=True)
    data.lat = data.lat % 90
    data.lon = data.lon % 180
    data.to_csv(filename, sep=" ", index=False)
    return data, "downloaded"


''' 
Download a station's data at a given year. No post-processing is done.
The output is a Python dictionary with keys the day of the year 
and values the corresponding Pandas DataFrame containing the GPS data.

Keyword arguments:
station: station name
year: desidered year
'''
#def ngl_rapid_5min(station,year):
#    url = "http://geodesy.unr.edu/gps_timeseries/kenv/%s/%s.%4i.kenv.zip"
#    try:
#        filehandle, _ = urllib.request.urlretrieve(url % (station, station, year))
#    except:
#        return None
#    zip_file_object = zipfile.ZipFile(filehandle, 'r')
#    dataframes = {}
#    for nl in zip_file_object.namelist():
#        file = zip_file_object.open(nl)
#        with gzip.open(file, 'rb') as f:
#            content = f.read().decode('ascii')
#            content_df = pd.read_csv(StringIO(content), sep="\s+")
#            doy = content_df['doy'].values[0]
#            dataframes[doy] = content_df            
#    return dataframes


''' 
Download and save a station's data at a given year. 
No post-processing is done.  Datetime is in GPS seconds from 01/01/2000.

Keyword arguments:
station: station name
year: desidered year
include_sigma: whether to include the uncertainties in the measurements
'''
#def ngl_save_rapid_5min(root, station,year, include_sigma=True):
#    df_dict = ngl_rapid_5min(station, year)
#    max_day = 365
#    if(366 in df_dict.keys()):
#        max_day += 1
#    rows = []
#    for day, df in df_dict.items():
#        north = df[NGLDownloader.north_ref_name]
#        east = df[NGLDownloader.east_ref_name]
#        vertical = df[NGLDownloader.vertical_ref_name]    
#        #seconds_day = df['s-day']
#        seconds_J2000 = df['sec-J2000']
#        if(include_sigma):


def str_to_datetime(s, limit_for_2000):
    str_to_month = {'JAN':1, 'FEB':2, 'MAR':3, 'APR' : 4, 'MAY':5, 'JUN':6, 'JUL' : 7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
    d = int(s[5:])
    m = str_to_month[s[2:5]]
    y = s[:2]
    if(int(y) > limit_for_2000):
        y = int("19" + y)
    else:
        y = int("20" + y)
    return datetime.datetime(y, m, d)

def str_to_datetime_2000(s):
    str_to_month = {'JAN':1, 'FEB':2, 'MAR':3, 'APR' : 4, 'MAY':5, 'JUN':6, 'JUL' : 7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
    d = int(s[5:])
    m = str_to_month[s[2:5]]
    y = s[:2]
    y = int("20" + y)
    return datetime.datetime(y, m, d)


def fraction_year_to_datetime(s):
    year_str, fraction_str = s.split(".")
    year = int(year_str)
    fraction = float("0." + fraction_str)
    # Calculate the day of the year (1 to 365 or 366)
    total_days = 365 if year % 4 != 0 else 366  # Check for leap year
    day_of_year = int(total_days * fraction)
    # Convert to yyyy mm dd format
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
    return date



#class NGLDownloader:
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#
#    @staticmethod
#    def merge_year_5min_rapid(dataframes_dict):
#        max_day = 365
#        if(366 in dataframes_dict.keys()):
#            max_day += 1
#        rows = []
#        for day in range(1, 1+max_day):
#            if(day in dataframes_dict):
#                north = dataframes_dict[day][NGLDownloader.north_ref_name]
#                east = dataframes_dict[day][NGLDownloader.east_ref_name]
#                vertical = dataframes_dict[day][NGLDownloader.vertical_ref_name]    
#                seconds_day = dataframes_dict[day]['s-day']
#                seconds_J2000 = dataframes_dict[day]['sec-J2000']
#                row = np.array((seconds_J2000,north, east, vertical)).T
#                rows.append(row)
#        rows = np.concatenate(rows)
#        return rows
#    
#    
#     
#    #valid options are:
#    # interval_type = 5, 24
#    # quality_type = 0, 1
#    def __init__(self, output_folder, interval_type, quality_type=1):
#        self.rootpath = output_folder
#        self.interval_type = interval_type
#        self.quality_type = quality_type
#        if(interval_type==5):
#            if(quality_type==0):
#                self.station_list = NGLDownloader.process_station_list(NGLDownloader.ngl_5min_1h30min)
#            else:
#                self.station_list = NGLDownloader.process_station_list(NGLDownloader.ngl_5min_24h)
#        else:
#            if(quality_type==0):
#                self.station_list = NGLDownloader.process_station_list(NGLDownloader.ngl_24h_24h)
#            else:
#                self.station_list = NGLDownloader.process_station_list(NGLDownloader.ngl_24h_2w)       
#    
#    
#    def get_all_stations_within_radius(self, lat_origin, lon_origin, maximal_radius, r0=6371):
#        dists = haversine_distances(np.radians(self.station_list[['lat','lon']].values), np.radians(np.array([lat_origin, lon_origin]))[None,:])[:,0]
#        return self.station_list.name.values[dists <= maximal_radius/r0].astype('str')
#    
#    
#    def download_data_year_loc_5min_rapid(self, year, lat_origin, lon_origin, maximal_radius, verbose=False,r0=6371):
#        stations_to_load = self.get_all_stations_within_radius(lat_origin, lon_origin, maximal_radius)
#        num_downloaded = 0
#        num_existing = 0
#        num_failed = 0
#        for counter,stat_name in enumerate(stations_to_load):
#            if(verbose):
#                print(stat_name, counter*100/len(stations_to_load))
#            filename = self.rootpath + stat_name + "_" + str(year) + ".npz"
#            if(os.path.exists(filename)):
#                num_existing += 1
#                continue
#            dfs = NGLDownloader.retrieve_year_5min_rapid(stat_name, year)
#            if(dfs is None):
#                num_failed += 1
#                continue
#            num_downloaded += 1
#            tseries = NGLDownloader.merge_year_5min_rapid(dfs)
#        
#            np.savez(filename,seconds=tseries[:,0],north=tseries[:,1],
#                     east=tseries[:,2], vertical=tseries[:,3])
#        return stations_to_load, num_downloaded, num_existing, num_failed
#            
#    def postprocess_year_5min_rapid(self, names, year):
#        dataframes = {}
#        for stat_name in names:
#            filename = self.rootpath + stat_name + "_" + str(year) + ".npz"
#            if(os.path.exists(filename)):
#                data = np.load(filename)
#                origin_datetime = np.datetime64(J2000_to_utc(data['seconds'][0]))
#                delta_times = (data['seconds'] - data['seconds'][0]).astype('timedelta64[s]')
#                correct_datetimes = origin_datetime + delta_times
#                df = pd.DataFrame()
#                df['datetime'] = correct_datetimes
#                df['north'] = data['north']
#                df['east'] = data['east']
#                df['vertical'] = data['vertical']
#                dataframes[stat_name] = df
#        return dataframes
#    
#    def retrieve_station_24h_final(self, station_name, data_type="tenv3"):
#        filename = self.rootpath + station_name + ".csv"
#        if(os.path.exists(filename)):
#            return pd.read_csv(self.rootpath + station_name + ".csv", sep =" ", parse_dates=['date']), "loaded"
#        labels_to_rename = {"_latitude(deg)" : "lat" ,"_longitude(deg)" : "lon", "__height(m)": "height"}
#        base_url = "http://geodesy.unr.edu/gps_timeseries/" + data_type
#        data =  pd.read_csv(base_url + "/IGS14/" + station_name + "." + data_type, sep=r"\s+")
#        #data = pd.read_csv(self.rootpath + name + ".csv")
#        data['date'] = [ NGLDownloader.str_to_datetime(s, 23) for s in data['YYMMMDD']]
#        data['date'] = data['date'].values.astype('datetime64[D]')
#        data.rename(labels_to_rename,axis=1, inplace=True)
#        data = data[['date','site','lat','lon','height']]
#        #data['lon'] = data['lon'] % 180
#        data.to_csv(filename, sep=" ", index=False)
#        return data, "downloaded"