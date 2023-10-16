import torch
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from io import StringIO
import datetime
from sklearn.metrics.pairwise import haversine_distances
import zipfile
import urllib
import re
from typing import Union, Optional

# inspired by https://zenodo.org/record/8161777
class Sequence:

    def __init__(self,
                 inter_times: Union[torch.Tensor, np.ndarray],
                 t_start : float = 0.0,
                 t_nll_start : Optional[float] = None,
                 features : Optional[Union[torch.Tensor, np.ndarray]] = None):
        self.inter_times =  torch.flatten(torch.as_tensor(inter_times))
        if not self.inter_times.dtype in [torch.float32, torch.float64]:
            raise ValueError(
                f"Supported types for inter_times are torch.float32 or torch.float64"
                "(got {self.inter_times.dtype})"
            )
        self.arrival_times = self.inter_times.cumsum(dim=-1)[:-1] + t_start
        self.t_start = float(t_start)
        self.t_end = float(self.inter_times.sum().item() + self.t_start)
        if t_nll_start is None:
            t_nll_start = t_start
        self.t_nll_start = float(t_nll_start)
        if(features is not None):
            self.features = torch.as_tensor(features)
            if self.features.shape[0] != self.arrival_times.shape[0]: # different sequence length!!!!
                raise ValueError(
                f"The length of features is different from the induced length of arrival times."
            )
        else:
            self.features = None

    def get_subsequence(self, start : float, end : float):
        if start < self.t_start or end > self.t_end:
            raise ValueError(
                f"Error in either start or end. start must be >= {self.t_start} and end must be <= {self.t_end}"
            )
        mask = (self.arrival_times >= start) & (self.arrival_times <= end)
        new_arrival_times = self.arrival_times[mask]
        if(len(new_arrival_times) > 0):
            # find the last gap, to add to the new inter_times
            last_inter_time = torch.tensor(
                [end - new_arrival_times[-1]],
                device=self.inter_times.device,
                dtype=self.inter_times.dtype,
            )
            # we skip the last element from inter_times
            new_inter_times = torch.cat([self.inter_times[:-1][mask], last_inter_time])
            # 
            first_inter_time = new_arrival_times[0] - start
    @staticmethod
    def compute_inter_times(arrival_times : Union[torch.tensor, np.ndarray, list],
                             t_start : float, 
                             t_end : float):
        return np.diff(arrival_times, prepend=[t_start], append=[t_end])
    


class irisRequests:
    stations_base_url = "http://service.iris.edu/fdsnws/station/1/query?"
    events_base_url = "http://service.iris.edu/fdsnws/event/1/query?"
    data_base_url = "http://service.iris.edu/fdsnws/dataselect/1/query?"
    query_formats = ["xml", "text","geocsv"]
    events_magnitude_types = ["ML", "Ms", "mb", "Mw","all","preferred"]
    
    
    # GENERIC PRIVATE METHODS

    @staticmethod
    def __format_time(start, end):
        time_format = "%Y-%m-%dT%H:%M:%S"
        return {"start" : start.strftime(time_format), "end" : end.strftime(time_format)}
    

    @staticmethod
    def __format_box(minlat, maxlat, minlon, maxlon):
        return {"minlat" : minlat, "maxlat" : maxlat, "minlon" : minlon, "maxlon" : maxlon}
    
    @staticmethod
    def __format_radial(lat, lon, maxradius, minradius=0, mindepth=None, maxdepth=None):
        format = {"latitude" : lat, "longitude" : lon, "maxradius" : maxradius, "minradius" : minradius}
        if(mindepth is not None):
            format['mindepth'] = mindepth
        if(maxdepth is not None):
            format['maaxdepth'] = maxdepth
        return format
    
    @staticmethod
    def __format_magnitude(minmag=None, maxmag=None, magtype= None):
        download_format = {}
        if(minmag is not None):
            download_format["minmag"] = minmag
        if(maxmag is not None):
            download_format["maxmag"] = maxmag
        if(magtype is None):
            download_format["magnitudetype"] = "preferred"
        return download_format
    
    @staticmethod
    def __format_url(base_url, formats):
        query_str = base_url 
        for format in formats:
            for key, val in format.items():
                query_str += str(key) + "=" +str(val) + "&"
        return query_str
    
    # STATIONS
    @staticmethod
    def process_stations_url(url):
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.content.decode()), comment="#", sep="|")
        if('Latitude' in df.columns):
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        if('Longitude' in df.columns):
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        if('Elevation' in df.columns):
            df['Elevation'] = pd.to_numeric(df['Elevation'], errors='coerce')
        #df.dropna(inplace=True)
        df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
        df['EndTime'] = pd.to_datetime(df['EndTime'], errors='coerce')
        return df
    
    @staticmethod
    def retrieve_all_stations():
        return irisRequests.process_stations_url( irisRequests.stations_base_url + "&format=geocsv")    
    @staticmethod
    def retrieve_networks():
        return irisRequests.process_stations_url( irisRequests.stations_base_url +  "level=network&format=geocsv")  
    @staticmethod
    def retrieve_stations(network):
        return irisRequests.process_stations_url( irisRequests.stations_base_url + "level=station&network=" + network + "&format=geocsv")
    @staticmethod
    def retrieve_channels(network, station):
        return irisRequests.process_stations_url( irisRequests.stations_base_url + "level=channel&network=" + network + "&station=" + station + "&format=geocsv")
    @staticmethod
    def url_stations_box(minlat, maxlat, minlon, maxlon, start_time=None, end_time=None, download_format="geocsv" ):
        if(start_time is not None and end_time is not None):
            time_f = irisRequests.__format_time(start_time, end_time)
        else:
            time_f = None
        box_f = irisRequests.__format_box(minlat, maxlat, minlon, maxlon)
        formats = []
        if(time_f != None):
            formats.append(time_f)
        formats.append(box_f)
        url = irisRequests.__format_url(irisRequests.stations_base_url, formats)+ "format=" + download_format
        return url
    @staticmethod
    def url_stations_radial(lat, lon, maxradius, minradius=0, start_time=None, end_time=None, download_format="geocsv" ):
        if(start_time is not None and end_time is not None):
            time_f = irisRequests.__format_time(start_time, end_time)
        else:
            time_f = None
        radial_f = irisRequests.__format_radial(lat, lon, maxradius, minradius)
        formats = []
        if(time_f is not None):
            formats.append(time_f)
        formats.append(radial_f)
        url = irisRequests.__format_url(irisRequests.stations_base_url, formats) + "format=" + download_format
        return url
    
    @staticmethod
    def url_data(network, station,start, end, channel=None, quality="B", repo="primary"):
        '''
            Creates the url for downloading a waveform from a specified station in a given network.

            Parameters:
                network (string): network's name
                station (string): station's name
                start (datetime): time origin of the waveform
                end (datetime): time end of the waveform
                channel (string): if None, all the available channels are included.

            Returns:
                url (string): the url to download waveform data.
        '''
        data_f = {"network" : network,
                    "station" : station,
                    "quality" : quality,
                    "repo" : repo}
        if(channel is not None):
            data_f["channel"] = channel
        time_f = irisRequests.__format_time(start, end)
        download_format = "geocsv"
        if(channel is None):
            download_format += ".zip"
        return irisRequests.__format_url(irisRequests.data_base_url, [data_f, time_f]) + "format=" + download_format

    @staticmethod
    def __process_data_file(file):
        numerical_fields  = ["sample_rate_hz", "latitude_deg", "longitude_deg",
                              "elevation_m", "depth_m", "azimuth_deg", "dip_deg", 
                              "scale_factor", "scale_frequency_hz"]
        header = {}
        for line in file.readlines():
            line = line.decode()
            if(line.startswith('#')):
                fields = re.split(r"\s+", line.strip())[1:]
                field_name, field_val = fields[0], fields[1]
                field_name = field_name[:-1]
                if(field_name in numerical_fields):
                    header[field_name] = float(field_val)          
        file.seek(0)
        df = pd.read_csv(file, comment="#", sep=",", parse_dates=["Time"])
        df.rename({" Sample" : "Sample"}, axis=1, inplace=True)
        return df, header
    
    @staticmethod
    def process_data_url(url):
        filehandle,_ = urllib.request.urlretrieve(url )
        if(".zip" in url):

            zip_file_object = zipfile.ZipFile(filehandle, 'r')
            dataframes = {}
            for nl in zip_file_object.namelist():
                channel = re.findall(r"\.CH[A-Za-z]\.", nl )[0][1:-1]
                file = zip_file_object.open(nl)
                df, header = irisRequests.__process_data_file(file)
                dataframes[channel] = (df,header) 
            return dataframes
        else:
            raise NotImplementedError("Download of one single channel is not yet available.")
    
    
    # EVENTS

    @staticmethod
    def get_events_catalogs():
        catalog_list = "http://service.iris.edu/fdsnws/event/1/catalogs"
        response = requests.get(catalog_list)
        names =  ET.parse(StringIO(response.content.decode('ascii'))).findall("Catalog")
        return [name.text.strip() for name in names]
    
    @staticmethod
    def url_events(formats):
        return irisRequests.__format_url(irisRequests.events_base_url, formats)
        
    @staticmethod
    def url_events_box(start_time, end_time, minlat, maxlat, minlon, maxlon, minmag=None, maxmag=None, magtype=None,format="geocsv"):
        time_f = irisRequests.__format_time(start_time, end_time)
        box_f = irisRequests.__format_box(minlat, maxlat, minlon, maxlon)
        mag_f = irisRequests.__format_magnitude(minmag,maxmag,  magtype)
        s = irisRequests.url_events([time_f, box_f, mag_f]) + "format=" + format
        return s
    
    @staticmethod
    def retrieve_events_box(start_time, end_time, minlat, maxlat, minlon, maxlon, minmag=None, maxmag=None, magtype=None):
        download_url = irisRequests.url_events_box(start_time, end_time, minlat, maxlat, minlon, maxlon, minmag, maxmag, magtype)
        df = pd.read_csv(download_url, sep="|", comment="#")
        df.Time = pd.to_datetime(df.Time, errors='coerce')
        df.dropna(axis=0, inplace=True)
        df.sort_values(by="Time", inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df