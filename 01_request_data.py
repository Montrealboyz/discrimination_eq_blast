import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader, RectangularDomain
import numpy as np
import pandas as pd
import os
df = pd.read_csv('./earthquake_catalog.csv')
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')


for time, latitude, longitude in zip(df['Time'], df['Latitude'], df['Longitude']):
    t = UTCDateTime(time)
    #waveformsfold = 'events/'+str(ID[0:6])+str(ID[8:])
    domain = CircularDomain(latitude=float(latitude), longitude=float(longitude),minradius=0.0, maxradius=1.5)
    restrictions = Restrictions(
        # Get data from 2 minutes before the event to 3 minutesr after the
        # event. This defines the temporal bounds of the waveform data.
        starttime= t - 30,
        endtime  = t + 90,
        # You might not want to deal with gaps in the data. If this setting is
        # True, any trace with a gap/overlap will be discarded.
        reject_channels_with_gaps=False,
        # And you might only want waveforms that have data for at least 70 % of
        # the requested time span. Any trace that is shorter than 70 % of the
        # desired total duration will be discarded.
        minimum_length=0.70,
        # No two stations should be closer than 10 km to each other. This is
        # useful to for example filter out stations that are part of different
        # networks but at the same physical station. Settings this option to
        # zero or None will disable that filtering.
        minimum_interstation_distance_in_m=1,
        # Only HH or BH channels. If a station has HH channels, those will be
        # downloaded, otherwise the BH. Nothing will be downloaded if it has
        # neither. You can add more/less patterns if you like.
        channel_priorities=["HH[ZNE12]", "BH[ZNE12]", "HN[ZNE12]", "EH[ZNE12]"],
        network=None,
        station=None,
        location=None,
        # Location codes are arbitrary and there is no rule as to which
        # location is best. Same logic as for the previous setting.
        location_priorities=["", "00", "10"])
    mdl = MassDownloader(providers=["IRIS"])
    waveformsfold = 'earthquake_waveform/'+t.strftime('%Y%m%d%H%M%S')
    stationfold  = 'earthquake_station/'+t.strftime('%Y%m%d%H%M%S')
    #print(waveformsfold)
    mdl.download(domain, restrictions, mseed_storage= waveformsfold,stationxml_storage= stationfold)