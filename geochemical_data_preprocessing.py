import functools
import glob
import numpy as np
import os
import pandas as pd

# concatenate csv files
geochem_raw_path = './Datasets/Geochemistry/surface_geochem_results.csv'

if os.path.isfile(geochem_raw_path):
    print('The geochemical data file already exists.')
    geochem_raw = pd.read_csv(geochem_raw_path, index_col=False)
else:
    files = os.path.join('./Datasets/Geochemistry/', '*_surface_geochem_results.csv')
    files = glob.glob(files)
    geochem_raw = pd.concat(map(functools.partial(pd.read_csv, index_col=False), files), ignore_index=True)
    geochem_raw.to_csv(geochem_raw_path, index=False)

# unique_source = geochem_raw.SAMPLE_SOURCE.unique()
unique_source = ['Soil', 'Rock outcrop / float', 'Stream sediment']
geochem_data = []

for i in unique_source:
    geochem_data.append(geochem_raw[geochem_raw.SAMPLE_SOURCE==i].values)

soil_raw = geochem_data[0]
rock_raw = geochem_data[1]
stream_raw = geochem_data[2]

soil_elements = np.unique(soil_raw[:, 32])
rock_elements = np.unique(rock_raw[:, 32])
stream_elements = np.unique(stream_raw[:, 32])

soil_temp = []
rock_temp = []
stream_temp = []

for i in soil_elements:
    soil_temp.append(soil_raw[np.where(soil_raw[:, 32] == i)])

for i in rock_elements:
    rock_temp.append(rock_raw[np.where(rock_raw[:, 32] == i)])

for i in stream_elements:
    stream_temp.append(stream_raw[np.where(stream_raw[:, 32] == i)])

soil_data = soil_temp
rock_data = rock_temp
stream_data = stream_temp

# replace censored values
for i in range(len(soil_data)):
    for j in range(soil_data[i].shape[0]):
        if type(soil_data[i][j, 33]) == str and soil_data[i][j, 33].startswith('<'):
            soil_data[i][j, 33] = float(soil_data[i][j, 33][1:])*(3/4)
        elif type(soil_data[i][j, 33]) == str and soil_data[i][j, 33].startswith('>'):
            soil_data[i][j, 33] = float(soil_data[i][j, 33][1:])*(4/3)
        else:
            soil_data[i][j, 33] = float(soil_data[i][j, 33])

for i in range(len(rock_data)):
    for j in range(rock_data[i].shape[0]):
        if type(rock_data[i][j, 33]) == str and rock_data[i][j, 33].startswith('<'):
            rock_data[i][j, 33] = float(rock_data[i][j, 33][1:])*(3/4)
        elif type(rock_data[i][j, 33]) == str and rock_data[i][j, 33].startswith('>'):
            rock_data[i][j, 33] = float(rock_data[i][j, 33][1:])*(4/3)
        else:
            rock_data[i][j, 33] = float(rock_data[i][j, 33])

for i in range(len(stream_data)):
    for j in range(stream_data[i].shape[0]):
        if type(stream_data[i][j, 33]) == str and stream_data[i][j, 33].startswith('<'):
            stream_data[i][j, 33] = float(stream_data[i][j, 33][1:])*(3/4)
        elif type(stream_data[i][j, 33]) == str and stream_data[i][j, 33].startswith('>'):
            stream_data[i][j, 33] = float(stream_data[i][j, 33][1:])*(4/3)
        else:
            stream_data[i][j, 33] = float(stream_data[i][j, 33])

# check and harmonize the unit of concentration values
geochem_unit = geochem_raw.UNIT.unique()

soil_units = []

for i in range(len(soil_data)):
    soil_unit = list(np.unique(soil_data[i][:, 34]))
    soil_units.extend(soil_unit)

soil_units = list(set(soil_units))

rock_units = []

for i in range(len(rock_data)):
    rock_unit = list(np.unique(rock_data[i][:, 34]))
    rock_units.extend(rock_unit)

rock_units = list(set(rock_units))

stream_units = []

for i in range(len(stream_data)):
    stream_unit = list(np.unique(stream_data[i][:, 34]))
    stream_units.extend(stream_unit)

stream_units = list(set(stream_units))

soil_elements_units = soil_elements.copy()
rock_elements_units = rock_elements.copy()
stream_elements_units = stream_elements.copy()

for i in range(len(soil_data)):
    if len(np.unique(soil_data[i][:, 34])) > 1:
        if sorted(np.unique(soil_data[i][:, 34])) == ['%', 'ppm']:
            soil_elements_units[i] = soil_elements_units[i] + '_ppm'
            for j in range(soil_data[i].shape[0]):
                if soil_data[i][j, 34] == '%': # convert % to ppm
                    soil_data[i][j, 33] = soil_data[i][j, 33]*10**4
        else:
            soil_elements_units[i] = soil_elements_units[i] + '_ppb'
            for j in range(soil_data[i].shape[0]):
                if soil_data[i][j, 34] == '%': # convert % to ppb
                    soil_data[i][j, 33] = soil_data[i][j, 33]*10**7
                elif soil_data[i][j, 34] == 'ppm': # convert ppm to ppb
                    soil_data[i][j, 33] = soil_data[i][j, 33]*10**3
        # print('The number of units for ' + soil_data[i][0, 32] + ' in the soil dataset is more than one')
    else:
        soil_elements_units[i] = soil_elements_units[i] + '_' + soil_data[i][0, 34]

for i in range(len(rock_data)):
    if np.unique(rock_data[i][:, 34]).tolist() == ['X']:
        print(f'Warning for X unit: rock_data, element = {i}')
        break
    if len(np.unique(rock_data[i][:, 34])) > 1:
        if sorted(np.unique(rock_data[i][:, 34])) == ['%', 'ppm']:
            rock_elements_units[i] = rock_elements_units[i] + '_ppm'
            for j in range(rock_data[i].shape[0]):
                if rock_data[i][j, 34] == '%': # convert % to ppm
                    rock_data[i][j, 33] = rock_data[i][j, 33]*10**4
        elif 'X' in np.unique(rock_data[i][:, 34]):
            rock_elements_units[i] = rock_elements_units[i] + '_ppb'
            mask = rock_data[i][:, 34] == 'X'
            for j in range(rock_data[i].shape[0]):
                if rock_data[i][j, 34] == 'ppm':
                    rock_data[i][j, 33] = rock_data[i][j, 33]*10**3
                elif rock_data[i][j, 34] == '%':
                    rock_data[i][j, 33] = rock_data[i][j, 33]*10**7
            rock_data[i] = rock_data[i][~mask]
        else:
            rock_elements_units[i] = rock_elements_units[i] + '_ppb'
            for j in range(rock_data[i].shape[0]):
                if rock_data[i][j, 34] == '%': # convert % to ppb
                    rock_data[i][j, 33] = rock_data[i][j, 33]*10**7
                elif rock_data[i][j, 34] == 'ppm': # convert ppm to ppb
                    rock_data[i][j, 33] = rock_data[i][j, 33]*10**3
        # print('The number of units for ' + rock_data[i][0,32] + ' in the rock dataset is more than one')
    else:
        rock_elements_units[i] = rock_elements_units[i] + '_' + rock_data[i][0, 34]

for i in range(len(stream_data)):
    if len(np.unique(stream_data[i][:, 34])) > 1:
        if sorted(np.unique(stream_data[i][:, 34])) == ['%', 'ppm']:
            stream_elements_units[i] = stream_elements_units[i] + '_ppm'
            for j in range(stream_data[i].shape[0]):
                if stream_data[i][j, 34] == '%': # convert % to ppm
                    stream_data[i][j, 33] = stream_data[i][j, 33]*10**4
        else:
            stream_elements_units[i] = stream_elements_units[i] + '_ppb'
            for j in range(stream_data[i].shape[0]):
                if stream_data[i][j, 34] == '%': # convert % to ppb
                    stream_data[i][j, 33] = stream_data[i][j, 33]*10**7
                elif stream_data[i][j, 34] == 'ppm': # convert ppm to ppb
                    stream_data[i][j, 33] = stream_data[i][j, 33]*10**3
        # print('The number of units for ' + stream_data[i][0, 32] + ' in the stream dataset is more than one')
    else:
        stream_elements_units[i] = stream_elements_units[i] + '_' + stream_data[i][0, 34]

# replace outliers
for i in range(len(soil_data)):
    q1 = np.quantile(soil_data[i][:, 33], 0.25)
    q3 = np.quantile(soil_data[i][:, 33], 0.75)
    iqr = q3-q1
    median = np.median(soil_data[i][:, 33])
    for j in range(soil_data[i].shape[0]):
        if soil_data[i][j, 33] < q1-1.5*iqr or soil_data[i][j, 33] > q3+1.5*iqr:
            soil_data[i][j, 33] = median

for i in range(len(rock_data)):
    q1 = np.quantile(rock_data[i][:, 33], 0.25)
    q3 = np.quantile(rock_data[i][:, 33], 0.75)
    iqr = q3-q1
    median = np.median(rock_data[i][:, 33])
    for j in range(rock_data[i].shape[0]):
        if rock_data[i][j, 33] < q1-1.5*iqr or rock_data[i][j, 33] > q3+1.5*iqr:
            rock_data[i][j, 33] = median

for i in range(len(stream_data)):
    q1 = np.quantile(stream_data[i][:, 33], 0.25)
    q3 = np.quantile(stream_data[i][:, 33], 0.75)
    iqr = q3-q1
    median = np.median(stream_data[i][:, 33])
    for j in range(stream_data[i].shape[0]):
        if stream_data[i][j, 33] < q1-1.5*iqr or stream_data[i][j, 33] > q3+1.5*iqr:
            stream_data[i][j, 33] = median
            
# create output files
soil_data_conc = np.concatenate(soil_data, axis=0)
soil_data_dict = {}

for row in soil_data_conc:
    sample_no = row[0]
    lon = row[25]
    lat = row[26]
    element = row[32]
    concentration = row[33]

    if sample_no not in soil_data_dict:
        soil_data_dict[sample_no] = {'sample_no': sample_no, 'lon_gda94': lon, 'lat_gda94': lat}

    soil_data_dict[sample_no][element] = concentration
    
soil_data_lst = list(soil_data_dict.values())
soil_data_df = pd.DataFrame(soil_data_lst)
# columns to keep in their current place
columns_to_keep = soil_data_df.columns[:3]
# columns to sort
columns_to_sort = soil_data_df.columns[3:]
# sort the columns
sorted_columns = sorted(columns_to_sort)
# concatenate columns in the desired order
soil_data_df = pd.concat([soil_data_df[columns_to_keep], soil_data_df[sorted_columns]], axis=1)
soil_data_df.columns = list(columns_to_keep) + sorted(soil_elements_units)
soil_data_df.to_csv('./Datasets/Geochemistry/soil_data.csv', index=False)

rock_data_conc = np.concatenate(rock_data, axis=0)
rock_data_dict = {}

for row in rock_data_conc:
    sample_no = row[0]
    lon = row[25]
    lat = row[26]
    element = row[32]
    concentration = row[33]

    if sample_no not in rock_data_dict:
        rock_data_dict[sample_no] = {'sample_no': sample_no, 'lon_gda94': lon, 'lat_gda94': lat}

    rock_data_dict[sample_no][element] = concentration
    
rock_data_lst = list(rock_data_dict.values())
rock_data_df = pd.DataFrame(rock_data_lst)
# columns to keep in their current place
columns_to_keep = rock_data_df.columns[:3]
# columns to sort
columns_to_sort = rock_data_df.columns[3:]
# sort the columns
sorted_columns = sorted(columns_to_sort)
# concatenate columns in the desired order
rock_data_df = pd.concat([rock_data_df[columns_to_keep], rock_data_df[sorted_columns]], axis=1)
rock_data_df.columns = list(columns_to_keep) + sorted(rock_elements_units)
rock_data_df.to_csv('./Datasets/Geochemistry/rock_data.csv', index=False)

stream_data_conc = np.concatenate(stream_data, axis=0)
stream_data_dict = {}

for row in stream_data_conc:
    sample_no = row[0]
    lon = row[25]
    lat = row[26]
    element = row[32]
    concentration = row[33]

    if sample_no not in stream_data_dict:
        stream_data_dict[sample_no] = {'sample_no': sample_no, 'lon_gda94': lon, 'lat_gda94': lat}

    stream_data_dict[sample_no][element] = concentration
    
stream_data_lst = list(stream_data_dict.values())
stream_data_df = pd.DataFrame(stream_data_lst)
# columns to keep in their current place
columns_to_keep = stream_data_df.columns[:3]
# columns to sort
columns_to_sort = stream_data_df.columns[3:]
# sort the columns
sorted_columns = sorted(columns_to_sort)
# concatenate columns in the desired order
stream_data_df = pd.concat([stream_data_df[columns_to_keep], stream_data_df[sorted_columns]], axis=1)
stream_data_df.columns = list(columns_to_keep) + sorted(stream_elements_units)
stream_data_df.to_csv('./Datasets/Geochemistry/stream_data.csv', index=False)
