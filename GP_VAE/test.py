import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import sys
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
mus = tf.constant([.2, .5, .5, .3])
sigmas = tf.constant([.1, .4, .9, .3])
mapped = tf.constant([1, 3.2, 4.5, 6.6])

#out = tfd.MultivariateNormalDiag(loc=mapped[..., :self.z_size], scale_diag=tf.nn.softplus(mapped[..., self.z_size:]))
def round_up_by_ws(num, ws):
    '''
    Returns the smallest multiple of ws larger than num
    e.g. if ws is 4 and num is 6, then this will return 8'''
    return -((-num)//ws)*ws

things = tf.constant([[1, 2, 34],
                      [5, 6, 78]])



pulse_df_1 = pd.DataFrame(data={'ID': [14567, 14567, 14567, 14567, 14567, 14567, 14567],
                        'time_stamp': [0, 1, 2, 3, 11, 12, 13],
                        'signal': [130, np.nan, np.nan, 144, 145, 144.4, 156.7], 
                        'map': [0, 1, 1, 0, 0, 0, 0], 
                        'gender': ['M', 'M', 'M', 'M', 'M', 'M', 'M'], 
                        'unit': ['BPM', 'BPM', 'BPM', 'BPM', 'BPM', 'BPM', 'BPM'],
                        'age': [43, 43, 43, 43, 43, 43, 43],
                        'encounter_ID': [0, 0, 0, 0, 1, 1, 1]})


SpO2_df_1 = pd.DataFrame(data={'ID': [14567, 14567, 14567, 14567, 14567, 14567, 14567],
                        'time_stamp': [0, 1, 2, 3, 11, 12, 13], 
                        'signal': [99.2, 99.0, np.nan, 96.6, 97.8, 94.4, 95.5], 
                        'map': [0, 0, 1, 0, 0, 0, 0], 
                        'gender': ['M', 'M', 'M', 'M', 'M', 'M', 'M'], 
                        'unit': ['%', '%', '%', '%', '%', '%', '%'],
                        'age': [43, 43, 43, 43, 43, 43, 43],
                        'encounter_ID': [0, 0, 0, 0, 1, 1, 1]})


# = = = = = = = = =

pulse_df_2 = pd.DataFrame(data={'ID': [1234, 1234, 1234, 1234, 1234, 1234],
                        'time_stamp': [0, 1, 2, 3, 4, 5], 
                        'signal': [125, 125.1, np.nan, 144, 145, 144.4], 
                        'map': [0, 0, 1, 0, 0, 0], 
                        'gender': ['M', 'M', 'M', 'M', 'M', 'M'], 
                        'unit': ['BPM', 'BPM', 'BPM', 'BPM', 'BPM', 'BPM'],
                        'age': [21, 21, 21, 21, 21, 21],
                        'encounter_ID': [0, 0, 0, 0, 0, 0]})


SpO2_df_2 = pd.DataFrame(data={'ID': [1234, 1234, 1234, 1234, 1234, 1234],
                        'time_stamp': [0, 1, 2, 3, 4, 5], 
                        'signal': [99.2, 100, np.nan, 96.6, 97.8, 94.4], 
                        'map': [0, 0, 1, 0, 0, 0], 
                        'gender': ['M', 'M', 'M', 'M', 'M', 'M'], 
                        'unit': ['%', '%', '%', '%', '%', '%'],
                        'age': [21, 21, 21, 21, 21, 21],
                        'encounter_ID': [0, 0, 0, 0, 0, 0]})


df = pulse_df_1.append(pulse_df_2)
df.index = [val for val in range(len(df))]
print(df)

PIDS = list(set(df['ID']))
indexes = []
x = []
masks = []
for id in PIDS:
    indexes.append(df[df['ID']==id].index.values[0])
indexes.sort()
indexes.append(len(df))
longest_encounter = 0

# set longest_encounter value
for i in range(len(indexes)-1):
    ind1 = indexes[i]
    ind2 = indexes[i+1]
    
    data_for_patient = df[ind1:ind2] # df for one patient for this particular signal
    encounters = sorted(list(data_for_patient['encounter_ID']))
    for encounter in encounters:
        if encounters.count(encounter) > longest_encounter:
            longest_encounter = encounters.count(encounter)




masks = np.array(masks)
signals = []

my_df = pd.DataFrame(columns=['ID', 'time_stamp', 'signal', 'map', 'encounter_ID'])
print(my_df)
if len(my_df) == 0:
    my_df = pulse_df_2[['ID', 'time_stamp', 'signal', 'map', 'encounter_ID']]


if len(my_df) != 1:
    my_df = pd.concat([my_df, pulse_df_1[['ID', 'time_stamp', 'signal', 'map', 'encounter_ID']] ])

my_df.rename({'signal': 'Pulse', 'map': 'Pulse_map'}, axis='columns', inplace=True)
my_df = my_df.sort_values(by=['ID', 'encounter_ID'])
signals.append(my_df)

print('===')

my_df2 = pd.DataFrame(columns=['ID', 'time_stamp', 'signal', 'map', 'encounter_ID'])
print(my_df2)
if len(my_df2) == 0:
    my_df2 = SpO2_df_2[['ID', 'time_stamp', 'signal', 'map', 'encounter_ID']]


if len(my_df) != 1:
    my_df2 = pd.concat([my_df2, SpO2_df_1[['ID', 'time_stamp', 'signal', 'map', 'encounter_ID']] ])

my_df2.rename({'signal': 'SpO2', 'map': 'SpO2_map'}, axis='columns', inplace=True)
my_df2 = my_df2.sort_values(by=['ID', 'encounter_ID'])
signals.append(my_df2)
print(my_df2)


from functools import reduce
normal_data = reduce(lambda left, right: left.merge(right, on=['ID', 'time_stamp'], how='outer'), signals)
print("=====================")
print(normal_data)

signal_list = ['Pulse', 'SpO2']
#for signal_type in signal_list:
#    normal_data['%s_map' % signal_type] = normal_data['%s_map' % signal_type].isnull().apply(int)
normal_data = normal_data.sort_values(by=['ID', 'time_stamp'])
#i = normal_data['time_stamp']
#encounter_ID = i.ge(i.shift() + 12 * 10 * 5).cumsum()
#normal_data.insert(1, 'encounter_ID', encounter_ID)
#normal_data[['etCO2', 'awRR', 'CVPm']] = \
#    normal_data[['etCO2', 'awRR', 'CVPm']].fillna(0)
#normal_data[signal_list] = normal_data[signal_list].fillna(method='ffill')
#normal_data = normal_data.fillna(0)

print('=#=#')
normal_data = normal_data.fillna(0)


signal_list = ['Pulse', 'SpO2']
signal_maps = [signal + '_map' for signal in signal_list]
num_features = len(signal_list)
window_size = 4
train_data = []
train_maps = []
for patient_data in normal_data.groupby('ID'):
    for encounter_data in patient_data[1].groupby('encounter_ID_x'):
        encounter_data = encounter_data[1]
        datapoint = encounter_data[signal_list].to_numpy().transpose()
        map = encounter_data[signal_maps].to_numpy().transpose() # transpose makes these num_features x longest_encounter
        print(datapoint)
        # This pads both the data and map with 0's. We don't pad map with 1's since we don't want the nll
        # of the GP VAE to be impacted by this.
        if datapoint.shape[1] > window_size:
            pad_amt = round_up_by_ws(datapoint.shape[1], window_size) - datapoint.shape[1]# how much to pad so that we can have an even # of datapoints of window_size len
            datapoint = np.pad(datapoint, ((0, 0), (0, pad_amt)), 'constant')
            map = np.pad(map, ((0, 0), (0, pad_amt)), 'constant')

            datapoint = np.transpose(datapoint.transpose().reshape((-1, window_size, num_features)), axes=(0, 2, 1))
            map = np.transpose(map.transpose().reshape((-1, window_size, num_features)), axes=(0, 2, 1))

            for dpt, mp in zip(datapoint, map):
                # dpt and mp are of size num_features x window_size
                train_data.append(dpt)
                train_maps.append(mp)

        else:
            datapoint = np.pad(datapoint, ((0, 0), (0, window_size-datapoint.shape[1])), 'constant')
            map = np.pad(map, ((0, 0), (0, window_size-map.shape[1])), 'constant')
            
            train_data.append(datapoint)
            train_maps.append(map)
        

train_data = np.dstack(train_data)
train_maps = np.dstack(train_maps)
train_data = np.rollaxis(train_data, -1)
train_maps = np.rollaxis(train_maps, -1)

