import numpy as np
import pandas as pd
import pyarrow as pa
import datetime
import random
import os, psutil
from data_generator.utils import round_up_by_ws



def df_to_numpy(merged_df, min_seq_len, max_seq_len, signal_list, missingness_threshold=0.4):
    print('min_seq_len: ', min_seq_len)
    print('max_seq_len: ', max_seq_len)
    print('signal list: ', signal_list)
    
    print()
    print()
    num_features = len(signal_list)

    data = []
    maps = []
    
    PIDs = []

    num_too_long = 0
    num_too_short = 0
    num_in_between = 0
    total_num_patients = 0
    samples_too_much_missingness = 0
    original_num_samples = 0

    for patient_data in merged_df.groupby('patientid'):
        total_num_patients += 1
        patientid, patient_data = patient_data
        
        patient_data = patient_data.drop_duplicates(subset='reldatetime')
        datapoint = patient_data[signal_list].to_numpy().transpose() # transpose makes these (num_features, encounter_len)
        #print("datapoint shape: ", datapoint.shape, patientid)
        map = np.ones_like(datapoint)
        if datapoint.shape[1] < min_seq_len:
            num_too_short += 1
            continue # seq len is too small, drop this datapoint
                

        # This pads the data with 0's and the map with 1's
        if datapoint.shape[1] > max_seq_len:
            pad_amt = round_up_by_ws(datapoint.shape[1], max_seq_len) - datapoint.shape[1]# how much to pad so that we can have an even # of datapoints of max_seq_len len
            pad_amt = int(pad_amt)

            # Pad on the left side so we dont pad near the end of the signal
            datapoint = np.pad(datapoint, ((0, 0), (pad_amt, 0)), 'constant')
            map = np.pad(map, ((0, 0), (pad_amt, 0)), 'constant', constant_values=((0, 0), (1, 0)))
            

            datapoint = np.transpose(datapoint.transpose().reshape((-1, max_seq_len, num_features)), axes=(0, 2, 1))
            map = np.transpose(map.transpose().reshape((-1, max_seq_len, num_features)), axes=(0, 2, 1))
            

            for i, (dpt, mp) in enumerate(zip(datapoint, map)):
                original_num_samples += 1
                # dpt and mp are of size num_features x window_size
                if i == 0: # first dpt in datapoint
                    if dpt.shape[1] - pad_amt < min_seq_len: # if the unpadded data is shorter than min_seq_len, then don't include it
                        num_too_short += 1
                        continue
                if np.count_nonzero(mp == 0) / (mp.shape[0]*mp.shape[1]) <= missingness_threshold: # If missingness is below threshold, include it in the dataset
                    data.append(dpt)
                    maps.append(mp)
                    PIDs.append(patientid)
                    
                else:
                    samples_too_much_missingness += 1

                

        else:
            num_in_between += 1
            original_num_samples += 1
            pad_amt = max_seq_len-datapoint.shape[1]
            datapoint = np.pad(datapoint, ((0, 0), (pad_amt, 0)), 'constant')
            map = np.pad(map, ((0, 0), (pad_amt, 0)), 'constant', constant_values=((0, 0), (1, 0)))
            
            
            if np.count_nonzero(map == 1) / (map.shape[0]*map.shape[1]) <= missingness_threshold:
                data.append(datapoint)
                maps.append(map)
                PIDs.append(patientid)
                
            else:
                samples_too_much_missingness += 1
    

    data = np.stack(data)
    maps = np.stack(maps)
    PIDs = np.array(PIDs)

    data_maps = np.hstack((data, maps)) # data_maps is of shape (num_samples, 2*num_features, seq_len). Each sample and map has been concatenated into one matrix
    data_maps = np.reshape(data_maps, (data_maps.shape[0], 2, num_features, max_seq_len)) # splits up the data from maps, so shape is now (num_samples, 2, num_features, max_seq_len)


    print('Num too short (dropped): ', num_too_short)
    print('Num too long (kept): ', num_too_long)
    print('Num in between (kept): ', num_in_between)
    print('Total num patients: ', total_num_patients)
    return data_maps, PIDs

def mortality_and_24hrs_df_to_numpy(merged_df, discharge_status, min_seq_len, max_seq_len, signal_list, truncate_amount):
    print('min_seq_len: ', min_seq_len)
    print('max_seq_len: ', max_seq_len)
    print('signal list: ', signal_list)
    print('truncate_amount: ', truncate_amount)
    print()
    print()
    num_features = len(signal_list)

    data = []
    maps = []
    mortality_labels = []
    PIDs = []
    first_24_hrs_data = []
    first_24_hrs_maps = []

    num_too_long = 0
    num_too_short = 0
    num_in_between = 0
    num_unknown_discharge = 0
    total_num_patients = 0

    for patient_data in merged_df.groupby('patientid'):
        total_num_patients += 1
        patientid, patient_data = patient_data
        
        patient_data = patient_data.drop_duplicates(subset='reldatetime')
        datapoint = patient_data[signal_list].to_numpy().transpose() # transpose makes these (num_features, encounter_len)
        #print("datapoint shape: ", datapoint.shape, patientid)

        if datapoint.shape[1] < min_seq_len or discharge_status[discharge_status['patientid']==patientid]['discharge_status'].isnull().values.any():
            if datapoint.shape[1] < min_seq_len:
                num_too_short += 1
            else:
                num_unknown_discharge += 1
            continue
        else:
            datapoint = datapoint[:, :datapoint.shape[1] - min(datapoint.shape[1], truncate_amount)]
            # print("datapoint shape: ", datapoint.shape, patientid)
            if datapoint.shape[1] > max_seq_len:
                num_too_long += 1
                datapoint = datapoint[:, -max_seq_len:]
                mask = np.ones(datapoint.shape) # all observed
                data.append(datapoint)
                PIDs.append(int(patientid))
                maps.append(mask)
                label = np.zeros(datapoint.shape[1])
                if discharge_status[discharge_status['patientid']==patientid]['discharge_status'].item() == 'dead':
                    label[-truncate_amount:] = 1
                mortality_labels.append(label)
                # Take first 24 hrs of data
                first_24_hrs_data.append(datapoint[:, :int((24*60*60)/300)])
                first_24_hrs_maps.append(np.ones_like(first_24_hrs_data[-1]))

            else:
                num_in_between += 1
                # In the case that the datapoint is between the min and max seq len, then left pad the data so its max_seq_len long
                pad_amt = max_seq_len-datapoint.shape[1]
                datapoint = np.pad(array=datapoint, pad_width=((0, 0), (pad_amt, 0)), mode='edge') # left pads data with the first observed value. e.g. 1, 2, 3 padded by 2 values becomes 1, 1, 1, 2, 3
                data.append(datapoint)
                PIDs.append(int(patientid))
                mask = np.ones(datapoint.shape)
                mask[:, 0:pad_amt] = 0 # mask the padded part as fully imputed
                maps.append(mask)
                label = np.zeros(datapoint.shape[1])
                if discharge_status[discharge_status['patientid']==patientid]['discharge_status'].item() == 'dead':
                    label[-truncate_amount:] = 1
                mortality_labels.append(label)
                # Take first 24 hrs of data
                first_24_hrs_data.append(datapoint[:, :int((24*60*60)/300)])
                first_24_hrs_maps.append(np.ones_like(first_24_hrs_data[-1]))
    

    data = np.stack(data)
    first_24_hrs_data = np.stack(first_24_hrs_data)
    maps = np.stack(maps)
    first_24_hrs_maps = np.stack(first_24_hrs_maps)
    mortality_labels = np.stack(mortality_labels)
    PIDs = np.array(PIDs)

    data_maps = np.hstack((data, maps)) # data_maps is of shape (num_samples, 2*num_features, seq_len). Each sample and map has been concatenated into one matrix
    data_maps = np.reshape(data_maps, (data_maps.shape[0], 2, num_features, max_seq_len)) # splits up the data from maps, so shape is now (num_samples, 2, num_features, max_seq_len)

    first_24_hrs_data_maps = np.hstack((first_24_hrs_data, first_24_hrs_maps)) # data_maps is of shape (num_samples, 2*num_features, seq_len). Each sample and map has been concatenated into one matrix
    first_24_hrs_data_maps = np.reshape(first_24_hrs_data_maps, (first_24_hrs_data_maps.shape[0], 2, num_features, int((24*60*60)/300))) # splits up the data from maps, so shape is now (num_samples, 2, num_features, max_seq_len)

    print('Num too short (dropped): ', num_too_short)
    print('Num unkonwn discharge status (dropped): ', num_unknown_discharge)
    print('Num too long (kept): ', num_too_long)
    print('Num in between (kept): ', num_in_between)
    print('Total num patients: ', total_num_patients)
    return data_maps, mortality_labels, PIDs, first_24_hrs_data_maps

def normalize_signals(train_signals, test_signals):
    '''
    Normalizes the signals
    '''
    train_signals_normalized = train_signals.copy()
    test_signals_normalized = test_signals.copy()
    
    if len(train_signals.shape)==4:
        n, c, f, l = train_signals.shape
    else:
        n, f, l = train_signals.shape

    train_signals_normalized[:, 1, :, :] = (train_signals[:, 0, :, :]==0).astype('float32')
    test_signals_normalized[:, 1, :, :] = (test_signals[:, 0, :, :]==0).astype('float32')
    
    train_means = []
    train_stds = []

    for feature in range(f):
        train_signal_mean = np.mean(train_signals[:,0,feature,:][train_signals[:,1,feature, :]==1].reshape(-1,)) # Takes the mean in the places where the map == 1 (i.e. non missing values)
        train_signal_std = np.std(train_signals[:,0,feature,:][train_signals[:,1,feature, :]==1].reshape(-1,))
    
        train_means.append(train_signal_mean)
        train_stds.append(train_signal_std)

        
        train_signals_normalized[:, 0, feature, :] = (train_signals[:, 0, feature, :] - train_signal_mean) / (
                1 if float(train_signal_std) == 0 else float(train_signal_std))
        test_signals_normalized[:, 0, feature, :] = (test_signals[:, 0, feature, :] - train_signal_mean) / (
                1 if float(train_signal_std) == 0 else float(train_signal_std))
    
    
    specs = np.array([train_means, train_stds])
    # specs is of shape (2, num_features). The first row is the average value for each feature in the train set, the second row is the std for each variable in the trainset.
    return train_signals_normalized, test_signals_normalized, specs


def get_circulatory_failure_data(merged_df, admission_times, time_of_vasopressor_or_inotrope, signal_list):
    '''
    According to the original paper (https://www.nature.com/articles/s41591-020-0789-4)
    A patient is defined as being in circulatory failure if '(1) arterial lactate is 
    elevated (≥2 mmol/l), and (2) either mean arterial pressure (MAP) <= 65 mmHg, 
    or the patient is receiving vasopressors or inotropes'

    Arterial lactate is vm136, MAP is vm5, and vassopressor/inotrope info is input through 
    the time_of_first_vasopressor_or_inotrope dictionary.

    So circulatory failure is at points in time where (vm136 >=2) AND (vm5 <= 65 OR vasopressor/inotrope given)
    
    '''
    data = []
    maps = []
    PIDs = []
    labels = []


    vm136_ind = signal_list.index('vm136')
    vm5_ind = signal_list.index('vm5')
    len_of_one_day = int(24*12) # one observation every 5 min means there are 12 observations per hr, 24*12 observations per day
    num_features = len(signal_list)

    for patient_data in merged_df.groupby('patientid'):
        patientid, patient_data = patient_data
        
        patient_data = patient_data.drop_duplicates(subset='reldatetime')
        datapoint = patient_data[signal_list].to_numpy().transpose() # transpose makes these (num_features, encounter_len)
        
        
        label = np.ones(datapoint.shape[1])
        vm136_label = datapoint[vm136_ind] >= 2
        vm5_label = datapoint[vm5_ind] <= 65

        pharma_label = np.zeros_like(label)
        
        if patientid in time_of_vasopressor_or_inotrope:
            for time in time_of_vasopressor_or_inotrope[patientid]:
                ind = int(((admission_times[patientid] - time).total_seconds()/60)//5) # get time since admission in seconds, divide by 60 to get minutes, then divide by 5 to get 5 min intervals
                
                # Set the hour after the inotrope or vasopressor to 1 for the pharma label
                # i.e. in that hour, they are considered 'on a vassopressor/inotrope'
                pharma_label[ind: min(ind+12, len(label))] = 1
        
        label = np.logical_and(label, np.logical_and(vm136_label, np.logical_or(vm5_label, pharma_label)))
        if 1 in label:
            first_circ_failure_ind = np.where(label == 1)[0][0]
            # Now for circulatory failure patients, the signal ends with circulatory failure
            datapoint = datapoint[:, :first_circ_failure_ind]
            label = label[:first_circ_failure_ind] # Now label has no 1's
            label[-int(len_of_one_day/48):] = 1  # So we add a 30 min pre circulatory failure window of 1's
        
        mask = np.ones_like(datapoint)

        if 1 not in label: # Meaning this is NOT a patient who experiences circulatory failure
            if datapoint.shape[1] > 2*len_of_one_day:
                ind = random.randint(0, datapoint.shape[1] - 2*len_of_one_day)
                data.append(datapoint[:, ind:ind+2*len_of_one_day])
                maps.append(mask[:, ind:ind+2*len_of_one_day])
                labels.append(label[ind:ind+2*len_of_one_day])
                
                PIDs.append(patientid)
                
            elif datapoint.shape[1] < 2*len_of_one_day and datapoint.shape[1] > 1*len_of_one_day:
                pad_amt = int(2*len_of_one_day)-datapoint.shape[1]
                datapoint = np.pad(array=datapoint, pad_width=((0, 0), (pad_amt, 0)), mode='edge') # left pads data with the first observed value. e.g. 1, 2, 3 padded by 2 values becomes 1, 1, 1, 2, 3
                data.append(datapoint)
                
                
                mask = np.ones(datapoint.shape)
                mask[:, 0:pad_amt] = 0 # mask the padded part as fully imputed
                
                maps.append(mask)
                label = np.zeros(datapoint.shape[1])
                labels.append(label) # 0 labels since this is the case of no circ failure
                
                if label.shape[0] != 2*len_of_one_day:
                    print('label shape: 2', label.shape)
                    assert 2==1
                PIDs.append(patientid)
        else: # Meaning this is a patient who DOES experience circulatory failure
            if datapoint.shape[1] > 2*len_of_one_day:
                data.append(datapoint[:, -2*len_of_one_day:])
                
                maps.append(mask[:, -2*len_of_one_day:])
                
                PIDs.append(patientid)
                labels.append(label[-2*len_of_one_day:])
                if label[-2*len_of_one_day:].shape[0] != 2*len_of_one_day:
                    print('label shape: 3', label.shape)
                    assert 2==1
                


            elif datapoint.shape[1] < 2*len_of_one_day and datapoint.shape[1] > 1*len_of_one_day:
                pad_amt = int(2*len_of_one_day)-datapoint.shape[1]
                datapoint = np.pad(array=datapoint, pad_width=((0, 0), (pad_amt, 0)), mode='edge') # left pads data with the first observed value. e.g. 1, 2, 3 padded by 2 values becomes 1, 1, 1, 2, 3
                data.append(datapoint)
                
                mask = np.ones(datapoint.shape)
                mask[:, 0:pad_amt] = 0 # mask the padded part as fully imputed
                maps.append(mask)

                label = np.concatenate([np.zeros(pad_amt), label]) # pad label
                if label.shape[0] != 2*len_of_one_day:
                    print('label shape: 4', label.shape)
                    assert 2==1

                labels.append(label)


                PIDs.append(patientid)
                
    


    data = np.stack(data)
    
    maps = np.stack(maps)
    
    PIDs = np.array(PIDs)
    data_maps = np.hstack((data, maps)) # data_maps is of shape (num_samples, 2*num_features, seq_len). Each sample and map has been concatenated into one matrix
    data_maps = np.reshape(data_maps, (data_maps.shape[0], 2, num_features, 2*len_of_one_day)) # splits up the data from maps, so shape is now (num_samples, 2, num_features, max_seq_len)

    labels = np.stack(labels) # of shape (num_samples, seq_len)
    return data_maps, labels, PIDs
    



if __name__ == '__main__':
    random.seed(100) 
    process = psutil.Process(os.getpid())
    signal_list = ['vm1', 'vm3', 'vm4', 'vm5', 'vm13', 'vm20', 'vm28', 'vm62', 'vm136', 'vm146', 'vm172', 'vm174', 'vm176', 'pm41', 'pm42', 'pm43', 'pm44', 'pm87']

    process_mortality = True     # process and save the mortality data
    process_circulatory = False  # process and save the circulatory failure data. You may want to do one at a time due to potential memory constraints.
    

    if process_mortality:
        print('Memory Usage 1:')
        print(process.memory_info().rss/1e9, 'GB')
        dfs = []
        for part in range(0, 250):
            df = pd.read_parquet('DONTCOMMITdata/hirid/1.1.1/imputed_stage/imputed_stage_parquet/parquet/part-%d.parquet'%part)
            dfs.append(df)


        print('=================')
        all_data = pd.concat(dfs, ignore_index=True)
        print(all_data.nunique())

        all_data = all_data.sort_values(by=['patientid', 'reldatetime'])
        print(all_data.nunique())

        print('Memory Usage 2:')
        print(process.memory_info().rss/1e9, 'GB')

        print('Processing encoder training data: ')

        min_seq_len = int((48*60*60)/300) # 48 hrs in seconds, divided by 300 seconds (5 min) to get the number of 5 min intervals in 48 hrs
        max_seq_len = int((96*60*60)/300) # sets max to 4 days
        encoder_data_maps, encoder_PIDs = df_to_numpy(merged_df=all_data, min_seq_len=min_seq_len, max_seq_len=max_seq_len, signal_list=signal_list)

        inds = np.arange(len(encoder_data_maps))
        random.shuffle(inds)

        n_train = int(0.8*(len(encoder_data_maps)))
        train_encoder_data_maps = encoder_data_maps[:n_train]
        train_encoder_PIDs = encoder_PIDs[:n_train]
        TEST_encoder_data_maps = encoder_data_maps[n_train:]
        TEST_encoder_PIDs = encoder_PIDs[n_train:]

        train_encoder_data_maps, TEST_encoder_data_maps, normalization_specs = normalize_signals(train_encoder_data_maps, TEST_encoder_data_maps)

        np.save('DONTCOMMITdata/hirid_numpy/train_encoder_data_maps.npy', train_encoder_data_maps)
        np.save('DONTCOMMITdata/hirid_numpy/TEST_encoder_data_maps.npy', TEST_encoder_data_maps)
        np.save('DONTCOMMITdata/hirid_numpy/train_encoder_PIDs.npy', train_encoder_PIDs)
        np.save('DONTCOMMITdata/hirid_numpy/TEST_encoder_PIDs.npy', TEST_encoder_PIDs)

        print('Memory Usage 3:')
        print(process.memory_info().rss/1e9, 'GB')

        del train_encoder_data_maps
        del train_encoder_PIDs
        del TEST_encoder_data_maps
        del TEST_encoder_PIDs



        print('Memory Usage 4:')
        print(process.memory_info().rss/1e9, 'GB')

        

        reference = pd.read_csv('DONTCOMMITdata/hirid/1.1.1/general_table.csv')
        discharge_status = reference[['patientid', 'discharge_status']]

        
        min_seq_len = int((48*60*60)/300) # 48 hrs in seconds, divided by 300 seconds (5 min) to get the number of 5 min intervals in 48 hrs
        max_seq_len = int((96*60*60)/300) # sets max to 4 days
        truncate_amount = int((12*60*60)/300) # 12 hrs truncate


        mortality_data_maps, mortality_labels, first_24_hrs_PIDs, first_24_hrs_data_maps = mortality_and_24hrs_df_to_numpy(merged_df=all_data, discharge_status=discharge_status, min_seq_len=min_seq_len, max_seq_len=max_seq_len, signal_list=signal_list, truncate_amount=truncate_amount)
        print('min seq len: ', min_seq_len, 'max_seq_len: ', max_seq_len, 'truncate_amount: ', truncate_amount)
        print(mortality_data_maps.shape)
        print(mortality_labels.shape)

        

        print('Memory Usage 5:')
        print(process.memory_info().rss/1e9, 'GB')


        del all_data
        del dfs

        # Loading raw data
        groups = {}
        for part in range(250):
            print("Part:")
            print(part)
            df = pd.read_parquet('DONTCOMMITdata/hirid/1.1.1/raw_stage/observation_tables/parquet/part-%d.parquet'%part)
            
            for pid in first_24_hrs_PIDs:
                if not df[df['patientid']==pid].empty: # If this PID is in df
                    if df[(df['patientid'] == pid) & (df['variableid'] == 9990004)].empty:
                        # If there's no apache 4 group value, return the value for 9990002, or apache 2 if it exists. Else add -1
                        if df[(df['patientid'] == pid) & (df['variableid'] == 9990002)]['value'].empty:
                            groups[pid] = -1
                        else:
                            group = int(df[(df['patientid'] == pid) & (df['variableid'] == 9990002)]['value'].mode()[0])
                            groups[pid] = group
                    else: # If apache 4 is available though, use that
                        group = int(df[(df['patientid'] == pid) & (df['variableid'] == 9990004)]['value'].mode()[0])
                        groups[pid] = group
            
            del df
            
            
        
        Apache_Groups = [groups[pid] for pid in first_24_hrs_PIDs]
        Apache_Groups = np.array(Apache_Groups)
        print('Apache_Groups shape: ', Apache_Groups.shape)
        (unique, counts) = np.unique(Apache_Groups, return_counts=True)
        print('Distribution of Apache states:')
        for i in range(len(unique)):
            print(unique[i], ': ', counts[i])

        num_deaths = len(np.where(np.sum(mortality_labels, axis=1).reshape(-1,) > 0)[0])

        print('Total num of deaths in data', num_deaths)

        print('Memory Usage 6:')
        print(process.memory_info().rss/1e9, 'GB')

        inds = np.arange(len(mortality_data_maps))
        random.shuffle(inds)
        mortality_data_maps = mortality_data_maps[inds]
        mortality_labels = mortality_labels[inds]
        first_24_hrs_PIDs = first_24_hrs_PIDs[inds]
        Apache_Groups = Apache_Groups[inds]
        first_24_hrs_data_maps = first_24_hrs_data_maps[inds]

        train_mortality_data_maps = mortality_data_maps[0:int(0.8*len(mortality_data_maps))]
        train_mortality_labels = mortality_labels[0:int(0.8*len(mortality_labels))]
        train_first_24_hrs_PIDs = first_24_hrs_PIDs[0:int(0.8*len(first_24_hrs_PIDs))]
        train_Apache_Groups = Apache_Groups[0:int(0.8*len(Apache_Groups))]
        train_first_24_hrs_data_maps = first_24_hrs_data_maps[0:int(0.8*len(first_24_hrs_data_maps))]

        TEST_mortality_data_maps = mortality_data_maps[int(0.8*len(mortality_data_maps)):]
        TEST_mortality_labels = mortality_labels[int(0.8*len(mortality_labels)):]
        TEST_first_24_hrs_PIDs = first_24_hrs_PIDs[int(0.8*len(first_24_hrs_PIDs)):]
        TEST_Apache_Groups = Apache_Groups[int(0.8*len(Apache_Groups)):]
        TEST_first_24_hrs_data_maps = first_24_hrs_data_maps[int(0.8*len(first_24_hrs_data_maps)):]


        print("train_mortality_data_maps shape: ", train_mortality_data_maps.shape)
        print("train_mortality_labels shape: ", train_mortality_labels.shape)
        print("train_first_24_hrs_PIDs shape: ", train_first_24_hrs_PIDs.shape)
        print("train_Apache_Groups shape: ", train_Apache_Groups.shape)
        print('train_first_24_hrs_data_maps shape: ', train_first_24_hrs_data_maps.shape)

        print("TEST_mortality_data_maps shape: ", TEST_mortality_data_maps.shape)
        print("TEST_mortality_labels shape: ", TEST_mortality_labels.shape)
        print("TEST_first_24_hrs_PIDs shape: ", TEST_first_24_hrs_PIDs.shape)
        print("TEST_Apache_Groups shape: ", TEST_Apache_Groups.shape)
        print('TEST_first_24_hrs_data_maps shape: ', TEST_first_24_hrs_data_maps.shape)


        # train_mortality_data_maps, TEST_mortality_data_maps, normalization_specs = normalize_signals(train_mortality_data_maps, TEST_mortality_data_maps)
        # Now to normalize the first_24_hrs_data_maps and mortality_data_maps based on the normalization specs
        for feature in range(normalization_specs.shape[1]):
            train_signal_mean = normalization_specs[0, feature]
            train_signal_std = normalization_specs[1, feature]
            train_first_24_hrs_data_maps[:, 0, feature, :] = (train_first_24_hrs_data_maps[:, 0, feature, :] - train_signal_mean) / (
                    1 if float(train_signal_std) == 0 else float(train_signal_std))
            TEST_first_24_hrs_data_maps[:, 0, feature, :] = (TEST_first_24_hrs_data_maps[:, 0, feature, :] - train_signal_mean) / (
                    1 if float(train_signal_std) == 0 else float(train_signal_std))
            train_mortality_data_maps[:, 0, feature, :] = (train_mortality_data_maps[:, 0, feature, :] - train_signal_mean) / (
                    1 if float(train_signal_std) == 0 else float(train_signal_std))
            TEST_mortality_data_maps[:, 0, feature, :] = (TEST_mortality_data_maps[:, 0, feature, :] - train_signal_mean) / (
                    1 if float(train_signal_std) == 0 else float(train_signal_std))


        num_deaths = len(np.where(np.sum(train_mortality_labels, axis=1).reshape(-1,) > 0)[0])
        print('Total num of deaths in train data', num_deaths)

        num_deaths = len(np.where(np.sum(TEST_mortality_labels, axis=1).reshape(-1,) > 0)[0])
        print('Total num of deaths in TEST data', num_deaths)

        np.save('DONTCOMMITdata/hirid_numpy/train_mortality_data_maps.npy', train_mortality_data_maps)
        np.save('DONTCOMMITdata/hirid_numpy/train_mortality_labels.npy', train_mortality_labels)
        np.save('DONTCOMMITdata/hirid_numpy/TEST_mortality_data_maps.npy', TEST_mortality_data_maps)
        np.save('DONTCOMMITdata/hirid_numpy/TEST_mortality_labels.npy', TEST_mortality_labels)
        np.save('DONTCOMMITdata/hirid_numpy/normalization_specs.npy', normalization_specs)
        np.save('DONTCOMMITdata/hirid_numpy/train_first_24_hrs_PIDs.npy', train_first_24_hrs_PIDs)
        np.save('DONTCOMMITdata/hirid_numpy/TEST_first_24_hrs_PIDs.npy', TEST_first_24_hrs_PIDs)
        np.save('DONTCOMMITdata/hirid_numpy/train_Apache_Groups.npy', train_Apache_Groups)
        np.save('DONTCOMMITdata/hirid_numpy/TEST_Apache_Groups.npy', TEST_Apache_Groups)
        np.save('DONTCOMMITdata/hirid_numpy/TEST_first_24_hrs_data_maps.npy', TEST_first_24_hrs_data_maps)
        np.save('DONTCOMMITdata/hirid_numpy/train_first_24_hrs_data_maps.npy', train_first_24_hrs_data_maps)
        print('Saved mortality data, first_24_hrs_PIDs, 24_hrs_data, and apache groups')
        del train_mortality_data_maps
        del train_mortality_labels
        del TEST_mortality_data_maps
        del TEST_mortality_labels
        del train_first_24_hrs_PIDs
        del TEST_first_24_hrs_PIDs
        del train_Apache_Groups
        del TEST_Apache_Groups
        del TEST_first_24_hrs_data_maps
        del train_first_24_hrs_data_maps
    
    elif process_circulatory

        print('Moving on to circulatory failure data')

        print('Memory Usage 7:')
        print(process.memory_info().rss/1e9, 'GB')

        all_pids = []
        admission_times = {}
        for part in range(250):
            print("Raw Part:")
            print(part)
            df = pd.read_parquet('DONTCOMMITdata/hirid/1.1.1/raw_stage/observation_tables/parquet/part-%d.parquet'%part)

            for patient_data in df.groupby('patientid'):
                patientid, patient_data = patient_data
                all_pids.append(patientid)
                patient_data = patient_data.sort_values(by=['datetime'], ignore_index=True)
                start_time = patient_data['datetime'][0]
                admission_times[patientid] = start_time
        
        
        time_of_vasopressor_or_inotrope = {}
        pharma_ids = [1000462, 1000656, 1000657, 1000658, 71, 1000750, 1000649, 1000650, 1000655, 426, 1000441, 112, 113]
        for part in range(250):
            print("Pharma Part:")
            print(part)
            df = pd.read_parquet('DONTCOMMITdata/hirid/1.1.1/raw_stage/pharma_records/parquet/part-%d.parquet'%part)
            
            for patient_data in df.groupby('patientid'):
                patientid, patient_data = patient_data
        
                patient_data = patient_data.loc[df['pharmaid'].isin(pharma_ids)] # locate rows with any of the pharma ids
                if not patient_data.empty: # If we did find one of those pharma values for this patient
                    patient_data = patient_data.sort_values(by='givenat') # sort by time the drugs were given
                    time_of_vasopressor_or_inotrope[patientid] = patient_data['givenat'].to_list()




        dfs = []
        for part in range(0, 250):
            df = pd.read_parquet('DONTCOMMITdata/hirid/1.1.1/imputed_stage/imputed_stage_parquet/parquet/part-%d.parquet'%part)
            dfs.append(df)

        print('=================')
        all_data = pd.concat(dfs, ignore_index=True)

        circulatory_data_maps, circulatory_labels, circulatory_PIDs = get_circulatory_failure_data(merged_df=all_data, admission_times=admission_times, time_of_vasopressor_or_inotrope=time_of_vasopressor_or_inotrope, signal_list=signal_list)
        print('circulatory_data_maps shape: ', circulatory_data_maps.shape)
        print('circulatory_labels shape: ', circulatory_labels.shape)
        print('circulatory_PIDs shape: ', circulatory_PIDs.shape)

        normalization_specs = np.load('DONTCOMMITdata/hirid_numpy/normalization_specs.npy')
        for feature in range(normalization_specs.shape[1]):
            train_signal_mean = normalization_specs[0, feature]
            train_signal_std = normalization_specs[1, feature]
            circulatory_data_maps[:, 0, feature, :] = (circulatory_data_maps[:, 0, feature, :] - train_signal_mean) / (
                    1 if float(train_signal_std) == 0 else float(train_signal_std))

        inds = np.arange(len(circulatory_data_maps))
        random.shuffle(inds)
        circulatory_data_maps = circulatory_data_maps[inds]
        circulatory_labels = circulatory_labels[inds]
        circulatory_PIDs = circulatory_PIDs[inds]
        
        n_train = int(0.8*len(circulatory_data_maps))
        train_circulatory_data_maps = circulatory_data_maps[0:n_train]
        train_circulatory_labels = circulatory_labels[0:n_train]
        train_circulatory_PIDs = circulatory_PIDs[0:n_train]

        TEST_circulatory_data_maps = circulatory_data_maps[n_train:]
        TEST_circulatory_labels = circulatory_labels[n_train:]
        TEST_circulatory_PIDs = circulatory_PIDs[n_train:]

        np.save('DONTCOMMITdata/hirid_numpy/train_circulatory_data_maps.npy', train_circulatory_data_maps)
        np.save('DONTCOMMITdata/hirid_numpy/train_circulatory_labels.npy', train_circulatory_labels)
        np.save('DONTCOMMITdata/hirid_numpy/train_circulatory_PIDs.npy', train_circulatory_PIDs)

        np.save('DONTCOMMITdata/hirid_numpy/TEST_circulatory_data_maps.npy', TEST_circulatory_data_maps)
        np.save('DONTCOMMITdata/hirid_numpy/TEST_circulatory_labels.npy', TEST_circulatory_labels)
        np.save('DONTCOMMITdata/hirid_numpy/TEST_circulatory_PIDs.npy', TEST_circulatory_PIDs)

        print('Memory Usage 8:')
        print(process.memory_info().rss/1e9, 'GB')

    


