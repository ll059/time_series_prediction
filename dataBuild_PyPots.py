import pandas as pd
import numpy as np
import itertools
import pickle 
from sklearn.model_selection import train_test_split
from collections import Counter

# problematic clients: medstart
clients = ['advctz', 'advetz','bjc', 'cone', 'hendrick', 'medstar', 'mtsinai', 'ohiohealth', 'qmc',
           'shce', 'sw','tenetca', 'tenetfl', 'wth']
#clients = ['advctz']
 

file_path = '/home/james/Projects/datasets/sequences/MI/pulled_MI_14-clients/'
store_path = '/home/james/Projects/datasets/sequences/MI/pulled_MI_14-clients/processed_data/'

def build_data(client):
    print(f"Processing: {client}")
    basic = pd.read_csv(file_path + f'Basic_{client}.csv')
    basicTimeTable = pd.read_csv(file_path + f'BasicTimeTable_{client}.csv')
    concepts = pd.read_csv(file_path + f'Concepts_{client}.csv')
    conceptsFullyDoc =  pd.read_csv(file_path + f'ConceptsFullyDoc_{client}.csv')
    medication = pd.read_csv (file_path + f'Medications_{client}.csv')
    medicationClass = pd.read_csv(file_path + f'medicationClass_{client}.csv')     
    observations = pd.read_csv(file_path + f'Observations_{client}.csv')    

    
    conceptsFullyDoc = conceptsFullyDoc.reset_index(drop=True)
    concepts = concepts.reset_index(drop=True)
    medication= medication.reset_index(drop=True)
    medicationClass = medicationClass.reset_index(drop=True)
    observations = observations.reset_index(drop=True)
    #services = services.reset_index(drop=True)

    basicTimeTable.rename(columns={'visit_id': 'visitId'}, inplace=True)
    conceptsFullyDoc.rename(columns={'creation_date': 'rounded_date'}, inplace=True)
    concepts.rename(columns={'creation_date': 'rounded_date'}, inplace=True)
    observations.rename(columns={'observation_date': 'rounded_date'}, inplace=True)
    medication.rename(columns={'order_date': 'rounded_date'}, inplace=True)
    medicationClass.rename(columns={'order_date': 'rounded_date'}, inplace=True)
    #services.rename(columns={'creation_date': 'rounded_date'}, inplace=True)

    #Filter out Non Troponin visit_ids
    troponin_columns = ['troponiniHs_value', 'troponint_value', 'troponini_value']

    df_troponin_cols = []

    for col in observations.columns:
        if col in troponin_columns:
            df_troponin_cols.append(col)
    
    troponin_ids = observations[~pd.isnull(observations[df_troponin_cols]).all(axis=1)]['visitId'].unique()
    
    #filter basic data
    basic_troponin = basic[basic['visitId'].isin(troponin_ids)]
    labels_troponin = np.array(basic_troponin['result'])
    basicTimeTable = basicTimeTable[basicTimeTable['visitId'].isin(troponin_ids)]
    index_visitIds = np.array(basic_troponin.index)

    '''
    basicTimeTable_ids = basicTimeTable['visitId'].unique()
    print('# of unique ids in basictimetable is: ', len(basicTimeTable_ids))
    basic_ids = basic_troponin['visitId'].unique()
    print('# of unique ids in basic is: ', len(basic_ids))
    
    diff = []
    for id in basic_ids:
        if id not in basicTimeTable_ids:
            diff.append(id)
    print('diff ids are: ', diff)
    '''
    
    sub = pd.merge(basicTimeTable,conceptsFullyDoc,on=['visitId', 'rounded_date'],how='left')
    sub = pd.merge(sub,concepts,on=['visitId', 'rounded_date'],how='left')
    sub = pd.merge(sub,observations,on=['visitId', 'rounded_date'],how='left')
    sub = pd.merge(sub,medication,on=['visitId', 'rounded_date'],how='left')
    sub = pd.merge(sub,medicationClass,on=['visitId', 'rounded_date'],how='left')

    sub = sub.drop(columns=['Unnamed: 0'])
    
    return sub, labels_troponin, index_visitIds

#IMPORTING & PROCESSING
dfs = []
labels = []
indicies = []

for client in clients:
    df, label, index = build_data(client) 
    dfs.append(df)
    labels.append(label)
    indicies.append(index)


sub = pd.concat(dfs, axis=0)
dfs = None 

#Labels and Indicies
labels = np.array(list(itertools.chain.from_iterable(labels)))
indicies = np.array(list(itertools.chain.from_iterable(indicies)))


#####STANDARDIZATION
print('Standardizing observations...')
sub['cholesterol_value'] = sub['cholesterol_value'] / 1000
sub['cholesterolHdl_value'] = sub['cholesterolHdl_value'] / 100
sub['troponini_value'] = sub['troponini_value']
sub['act_value'] = sub['act_value'] / 1000
sub['cholesterolLdl_value'] = sub['cholesterolLdl_value'] / 100
sub['ptt_value'] = sub['ptt_value'] / 100
sub['troponint_value'] = sub['troponint_value'] * 100000
sub['map_value'] = sub['map_value'] / 100
sub['troponiniHs_value'] = sub['troponiniHs_value']

#Medications
print('Standardizing medcations...')
sub['norepinephrine_drug'] = sub['norepinephrine_drug'] / 10
sub['metoprolol_drug'] = sub['metoprolol_drug'] / 10
sub['insulin_drug'] = sub['insulin_drug'] / 10
sub['aspirin_drug'] = sub['aspirin_drug'] / 10
sub['furosemide_drug'] = sub['furosemide_drug'] / 10
sub['heparin_drug'] = sub['heparin_drug'] / 10
sub['nitroglycerin_drug'] = sub['nitroglycerin_drug'] / 10
sub['carvedilol_drug'] = sub['carvedilol_drug'] / 10
sub['clopidogrel_drug'] = sub['clopidogrel_drug'] / 10
sub['ticagrelor_drug'] = sub['ticagrelor_drug'] / 10
sub['ciprofloxacin_drug'] = sub['ciprofloxacin_drug'] / 10
sub['apixaban_drug'] = sub['apixaban_drug'] / 10

#Drug Class
print('Standardizing drug class...')
sub['betablocker_drugclass'] = sub['betablocker_drugclass'] / 10
sub['diuretic_drugclass'] = sub['diuretic_drugclass'] / 10
sub['anticoagulant_drugclass'] = sub['anticoagulant_drugclass'] / 10
sub['vasodilator_drugclass'] = sub['vasodilator_drugclass'] / 10
sub['vasopressor_drugclass'] = sub['vasopressor_drugclass'] / 10
sub['antiplatelet_drugclass'] = sub['antiplatelet_drugclass'] / 10


sub = sub.set_index('visitId')
sub = sub.drop(['rounded_date'], axis = 1)

 


######PROCESS AND TURN INTO NUMPY ARRAY
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_pad(df):
    total_length = 60

    array_list = []

    for v_id in df.index.unique():
        v_df = np.array(df.loc[[v_id]])
        #print(v_id)

        if v_df.shape[0] < total_length:
            #pad_array = np.vstack([v_df, np.empty((total_length - v_df.shape[0], v_df.shape[1]))*np.nan])
            pad_array = np.vstack([v_df, np.full([total_length-v_df.shape[0], v_df.shape[1]], np.nan)])
            array_list.append(pad_array)

        elif v_df.shape[0] > total_length:
            pad_array = v_df[:total_length]
            array_list.append(pad_array)

        else:
            array_list.append(v_df)
    array_dfs = np.stack(array_list)
    return array_dfs

#Using chunks of 1000 - hopefully the lower number will help process faster
unique_ids = sub.index.unique()
id_groups = list(chunks(unique_ids, 1000))

#break up into subgroups
sub_groups = []
for e, i in enumerate(id_groups):
    subsub = sub[sub.index.isin(id_groups[e])]
    sub_groups.append(subsub)

all_arrays = []

for e, s in enumerate(sub_groups):
    print(f"Processing group: {e}")

    processed = process_pad(s)
    all_arrays.append(processed)

##Final Array
master_arrays = np.vstack(all_arrays)
print("master_arrays shape is: ", master_arrays.shape)

###TRAIN TEST SPLIT AND SAVE
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(master_arrays, labels, indicies, test_size=0.2)


#Save Point


with open(store_path + 'MI_14-clients_XTrain_PyPots.pickle', 'wb') as handle:
    pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(store_path + 'MI_14-clients_XTest_PyPots.pickle', 'wb') as handle:
    pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(store_path + 'MI_14-clients_yTrain_PyPots.pickle', 'wb') as handle:
    pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(store_path + 'MI_14-clients_yTest_PyPots.pickle', 'wb') as handle:
    pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(store_path + 'MI_14-clients_indexTrain_PyPots.pickle', 'wb') as handle:
    pickle.dump(index_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(store_path + 'MI_14-clients_indexTest_PyPots.pickle', 'wb') as handle:
    pickle.dump(index_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
