

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import importlib
from os.path import isfile
import bz2
import pickle
import _pickle as cPickle
import random
import statsmodels.api as sm
from sklearn.metrics import r2_score

neuron_quality = pd.read_csv('neuron_quality.csv')
neuron_quality_np = neuron_quality.to_numpy()
unique_neuron_types, counts = np.unique( neuron_quality_np[:,1], return_counts = True )

unique_neuron_types = unique_neuron_types[ counts >= 10 ]
counts = counts[ counts >= 10 ] # only use LC neurons with at least 50 neurons
analyze_neurons = np.append( unique_neuron_types[-3:], unique_neuron_types[:-3] )[:10]
analyze_neurons = list( analyze_neurons[ np.all([analyze_neurons != 'LC14',analyze_neurons != 'LC10'],axis=0) ] )

node_class_dict = {"soma": 1,"axon": 2,"dendrite": 3,"cell body fiber": 4,"connecting cable": 5,"other": 6}

section_colors = {'axon': [0.371764705882353, 0.717647058823529, 0.361176470588235],
                  'dendrite': [152/255, 78/255, 163/255],
                  'connecting cable': [1, 0.548235294117647, 0.100000000000000],
                  'cell body fiber': [0, 0.30196078431372547, 0.25098039215686274],
                  'soma': [0, 0.6941176470588235, 0.6901960784313725],
                  'other': [0.7686274509803922, 0.4980392156862745, 0.19607843137254902]}

def compressed_pickle(title, data):
    with bz2.BZ2File(title , 'w') as f: 
        cPickle.dump(data, f)
        
# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

def save_biggest_size():
    overall_max_size = np.zeros(3, dtype=int)
    for section in ['dendrite', 'connecting cable', 'axon']:
        max_size = np.zeros(3, dtype=int)
        for i_type, neuron_type in enumerate(analyze_neurons):
            for bodyId in neuron_quality_np[neuron_quality_np[:,1] == neuron_type,0]:
                mito_file = f'segmentations/mito_df/{neuron_type}_{bodyId}_mito_df.csv'
                if isfile(mito_file):
                    # this neuron has a saved mitochondria csv file
                    mito_df = pd.read_csv(mito_file)
    
                    # node_class_dict takes a section as input about outputs the class integer assigned to that section
                    is_in_section = mito_df['class'].to_numpy() == node_class_dict[section]
                    if np.any(is_in_section):
                        for i_mito in np.where(is_in_section)[0]:
                            file = f'segmentations/mitochondria/{section}/{neuron_type}_{bodyId}_IMito_{i_mito}.pbz2'
    
                            if isfile(file):
                                # load this mitochondrion
                                mito_subvol = decompress_pickle(file)
                                max_size = np.max([max_size, mito_subvol.shape],axis=0)
                                overall_max_size = np.max([overall_max_size, mito_subvol.shape],axis=0)
        np.save(f'presaved_data/{section}_biggest_size.npy', max_size)
        print(f'Finished with {section.title()}')
    np.save(f'presaved_data/overall_biggest_size.npy', overall_max_size)
    return None

def zero_pad_img(subvol, new_size):
    
    for i_axes in range(3):
        dh = new_size[i_axes] - subvol.shape[i_axes]
        
        pre_dims, post_dims = list(subvol.shape), list(subvol.shape)
        pre_dims[i_axes] = int(dh/2)
        post_dims[i_axes]= int(dh - int(dh/2))
        subvol = np.concatenate( [np.zeros(pre_dims), subvol, np.zeros(post_dims)], axis=i_axes)
    return subvol

def get_sample_weight(neuron_type, section):
    n_mitos = 0
    for bodyId in neuron_quality_np[neuron_quality_np[:,1] == neuron_type,0]:
        mito_file = f'segmentations/mito_df/{neuron_type}_{bodyId}_mito_df.csv'
        if isfile(mito_file):
            # this neuron has a saved mitochondria csv file
            mito_df = pd.read_csv(mito_file)

            # node_class_dict takes a section as input about outputs the class integer assigned to that section
            is_in_section = mito_df['class'].to_numpy() == node_class_dict[section]
            if np.any(is_in_section):
                for i_mito in np.where(is_in_section)[0]:
                    n_mitos += int(isfile(f'segmentations/mitochondria/{section}/{neuron_type}_{bodyId}_IMito_{i_mito}.pbz2'))
    return 1 / n_mitos

def get_batches(ids, k):
    random.shuffle(ids)
    k_groups = [ [] for _ in range(k) ]
    group_size = int(len(ids) / k)
    for i_k in range(k):
        for i in range(group_size):
            k_groups[i_k].append( ids[i + i_k*group_size] )
    for i in range( len(ids) % k ):
        k_groups[i].append( ids[ -(i+1) ] )
    return k_groups

def get_cross_val_groups(ids, k):
    '''
    Inputs:
        - ids : list of ids to separate into "k" groups
        - k : number of groups
    Outputs:
        - k length list of training ids
    '''

    unique_ids = np.unique( ids )
    np.random.shuffle(unique_ids)
    k_groups = [ [] for _ in range(k) ]
    group_size = int(len(unique_ids) / k)
    for i_k in range(k):
        for i in range(group_size):
            k_groups[i_k].append( unique_ids[i + i_k*group_size] )
    for i in range( len(unique_ids) % k ):
        k_groups[i].append( unique_ids[ -(i+1) ] )
    k_groups_train = [ ids[~np.isin(ids,k_groups[i_k])] for i_k in range(k) ]
    return k_groups_train, k_groups

def fit_power_laws(X, Y, Y_norm):
    assert np.all(Y > 0)
    assert np.all(X > 0)
    
    x = sm.add_constant(np.log10(X))
    y = np.log10(Y)
    reg = sm.OLS(y, x).fit()
    CoD = r2_score(y, reg.predict(x))

    # scale X such that Y is all 1 um^2
    y_factor = np.ones( X.shape )
    #])
    for i_feat in range(X.shape[1]):
        if reg.params[1+i_feat] != 0:
            y_factor[:,i_feat] = (Y_norm / Y)**(1/reg.params[1+i_feat])
    
    x_scaled = X * y_factor
    
    
    return reg, CoD, x_scaled
    