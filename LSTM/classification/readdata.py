import os, sys, json, random, h5py
import numpy as np
from scipy.fftpack import fft, ifft

def readdata(path, n, ftwindowsize, windowsize, stride, channel, option):
    zlabel = []
    rlabel = []
    G = []
    #Dataset = []
    print(option)
    metafilelst=[]
    datafilelst=[]
    print(path)
    LB = []
    FT = []
    for x in os.listdir(path):
        if x.endswith('.meta'):
            metafilelst.append(x)
        elif x.endswith('.data'):
            datafilelst.append(x) 
        else:
            continue
    print('the total number of samples available in this directory is: ', len(metafilelst))
    n = min(len(metafilelst), n)
    print('the number of samples in this directory is: ',n)
    F = zip(metafilelst, datafilelst) # full list of all data in the directory "folder"
    np.random.shuffle(F)
    seqcount = 0
    #for i in range(len(metafilelst)):
    for i in range(n):
        #print(F[i][0])
        with open(path+F[i][0]) as json_file:
            metadata = json.load(json_file)
            max_channel = len(metadata[u'core:capture'])
            #trLen = metadata[u'core:capture'][0][u'PFP:length']
            if max_channel > 1:
                channellst = [metadata[u'core:capture'][n][u'PFP:channel'] for n in range(len(metadata[u'core:capture']))]
                idx = index(channellst, channel)
                #for idx, val in enumerate(channellst):
                #    if val == channel:
                trLen = metadata[u'core:capture'][idx][u'PFP:length']
                start = int(metadata[u'core:capture'][idx][u'core:sample_start'])
            else:
                start = 0
                trLen = metadata[u'core:capture'][0][u'PFP:length']
            
            rlabel.append((start, start+trLen-1))
            zlabel.append(metadata[u'core:global'][u'PFP:label'])
            G.append(F[i])
    #print(np.shape(G))
    for mf, df in G:
        sample = file(path+df)
        st = rlabel[seqcount][0]
        ed = rlabel[seqcount][1]
        signalIterA = readbychannel(sample, channel, st, ed)
        newset = split_len(signalIterA, windowsize, stride)
        #Dataset.append(newset)
        #print(np.shape(newset))
        if option == 1:
            dummy = normalize(ftwindow_genII(newset, ftwindowsize))
        else:
            dummy = normalize(window_gen(newset))
        #print(np.shape(dummy))
        FT.append(dummy)
        LB.append(zlabel[seqcount]*np.ones(len(FT[seqcount])))
        seqcount+=1
    return FT, LB, n

def split_len(seq, length, step):
    #seq = seq[:-399]
    #seq = seq[:-300]
    #seq_list = [seq[ind:ind+length] for ind in range(0, len(seq), length-stride)]
    #print(np.shape(seq_list[-2]))
    numstps = (len(seq)-length)/step+1
    seq_list = [seq[i*step:(length+i*step)] for i in range(numstps)]
    return seq_list

def window_gen(array):
    return [array[k] for k in range(len(array))]

def ftwindow_gen(array):
    #print(len(array))
    #print(array[1])
    return [np.abs(np.fft.fft(array[k])) for k in range(len(array))]

def index(a_list, value):
    try:
        return a_list.index(value)
    except ValueError:
        return None

def ftwindow_genII(array, ftwindowsize):
    dummy = []
    for k in range(len(array)):
        vec = np.abs(np.fft.fft(array[k]))
        dummy.append(np.mean(np.reshape(vec, (ftwindowsize, -1)), axis = 1))
    return dummy

def readbychannel(filename, channel, st, ed):
    data_type = np.dtype ('float32').newbyteorder ('=')
    signal_all = np.fromfile(filename, dtype=data_type)
    return signal_all[st:ed]  

def normalize(array):
    a = []
    #print(np.shape(array))
    #print('length of each fft array is ', len(array))
    #print(np.shape(array)) #(399, 1000)
    for j in range(len(array)):
        #print(len(array[j]))
        a.append((array[j] - min(array[j]))/(max(array[j]) - min(array[j])))
    return a

def write_to_hdf5(self):
    #global windowsize
    dpts = self['parameters']['num_trace']
    windowsize = self['parameters']['windowsize']
    stride = self['parameters']['stride']
    Nfft = self['parameters']['num_fft_windows']
    folder = self['data_dir']
    channel = self['parameters']['channel']
    option = self['parameters']['option']
    n = dpts/len(folder)
    Labelset = []
    FTset = []
    ft_data = []
    state = []
    #if windowsize > 1000:
    #    ftwindowsize = 1024
    #else:
    ftwindowsize = windowsize
    for f in folder:
        ft_data, state, n = readdata(f, n, ftwindowsize, windowsize, stride, channel, option)
        n = dpts-n 
        #print('the number of samples of the next directory is: ', n)
        #print(np.shape(ft_data))
        for d in ft_data:
            FTset.append(d)
        for s in state:
            Labelset.append(s)   
    print('Complete!') 
    #Parameters for creating hdf5 files
    #print('the shape of FT data set is: ', np.shape(FTset))
    L = list(len(FTset[i]) for i in range(len(FTset)))
    #print('FTset contains: ', np.shape(FTset))
    #print('FTset first signal contains: ', np.shape(FTset[0]))
    #print('FTset first signal first fft window contains: ', np.shape(FTset[0][0]))
    #print('Length vector  of contains: ', L)
    nbatch = map(lambda l: l/Nfft, L)
    #print('total number of batches as in sum: ', nbatch)
    #Data frame with key "seqs"
    seqs = np.empty(shape = (0, Nfft, ftwindowsize))
    #print('shape of the data frame is: ', np.shape(seqs))
    #Machine state with key "labels"
    labels = np.empty(shape = (0, 1, 1))
    #randomize the sequences
    ind = np.random.permutation(range(len(L)))
    #print('index range is', range(len(ind)))
    #print('length of Label set is ', len(Labelset))
    #createing hdf5 file
    print('creating hdf5 file...')
    #print('The length of Labelset sequence 0 is ', np.shape(Labelset[0]))
    #print(Labelset[0][0])
    for j in range(len(ind)):
        try:
            l = Labelset[ind[j]][0];
            state = np.reshape([l]*nbatch[ind[j]],(nbatch[ind[j]],1,1))            
            #if Labelset[ind[j]][0]== 1:
            #    state = np.reshape([1]*nbatch[ind[j]],(nbatch[ind[j]],1,1))
            #else:
            #    state = np.reshape([0]*nbatch[ind[j]],(nbatch[ind[j]],1,1))
            #print(np.shape(state))
            #print(np.shape(labels))
            labels = np.concatenate((state,labels),axis = 0)
            bn = len(FTset[ind[j]][:])//Nfft
            data = np.reshape(FTset[ind[j]][0:bn*Nfft], (bn,Nfft,ftwindowsize))
            #print(np.shape(data))
            seqs = np.concatenate((data,seqs),axis = 0)
        except IndexError:
            pass
    #print(np.shape(seqs))
    #print(np.shape(labels))
    #output = h5py.File('train_DNI_Router_Sparse.hdf5','w')
    #print(np.shape(seqs[0]))
    #print(np.shape(labels[0]))
    #filename = self.result_dir+self.result_file+'.hdf5'
    output = h5py.File(self['result_dir']+self['result_name']+'.hdf5','w')
    output.create_dataset('input', data = seqs)
    output.create_dataset('labels', data = labels)
    output.close()
    print('Done!')
    #print(np.shape(FTset))

