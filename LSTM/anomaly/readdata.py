import os, sys, json, random, h5py
import numpy as np
from scipy.fftpack import fft, ifft

def readdata(path, n, windowsize,stride):
    zlabel = []
    G = []
    #Dataset = []
    metafilelst=[]
    datafilelst=[]
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
            trLen = metadata[u'core:capture'][0][u'PFP:length']
            if trLen <11000:
                F.remove(F[i])
            else:
                zlabel.append(metadata[u'core:global'][u'PFP:label'])
                G.append(F[i])
    #print(np.shape(G))
    for mf, df in G:
        sample = file(path+df)
        data_type = np.dtype ('float32').newbyteorder ('=')
        signalIterA = np.fromfile(sample, dtype=data_type)
        newset = split_len(signalIterA, windowsize, stride)
        #Dataset.append(newset)
        #print(np.shape(newset))
        dummy = ftwindow_gen(newset)
        #print(np.shape(dummy)))
        FT.append(dummy)
        LB.append(zlabel[seqcount]*np.ones(len(FT[seqcount])))
        seqcount+=1
    return FT, LB, n

def split_len(seq, length, step):
    seq = seq[:-900]
    #seq = seq[:-300]
    #seq_list = [seq[ind:ind+length] for ind in range(0, len(seq), length-stride)]
    #print(np.shape(seq_list[-2]))
    numstps = (len(seq)-length)/step+1
    seq_list = [seq[i*step:(length+i*step)] for i in range(numstps)] 
    #print(len(seq_list[-1]))
    return seq_list

def ftwindow_gen(array):
    #print(len(array))
    #print(array[1])
    return [np.abs(np.fft.fft(array[k])) for k in range(len(array))]

def write_to_hdf5(self):
    #global windowsize
    dpts = self['parameters']['num_trace']
    windowsize = self['parameters']['windowsize']
    stride = self['parameters']['stride']
    Nfft = self['parameters']['num_fft_windows']
    folder = self['data_dir']
    n = dpts/len(folder)
    Labelset = []
    FTset = []
    ft_data = []
    state = []
    for f in folder:
        ft_data, state, n = readdata(f, n, windowsize,stride)
        n = dpts-n 
        print('the number of samples of the next directory is: ', n)
        print(np.shape(ft_data))
        for d in ft_data:
            FTset.append(d)
        for s in state:
            Labelset.append(s)   
    print('Complete!') 
    #Parameters for creating hdf5 files
    print('the shape of FT data set is: ', np.shape(FTset))
    L = list(len(FTset[i]) for i in range(len(FTset)))
    print('FTset contains: ', np.shape(FTset))
    print('FTset first signal contains: ', np.shape(FTset[0]))
    print('FTset first signal first fft window contains: ', np.shape(FTset[0][0]))
    print('Length vector  of contains: ', L)
    nbatch = map(lambda l: l/Nfft, L)
    print('total number of batches as in sum: ', nbatch)
    #Data frame with key "seqs"
    seqs = np.empty(shape = (0, Nfft, windowsize))
    #print('shape of the data frame is: ', np.shape(seqs))
    #Machine state with key "labels"
    labels = np.empty(shape = (0, 1, 1))
    #randomize the sequences
    ind = np.random.permutation(range(len(L)))
    print('index range is', range(len(ind)))
    print('length of Label set is ', len(Labelset))
    #createing hdf5 file
    print('creating hdf5 file...')
    print('The length of Labelset sequence 0 is ', np.shape(Labelset[0]))
    #print(Labelset[0][0])
    for j in range(len(ind)):
        try:
            if Labelset[ind[j]][0]== 1:
                state = np.reshape([1]*nbatch[ind[j]],(nbatch[ind[j]],1,1))
            else:
                state = np.reshape([0]*nbatch[ind[j]],(nbatch[ind[j]],1,1))
            #print(np.shape(state))
            #print(np.shape(labels))
            labels = np.concatenate((state,labels),axis = 0)
            bn = len(FTset[ind[j]][:])//Nfft
            data = np.reshape(FTset[ind[j]][0:bn*Nfft], (bn,Nfft,windowsize))
            #print(np.shape(data))
            seqs = np.concatenate((data,seqs),axis = 0)
        except IndexError:
            pass
    print(np.shape(seqs))
    print(np.shape(labels))
    #output = h5py.File('train_DNI_Router_Sparse.hdf5','w')
    #print(np.shape(seqs[0]))
    #print(np.shape(labels[0]))
    #filename = self.result_dir+self.result_file+'.hdf5'
    output = h5py.File(self['result_dir']+self['result_name']+'.hdf5','w')
    output.create_dataset('input', data = seqs)
    output.create_dataset('labels', data = labels)
    output.close()
    print('Done')
    #print(np.shape(FTset))

