import numpy as np
import h5py
import scipy.io

def rootofsumsq(pair):
    return([(pair[ind][0]**2+ pair[ind][1]**2)**(0.5) for ind in range(len(pair))])

def labelgen(steps):
    return np.repeat([0,1,2,3], steps)

def shuffled_sets(d, l):
    print(len(d))
    print(len(l))
    if len(d) != len(l):
        print('ERROR!! Data and label set is not in the same length!')
    else:
        p = np.random.permutation(len(d))
        return d[p], l[p]

def datagen(path):
    # there are allIdxAbs as 4 separate sequences (i.e. 4 classes) of length 610 of length 100 vector
    mat = scipy.io.loadmat(path)
    #windowlength = mat['winSize'][0][0]
    num_class = np.shape(mat['allIdxAbs'])[2]
    total_windows = np.shape(mat['allIdxAbs'])[1]
    windowsize = np.shape(mat['allIdxAbs'])[0]
    num_train = 500
    num_test = total_windows-num_train
    train = np.zeros([num_train*num_class, windowsize])
    test = np.zeros([num_test*num_class, windowsize])

    for i in range(num_class):
        trainvec = np.transpose(mat['allIdxAbs'][:,0:num_train,i], (1,0))
        #print(np.shape(trainvec))
        #print(np.shape(train))
        np.concatenate((train,trainvec),axis=0)
        testvec = np.transpose(mat['allIdxAbs'][:,num_train:,i], (1,0))
        #print(np.shape(testvec))
        #print(np.shape(test))
        np.concatenate((test,testvec),axis=0)
    train = np.reshape(train, [num_train*num_class,1,windowsize])
    test = np.reshape(test,[num_test*num_class,1,windowsize])
    trainlabel= np.reshape(labelgen(num_train), [num_train*num_class,1,1])
    testlabel = np.reshape(labelgen(num_test), [num_test*num_class,1,1])
    #labelset = np.reshape(rawlabels[], [num_class,num_windows,1])
    tn_data, tn_label = shuffled_sets(train, trainlabel)
    tt_data, tt_label = shuffled_sets(test, testlabel)
    return tn_data, tn_label, tt_data, tt_label

path = 'winSize4096numWin100natom20/testMatchingPursuit003Results.mat'
print('Processing parameters...')
traindata, trainlabels, testdata, testlabels = datagen(path)

print('Generating Training Datasets...')
output1 = h5py.File('trainMP_first500.hdf5','w')
output1.create_dataset('input', data = traindata)
output1.create_dataset('labels', data = trainlabels)
output1.close()
print('Complete!!')

print('Generating Testing Datasets...')
output2 = h5py.File('testMP_first500.hdf5.hdf5','w')
output2.create_dataset('input', data = testdata)
output2.create_dataset('labels', data = testlabels)
output2.close()
print('Complete!!')
 


