import json
import sys, os
from subprocess import call
from readdata import *
#runs on python 2.7.x
#import NN_Module

#jsonfname = 'exploit_241.json'
#jsonfname = 's71200_longcap.json'
#with open(jsonfname) as json_file:
#    config = json.load(json_file, encoding='utf-8')
# the json file
if len(sys.argv) < 2:
    print('Please pass the configuration json file.')
    sys.exit(-1)
jsonfname=sys.argv[1]
print('the configuration file: {}'.format(jsonfname))
if(not os.path.exists(jsonfname)):
    print('The configuration file {} does not exist.'.format(jsonfname))
    sys.exit(-1)
with open(jsonfname) as json_file:
    config=json.load(json_file, encoding='utf-8')
    
# Preprocessing parameters
windowsize = config['preprocess']['windowsize'] #fft window size
stride = config['preprocess']['stride'] #window_stride eq. 100 overlap
num_fft_windows = config['preprocess']['num_fft_windows'] # numbers of windows to feed to NN each iteration, if we do not have enough samples in a trace, we have to change this number. extra samples are discarded.
fft_option = config['preprocess']['option']
channel = config['preprocess']['channel']
#NN memory cell are reset each iteration / trace.

# Training and testing set
num_trace_train = str(config['preprocess']['num_trace_train'])
num_trace_test = str(config['preprocess']['num_trace_test'])

# hyperparameters & regularization
hidden_size = str(config['nn_var']['hidden_size'])
fc_size = str(config['nn_var']['fc_size'])
epochs = str(config['nn_var']['epochs'])
learning_rate = str(config['nn_var']['learning_rate'])
batch_size = str(config['nn_var']['batch_size']) #how many of those 50-fft-windows are fed into NN before weights are updated.
output_size = str(config['nn_var']['output_size']) #output size, number of classes, for anomaly detection, no need to specify (it should be the number of input nodes-1000 here.
dropout_rate = str(config['nn_var']['dropout_rate']) # fraction of neurons in the hidden layers being dropped out


# model
data_dir = config['paths']['data_dir']
data_name = str(config['paths']['data_name'])# to distinguish stored results between datasets / cases
# saved dataframe, model
preprocess_dir = str(config['paths']['preprocess_dir']) #for preprocessing tempory storage
#log stats, etc
out_dir = str(config['paths']['result_dir'])
gpu = str(config['tasks']['gpu']) #0=default, currently can only specify only 1 gpu


##*******************************************************
## Menu for data generation and network training/testing
##*******************************************************
        

if config['tasks']['preprocess_trainset'] == 1:
    # i.e. DNI_Router dataset
    preprocess_fname = data_name + '_' + num_trace_train
    conf = {'data_dir':data_dir, 'result_dir':preprocess_dir, 'result_name': preprocess_fname, 'parameters': {'num_trace': config['preprocess']['num_trace_train'],'windowsize': windowsize, 'channel': channel, 'stride': stride, 'num_fft_windows': num_fft_windows, 'option': fft_option}}
    write_to_hdf5(conf)
if config['tasks']['train_LSTM_module'] == 1:
    preprocess_fname = data_name + '_' + num_trace_train
    traindata = preprocess_dir + preprocess_fname + '.hdf5'
    try:
        call(["python", "pfp.py", "--hidden_size", hidden_size, hidden_size, "--fc_size", fc_size, "--epoch", epochs, "--learning_rate", learning_rate, "--output_size", output_size, "--dropout_rate" ,dropout_rate, "--batch_size", batch_size, "--file", traindata, "--checkpoint_dir", out_dir, "--gpu_id", gpu]) 
        config.update({'output': {'train_result_dir':config['paths']['result_dir']+preprocess_fname+'/', 'model': 'checkpoint.tf','log_file': 'output_train.log'}})
        with open(jsonfname,'w') as json_file:
            json.dump(config, json_file)
    except KeyboardInterrupt:
        pass
    
#testing set generation and testing
if config['tasks']['preprocess_testset'] == 1:
    # i.e. DNI_Router dataset
    preprocess_fname = data_name + '_' + num_trace_test
    conf = {'data_dir':data_dir, 'result_dir':preprocess_dir, 'result_name': preprocess_fname, 'parameters': {'num_trace': config['preprocess']['num_trace_test'],'windowsize': windowsize, 'channel':channel, 'stride': stride, 'num_fft_windows': num_fft_windows, 'option': fft_option}}
    write_to_hdf5(conf)

### nned to fix
if config['tasks']['test_LSTM_module'] == 1:
    preprocess_fname = data_name + '_' + num_trace_test 
    #preprocess_dir = 'test' + data_name + str(test_var['num_trace']) + 'combinedsamples'
    testdata = preprocess_dir + preprocess_fname + '.hdf5'
    model = config['output']['train_result_dir'] + config['output']['model']
    #model = out_dir + 'pfp_hs_' + hidden_size + '_' + hidden_size + '_fs_' + fc_size + '_train' + data_name + num_trace_train + 'combinedsamples/checkpoint.tf'
    try:
        call(["python", "pfp.py", "--test", "--hidden_size", hidden_size, hidden_size, "--fc_size", fc_size, "--learning_rate", learning_rate, "--output_size", output_size, "--init_from", model, "--batch_size", batch_size, "--file", testdata, "--checkpoint_dir", out_dir, "--gpu_id", gpu])
        config['output'].update({'test_result_dir':config['paths']['result_dir']+preprocess_fname+'/', 'log_test': 'output_test.log'})
        with open(jsonfname,'w') as json_file:
            json.dump(config, json_file)
    except KeyboardInterrupt:
        pass



