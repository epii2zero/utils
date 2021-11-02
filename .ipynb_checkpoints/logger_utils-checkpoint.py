import os, sys
import datetime
import json
import torch

def dprint(txt, file_pointer):
    print(txt)
    print(txt, file=file_pointer)
    
    
class Stream_Duplicator():
    """
        Duplicate stdout pipe
    """
    def __init__(self, file_pointer, b):
        self.fp = file_pointer
        self.b = b
        print('init')
            
    def write(self, data):
        self.fp.write(data)
        self.b.write(data)
                
    def flush(self):
        self.fp.flush()
        self.b.flush()
        
class jupyter_experiment_logger():
    """
        Make log file during jupyter operation
    """
    def __init__(self, log_path, exp_name):
        self.exp_name = exp_name
        self.logfile_path = log_path + exp_name + '.txt'
        try:
            self.fp = open(self.logfile_path, 'a+')
        except Exception as e:
            print('Failed to initialize log file')
            print('    {}'.format(e))
    
    def _check_recorded(self, section):
        # read current log file data
        self.fp.flush()
        cur_point = self.fp.tell()
        self.fp.seek(0)
        lines = self.fp.readlines()
        self.fp.seek(cur_point)
        
        # find section data
        sections = []
        for i in range(len(lines)-2):
            if lines[i][:2] == '--' and lines[i+2][:2] == '--':
                sections.append(lines[i+1].strip())
                
        # bool type return
        return section in sections
    
    def _insert_str_middle(self, str, point):
        ##############################
        # test
        self.fp.seek(10)
        print(str, file=self.fp)
        print('done')
        ##############################
        
    def init_experiment(self):
        if self._check_recorded('Header'):
            print('Header section is already recorded')
            return
        print('--------------------\n       Header\n--------------------', file=self.fp)
        
        start_time = datetime.datetime.now()
        dprint('Experiment name: {}'.format(self.exp_name), self.fp)
        dprint('Code path: {}'.format(os.getcwd()), self.fp)
        dprint('Excuted time: {}'.format(start_time), self.fp)
        print('', file=self.fp)
        self.fp.flush()
        
    def configurations(self, config):
        if self._check_recorded('Configurations'):
            print('Configurations section is already recorded')
            return
        print('--------------------\n   Configurations\n--------------------', file=self.fp)
        
        dprint('From {}.json:'.format(self.exp_name), self.fp)
        dprint(json.dumps(config, sort_keys=True, indent=4), self.fp)
        print('', file=self.fp)
        self.fp.flush()
        
    def set_device(self, GPU_ID, device):
        if self._check_recorded('Device'):
            print('Device section is already recorded')
            return
        print('--------------------\n       Device\n--------------------', file=self.fp)
        
        dprint('Using GPU ID: {}'.format(GPU_ID), self.fp)
        dprint('Device: {}'.format(device), self.fp)
        print('', file=self.fp)
        self.fp.flush()
        
    def dataset(self, dataset_dir_dict):
        self._insert_str_middle('insertion test', 0)
    
    def close(self):
        self.fp.close()
    
    