from collections import OrderedDict
import torch
import os

class checkpoint_scanner():
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_list = sorted(os.listdir(checkpoint_dir))
        self.epoch_list = []
        for file in self.checkpoint_list:
            file_name = os.path.splitext(file)[0]
            file_type = os.path.splitext(file)[1]
            if file_type=='.pkl':  # check whether checkpoint file
                self.epoch_list.append(file_name.split('_')[-1])
        self.epoch_list.sort()
        self.val_last_epoch = int(self.epoch_list[-1])
        
    def last_epoch(self, verbose=False):
        if verbose:
            print(f'The last epoch is {self.val_last_epoch}\n')
        return self.val_last_epoch
    
    def remove_auto(self, keep_list, keep_last=True):
        cp_keep_list = keep_list.copy()
        if keep_last:
            cp_keep_list.append(self.val_last_epoch)
        sorted_keep_list = sorted(list(set(cp_keep_list)))
        in_type = input(f"{sorted_keep_list}\n To delete checkpoints except above, type 'confirm'\n> ")
        if in_type != 'confirm':
            print('Canceled')
            return
        for file in self.checkpoint_list:
            file_name = os.path.splitext(file)[0]
            file_type = os.path.splitext(file)[1]
            if file_type != '.pkl':
                continue
            file_path = f'{self.checkpoint_dir}/{file}'
            file_epoch = int(file_name.split('_')[-1])
            if os.path.isfile(file_path) and file_epoch not in sorted_keep_list:
                os.remove(file_path)
                print(f'Deleted {file}')
            

class history_container():
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.train_history = OrderedDict()
        self.validation_history = OrderedDict()
        self.test_history = OrderedDict()
        self.hparam_history = OrderedDict()
        self.epoch_history = []
        
    def history_show(self):
        print(f'experiment name: {self.experiment_name}\n'
              f'train_history: {self.train_history}\n'
              f'validation_history: {self.validation_history}\n'
              f'test_history: {self.test_history}\n'
              f'hparam_history: {self.hparam_history}\n'
              f'epoch_history: {self.epoch_history}')
        
#     def brief_epoch(self, epoch, name_list=None):
#         # Brief loss at certain epoch
#         print(f'Briefing train history at {epoch} epoch:')
#         self._scanner(self.history_container.train_history name_list)
#         print(f'\nBriefing validation history at {epoch} epoch:')
#         self._scanner(self.history_container.validation_history, name_list)
#         print(f'\nBriefing test history at {epoch} epoch:')
#         self._scanner(self.history_container.test_history, criterion, name_list)
#         print(f'')
#         return
    
#     def _brief_one_set


  
class history_scanner():
    def __init__(self, history_container):
        self.history_container = history_container
        
    def brief_history(self, criterion='minimum', name_list=None):
        self.keep_list = []
        # Brief train history
        print(f'Scanning train history:')
        self._scanner(self.history_container.train_history, criterion, name_list)
        print(f'\nScanning validation history:')
        self._scanner(self.history_container.validation_history, criterion, name_list)
        print(f'\nScanning test history:')
        self._scanner(self.history_container.test_history, criterion, name_list)
        print(f'')
        return self.keep_list
        
    def _scanner(self, histories, criterion, name_list):
        keys = histories.keys()
        for key in keys:
            if name_list and (key not in name_list):
                continue
            try:
                losses = histories[key]
                min_val = min(losses)
                min_idx = losses.index(min_val)
                print(f' {min_idx+1} epoch has {criterion} value {min_val} in {key}')
                self.keep_list.append(min_idx+1)
            except:
                print(f" Can't scan {key}")
                
    def brief_epoch(self, epoch, name_list=None):
        # Brief loss at certain epoch
        print(f'Briefing train history at {epoch} epoch:')
        self._epoch_reader(self.history_container.train_history, epoch, name_list)
        print(f'\nBriefing validation history at {epoch} epoch:')
        self._epoch_reader(self.history_container.validation_history, epoch, name_list)
        print(f'\nBriefing test history at {epoch} epoch:')
        self._epoch_reader(self.history_container.test_history, epoch, name_list)
        print(f'')
        return
    
    def _epoch_reader(self, histories, epoch, name_list):
        keys = histories.keys()
        for key in keys:
            if name_list and (key not in name_list):
                continue
            try:
                losses = histories[key]
                target_val = losses[epoch-1]
                print(f' {target_val} in {key}')
            except:
                print(f" Can't read {key}")
    
        
class checkpoint_maker():
    def __init__(self, model_odict, optimizer_odict, history_container):
        self.model_odict = model_odict
        self.optimizer_odict = optimizer_odict
        self.history_container = history_container
        self.checkpoint = OrderedDict()
        
    def make(self):
        self.checkpoint['models_state_dict'] = OrderedDict()
        for name, model in self.model_odict.items():
            self.checkpoint['models_state_dict'][name] = model.state_dict()
        
        self.checkpoint['optimizers_state_dict'] = OrderedDict()
        for name, optimizer in self.optimizer_odict.items():
            self.checkpoint['optimizers_state_dict'][name] = optimizer.state_dict()
            
        self.checkpoint['history_container'] = self.history_container
                    
        return self.checkpoint
    
        
class checkpoint_loader():
    """

    """    
    def __init__(self, checkpoint_dir, loading_epoch, map_location=None):
        self.checkpoint = None
        self.loading_epoch = loading_epoch
        self.start_epoch = None
        
        for try_epoch in range(loading_epoch, -1, -1):
            if try_epoch == 0:
                print('start with initial state\n')
                self.start_epoch = 1
                break
            try:
                load_path = f'{checkpoint_dir}/checkpoint_{try_epoch:0=4d}.pkl'
                self.checkpoint = torch.load(load_path, map_location=map_location)        
                print(f'checkpoint at {try_epoch} epoch will be loaded\n file path: {load_path}\n')
                self.start_epoch = try_epoch + 1
                break
            except Exception as e:
                print(e)
                print(f'checkpoint at {try_epoch} epoch can not be loaded\n')
            
    def starting_point(self):
        return self.start_epoch
    
    def loaded_epoch(self):
        return self.start_epoch - 1
    
    def model_loader(self, model_odict):
        if not self.checkpoint:
            print('initial state model')
            return model_odict
        else:
            for name, model in model_odict.items():
                try:
                    model.load_state_dict(self.checkpoint['models_state_dict'][name])
                    print(f'load model: {name}')
                except Exception as e:
                    print(e)
                    print(f'Fail to load model: {name}')
            print()
            return model_odict
        
    def optimizer_loader(self, optimizer_odict):
        if not self.checkpoint:
            print('initial state optimizer')
            return optimizer_odict
        else:
            for name, optimizer in optimizer_odict.items():
                try:
                    optimizer.load_state_dict(self.checkpoint['optimizers_state_dict'][name])
                    print(f'load optimizer: {name}')
                except Exception as e:
                    print(e)
                    print(f'Fail to load optimizer: {name}')
            print()
            return optimizer_odict
            
    def history_loader(self, history_container):
        if not self.checkpoint:
            print('initial state history')
            return history_container
        else:
            try:
                history_container = self.checkpoint['history_container']
                print(f'test load history container\n'
                      f' experiment name: {history_container.experiment_name}\n')
                return history_container
            except Exception as e:
                print(e)
                return history_container
            
#     def hparam_loader(self, hparam_odcit):
#         if not self.checkpoint:
#             print('initial state hyper-parameters')
#             return hparam_odict
#         else:
#             try:
#                 hparam_odict = self.checkpoint['hpa']
