import os
import librosa

def train_test_files_match_test(dataset_dir_1, dataset_dir_2, name_test=True):
    """
    Check whether the file number of two dataset are same 
    and optionaly its name
    
    Arguments:
        dataset_dir_1 (folder_path): dataset1 path
        dataset_dir_2 (folder_path): dataset2 path
        name_test (bool, optional): whether to check name matching of the datasets 
    """
    flen_1 = len(os.listdir(dataset_dir_1))
    flen_2 = len(os.listdir(dataset_dir_2))
    if flen_1 != flen_2:
        raise ValueError('Num of files are not same')
    else:
        print('Num of files are same')
    if name_test:
        for i in range(flen_1):
            fname1 = os.listdir(dataset_dir_1)[i]
            fname2 = os.listdir(dataset_dir_2)[i]
            if fname1 != fname2:
                raise ValueError(fname1, fname2, 'are not same name')
        print('All file names are same')
    return True

def make_file_list(dataset_dir, object_type=None):
    """
    Get dataset path and return filelist of path
    To limit the file type, set parameter object_type=['.np', '.wav']
    """
    
    file_list = []
    for file in os.listdir(dataset_dir):
        file_name = os.path.splitext(file)[0]
        file_type = os.path.splitext(file)[1]
        if object_type and (file_type not in object_type):
            print('wrong type:', file)
            continue
        file_list.append(os.path.join(dataset_dir, file))
    return file_list

def make_file_dict(dataset_dir, object_type=None):
    file_list = make_file_list(dataset_dir, object_type)
    file_dict = {}
    for file in file_list:
        key = file.split('/')[-1]
        file_dict[key]=file
    return  file_dict

def random_segmentation(path, frame_len, frame_num, seed=None, object_type=['.wav']):
    if not os.path.exists(path):
        raise ValueError('worng path')
    if not os.path.exists(os.path.join(path, 'segment')):
        os.mkdir(os.path.join(path, 'segment'))
        
    file_list = os.listdir(path)
    s_cnt = 0
    l_cnt = 0
    for i, file in enumerate(file_list):
        file_name = os.path.splitext(file)[0]
        file_type = os.path.splitext(file)[1]
        
        if file_type not in object_type:
            print("{}/{}: wrong type {}".format(i+1, len(file_list), file))
            continue
        
        wav, sr = librosa.load(os.path.join(path, file), sr=None)
        wav_len = len(wav)
        if seed:
            np.random.seed(seed)
        points = np.arange(0, wav_len-frame_len, frame_len//2)
        if len(points) > frame_num:
            points = np.random.choice(points, frame_num, replace=False)
            l_cnt += 1
        else:
            s_cnt += 1
        np_frames = np.zeros((len(points), frame_len))
        for j, point in enumerate(points):
            np_frames[j,:] = wav[point:point+frame_len]
        save_path = os.path.join(path, 'segment', file_name+'.npy')
        np.save(save_path, np_frames)
        print('{}/{} done: {} array is saved at {}'.format(i+1, len(file_list), np_frames.shape, file_name+'.npy'))
    return