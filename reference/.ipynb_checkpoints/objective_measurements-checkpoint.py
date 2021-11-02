import numpy as np
import scipy.io.wavfile as wav
import os
from tqdm import tqdm
import logging
import time
import sys
import mir_eval
from pystoi.stoi import stoi

# logger to save the result 
def lets_log(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_hander = logging.StreamHandler()
    logger.addHandler(stream_hander)
    file_handler = logging.FileHandler(os.path.join(log_dir))
    logger.addHandler(file_handler)
    return logger

# function to calculate sdr, stoi and other score, in our project, we evaluate with sdr and stoi
def objective_measurements(clean, enhanced, fs):
    if len(clean) > len(enhanced):
        clean = clean[:len(enhanced)]
    else:
        enhanced = enhanced[:len(clean)]

    sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(clean, enhanced)
    score_stoi = stoi(clean, enhanced, fs)
    return sdr[0], sir, sar, perm, score_stoi
	
# # create txt file to save the result 
# result_dir = ...  # directory of result file
# logger = lets_log(os.path.join(result_dir, 'score.txt'))


# project_dir = os.getcwd()
# # path to clean files 
# DB_dir = '/clean/TEST_CORE'

# # LIST OF ENHANCED FILES OF ALL TEST FILES
# enhanced_wav_list = ...

# # list of noise type and SNR
# ntype_lib = ["aurora4_airport_1", "aurora4_lobby_1", "aurora4_station_1", "babble", "destroyerops", "DLIVING", "factory1", "leopard", "pink", "SPSQUARE", "STRAFFIC", "volvo", "aurora4_benz_1", "aurora4_exhibition_1", "dcube_record", "destroyerengine", "white", "OMEETING"]
# snrtype_lib = ['-10dB', '-5dB', '0dB', '5dB', '10dB']

# score_sdr = [[[] for n in range(len(snrtype_lib))] for m in range(len(ntype_lib))]
# score_stoi = [[[] for n in range(len(snrtype_lib))] for m in range(len(ntype_lib))]

# start_time = time.time()

# # loop through all enhanced files
# for n in tqdm(range(len(enhanced_wav_list))):
# 	# path to clean file of correspoding enhanced file, I save enhanced filename following the name of test file, then I can get the name of the correspoding clean file from that (see readme.txt file of the data about filename matching)
#     addr_clean = os.path.join(DB_dir, os.sep.join(enhanced_wav_list[n].split(os.sep)[-1].split('_')[:3]))
# 	# read clean file 
#     _, _clean = wav.read(addr_clean)
# 	# read enhanced file
#     fs, enhanced = wav.read(enhanced_wav_list[n])
#     # normalize into [0, 1] range 
#     clean = _clean[:]
#     clean = clean/32768
#     enhanced = enhanced/32768

#     ntype_idx = enhanced_wav_list[n].split(os.sep)[-3]
#     m = ntype_lib.index(ntype_idx)
#     snrtype_idx = enhanced_wav_list[n].split(os.sep)[-2]
#     l = snrtype_lib.index(snrtype_idx)
	
# 	# calculate sdr and stoi 
#     vsdr, _, _, _, vstoi = ssplib.objective_measurements(clean, enhanced, fs)
# 	# calculate sdr improvement and save into array
#     score_sdr[m][l].append(vsdr - np.float(snrtype_idx.split('dB')[0]))
# 	# stoi
#     score_stoi[m][l].append(vstoi)

# # score_sdr = np.array(score_sdr)
# print("score_sdr", score_sdr)
# mean_sdr = np.mean(score_sdr, 2)
# mean_stoi = np.mean(score_stoi, 2)
# end_time = time.time()
# print('time: {}'.format(end_time - start_time))

# logger.info('SDR')
# logger.info("\n".join(" ".join(map(str, line)) for line in mean_sdr))

# logger.info('STOI')
# logger.info("\n".join(" ".join(map(str, line)) for line in mean_stoi))