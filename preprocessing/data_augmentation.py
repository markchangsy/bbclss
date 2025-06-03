import os
import sys
import json
import wave
import shlex
import random
import argparse
import subprocess
import contextlib
from glob import glob
import multiprocessing
cpus = multiprocessing.cpu_count()
import concurrent.futures
from tqdm import tqdm
import pandas as pd

def parse_rir_list(rir_set_para_array, rir_root, sampling_rate = None):
    """ This function creates the RIR list
        Each rir object in the list contains the following attributes:
        rir_id, room_id, receiver_position_id, source_position_id, rt60, drr, probability
        Please refer to the help messages in the parser for the meaning of these attributes
    """
    rir_parser = argparse.ArgumentParser()
    rir_parser.add_argument('--rir-id', type=str, required=True, help='This id is unique for each RIR and the noise may associate with a particular RIR by refering to this id')
    rir_parser.add_argument('--room-id', type=str, required=True, help='This is the room that where the RIR is generated')
    rir_parser.add_argument('--probability', type=float, default=None, help='probability of the impulse response.')
    rir_parser.add_argument('rir_rspecifier', type=str, help="""rir rspecifier, it can be either a filename or a piped command.
                            E.g. data/impulses/Room001-00001.wav or "sox data/impulses/Room001-00001.wav -t wav - |" """)

    set_list = parse_set_parameter_strings(rir_set_para_array)

    rir_list = []
    for rir_set in set_list:
        current_rir_list = [rir_parser.parse_args(shlex.split(x.strip())) for x in open(rir_set.filename)]
        for rir in current_rir_list:
            if sampling_rate is not None:
                # check if the rspecifier is a pipe or not
                if len(rir.rir_rspecifier.split()) == 1:
                    rir.rir_rspecifier = rir.rir_rspecifier.replace("/data", rir_root)
                    rir.rir_rspecifier = "sox {0} -r {1} -t wav - |".format(rir.rir_rspecifier, sampling_rate)
        rir_list += current_rir_list
    return rir_list

def parse_set_parameter_strings(set_para_array):
    """ This function parse the array of rir set parameter strings.
        It will assign probabilities to those rir sets which don't have a probability
        It will also check the existence of the rir list files.
    """
    set_list = []
    for set_para in set_para_array:
        set = lambda: None
        setattr(set, "filename", None)
        setattr(set, "probability", None)
        parts = set_para.split(',')
        if len(parts) == 2:
            set.probability = float(parts[0])
            set.filename = parts[1].strip()
        else:
            set.filename = parts[0].strip()

        if not os.path.isfile(set.filename):
            raise Exception(set.filename + " not found")
        set_list.append(set)

    return set_list

def parse_file_to_dict(file, assert2fields = False, value_processor = None):
    """ This function parses a file and pack the data into a dictionary
        It is useful for parsing file like wav.scp, utt2spk, text...etc
    """
    if value_processor is None:
        value_processor = lambda x: x[0]
    dict = {}
    for line in open(file, 'r', encoding='utf-8'):
        parts = line.split()
        if assert2fields:
            assert(len(parts) == 2)

        dict[parts[0]] = value_processor(parts[1:])
    return dict

def augment_bg_wav(utt, wav, dur, method, bg_snr_opts, bg_noise_utts, noise_wavs, noise2dur, num_opts, output_dir):
    # This section is common to both foreground and background noises
    new_wav = ""
    dur_str = str(dur)
    noise_dur = 0
    tot_noise_dur = 0
    snrs=[]
    noises=[]
    start_times=[]

    # Now handle the background noises
    if len(bg_noise_utts) > 0:
        num = random.choice(num_opts)
        for i in range(0, num):
            noise_utt = random.choice(bg_noise_utts)
            noise = "./wav-reverberate --duration=" \
            + dur_str + " \"" + noise_wavs[noise_utt] + "\" - |"
            snr = random.choice(bg_snr_opts)
            snrs.append(snr)
            start_times.append(0)
            noises.append(noise)

    start_times_str = "--start-times='" + ",".join([str(i) for i in start_times]) + "'"
    snrs_str = "--snrs='" + ",".join([str(i) for i in snrs]) + "'"
    noises_str = "--additive-signals='" + ",".join(noises).strip() + "'"

    output_path = os.path.join(output_dir, utt)
    # If the wav is just a file
    if wav.strip()[-1] != "|":
        new_wav = "./wav-reverberate --shift-output=true " + noises_str + " " \
            + start_times_str + " " + snrs_str + " " + wav + " " + output_path + "-" + method + ".wav"
    # Else if the wav is in a pipe
    else:
        new_wav = wav + "./wav-reverberate --shift-output=true " + noises_str + " " \
            + start_times_str + " " + snrs_str + " - - |"
    
    aug_name = f"{output_path}-{method}.wav"
    
    return new_wav, aug_name

def augment_fg_wav(utt, wav, dur, fg_snr_opts, fg_noise_utts, noise_wavs, noise2dur, interval, output_dir):
    # This section is common to both foreground and background noises
    new_wav = ""
    dur_str = str(dur)
    noise_dur = 0
    tot_noise_dur = 0
    snrs=[]
    noises=[]
    start_times=[]

    # Now handle the foreground noises
    if len(fg_noise_utts) > 0:
        while tot_noise_dur < dur:
            noise_utt = random.choice(fg_noise_utts)
            noise = noise_wavs[noise_utt]
            snr = random.choice(fg_snr_opts)
            snrs.append(snr)
            noise_dur = noise2dur[noise_utt]
            start_times.append(tot_noise_dur)
            tot_noise_dur += noise_dur + interval
            noises.append(noise)

    start_times_str = "--start-times='" + ",".join([str(i) for i in start_times]) + "'"
    snrs_str = "--snrs='" + ",".join([str(i) for i in snrs]) + "'"
    noises_str = "--additive-signals='" + ",".join(noises).strip() + "'"

    output_path = os.path.join(output_dir, utt)
    # If the wav is just a file
    if wav.strip()[-1] != "|":
        new_wav = "./wav-reverberate --shift-output=true " + noises_str + " " \
            + start_times_str + " " + snrs_str + " " + wav + " " + output_path + "-noise.wav"
    # Else if the wav is in a pipe
    else:
        new_wav = wav + "./wav-reverberate --shift-output=true " + noises_str + " " \
            + start_times_str + " " + snrs_str + " - - |"
    
    aug_name = f"{output_path}-noise.wav"
    
    return new_wav, aug_name

def augment_rvb_wav(wav, wav_id, rir_list, output_dir):
    noise_utt = random.choice(rir_list)
    output_path = os.path.join(output_dir, wav_id)
    noise_rvb_command = """./wav-reverberate --shift-output=true --impulse-response="{}" {} {}-reverb.wav """.format(noise_utt.rir_rspecifier, wav, output_path)
    aug_name = f"{output_path}-reverb.wav"
    return noise_rvb_command, aug_name

def augment_speed_wav(wav, wav_id, speed_rates, output_dir):
    speed_rate = random.choice(speed_rates)
    output_path = os.path.join(output_dir, wav_id)
    command = "sox -t wav {0} {1}-sp{2}.wav speed {2}".format(wav, output_path, speed_rate)
    aug_name = f"{output_path}-sp{speed_rate}.wav"
    return command, aug_name

def get_noise_list(noise_wav_scp_filename, noise_root):
    noise_wav_scp_file = open(noise_wav_scp_filename, 'r', encoding='utf-8').readlines()
    noise_wavs = {}
    noise_utts = []
    for line in noise_wav_scp_file:
        toks=line.split(" ")
        wav = " ".join(toks[1:])
        noise_utts.append(toks[0])
        wav = wav.replace("/data", noise_root)
        noise_wavs[toks[0]] = wav.rstrip()
    return noise_utts, noise_wavs


def RunKaldiCommand(command, wait = True):
    """ Runs commands frequently seen in Kaldi scripts. These are usually a
        sequence of commands connected by pipes, so we use shell=True """
    #logger.info("Running the command\n{0}".format(command))
    p = subprocess.Popen(command, shell = True,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE)

    if wait:
        [stdout, stderr] = p.communicate()
        if p.returncode is not 0:
            raise Exception("There was an error while running the command {0}\n------------\n{1}".format(command, stderr))
        return stdout, stderr
    else:
        return p

def get_noise(noise_dir, noise_root):
    noise_wavs = {}
    noise_reco2dur = {}
    noise_utts = []

    noise_wav_filename = noise_dir + "/wav.scp"
    noise_utts, noise_wavs = get_noise_list(noise_wav_filename, noise_root)
    noise_reco2dur = parse_file_to_dict(noise_dir + "/reco2dur",
        value_processor = lambda x: float(x[0]))
    noise_wavs.update(noise_wavs)
    noise_reco2dur.update(noise_reco2dur)

    return noise_utts, noise_wavs, noise_reco2dur

class DynamicAugmentation(object):

    def __init__(self, musan_dir, rir_dir, output_dir, pool):
        self.sr = 16000
        self.music_bg_snrs = [15, 10, 8, 5]
        self.babble_bg_snrs = [20, 17, 15, 13, 5]
        self.fg_snrs = [15, 10, 5, 0]
        self.speed_ration = [0.9, 1.1]

        self.music_num_bg_noises = [1]
        self.babble_num_bg_noises = [3, 4, 5, 6, 7] 

        self.fg_interval = 1
        self.output_dir = output_dir
        self.pool = pool

        music_dir = os.path.join(musan_dir, 'musan_music')
        babble_dir = os.path.join(musan_dir, 'musan_speech')
        noise_dir = os.path.join(musan_dir, 'musan_noise')

        smallroom_rir = '0.5, {}/simulated_rirs/smallroom/rir_list'.format(rir_dir)
        mediumroom_rir = '0.5, {}/simulated_rirs/mediumroom/rir_list'.format(rir_dir)
        rir_set_para_array = [smallroom_rir, mediumroom_rir]

        noise_root = os.path.dirname(musan_dir)
        rir_root = os.path.dirname(rir_dir)

        self.music_utts, self.music_wavs, self.music_reco2dur = get_noise(music_dir, noise_root)
        self.babble_utts, self.babble_wavs, self.babble_reco2dur = get_noise(babble_dir, noise_root)
        self.noise_utts, self.noise_wavs, self.noise_reco2dur = get_noise(noise_dir, noise_root)
        self.rir_list = parse_rir_list(rir_set_para_array, rir_root, self.sr)

    def random_augment_noise(self, sample):

        wav = sample['audio_path']
        utt = sample['id'].rstrip('.wav')
        dur = sample['duration']
        
        commands = [] 
        aug_files = []

        new_wav, aug_name = augment_bg_wav(utt, wav, dur, 'music', self.music_bg_snrs, self.music_utts, \
            self.music_wavs, self.music_reco2dur, self.music_num_bg_noises, self.output_dir)
        commands.append(new_wav)
        aug_files.append(aug_name)

        new_wav, aug_name = augment_bg_wav(utt, wav, dur, 'babble', self.babble_bg_snrs, self.babble_utts, \
            self.babble_wavs, self.babble_reco2dur, self.babble_num_bg_noises, self.output_dir)
        commands.append(new_wav)
        aug_files.append(aug_name)

        new_wav, aug_name = augment_fg_wav(utt, wav, dur, self.fg_snrs, self.noise_utts, \
            self.noise_wavs, self.noise_reco2dur, self.fg_interval, self.output_dir)
        commands.append(new_wav)
        aug_files.append(aug_name)

        new_wav, aug_name = augment_rvb_wav(wav, utt, self.rir_list, self.output_dir)
        commands.append(new_wav)
        aug_files.append(aug_name)

        new_wav, aug_name = augment_speed_wav(wav, utt, self.speed_ration, self.output_dir)
        commands.append(new_wav)
        aug_files.append(aug_name)

        sample['aug_files'] = aug_files

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(RunKaldiCommand, commands)
        
        return sample
        # self.pool.map(RunKaldiCommand, commands)

def data_processing(input_csv, audio_dir):
    df =  pd.read_csv(input_csv, skipinitialspace=True)
    audio_dict = []

    for index, data in df.iterrows():
        audio = data["slice_file_name"]
        classid = data["classID"]
        label = data["class"]

        audio_path = os.path.join(audio_dir, audio)

        if os.path.exists(audio_path):
            audio_id = os.path.basename(audio)

            with contextlib.closing(wave.open(audio_path,'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)

            audio_dict.append({'audio_path': audio_path, 'id': audio_id, 'duration': duration,
                            'classID': classid, 'class': label})

    print('Number of input audios: {}'.format(len(audio_dict)))
    
    return audio_dict

def create_aug_meta(input_csv, aug_metadata):
    df =  pd.read_csv(input_csv, skipinitialspace=True)
    new_df = df.copy()

    for aug_data in aug_metadata:
        for aug_file in aug_data['aug_files']:
            aug_filename = os.path.basename(aug_file)
            new_df.loc[len(new_df)] = [aug_filename, 0, 0, 0, 0, 0, aug_data["classID"], aug_data["class"]]

    return new_df

def main(args):
    aug_metadata = []
    pool = multiprocessing.Pool(cpus)

    input_csv = args.input_csv
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_dict = data_processing(input_csv, args.audio_dir)
    audio_dict = audio_dict
    dynamic_aug = DynamicAugmentation(args.musan_dir, args.rir_dir, output_dir, None)

    for sample in tqdm(pool.imap_unordered(dynamic_aug.random_augment_noise, audio_dict), total=len(audio_dict)):
        aug_metadata.append(sample) 

    pool.close()
    pool.join()

    aug_df = create_aug_meta(args.input_csv, aug_metadata)
    aug_df.to_csv(args.output_csv, index=False)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--audio_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--output_csv', type=str)

    parser.add_argument('--musan_dir', type=str, default='musan')
    parser.add_argument('--rir_dir', type=str, default='RIRS_NOISES')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
        