import subprocess
import os
import csv
from math import ceil

from mmsdk import mmdatasdk
import torchtext
import numpy as np

from .aligner.align import align


DATA_DIR = os.path.join('pipeline', 'data')
BIN_DIR = os.path.join('..', 'venv', 'bin')
WHISPER_EXE = os.path.join(BIN_DIR, 'whisper')
OPENFACE_EXE = os.path.join('pipeline', 'FeatureExtraction')

glove_emb = torchtext.vocab.GloVe()


def split_audio(video_id: str):
    raw_video = os.path.join(DATA_DIR, 'raw', video_id, 'video.mp4')
    raw_audio = os.path.join(DATA_DIR, 'raw', video_id, 'audio.wav')

    command = 'ffmpeg -i %s -ab 160k -ac 2 -ar 44100 -vn %s' % (
        raw_video,
        raw_audio,
    )

    subprocess.check_call(
        command,
        shell=True
    )


def transcribe(video_id: str):
    raw_audio = os.path.join(DATA_DIR, 'raw', video_id, 'audio.wav')
    data_dir = os.path.join(DATA_DIR, 'whisper', video_id)

    subprocess.check_call(
        WHISPER_EXE + ' --model tiny --language en %s -o %s' % (
            raw_audio,
            data_dir,
        ),
        shell=True,
    )


def openface(video_id: str):
    raw_video = os.path.join(DATA_DIR, 'raw', video_id, 'video.mp4')
    data_dir = os.path.join(DATA_DIR, 'openface', video_id)

    options = '-2Dfp -3Dfp -pdmparams -pose -aus -gaze'

    subprocess.check_call(
        OPENFACE_EXE + ' -f %s -out_dir %s %s' % (
            raw_video,
            data_dir,
            options,
        ),
        shell=True,
    )


def prepare_dataset(video_id: str, word_intervals: list[tuple[tuple, str]], granularity: float):
    openface_data_dir = os.path.join(DATA_DIR, 'openface', video_id)

    openface_data = {}
    openface_data['video'] = {
        'features': [],
        'intervals': [],
    }

    last_timestamp = None
    with open(os.path.join(openface_data_dir, 'video.csv')) as f:
        reader = csv.reader(f)

        # Skip the header
        reader.__next__()

        prev_timestamp = None
        features = []
        intervals = []
        for row in reader:
            timestamp = float(row[2])

            if prev_timestamp is None:
                intervals = [timestamp, timestamp]
            else:
                intervals = [prev_timestamp, timestamp]

            prev_timestamp = timestamp
            last_timestamp = timestamp

            features = [float(x) for x in row[1:]]

            openface_data['video']['features'].append(features)
            openface_data['video']['intervals'].append(intervals)

    glove_data = {}
    glove_data['video'] = {
        'features': [],
        'intervals': [],
    }

    for interval_tuple, word in word_intervals:
        intervals = list(interval_tuple)
        features = glove_emb[word]

        glove_data['video']['features'].append(features)
        glove_data['video']['intervals'].append(intervals)

    openface_data['video']['intervals'] = np.array(openface_data['video']['intervals'])
    openface_data['video']['features'] = np.array(openface_data['video']['features'])

    glove_data['video']['intervals'] = np.array(glove_data['video']['intervals'])
    glove_data['video']['features'] = [t.numpy() for t in glove_data['video']['features']]
    glove_data['video']['features'] = np.array(glove_data['video']['features'])

    fake_labels = {}
    fake_labels['video'] = {
        'features': [],
        'intervals': [],
    }

    start = 0.0
    finish = last_timestamp
    duration = finish - start
    num_segments = int(ceil(duration / granularity))

    features = []
    intervals = []
    for i in range(num_segments):
        features = [0]
        intervals = [i * granularity, (i + 1) * granularity]
        fake_labels['video']['features'].append(features)
        fake_labels['video']['intervals'].append(intervals)

    fake_labels['video']['intervals'] = np.array(fake_labels['video']['intervals'])
    fake_labels['video']['features'] = np.array(fake_labels['video']['features'])

    openface_comp_seq = mmdatasdk.computational_sequence('OpenFace_2')
    glove_comp_seq = mmdatasdk.computational_sequence('glove_vectors')
    fake_labels_seq = mmdatasdk.computational_sequence('labels')

    openface_comp_seq.set_data(openface_data)
    glove_comp_seq.set_data(glove_data)
    fake_labels_seq.set_data(fake_labels)

    metadata_template = [
        'root name',
        'computational sequence description',
        'dimension names'
        'computational sequence version',
        'alignment compatible',
        'dataset name',
        'dataset version',
        'creator',
        'contact',
        'featureset bib citation',
        'dataset bib citation'
    ]

    openface_metadata = {key: '' for key in metadata_template}
    openface_metadata['root name'] = 'OpenFace_2'

    glove_metadata = {key: '' for key in metadata_template}
    glove_metadata['root name'] = 'glove_vectors'

    fake_labels_metadata = {key: '' for key in metadata_template}
    fake_labels_metadata['root name'] = 'labels'

    openface_comp_seq.set_metadata(openface_metadata)
    glove_comp_seq.set_metadata(glove_metadata)
    fake_labels_seq.set_metadata(fake_labels_metadata)

    raw_dataset_dir = os.path.join(DATA_DIR, 'dataset', 'raw', video_id)
    if not os.path.exists(raw_dataset_dir):
        os.makedirs(raw_dataset_dir)

    openface_comp_seq.deploy(os.path.join(raw_dataset_dir, 'openface.csd'))
    glove_comp_seq.deploy(os.path.join(raw_dataset_dir, 'glove.csd'))
    fake_labels_seq.deploy(os.path.join(raw_dataset_dir, 'labels.csd'))

    mydataset_recipe = {
        'OpenFace_2': os.path.join(raw_dataset_dir, 'openface.csd'),
        'glove_vectors': os.path.join(raw_dataset_dir, 'glove.csd'),
    }
    mydataset = mmdatasdk.mmdataset(mydataset_recipe)
    mydataset.align('glove_vectors')
    mydataset.impute('glove_vectors')

    mydataset.add_computational_sequences({'labels': os.path.join(raw_dataset_dir, 'labels.csd')}, destination=None)
    mydataset.align('labels')

    return mydataset


def preprocess(video_id: str):
    split_audio(video_id)

    transcribe(video_id)

    openface(video_id)

    word_intervals = align(video_id)

    video_dataset = prepare_dataset(video_id, word_intervals, 5.0)

    aligned_dataset_dir = os.path.join(DATA_DIR, 'dataset', 'aligned', video_id)
    if not os.path.exists(aligned_dataset_dir):
        os.makedirs(aligned_dataset_dir)

    deploy_files = {x: x for x in video_dataset.keys()}
    video_dataset.deploy(destination=aligned_dataset_dir, filenames=deploy_files)


if __name__ == '__main__':
    preprocess()
