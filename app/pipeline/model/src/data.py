import os
import h5py
import numpy as np

from mmsdk import mmdatasdk

from .dataset import MOSI, MOSEI, IEMOCAP
from .utils import save, load, load2


def get_data(args, phase, infer_sample=None):
    dataset = args["dataset"]
    seq_len = args["data_seq_len"]
    file_folder = args["data_folder"]
    aligned = args["aligned"]

    zsl = args["zsl"]
    fsl = args["fsl"] if phase == "train" else -1

    if phase == "infer":
        phase = "test"

    processed_path = f'./processed_datasets/{dataset}_{seq_len}_{phase}{"" if aligned else "_noalign"}.pt'
    if os.path.exists(processed_path) and zsl == -1 and fsl == -1:
        print(f"Load processed dataset! - {phase}")
        return load(processed_path)

    if dataset == "mosi":
        if seq_len == 20:
            data_path = os.path.join(file_folder, f"X_{phase}.h5")
            label_path = os.path.join(file_folder, f"y_{phase}.h5")
            data = np.array(h5py.File(data_path, "r")["data"])
            labels = np.array(h5py.File(label_path.replace("X", "y"), "r")["data"])
            text = data[:, :, :300]
            audio = data[:, :, 300:305]
            vision = data[:, :, 305:]
            this_dataset = MOSI(
                list(range(len(labels))), text, audio, vision, labels, is20=True
            )
        else:
            data_path = os.path.join(
                file_folder, f'mosi_data{"" if aligned else "_noalign"}.pkl'
            )
            data = load(data_path)
            data = data[phase]
            this_dataset = MOSI(
                data["id"], data["text"], data["audio"], data["vision"], data["labels"]
            )
    elif dataset == "mosei_emo":
        path = "/raid/kirill_kirillov/dmer/Modality-Transferable-MER/data/cmu-mosei/"
        path = path + "processed/"

        aligned_cmumosei_highlevel = mmdatasdk.mmdataset(path)

        tensors = aligned_cmumosei_highlevel.get_tensors(
            seq_len=20,
            direction=False,
            non_sequences=["All Labels"],
            folds=[
                mmdatasdk.cmu_mosei.standard_folds.standard_train_fold,
                mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold,
                mmdatasdk.cmu_mosei.standard_folds.standard_test_fold,
            ],
        )
        fold_names = ["train", "valid", "test"]
        fold_index = fold_names.index(phase)

        text_data = tensors[fold_index]["glove_vectors"]
        audio_data = tensors[fold_index]["COVAREP"]
        vision_data = tensors[fold_index]["FACET 4.2"]

        # Class order: happy, sad, anger, surprise, disgust, fear
        labels = tensors[fold_index]["All Labels"]
        # Delete sentiment
        labels = np.delete(labels, 0, axis=2)
        labels = np.array(labels > 0, np.int32)
        # Remove unneeded dimension
        labels = labels.reshape(labels.shape[0], labels.shape[2])

        this_dataset = MOSEI(
            list(range(len(labels))),
            text_data,
            audio_data,
            vision_data,
            labels,
            zsl=zsl,
            fsl=fsl,
        )
    elif dataset == "iemocap":
        data_path = os.path.join(
            file_folder, f'iemocap_data{"" if aligned else "_noalign"}.pkl'
        )
        data = load(data_path)
        data = data[phase]

        # iemocap4 Distribution
        # neutral happy sad angry
        # [954    338   690 735]
        # [358    116   188 136]
        # [383    135   193 227]
        text_data = data["text"]
        audio_data = data["audio"]
        vision_data = data["vision"]
        labels = data["labels"]
        labels = np.argmax(labels, axis=-1)

        if zsl != -1:
            # iemocap9 Distribution
            # 0     1       2    3   4         5          6     7       8
            # Anger Excited Fear Sad Surprised Frustrated Happy Neutral Disgust
            # [735  686     19   690  65       1235       338   954     1] (Train)
            # [136  206     9    188  17       319        116   358     0] (Valid)
            # [227  141     12   193  25       278        135   383     1] (Test)
            iemocap9_text_data = load2(os.path.join(file_folder, f"text_{phase}.p"))
            iemocap9_audio_data = load2(os.path.join(file_folder, f"covarep_{phase}.p"))
            iemocap9_vision_data = load2(os.path.join(file_folder, f"facet_{phase}.p"))
            iemocap9_labels = load2(os.path.join(file_folder, f"y_{phase}.p"))
            iemocap9_labels = iemocap9_labels[:, 1:-1]
            iemocap9_labels = np.expand_dims(iemocap9_labels[:, zsl], axis=1)

            nonzeros = [i for i, l in enumerate(iemocap9_labels) if np.sum(l) != 0]
            zsl_text_data = iemocap9_text_data[nonzeros]
            zsl_audio_data = iemocap9_audio_data[nonzeros]
            zsl_vision_data = iemocap9_vision_data[nonzeros]
            zsl_labels = iemocap9_labels[nonzeros]

            # Align seq len to 20
            zsl_text_data = zsl_text_data[:, :-1, :]
            zsl_audio_data = zsl_audio_data[:, :-1, :]
            zsl_vision_data = zsl_vision_data[:, :-1, :]

            text_data = np.concatenate((text_data, zsl_text_data), axis=0)
            audio_data = np.concatenate((audio_data, zsl_audio_data), axis=0)
            vision_data = np.concatenate((vision_data, zsl_vision_data), axis=0)

            labels = np.concatenate((labels, np.zeros((len(labels), 1))), axis=1)
            zsl_labels = np.concatenate(
                (np.zeros((len(zsl_labels), 4)), zsl_labels), axis=1
            )
            labels = np.concatenate((labels, zsl_labels), axis=0)
        this_dataset = IEMOCAP(
            list(range(len(labels))), text_data, audio_data, vision_data, labels
        )
    else:
        raise ValueError("Wrong dataset!")

    if zsl == -1 and fsl == -1:
        save(this_dataset, processed_path)

    return this_dataset


def get_glove_emotion_embs(path):
    dataDict = load(path)
    return dataDict
