import os
import torch
from datetime import date, datetime, time, timedelta

from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from webvtt import WebVTT, Caption

from mmsdk import mmdatasdk

from .src.cli import get_args
from .src.utils import capitalize_first_letter, load
from .src.data import get_glove_emotion_embs, MOSEI
from .src.models.eea import EmotionEmbAttnModel
from .src.config import EMOTIONS
from .src.evaluate import infer_mosei_emo


def activated_emotions(pred: list):
    emotion_names = EMOTIONS['mosei_emo']
    emotion_names = capitalize_first_letter(emotion_names)

    activated = []
    for i, emo in enumerate(pred):
        if emo >= 0.5:
            activated.append(i)

    if len(activated) == 0:
        return ''

    emotions = []
    for emo_index in activated:
        emotions.append(emotion_names[emo_index])

    emotions_as_caption = ''
    for i, emo in enumerate(emotions):
        emotions_as_caption += emo
        if i != len(emotions) - 1:
            emotions_as_caption += ', '

    return emotions_as_caption


def write_subtitles(video_id: str, preds: list):
    vtt = WebVTT()

    start = time()
    end = time(second=5)
    interval = timedelta(seconds=5)
    for pred in preds:
        start_str = start.strftime('%H:%M:%S.%f')[:-3]
        end_str = end.strftime('%H:%M:%S.%f')[:-3]

        start = (datetime.combine(date.today(), start) + interval).time()
        end = (datetime.combine(date.today(), end) + interval).time()

        text = activated_emotions(pred)

        caption = Caption(
            start_str,
            end_str,
            text,
        )

        vtt.captions.append(caption)

    vtt_dir = os.path.join('pipeline', 'data', 'results', video_id)
    vtt.save(os.path.join(vtt_dir, 'captions.vtt'))


def infer(video_id: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device {device}")

    path = os.path.join('pipeline', 'data', 'dataset', 'aligned', video_id)

    aligned_cmumosei_highlevel = mmdatasdk.mmdataset(path)

    tensors = aligned_cmumosei_highlevel.get_tensors(
        seq_len=20,
        direction=False,
    )

    text_data = tensors[0]["glove_vectors"]
    vision_data = tensors[0]["OpenFace_2"]

    infer_data = MOSEI(
        list(range(len(text_data))),  # Number of 20-seq-long sections
        text_data,
        [],
        vision_data,
        [],
    )

    infer_loader = DataLoader(infer_data, 512, shuffle=False)

    modal_dims = list(infer_data.get_dim())

    emo_list = EMOTIONS['mosei_emo']
    emo_list = capitalize_first_letter(emo_list)

    emo_weights = get_glove_emotion_embs(os.path.join('pipeline', 'data', 'glove.emotions.840B.300d.pt'))
    emo_weight = []
    for emo in emo_list:
        emo_weight.append(emo_weights[emo])

    MODEL = EmotionEmbAttnModel
    model = MODEL(
        num_classes=len(emo_list),
        input_sizes=modal_dims,
        hidden_size=300,
        hidden_sizes=[300, 200, 100],
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        modalities='tv',
        device=device,
        emo_weight=emo_weight,
        gru=False
    )

    model = model.to(device=device)

    # Load model checkpoint
    state_dict = load(os.path.join('pipeline', 'model', 'tv.pt'))
    state_dict.pop('textEmoEmbs.weight')
    if state_dict['modality_weights.weight'].size(0) != len('tv'):
        state_dict.pop('modality_weights.weight')

    model.load_state_dict(state_dict, strict=False)

    pos_weight = infer_data.get_pos_weight()
    pos_weight = pos_weight.to(device)

    model.eval()
    total_logits = None
    total_Y = None
    for X, Y, _ in tqdm(infer_loader, desc='infer'):
        X_text, X_audio, X_vision = X
        X_text = X_text.to(device=device)
        X_audio = X_audio.to(device=device)
        X_vision = X_vision.to(device=device)
        Y = Y.to(device=device)

        with torch.set_grad_enabled(False):
            logits = model(X_text, X_audio, X_vision)

        total_logits = torch.cat((total_logits, logits), dim=0) if total_logits is not None else logits
        total_Y = torch.cat((total_Y, Y), dim=0) if total_Y is not None else Y

    emotions = EMOTIONS['mosei_emo']
    preds, truths = infer_mosei_emo(total_logits, total_Y, 0.5, False)

    results_dir = os.path.join('pipeline', 'data', 'results', video_id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        for p in range(len(preds)):
            table = [['Emotion', 'Prediction']]
            for index, emotion in enumerate(emotions):
                table.append([emotion, preds[p][index]])

            f.write(tabulate(table))
            f.write('\n\n')

    write_subtitles(video_id, preds)


def main():
    args = get_args()

    device = torch.device(f"cuda:{args['cuda']}" if torch.cuda.is_available() else 'cpu')
    print(f"Running on device {device}")

    video_id = args['video_id']

    path = os.path.join('..', 'data', 'dataset', 'aligned', video_id)

    aligned_cmumosei_highlevel = mmdatasdk.mmdataset(path)

    tensors = aligned_cmumosei_highlevel.get_tensors(
        seq_len=20,
        direction=False,
    )

    text_data = tensors[0]["glove_vectors"]
    vision_data = tensors[0]["OpenFace_2"]

    print(text_data[0].shape)
    print(text_data[0][:5])
    print(text_data[-1][:5])
    print(vision_data[0].shape)
    print(vision_data[0][:5])
    print(vision_data[-1][:5])

    infer_data = MOSEI(
        list(range(len(text_data))),  # Number of 20-seq-long sections
        text_data,
        [],
        vision_data,
        [],
    )

    infer_loader = DataLoader(infer_data, batch_size=args['batch_size'], shuffle=False)

    modal_dims = list(infer_data.get_dim())

    zsl = args['zsl']
    emo_list = EMOTIONS[args['dataset']]
    if zsl != -1:
        if args['dataset'] == 'iemocap':
            emo_list.append(EMOTIONS['iemocap9'][zsl])
        else:
            emo_list = emo_list[:zsl] + emo_list[zsl + 1:]

    if args['cap']:
        emo_list = capitalize_first_letter(emo_list)

    emo_weights = get_glove_emotion_embs(args['glove_emo_path'])
    emo_weight = []
    for emo in emo_list:
        emo_weight.append(emo_weights[emo])

    MODEL = EmotionEmbAttnModel
    model = MODEL(
        num_classes=len(emo_list),
        input_sizes=modal_dims,
        hidden_size=args['hidden_size'],
        hidden_sizes=args['hidden_sizes'],
        num_layers=args['num_layers'],
        dropout=args['dropout'],
        bidirectional=args['bidirectional'],
        modalities=args['modalities'],
        device=device,
        emo_weight=emo_weight,
        gru=args['gru']
    )

    model = model.to(device=device)

    # Load model checkpoint
    state_dict = load(args['ckpt'])
    state_dict.pop('textEmoEmbs.weight')
    if state_dict['modality_weights.weight'].size(0) != len(args['modalities']):
        state_dict.pop('modality_weights.weight')

    model.load_state_dict(state_dict, strict=False)

    pos_weight = infer_data.get_pos_weight()
    pos_weight = pos_weight.to(device)

    model.eval()
    total_logits = None
    total_Y = None
    for X, Y, _ in tqdm(infer_loader, desc='infer'):
        X_text, X_audio, X_vision = X
        X_text = X_text.to(device=device)
        X_audio = X_audio.to(device=device)
        X_vision = X_vision.to(device=device)
        Y = Y.to(device=device)

        with torch.set_grad_enabled(False):
            logits = model(X_text, X_audio, X_vision)
            # loss = criterion(logits, Y)

        total_logits = torch.cat((total_logits, logits), dim=0) if total_logits is not None else logits
        total_Y = torch.cat((total_Y, Y), dim=0) if total_Y is not None else Y

    emotions = EMOTIONS['mosei_emo']
    preds, truths = infer_mosei_emo(total_logits, total_Y, 0.5, False)

    print(preds)

    with open('../data/result.txt', 'w') as f:
        for p in range(len(preds)):
            table = [['Emotion', 'Prediction']]
            for index, emotion in enumerate(emotions):
                table.append([emotion, preds[p][index]])

            f.write(tabulate(table))
            f.write('\n\n')

            if args['verbose'] == '1':
                print(tabulate(table))
                print()


if __name__ == '__main__':
    main()
