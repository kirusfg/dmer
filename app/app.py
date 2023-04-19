import os
import shutil
import threading

import numpy as np
from flask import Flask, render_template, request, url_for, redirect, send_from_directory, stream_with_context, stream_template
from flask_uploads import ALL, UploadSet, configure_uploads
from flask_executor import Executor

from pipeline.main import preprocess
from pipeline.model.infer import infer


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your secret key'
app.config['UPLOADED_VIDEOS_DEST'] = os.path.join('pipeline', 'data', 'raw')

videos = UploadSet('videos', ALL)
configure_uploads(app, videos)

executor = Executor(app)


if os.path.exists(os.path.join('pipeline', 'data', 'raw')):
    shutil.rmtree(os.path.join('pipeline', 'data', 'raw'), ignore_errors=True)
if os.path.exists(os.path.join('pipeline', 'data', 'dataset')):
    shutil.rmtree(os.path.join('pipeline', 'data', 'dataset'), ignore_errors=True)
if os.path.exists(os.path.join('pipeline', 'data', 'mfa')):
    shutil.rmtree(os.path.join('pipeline', 'data', 'mfa'), ignore_errors=True)
if os.path.exists(os.path.join('pipeline', 'data', 'openface')):
    shutil.rmtree(os.path.join('pipeline', 'data', 'openface'), ignore_errors=True)
if os.path.exists(os.path.join('pipeline', 'data', 'results')):
    shutil.rmtree(os.path.join('pipeline', 'data', 'results'), ignore_errors=True)
if os.path.exists(os.path.join('pipeline', 'data', 'whisper')):
    shutil.rmtree(os.path.join('pipeline', 'data', 'whisper'), ignore_errors=True)

video_id = 0


def video_processing(video_id: str):
    yield True
    preprocess(video_id)
    infer(video_id)
    yield False


@app.route('/')
@app.route('/home')
@app.route('/index')
@app.route('/index.html')
def start():
    return render_template('index.html')


@app.route('/data/<path:path>')
def send_data(path):
    return send_from_directory('pipeline', path)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global video_id
    if request.method == 'POST' and 'video' in request.files:
        _ = videos.save(request.files['video'], str(video_id), 'video.')

        url = url_for('results_video', video_id=str(video_id))
        video_id += 1

        return redirect(url)
    return render_template('upload-video.html')


@app.route('/process/<video_id>')
def process(video_id):
    # executor.submit(thread_function, video_id)
    # t = threading.Thread(target=thread_function, args=[video_id])
    # t.start()

    return render_template(
        'results-video.html',
        video_id=video_id,
        processing=True,
    )


@app.route('/results-video/<video_id>')
def results_video(video_id):
    return stream_template(
        'results-video.html',
        video_id=video_id,
        processing=video_processing(video_id),
    )


@app.route('/result', methods=['GET', 'POST'])
@app.route('/result.html', methods=['GET', 'POST'])
def result():
    image = "emotion.jpg"
    if request.method == 'POST':
        req = request.form
        # TODO

    with open("data.txt", "r") as file:
        lines = file.readlines()
        headers = lines[1].split()
        data = [line.split()[:2] for line in lines[2:-1]]
        maxbar = np.array(data)
        maxindex = np.argmax(maxbar[:, 1])

    return render_template(
        'result.html',
        headers=headers[:2],
        data=data,
        maxbar=maxindex,
        linesep="---------",
        image=image
    )


@app.route('/test')
@app.route('/test.html')
def test():
    with open("data.txt", "r") as file:
        lines = file.readlines()
        headers = lines[1].split()
        data = [line.split()[:2] for line in lines[2:-1]]
        maxbar = np.array(data)
        maxindex = np.argmax(maxbar[:, 1])
    return render_template('test.html', headers=headers[:2], data=data, maxbar=maxindex, linesep="---------")
