import os
import zipfile
import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from starlette.responses import FileResponse
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

import cnn_check



DATA_DIR_NAME = 'datasets'

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]


app = FastAPI(
    title='Blood classifier API',
    middleware=middleware,
)


@app.post("/upload/")
def infer(file: UploadFile = File(...)):
    f_name = file.filename.split('.')
    content_type = file.content_type.split('/')
    if 'zip' not in f_name and 'zip' not in content_type:
        return {'class_name': 'Wrong file format!'}

    if os.path.isdir(DATA_DIR_NAME):
        shutil.rmtree(DATA_DIR_NAME)
    os.mkdir(DATA_DIR_NAME)

    bf = file.file.read()
    with open('ds.zip', 'wb') as f:
        f.write(bf)

    with zipfile.ZipFile('ds.zip', 'r') as zip:
        zip.extractall(DATA_DIR_NAME)

    results = list()
    main_path = Path(DATA_DIR_NAME) / ''.join(f_name[:-1])
    for p in main_path.iterdir():
        class_name = cnn_check.classify(p)
        results.append({
            'path': p.name,
            'class_name': class_name,
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv('results.tsv', sep='\t', index=False)
    d = pd.DataFrame(results_df['class_name'].value_counts())
    d.to_csv('summary.tsv', sep='\t')


@app.get('/download/')
def download():
    return FileResponse(
        'results.tsv',
        media_type='application/octet-stream',
        filename='results.tsv'
    )


@app.get('/download-summary/')
def download_summary():
    d = pd.read_table('summary.tsv')
    fig = Figure(figsize=(10, 10))
    ax = fig.subplots()
    ax.barh(d['Unnamed: 0'], d['class_name'], color=[np.random.rand(3,) for _ in range(len(d))])
    fig.savefig('summary.png', format="png")
    return FileResponse(
        'summary.png',
        media_type='application/octet-stream',
        filename='summary.png'
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8082,
        reload=False,
        debug=False,
    )
