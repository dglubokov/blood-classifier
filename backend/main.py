import os
import zipfile
import shutil

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from blood_helpers import sculptor, coach, inspector


DATA_DIR_NAME = 'ds'

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


@app.post("/train/")
async def train_all(file: UploadFile = File(...)):
    f_name = file.filename.split('.')
    content_type = file.content_type.split('/')
    if 'zip' in f_name or 'zip' in content_type:
        bf = await file.read(file.spool_max_size)
        with open('ds.zip', 'wb') as f:
            f.write(bf)
        if os.path.isdir(DATA_DIR_NAME):
            shutil.rmtree(DATA_DIR_NAME)
        os.mkdir(DATA_DIR_NAME)

        unzipped_path = os.path.join(DATA_DIR_NAME, 'unzipped')
        zipfile.ZipFile('ds.zip').extractall(unzipped_path)

        peeled_path = os.path.join(DATA_DIR_NAME, 'peeled')
        sculptor.actions.sculpt(unzipped_path, peeled_path)

        ml_results = coach.ml.train_all()
        dl_results = coach.dl.train_cnn()
        
        return [ml_results, dl_results]


@app.post("/test/")
async def create_upload_file(file: UploadFile = File(...)):
    f_name = file.filename.split('.')
    content_type = file.content_type.split('/')
    if 'png' in f_name or 'png' in content_type:
        bf = await file.read(file.spool_max_size)
        with open('receive.png', 'wb') as f:
            f.write(bf)
        class_name = inspector.cnn_check.classify('receive.png')
        return {'class_name': class_name}
    return {'class_name': 'Wrong file format!'}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8082,
        reload=False,
        debug=False,
    )
