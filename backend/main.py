import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from utils import classifier


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


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    f_name = file.filename.split('.')
    content_type = file.content_type.split('/')
    if 'png' in f_name or 'png' in content_type:
        bf = await file.read(file.spool_max_size)
        with open('receive.png', 'wb') as f:
            f.write(bf)
        class_name = classifier.classify('receive.png')
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
