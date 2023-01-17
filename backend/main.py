import os
import zipfile
from pathlib import Path

import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from matplotlib.figure import Figure


import inference


DATA_DIR_NAME = Path("datasets")

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
]


app = FastAPI(
    title="Blood classifier API",
    middleware=middleware
)


@app.post("/classify/", tags=["Endpoints"])
def infer(file: UploadFile = File(...)):
    f_name = file.filename.split(".")
    content_type = file.content_type.split("/")
    if "zip" not in f_name and "zip" not in content_type:
        return HTTPException(status_code=404, detail="Wrong file format!")

    if not os.path.isdir(DATA_DIR_NAME):
        os.mkdir(DATA_DIR_NAME)

    with zipfile.ZipFile(file.file.read(), "rb") as zip:
        zip.extractall(DATA_DIR_NAME)

    results = []
    samples_path = DATA_DIR_NAME / "".join(f_name[:-1])
    for p in samples_path.iterdir():
        class_name = inference.classify(p)
        results.append({
            "path": p.name,
            "class_name": class_name,
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.tsv", sep="\t", index=False)
    d = pd.DataFrame(results_df["class_name"].value_counts())
    d.to_csv("summary.tsv", sep="\t")


@app.get("/download-results/", tags=["Endpoints"])
def download():
    return FileResponse(
        "results.tsv",
        media_type="application/octet-stream",
        filename="results.tsv"
    )


@app.get("/download-summary/", tags=["Endpoints"])
def download_summary():
    d = pd.read_table("summary.tsv")
    fig = Figure(figsize=(10, 10))
    ax = fig.subplots()
    ax.barh(d["Unnamed: 0"], d["class_name"], color=[np.random.rand(3,) for _ in range(len(d))])
    fig.savefig("summary.png", format="png")
    return FileResponse(
        "summary.png",
        media_type="application/octet-stream",
        filename="summary.png"
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8082,
        reload=False,
        debug=False,
    )
