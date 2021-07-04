def change_type(path: str):
    pass

def resample(path: str):
    pass

def augmentate(path: str):
    pass

def carve_features(path: str):
    pass

def sculpt(path: str, save_to: str):
    resample(path)
    change_type(path)
    augmentate(path)
    carve_features(path)
