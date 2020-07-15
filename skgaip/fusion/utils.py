import os
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver

def experiment(name):
    ex = Experiment(name)
    mongo_observer = MongoObserver(url="mongodb://dsuo:asdf@localhost:27017")
    file_observer = FileStorageObserver(basedir=os.path.join("sacred/experiments", name))
    ex.observers.append(mongo_observer)
    ex.observers.append(file_observer)
    return ex
