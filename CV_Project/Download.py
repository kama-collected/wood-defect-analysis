from roboflow import Roboflow
rf = Roboflow(api_key="GUfFf1O7SPcClfrFzK9a")
project = rf.workspace("darren-n9avh").project("wood-7ao2p-9ivge")
version = project.version(1)
dataset = version.download("yolo26")
                