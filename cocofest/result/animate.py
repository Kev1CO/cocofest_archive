import pyorerun as prr
import biorbd
import pickle
import numpy as np


class PickleAnimate:
    def __init__(self, path: str):
        self.path = path
        self.data = None
        self.model = None
        self.time = None
        self.state_q = None
        self.frames = None

    def load(self):

        with open(self.path, "rb") as f:
            self.data = pickle.load(f)

        model_path = self.data["bio_model_path"] if "bio_model_path" in self.data.keys() else None
        if model_path is None:
            raise ValueError("The bio model path is not available, please provide it to animate the solution.")

        # Load a predefined model
        self.model = biorbd.Model(model_path)
        self.time = self.data["time"]
        self.state_q = self.data["states"]["q"]
        self.frames = self.state_q.shape[1]

    def animate(self):
        self.load()

        # pyorerun animation
        prr_model = prr.BiorbdModel.from_biorbd_object(self.model)

        nb_seconds = self.time[-1]
        t_span = np.linspace(0, nb_seconds, self.frames)

        viz = prr.PhaseRerun(t_span)
        viz.add_animated_model(prr_model, self.state_q)
        viz.rerun("msk_model")
