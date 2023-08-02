import cv2
import numpy as np

from mivolo.model.create_timm_model import create_model
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import argparse
import torch
import onnx
import logging
import io
import os

from mivolo.model.mi_volo import MiVOLO
from timm.data import resolve_data_config
from mivolo.model.yolo_detector import Detector
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult

def get_parser():
    parser = argparse.ArgumentParser(description="Convert Pytorch to ONNX model")

    parser.add_argument(
        '--name', type= str,
        default='mivolo'
    )
    parser.add_argument(
        "--output",
        default='onnx_model',
        help='path to save converted onnx model'
    )
    parser.add_argument(
        "--checkpoint", 
        default="",
        type=str, 
        required=True, 
        help="path to mivolo checkpoint")
    parser.add_argument(
        "--device", 
        default="cpu", 
        type=str, 
        help="Device (accelerator) to use.")
    parser.add_argument(
        "--with-persons", 
        action="store_true", 
        default=False, 
        help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", 
        action="store_true", 
        default=False, 
        help="If set model will use only persons if available"
    )
    parser.add_argument(
        '--dir_images', type= str, default= ''
    )
    parser.add_argument(
        '--batch_size', type= int, default=1
    )
    parser.add_argument("--half", 
        action="store_true", 
        default=False, help="use half-precision model")
    return parser

class Meta:
    def __init__(self):
        self.min_age = None
        self.max_age = None
        self.avg_age = None
        self.num_classes = None

        self.in_chans = 3
        self.with_persons_model = False
        self.disable_faces = False
        self.use_persons = True
        self.only_age = False

        self.num_classes_gender = 2

    def load_from_ckpt(self, ckpt_path: str, disable_faces: bool = False, use_persons: bool = True) -> "Meta":

        state = torch.load(ckpt_path, map_location="cpu")

        self.min_age = state["min_age"]
        self.max_age = state["max_age"]
        self.avg_age = state["avg_age"]
        self.only_age = state["no_gender"]

        only_age = state["no_gender"]

        self.disable_faces = disable_faces
        if "with_persons_model" in state:
            self.with_persons_model = state["with_persons_model"]
        else:
            self.with_persons_model = True if "patch_embed.conv1.0.weight" in state["state_dict"] else False

        self.num_classes = 1 if only_age else 3
        self.in_chans = 3 if not self.with_persons_model else 6
        self.use_persons = use_persons and self.with_persons_model

        if not self.with_persons_model and self.disable_faces:
            raise ValueError("You can not use disable-faces for faces-only model")
        if self.with_persons_model and self.disable_faces and not self.use_persons:
            raise ValueError("You can not disable faces and persons together")

        return self

    def __str__(self):
        attrs = vars(self)
        attrs.update({"use_person_crops": self.use_person_crops, "use_face_crops": self.use_face_crops})
        return ", ".join("%s: %s" % item for item in attrs.items())

    @property
    def use_person_crops(self) -> bool:
        return self.with_persons_model and self.use_persons

    @property
    def use_face_crops(self) -> bool:
        return not self.disable_faces or not self.with_persons_model

def create(args) :
    half: bool = True
    half = args.half and args.device != "cpu"
    meta: Meta = Meta().load_from_ckpt(args.checkpoint, 
                                       args.disable_faces, args.use_persons)
    model_name = "mivolo_d1_224"
    mivolo_model = create_model(
            model_name=model_name,
            num_classes=meta.num_classes,
            in_chans=meta.in_chans,
            pretrained=False,
            checkpoint_path=args.checkpoint,
            filter_keys=["fds."],
        )
    data_config = resolve_data_config(
            model=mivolo_model,
            verbose=True,
            use_test_size=True,
        )
    data_config["crop_pct"] = 1.0
    c, h, w = data_config["input_size"]
    mivolo_model = mivolo_model.to(args.device)
    mivolo_model.eval()
    if half :
        mivolo_model = mivolo_model.half()

    return mivolo_model

class Predictor:
    def __init__(self, args, verbose: bool = True):
        self.detector = Detector(args.detector_weights, args.device, verbose=verbose)
        self.age_gender_model = MiVOLO(
            args.checkpoint,
            args.device,
            half=True,
            use_persons=args.with_persons,
            disable_faces=args.disable_faces,
            verbose=verbose,
        )

    def recognize(self, image: np.ndarray) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        self.age_gender_model.predict(image, detected_objects)

        out_im = None
        if self.draw:
            # plot results on image
            out_im = detected_objects.plot()

        return detected_objects, out_im
    
def main():
    args = get_parser().parse_args()

if __name__ == "__main__":
    main()
