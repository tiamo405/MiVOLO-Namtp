
import cv2
import numpy as np

from mivolo.model.create_timm_model import create_model

import argparse
import torch
import onnx
import logging
import io
import os
import torchvision.transforms as transforms


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
        '--batch-size',
        default=1,
        type=int,
        help="the maximum batch size of onnx runtime"
    )
    parser.add_argument("--checkpoint", default="", type=str, required=True, help="path to mivolo checkpoint")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device (accelerator) to use.")
    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If set model will use only persons if available"
    )
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


def export_onnx_model(model, inputs):
    """
    Trace and export a model to onnx format.
    Args:
        model (nn.Module):
        inputs (torch.Tensor): the model will be called by `model(*inputs)`
    Returns:
        an onnx model
    """
    assert isinstance(model, torch.nn.Module)

    # make sure all modules are in eval mode, onnx may change the training state
    # of the module if the states are not consistent
    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)

    # logger.info("Beginning ONNX file converting")
    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            input_names = ['input_1']
            output_names = ['output_1']
            dynamic_axes = {input_names[0]: {0:'batch'}}
            for _, name in enumerate(output_names):
                dynamic_axes[name] = dynamic_axes[input_names[0]]
            extra_args = {'opset_version': 13, 'verbose': False,
                'input_names': input_names, 'output_names': output_names,
                'dynamic_axes': dynamic_axes}
            torch.onnx.export(model, inputs, f, **extra_args)

            onnx_model = onnx.load_from_string(f.getvalue())

    # logger.info("Completed convert of ONNX model")

    # # Apply ONNX's Optimization
    # logger.info("Beginning ONNX model path optimization")
    # all_passes = onnxoptimizer.get_available_passes()
    # passes = ["extract_constant_to_initializer", "eliminate_unused_initializer", "fuse_bn_into_conv"]
    # assert all(p in all_passes for p in passes)
    # onnx_model = onnxoptimizer.optimize(onnx_model, passes)
    # logger.info("Completed ONNX model path optimization")
    return onnx_model


if __name__ == '__main__':
    args = get_parser().parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    transform = transforms.Compose([
        transforms.ToPILImage(),# Chuyển đổi từ numpy array sang PIL Image
        transforms.Resize((224, 224)),
        transforms.ToTensor(),                     # Chuyển đổi PIL Image sang tensor
        transforms.Normalize((0.485, 0.456, 0.406), # Chuẩn hóa giá trị pixel
                            (0.229, 0.224, 0.225))
    ])
    meta: Meta = Meta().load_from_ckpt(args.checkpoint, use_persons=args.with_persons,
            disable_faces=args.disable_faces,)

    model_name = "mivolo_d1_224"

    model = create_model(
            model_name=model_name,
            num_classes=meta.num_classes,
            in_chans=meta.in_chans,
            pretrained=False,
            checkpoint_path=args.checkpoint,
            filter_keys=["fds."],
        )
    half : bool = True
    half = half and args.device != "cpu"

    if half:
        model = model.half()
    model.eval()
    print(model)
    inputs = cv2.imread("test/jennifer_lawrence.jpg")
    inputs = transform(inputs)

    # Chuyển đổi tensor về kích thước phù hợp với mô hình (nếu cần) 
    # inputs = torch.stack([inputs]*6, dim= 1)
    inputs = inputs.unsqueeze(0) 
    # inputs = inputs.repeat(1,2,1,1)
    print(inputs.shape)
    
    # inputs = torch.randn(args.batch_size, 6, 224, 224).to(args.device)
    with torch.no_grad():
        if half:
            inputs = inputs.half()
        output = model(inputs)

    age_output = output[:, 2]
    gender_output = output[:, :2].softmax(-1)
    gender_probs, gender_indx = gender_output.topk(1)

    age = age_output.item()
    age = age * (meta.max_age - meta.min_age) + meta.avg_age
    age = round(age, 2)
    print(age)
    print("male" if gender_indx.item() == 0 else "female")

    # PathManager.mkdirs(args.output)
    # onnx_model = export_onnx_model(model, inputs)
    # os.makedirs(args.output, exist_ok= True)
    # save_path = os.path.join(args.output, args.name+'.onnx')
    # onnx.save_model(onnx_model, save_path)
    
