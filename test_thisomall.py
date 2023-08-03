import argparse
import logging
import os

import cv2
import torch
from pedes_thiso import PedesAttr, getitem
from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging
import glob

_logger = logging.getLogger("inference")



def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    parser.add_argument("--output", type=str, default=None, required=True, help="folder for output results")
    parser.add_argument("--detector-weights", type=str, default=None, required=True, help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", default="", type=str, required=True, help="path to mivolo checkpoint")

    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If set model will use only persons if available"
    )

    parser.add_argument("--draw", action="store_true", default=False, help="If set, resulted images will be drawn")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")
    parser.add_argument('--dir_json', type= str, default='./')
    parser.add_argument('--dir_image', type= str, default= './')
    parser.add_argument('--ckpt_torchscript', type= str, default='models/traced_model.pt')
    parser.add_argument('--use_torchscript', action="store_true", default= False)
    parser.add_argument('--input', type= str, default='input_custom', choices=['input_default', 'input_custom'])
    return parser


def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)

    predictor = Predictor(args, verbose=True)

    data_thiso = PedesAttr(cfg= None, split= "train", dir_json= args.dir_json, dir_image= args.dir_image)
    num_item = data_thiso.__len__()
    count_pre = 0
    num_img = 0
    img_fail = []
    img_error = []
    for index in range(num_item):
        imgpath, gender_tt, age_tt = getitem(data_thiso, index)
        img =cv2.imread(imgpath)
        print(imgpath)
        try:
            detected_objects, out_im, ages, genders = predictor.recognize(img)
            num_img += 1
            age, gender = ages[0], genders[0]
            if gender == gender_tt : count_pre += 1
            else : 
                img_fail.append(imgpath)
            if args.draw:
                bname = os.path.splitext(os.path.basename(imgpath))[0]
                filename = os.path.join(args.output, f"out_{bname}.jpg")
                cv2.imwrite(filename, out_im)
                _logger.info(f"Saved result to {filename}")
            
        except :
            img_error.append(imgpath)
            continue
        print('{} : du doan dung tren {}'.format(count_pre, num_img))
        print('img_error: ',img_error)
        print('img fail: ', img_fail)
if __name__ == "__main__":
    main()
