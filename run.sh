# docker run --gpus device=0 --name mivolo --runtime nvidia -dit -v /mnt/nvme0n1/phuongnam/MiVOLO-Namtp:/workspace nvcr.io/nvidia/pytorch:21.03-py3
# docker exec -it mivolo bash

# python demo.py \
#     --input "test/jennifer_lawrence.jpg" \
#     --output "output" \
#     --detector-weights "models/yolov8x_person_face.pt " \
#     --checkpoint "models/model_imdb_cross_person_4.24_99.46.pth.tar" \
#     --device "cpu" \
#     --with-persons \
#     --draw

# python onnx_export.py \
#     --output "output_onnx" \
#     --with-persons \
#     --checkpoint models/model_utk_age_gender_4.23_97.69.pth.tar \
#     --device "cpu"

# python onnx_export.py \
#     --output "output_onnx" \
#     --with-persons \
#     --checkpoint models/model_imdb_cross_person_4.22_99.46.pth.tar \
#     --device "cpu"

# python onnx_export.py \
#     --output "output_onnx" \
#     --with-persons \
#     --checkpoint models/model_imdb_cross_person_4.24_99.46.pth.tar \
#     --device "cpu"


python torchscript_export.py \
    --output "output_torchscript" \
    --with-persons \
    --checkpoint models/model_imdb_cross_person_4.24_99.46.pth.tar \
    --device "cpu"

# python test_thisomall.py \
#     --with-persons \
#     --checkpoint "models/model_utk_age_gender_4.23_97.69.pth.tar" \
#     --detector-weights "models/yolov8x_person_face.pt " \
#     --device "cpu" \
#     --use_torchscript \
#     --ckpt_torchscript models/traced_model.pt
