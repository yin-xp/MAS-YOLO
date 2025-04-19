As the performance requirements for printed circuit boards (PCBs) in electronic devices continue to increase, reliable defect detection during PCB manufacturing is vital. However, due to the small size, complex categories, and subtle differences of defect features, traditional detection methods are limited in accuracy and robustness. To overcome these challenges, this paper proposes MAS-YOLO, a lightweight detection algorithm for PCB defect detection based on improved YOLOv12 architecture. In the Backbone, a Median-enhanced Channel and Spatial Attention Block (MECS) expands the receptive field through median enhancement and depthwise convolution to gener-ate attention maps that effectively capture subtle defect features. In the Neck, an Adaptive Hierarchical Feature Integration Network (AHFIN) adaptively fuses mul-ti-scale features through weighted integration, enhancing feature utilization and focus on defect regions. Moreover, the original YOLOv12 loss function is replaced with the Slide Alignment Loss (SAL) to improve bounding box localization and detect complex defect types. Experimental results demonstrate that MAS-YOLO significantly improves mean Average Precision (mAP) and Frames Per Second (FPS) compared to the original YOLOv12, fulfilling real-time industrial detection requirements.


conda create -n yolov12 python=3.11
conda activate yolov12
pip install -r requirements.txt
pip install -e .

Training

from ultralytics import YOLO
model = YOLO('yolov12n.yaml')
results = model.train(
  data='coco.yaml',
  epochs=600, 
  batch=256, 
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1,2,3",
)
metrics = model.val()
results = model("path/to/image.jpg")
results[0].show()

Prediction

from ultralytics import YOLO
model = YOLO('yolov12{n/s/m/l/x}.pt')
model.predict()

Export

from ultralytics import YOLO
model = YOLO('yolov12{n/s/m/l/x}.pt')
model.export(format="engine", half=True)  # or format="onnx"




