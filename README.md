# 3D Point Cloud Teeth Segmentation Model compression for CPU Inference

This work is done by : 
# - ABAYAD Mehdi
# - ACHAKOURI Mohamed

Our project focuses on making a model that segments the teeth (computer vision 3D), while it makes it compressed and still much accurate to run effectively on CPU (while inference) with less RAM without much drop in performace.

What we did : 
- Experience most of the technics for model compression (such as knowledge distillation, quantization and structured/non-structured pruning)
- We tried the "Fix for softmax and cross-entropy" paper by DeepMind (posted around October 2026) that tries to solve numerical error while computing, and applied it in our knowledge distillation, the difference is very neglated.
- Train a DGCNN as teacher model, and model compression technics to the teacher model (and training) pointnet/pointnet++

A short presentation about it can be found in the pdf : "Teeth3DS-S3-S4.pdf"

The results are unexpected, which led us to conclude the following : 
- Knowledge Distillation is not effective when teacher and student do not have same structure.
- Pruning drops the mAP (mean Average Precision) when the model isnt that much large (like in our case).
- We tried models with sizes that are not that big enough, therefore there is not much to compress in a way to preserve mAP.

Our next steps (currently working on):
- Training a bigger DGCNN teacher model for sake of better mAP.
- Train a smaller DGCNN student model using knowledge distillation with the bigger teacher model (Intuitively, this should help the smaller DGCNN student model to have better performance than the one without distillation).
- We will then attempt to use Quantization Aware Training (QAT) for the teacher & student models, and compare performance.
- At last, we would draw a final pareto graph that puts different models with inference speed and accuracy (mAP).
