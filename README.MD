# Depth Anything - python infer by QNN and TFLite

## Model
QNN model and TFLite model are converted from small scale Depth-Anything ONNX model depth_anything_vits14.onnx.

## Infer by QNN 
Prapare qnn-net-run env as in Qualcomm doc. This part is tested under ubuntu20.4, python==3.8, qnn==2.20.0.240223.

Run:
```bash
cd qnn_python
python infer.py
```

## Infer by TFLite
Prapare tensorflow, onnx2tf pip package. This part is tested under python==3.10.14, onnx==1.16.1, tensorflow==2.17.0 and onnx2tf==1.25.8. ref:
```bash
pip install -U onnx==1.16.1 && pip install -U nvidia-pyindex && pip install -U onnx-graphsurgeon && pip install -U onnxruntime==1.18.1 && pip install -U onnxsim==0.4.33 && pip install -U simple_onnx_processing_tools && pip install -U sne4onnx>=1.0.13 && pip install -U sng4onnx>=1.0.4 && pip install -U tensorflow==2.17.0 && pip install -U protobuf==3.20.3 && pip install -U onnx2tf && pip install -U h5py==3.11.0 && pip install -U psutil==5.9.5 && pip install -U ml_dtypes==0.3.2 && pip install -U tf-keras~=2.16 && pip install flatbuffers>=23.5.26
```

Run:
```bash
cd tflite
python infer.py
```

## Result
![input](image.jpg)
![depth result - QNN](qnn_python/res_qnn.jpg)
![depth result - TFLite](tflite/res_tflite.jpg)


## Reference

https://github.com/fabio-sim/Depth-Anything-ONNX 

https://github.com/LiheYoung/Depth-Anything 

https://github.com/PINTO0309/onnx2tf
