import tensorflow as tf
import cv2
import numpy as np

# 调用tflite
mp= './saved_model/depth_anything_vits14_float32.tflite'
interpreter = tf.lite.Interpreter(model_path= mp)
interpreter.allocate_tensors()

img= '../image.jpg'
image = cv2.imread(img)
origin_h, origin_w, _ = image.shape
print('ori ', origin_h, origin_w)
length = 518
if origin_h > origin_w:
    new_w = round(origin_w * float(length) / origin_h)
    new_h = length
else:
    new_h = round(origin_h * float(length) / origin_w)
    new_w = length
scale_w = new_w / origin_w
sclae_h = new_h / origin_h
input_var = cv2.resize(image, (new_w, new_h))
mean_raw= [123.675, 116.28, 103.53]
std= [1/58.395, 1/57.12, 1/57.375]
snpe_raw = input_var - mean_raw
snpe_raw= snpe_raw* std
snpe_raw = snpe_raw.astype(np.float32)
print('snpe_raw after norm', snpe_raw.shape)
# 计算需要添加的填充
top, left= 0,0
bottom = length - new_h  # 计算底部填充行数
right = length - new_w  # 右侧填充列数
snpe_raw = cv2.copyMakeBorder(snpe_raw, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
print('snpe_raw after pad', snpe_raw.shape)
input_image = snpe_raw[..., ::-1] # bgr变rgb
input_image = np.expand_dims(input_image, axis=0)
input_image = tf.cast(input_image, dtype=tf.float32)

# Inference
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
interpreter.invoke()

output_details = interpreter.get_output_details()
print('input_details ', input_details)
print('output_details ', output_details)

depth_map = interpreter.get_tensor(output_details[0]['index'])

depth= depth_map.reshape((length, length))
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth= depth[:new_h, :new_w]
depth = cv2.resize(depth, (origin_w, origin_h))
depth = depth.astype(np.uint8)
print('after depth ', depth.shape)
depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
print('after depth ', depth.shape)

image = cv2.addWeighted(image, 0.2, depth_color.astype(np.uint8), 0.8, 0)
cv2.imwrite('res_tflite.jpg', image)




# 按照知乎， 配置onnx-tf-tflite的环境，tensorflow的环境非常麻烦，各种不好用
# pip install tensorflow
# pip install onnx_tf 
# pip install tensorflow
# pip install tensorflow-probability
# pip install tf_keras
# onnx-->tf
if False:
    onnxpath= '/home/local/sota/depthanything/Depth-Anything-ONNX/weights/depth_anything_vits14.onnx'
    onnx_model = onnx.load(onnxpath)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('dam.tf')

    # tf-->tflite
    converter = tf.lite.TFLiteConverter.from_saved_model('dam.tf')
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
    tflite_model = converter.convert()
    with open('dam.tflite', 'wb') as f:
        f.write(tflite_model)
# /home/local/miniconda3/envs/dinov2/lib/python3.9/site-packages/tensorflow_addons/utils/ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.13.0 and strictly below 2.16.0 (nightly versions are not supported). 


# 使用 onnx2tf库 取代 onnx-tf-tflite，取代tinynn
# conda create -n py310 python=3.10
# 按照官方安装方法 https://github.com/PINTO0309/onnx2tf
# pip install -U onnx==1.16.1 && pip install -U nvidia-pyindex && pip install -U onnx-graphsurgeon && pip install -U onnxruntime==1.18.1 && pip install -U onnxsim==0.4.33 && pip install -U simple_onnx_processing_tools && pip install -U sne4onnx>=1.0.13 && pip install -U sng4onnx>=1.0.4 && pip install -U tensorflow==2.17.0 && pip install -U protobuf==3.20.3 && pip install -U onnx2tf && pip install -U h5py==3.11.0 && pip install -U psutil==5.9.5 && pip install -U ml_dtypes==0.3.2 && pip install -U tf-keras~=2.16 && pip install flatbuffers>=23.5.26

# 直接执行失败，固定了onnx的输入后成功
# onnx2tf -i depth_anything_vits14.onnx -ois image:1,3,518,518
# 代码如上，已经跑通