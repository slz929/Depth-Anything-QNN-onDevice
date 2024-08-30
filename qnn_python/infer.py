import cv2, os
import numpy as np

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
# scalar data divide
# snpe_raw /= 255
snpe_raw = snpe_raw[..., ::-1] # bgr变rgb

print('snpe_raw after trans', snpe_raw.shape)

snpe_raw_filename = 'building.raw'
snpe_raw.tofile(snpe_raw_filename)
os.system('echo "building.raw" > input_list.txt')

cmd= 'qnn-net-run --backend /opt/qcom/aistack/qnn/2.20.0.240223/lib/x86_64-linux-clang/libQnnCpu.so  --input_list input_list.txt  --output_dir ./out/ --model x86_64-linux-clang/libdam.so '
os.system(cmd)

file_path= 'building.raw'
data = np.fromfile(file_path, dtype= np.float32)
print(data.shape)

file_path= 'out/Result_0/depth.raw'
depth = np.fromfile(file_path, dtype= np.float32)
print(data.shape, origin_w, origin_h)
depth= depth.reshape((length, length))
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
depth= depth[:new_h, :new_w]
depth = cv2.resize(depth, (origin_w, origin_h))
depth = depth.astype(np.uint8)
print('after depth ', depth.shape)
depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
print('after depth ', depth.shape)

image = cv2.addWeighted(image, 0.2, depth_color.astype(np.uint8), 0.8, 0)
cv2.imwrite('res_qnn.jpg', image)
