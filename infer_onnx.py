import time
import cv2
import os
from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Img

def resize_image(image_path, scale=1.25, max_size=8000):
    """调整图片大小，确保宽度或高度不超过最大尺寸"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图片 {image_path} 不存在！")

    original_height, original_width = img.shape[:2]
    #print(f"原始宽高: {original_width}x{original_height}")

    # 计算缩放后的宽高
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 如果缩放后尺寸超过最大尺寸，则重新计算缩放比例
    if max(new_width, new_height) > max_size:
        scale = max_size / max(new_width, new_height)
        new_width = int(new_width * scale)
        new_height = int(new_height * scale)

    #print(f"调整后的宽高: {new_width}x{new_height}")
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

def process_images(model, input_folder):
    """处理输入文件夹中的所有图片，并将结果保存到输出文件夹"""
    output_folder = f'./Results/{input_folder.split("/")[-1]}_{model_name}_output'
    os.makedirs(output_folder, exist_ok=True)

    total_time = 0.0  # 累计总时间
    image_count = 0    # 处理的图片数量

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            resized_img = resize_image(image_path)

            result,rec_time = model.ocr(resized_img)
            total_time += rec_time
            image_count += 1

            #print(f"------------------------ 推理 {filename} 花费时间：{rec_time:.6f} 秒 ----------------------")

            sav2Img(resized_img, result, output_path)
            # print(f"处理完成并保存: {output_path}")

    if image_count > 0:
        average_time = total_time / image_count
        print(f"------------------------ 平均推理时间：{average_time:.6f} 秒 ----------------------")
    else:
        print("没有找到可处理的图片。")

def load_model(config):
    """根据配置加载模型"""
    return ONNXPaddleOcr(
        cls = False,
        use_angle_cls=False,
        use_gpu=True,
        det_model_dir=config["det_model_dir"],
        rec_model_dir=config["rec_model_dir"],
        cls_model_dir=config["cls_model_dir"],
        rec_char_dict_path=config["rec_char_dict_path"],
        vis_font_path=config["vis_font_path"],
        drop_score=0.1,
    )

# 模型配置
MODEL_CONFIGS = {
    "paddle": {
        "det_model_dir": "./models/ppocrv4_paddle/det.onnx",
        "rec_model_dir": "./models/ppocrv4_paddle/rec.onnx",
        "cls_model_dir": "./models/ppocrv4_paddle/cls.onnx",
        "rec_char_dict_path": "./models/ppocrv4_paddle/ppocr_keys_paddle.txt",
        "vis_font_path": "./onnxocr/fonts/simfang.ttf",
    },
    "new": {
        "det_model_dir": "./models/ppocrv4_new/det.onnx",
        "rec_model_dir": "./models/ppocrv4_new/rec.onnx",
        "cls_model_dir": "./models/ppocrv4_new/cls.onnx",
        #"rec_image_shape": "3,64,320",
        "rec_char_dict_path": "./models/ppocrv4_new/ppocr_keys_new.txt",
        "vis_font_path": "./onnxocr/fonts/simfang.ttf",
    },
    "distill":{
        "det_model_dir": "./models/ppocrv4_distill/det.onnx",
        "rec_model_dir": "./models/ppocrv4_distill/rec.onnx",
        "cls_model_dir": "./models/ppocrv4_distill/cls.onnx",
        #"rec_image_shape": "3,64,320",
        "rec_char_dict_path": "./models/ppocrv4_distill/ppocr_keys_new.txt",
        "vis_font_path": "./onnxocr/fonts/simfang.ttf",
    }
}

if __name__ == "__main__":
    input_folder = "./data/test"  # 输入文件夹路径   meviy_test_img_250122   ， test

    model_name = "distill"  # 选择模型：paddle 或 new 或 distill

    model = load_model(MODEL_CONFIGS[model_name])

    process_images(model, input_folder)

'''
    model_name = "paddle"  # 选择模型：paddle 或 new

    model = load_model(MODEL_CONFIGS[model_name])

    process_images(model, input_folder)
'''