import json
import os
import random
import re
from collections import defaultdict
from urllib.parse import unquote

import cv2
import math
import numpy as np
from PIL import Image


class JsonParser:
    def __init__(self, json_path,image_folder,img_output,train_txt):
        """
        初始化JsonParser类，接收JSON数据。

        :param json_data: 可以是JSON字符串或Python字典。
        """
        with open(json_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        if isinstance(json_data, str):
            self.json_data = json.loads(json_data)
        else:
            self.json_data = json_data

        self.image_folder = image_folder

        self.image_output = img_output

        self.train_txt = train_txt

        self.counter = 0

        try:
            # 清空 train.txt 文件
            with open(self.train_txt, 'w', encoding='utf-8') as f:
                f.write("")
        except:
            pass

    def write_data_to_file(self):
        """
        将多个字典数据写入文件，用 \t 分割。

        :param data_list: 包含多个字典数据的列表。
        :param output_file: 输出文件路径。
        :param mode: 文件写入模式，默认为 "w"（覆盖写入），可选 "a"（追加写入）。
        """
        # 检查输出文件夹是否存在，如果不存在则创建
        """
        入口方法，遍历JSON数据并处理每一条数据。
        """

        if isinstance(self.json_data, list):
            for item in self.json_data:
                self.process_item(item)
        elif isinstance(self.json_data, dict):
            for key, value in self.json_data.items():
                self.process_item({key: value})
        else:
            raise ValueError("Unsupported JSON format. Expected a list or dictionary.")



    def process_item(self, item):
        """

        :param item: JSON中的一条数据(完整图)。

        id:item['id']   ,    item['data']['ocr']:图片名字
        """
        content = {}
        # 打印解析后的数据

        img_path = unquote(item['data']['ocr'].split('/')[-1])

        #if img_path == "1-10_5.png":
        content['name'] = img_path

        res = item['annotations'][0]['result']

        GPS_list = self.process_single_item(res)  # 处理单张图片，返回坐标
        # 输入所有坐标数据

        # 过滤矩形框
        filtered_boxes = self.filter_boxes(GPS_list)

        # 输出结果
        content['GPS'] = filtered_boxes
        # print(content)
        # self.re_plot(content)  # 重绘验证，可去掉
        self.crop_img(content)

        return content



    def process_single_item(self, res):
        """
        处理单条数据，按 id 分组并过滤 IOU 大于 0.5 且面积较小的框。
        同时保留包含 "text" 的 item，并提取坐标信息和 text 及其值。
        :param res: JSON 数据列表
        :return: 过滤后的 GPS 列表
        """
        grouped_data = defaultdict(list)
        for item in res:
            # 检查是否包含 "text" 字段
            if 'text' in item['value']:
                extracted_item = {
                    'original_width': item['original_width'],
                    'original_height': item['original_height'],
                    'x': item['value']['x'] / 100 * item['original_width'],
                    'y': item['value']['y'] / 100 * item['original_height'],
                    'width': item['value']['width'] / 100 * item['original_width'],
                    'height': item['value']['height'] / 100 * item['original_height'],
                    'rotation': item['value']['rotation'],
                    'text': item['value']['text'] # 保留 text 及其值
                }
                grouped_data[item['id']].append(extracted_item)
            '''
            else:
                extracted_item = {
                    'original_width': item['original_width'],
                    'original_height': item['original_height'],
                    'x': item['value']['x'] / 100 * item['original_width'],
                    'y': item['value']['y'] / 100 * item['original_height'],
                    'width': item['value']['width'] / 100 * item['original_width'],
                    'height': item['value']['height'] / 100 * item['original_height']
                }
            '''
            #  grouped_data[item['id']].append(extracted_item)

        GPS_list = []

        # 打印分组结果(坐标)
        for id_value, group in grouped_data.items():
            # 去重逻辑
            unique_group = []
            for item in group:
                if item not in unique_group:
                    unique_group.append(item)

            GPS_list.extend(unique_group)

        return GPS_list

    def crop_img(self, data):
        # 创建保存裁剪图片的文件夹
        Image.MAX_IMAGE_PIXELS = None
        output_folder = self.image_output
        os.makedirs(output_folder, exist_ok=True)

        # 打开图片
        image_path = os.path.join(self.image_folder, data['name'])
        try:
            image = Image.open(image_path)
        except:
            return

        # 遍历字典中的每个坐标信息
        for gps_info in data['GPS']:
            x = int(gps_info['x'])
            y = int(gps_info['y'])
            width = int(gps_info['width'])
            height = int(gps_info['height'])
            rotation = int(gps_info['rotation'])
            text = self.check_text(gps_info['text'])

            try:
                # 如果 rotation 不等于 0，则计算旋转后的矩形框
                if rotation != 0:
                    # 将角度转换为弧度
                    angle = math.radians(rotation)
                    # 计算矩形框的四个顶点坐标
                    points = [
                        (x, y),
                        (x + width, y),
                        (x + width, y + height),
                        (x, y + height)
                    ]
                    # 旋转后的顶点坐标
                    rotated_points = []
                    for px, py in points:
                        # 以 (x, y) 为旋转中心，计算旋转后的坐标
                        dx = px - x
                        dy = py - y
                        rotated_x = x + dx * math.cos(angle) - dy * math.sin(angle)
                        rotated_y = y + dx * math.sin(angle) + dy * math.cos(angle)
                        rotated_points.append((rotated_x, rotated_y))

                    # 找到旋转后的最小外接矩形
                    min_x = min(p[0] for p in rotated_points)
                    max_x = max(p[0] for p in rotated_points)
                    min_y = min(p[1] for p in rotated_points)
                    max_y = max(p[1] for p in rotated_points)

                    # 裁剪旋转后的矩形区域
                    cropped_image = image.crop((min_x, min_y, max_x, max_y))
                else:
                    # 如果 rotation 为 0，直接裁剪
                    cropped_image = image.crop((x, y, x + width, y + height))

                # 如果裁剪后的小图宽度小于高度，则逆时针旋转 90°
                # if cropped_image.width < cropped_image.height:
                cropped_image = cropped_image.rotate(rotation, expand=True)

                # 生成文件名，使用 data 中的信息来构造文件名
                output_filename = f"[{data['name']}]_{x}_{y}_[{text}].png"
                output_path = os.path.join(output_folder, output_filename)

                # 保存裁剪后的图片
                cropped_image.save(output_path)

                # 将图片路径和文本标签写入训练文本文件
                text = text.replace("_", "/")  # 替换斜杠
                with open(self.train_txt, 'a', encoding='utf-8') as f:
                    if text != "":
                        if text.isdigit() or text.isalpha():  # 如果是纯数字或纯字母
                            if random.random() >= 0.3:  # 70% 的概率丢弃
                                continue  # 跳过写入
                        f.write(f"{output_path}\t{text}\n")  # 否则写入文件

            except Exception as e:
                print(f"Error processing {text}: {e}")

    def check_text(self, text):
        # 去掉前后的全角和半角空格
        text = "".join(text)
        text = text.strip(' \u3000')  # \u3000 是全角空格

        # 移除换行符
        text = text.replace("\n", "")

        # 替换斜杠
        text = text.replace("/", "_").replace("\\", "_")

        # 替换特殊字符
        text = (text.replace("$ag$", "°").replace("$pm$", "±").replace("$min$", "′")
                .replace("$dt$","▲").replace("$fiag$","⏢ ⦿").replace("$thag$","⦿ ⏢")
                .replace("$tg$","").replace("$cb$","").replace("$dp$","↓")
                .replace("$vtc$","⊥").replace("$smt$","亖").replace("$rbt$","→")
                .replace("$flt$","▱").replace("$ln$","一").replace("$lct$","⦿")
                .replace("$cc$","cc").replace("$corn$","corn"))

        # 定义全角到半角的映射表，排除“。”和“、”
        fullwidth_to_halfwidth = {
            '，': ',', '！': '!', '？': '?', '；': ';', '：': ':',
            '“': '"', '”': '"', '‘': "'", '’': "'", '（': '(', '）': ')',
            '【': '[', '】': ']', '＜': '<', '＞': '>', '～': '~',
            '＠': '@', '＃': '#', '＄': '$', '％': '%', '＆': '&', '＊': '*',
            '＋': '+', '－': '-', '＝': '=', '＾': '^', '＿': '_', '｀': '`',
            '｜': '|', '｛': '{', '｝': '}', '　': ' ', '·': '.'
        }

        # 替换所有全角字符为半角字符，排除“。”和“、”
        def replace_all_fullwidth(text):
            result = []
            for char in text:
                if char in fullwidth_to_halfwidth and char not in {'。', '、'}:
                    result.append(fullwidth_to_halfwidth[char])  # 替换为半角
                else:
                    result.append(char)  # 保留原字符
            return ''.join(result)

        text = replace_all_fullwidth(text)

        # 将类似 ① 的符号转换为数字
        circle_numbers = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
                          '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10'}
        for circle, num in circle_numbers.items():
            text = text.replace(circle, num)

        # 去掉数字与数字之间的空格
        def remove_space_between_numbers(match):
            return match.group(1).replace(' ', '')

        text = re.sub(r'(\d[\d ]*\d)', remove_space_between_numbers, text)

        return text


    def re_plot(self, content):
        """
        根据 content 中的信息绘制矩形框并保存结果图片。

        :param content: 包含图片名和 GPS 数据的字典。
        """
        # 检查 content 是否为字典
        if not isinstance(content, dict):
            raise TypeError("content 必须是字典类型")

        # 检查 content 是否包含 name 字段
        if 'name' not in content:
            raise ValueError("content 必须包含 'name' 字段")

        # 获取图片名
        image_name = content['name']

        # 检查 image_name 是否为 None 或空字符串
        if image_name is None or not isinstance(image_name, str) or not image_name.strip():
            raise ValueError("image_name 不能为 None 或空字符串")

        # 构建图片路径
        image_path = os.path.join(self.image_folder, image_name)

        # 检查图片是否存在
        if not os.path.exists(image_path):
            # print(f"图片 {image_name} 未找到！")
            return

        # 使用 OpenCV 加载图片
        image = cv2.imread(image_path)

        # 检查图片是否成功加载
        if image is None:
            # print(f"图片 {image_name} 加载失败！")
            return

        # 遍历 GPS 数据，绘制矩形框
        for gps in content.get('GPS', []):
            # 获取坐标、宽高和旋转角度
            x = int(gps['x'])
            y = int(gps['y'])
            width = int(gps['width'])
            height = int(gps['height'])
            rotation = int(gps.get('rotation', 0))  # 如果没有 rotation 字段，默认为 0

            # 计算矩形框的四个顶点坐标
            points = [
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height)
            ]

            # 如果 rotation 不等于 0，则旋转矩形框
            if rotation != 0:
                # 将角度转换为弧度
                angle = math.radians(rotation)
                # 旋转后的顶点坐标
                rotated_points = []
                for px, py in points:
                    # 以 (x, y) 为旋转中心，计算旋转后的坐标
                    dx = px - x
                    dy = py - y
                    rotated_x = x + dx * math.cos(angle) - dy * math.sin(angle)
                    rotated_y = y + dx * math.sin(angle) + dy * math.cos(angle)
                    rotated_points.append((int(rotated_x), int(rotated_y)))
                points = rotated_points

            # 将顶点坐标转换为 NumPy 数组
            points = np.array(points, dtype=np.int32)

            # 绘制旋转后的矩形框
            cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

        # 创建 output 文件夹在 imgpath 中
        imgpath = os.path.dirname(self.image_folder)  # 获取 imgpath（假设是 image_folder 的父目录）
        output_folder = os.path.join(imgpath, "output")  # 在 imgpath 中创建 output 文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 保存结果
        output_path = os.path.join(output_folder, f"annotated_{image_name}")
        cv2.imwrite(output_path, image)
        print(f"结果已保存到: {output_path}")

    def filter_boxes(self,boxes, iou_threshold=0.5):
        # 过滤掉 IoU 大于阈值的较小矩形框，以及被完全包围的小图片
        to_remove = set()
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j or i in to_remove or j in to_remove:
                    continue
                # 检查是否完全包围
                if self.is_inside(boxes[j], boxes[i]):
                    to_remove.add(j)
                elif self.is_inside(boxes[i], boxes[j]):
                    to_remove.add(i)
                else:
                    # 检查 IoU
                    iou = self.calculate_iou(boxes[i], boxes[j])
                    if iou > iou_threshold:
                        # 保留面积较大的矩形框
                        area_i = boxes[i]['width'] * boxes[i]['height']
                        area_j = boxes[j]['width'] * boxes[j]['height']
                        if area_i > area_j:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)

        # 保留未被标记为移除的矩形框
        filtered_boxes = [box for idx, box in enumerate(boxes) if idx not in to_remove]
        return filtered_boxes

    def calculate_iou(self,box1, box2):
        # 计算两个矩形框的 IoU
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
        y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])

        # 计算交集面积
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # 计算并集面积
        box1_area = box1['width'] * box1['height']
        box2_area = box2['width'] * box2['height']
        union_area = box1_area + box2_area - inter_area

        # 计算 IoU
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    def is_inside(self,inner_box, outer_box):
        # 判断 inner_box 是否完全在 outer_box 内部
        return (inner_box['x'] >= outer_box['x'] and
                inner_box['y'] >= outer_box['y'] and
                inner_box['x'] + inner_box['width'] <= outer_box['x'] + outer_box['width'] and
                inner_box['y'] + inner_box['height'] <= outer_box['y'] + outer_box['height'])


# 示例用法
if __name__ == "__main__":
    # 入口

    parser = JsonParser(
            json_path="data/part1.json",    # data/part1.json
            image_folder="data/part1",  #  data/images/part1    data/img
            img_output="train_data/rec/val", #  train_data/rec/val
            train_txt="train_data/rec/rec_gt_val.txt" # train_data/rec/rec_gt_val.txt
          )
    parser.write_data_to_file()
    print("done")

    #'''
    parser = JsonParser(
            json_path="data/part2.json",
            image_folder="data/part2",
            img_output="train_data/rec/train",
            train_txt="train_data/rec/rec_gt_train.txt"
          )
    parser.write_data_to_file()
    #'''
    print("done")



