import json
import os
from collections import defaultdict
from urllib.parse import unquote

import cv2


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
        data_list = []
        if isinstance(self.json_data, list):
            for item in self.json_data:
                data_list.append(self.process_item(item))
        elif isinstance(self.json_data, dict):
            for key, value in self.json_data.items():
                data_list.append(self.process_item({key: value}))
        else:
            raise ValueError("Unsupported JSON format. Expected a list or dictionary.")

        output_dir = os.path.dirname(self.train_txt)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(data_list)

        # 打开文件并写入数据
        with open(self.train_txt, "w", encoding="utf-8") as f:
            # 将每个字典数据转换为 PaddleOCR 格式的字符串
            paddleocr_lines = [self.convert_to_paddleocr_format(data) for data in data_list]

            # 用 \t 分割所有字符串，并写入文件
            f.write("\n".join(paddleocr_lines))

            print("successfully converted paddleocr_format to paddleocr_format")

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
                    'text': item['value']['text']  # 保留 text 及其值
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

    def convert_to_paddleocr_format(self,data):
        """
        将单个字典数据转换为 PaddleOCR 数据集格式的字符串。

        :param data: 输入的字典数据。
        :return: PaddleOCR 格式的字符串。
        """
        # 获取图片路径
        image_name = data["name"]
        image_path = os.path.join(self.image_folder, image_name)  # 假设图片在 images 文件夹下

        # 遍历 GPS 数据
        annotations = []
        for gps in data.get("GPS", []):
            # 获取文本内容
            text = "".join(gps.get("text", []))

            # 获取矩形框的坐标
            x = gps["x"]
            y = gps["y"]
            width = gps["width"]
            height = gps["height"]

            # 计算多边形坐标（矩形框的四个顶点）
            polygon = [
                [x, y],  # 左上角
                [x + width, y],  # 右上角
                [x + width, y + height],  # 右下角
                [x, y + height],  # 左下角
            ]

            # 构建标注信息
            annotation = {
                "transcription": text,  # 文本内容
                "points": polygon,  # 多边形坐标
            }
            annotations.append(annotation)

        # 将标注信息转换为 JSON 字符串
        annotations_json = json.dumps(annotations, ensure_ascii=False)

        # 返回 PaddleOCR 格式的字符串
        return f"{image_path}\t{annotations_json}"

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
            print(f"图片 {image_name} 未找到！")
            return

        # 使用 OpenCV 加载图片
        image = cv2.imread(image_path)

        # 检查图片是否成功加载
        if image is None:
            print(f"图片 {image_name} 加载失败！")
            return

        # 遍历 GPS 数据，绘制矩形框
        for gps in content.get('GPS', []):
            # 获取坐标和宽高
            x = int(gps['x'])
            y = int(gps['y'])
            width = int(gps['width'])
            height = int(gps['height'])

            # 计算矩形框的左上角和右下角坐标
            left = x
            top = y
            right = x + width
            bottom = y + height

            # 绘制矩形框
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)  # 红色矩形框

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
            json_path="data/part1.json",
            image_folder="data/images/part1",
            img_output="data/images/output",
            train_txt="data/images/train.txt"
          )
    parser.write_data_to_file()


