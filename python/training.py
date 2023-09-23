import tkinter.ttk as ttk
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

import os
import cv2
import dlib
import numpy as np
import json

if not os.path.isfile("shape_predictor_68_face_landmarks.dat"):
    messagebox.showerror("错误", "shape_predictor_68_face_landmarks.dat 文件不存在")
    exit(-1)

if not os.path.isfile("dlib_face_recognition_resnet_model_v1.dat"):
    messagebox.showerror("错误", "dlib_face_recognition_resnet_model_v1.dat 文件不存在")
    exit(-1)

# 导入人脸检测数据
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


def detect_face(img):
    dets = detector(img, 1)  # 使用检测算子检测人脸，返回的是所有的检测到的人脸区域
    bbs = []

    for d in dets:
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(d.left(), 0)
        bb[1] = np.maximum(d.top(), 0)
        bb[2] = np.minimum(d.right(), img.shape[1])
        bb[3] = np.minimum(d.bottom(), img.shape[0])

        # 筛选符合大小的脸 识别到的脸必须为原图的5分之一大小
        if bb[2] - bb[0] < (img.shape[0] // 5):
            continue
        if bb[3] - bb[1] < (img.shape[1] // 5):
            continue
        bbs.append(bb)

    return bbs


# 生成人脸特征数据保存到文件
# 返回 【提取结果】 和 【消息】
# 如果提取失败则【提取结果】为None
# 如果出现【多张人脸】或者【无人脸】的照片对应的id所有特征信息将会被忽略（也就是不会保存到文件中）
def make_features(path, featurePath, labelPath, result_tree_view=None):
    data = np.zeros((1, 128))  # 定义一个128维的空向量data
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    # 先检查一遍每个文件的格式是否正确
    for imagePath in imagePaths:
        fileName = os.path.split(imagePath)[1]
        fileNameSplit = fileName.split('_')
        if len(fileNameSplit) != 3:
            return None, f'文件:{fileName} 格式错误,标准格式为:序号_姓名_随机字符.xxx'

    feature_dict = {}  # 记录序号对应的特征值
    result = {}  # 记录处理结果
    # 遍历文件夹中的图片
    for imagePath in imagePaths:

        # 获取每张图片的id和姓名
        fileName = os.path.split(imagePath)[1]
        fileNameSplit = fileName.split('_')
        id = fileNameSplit[0]
        name = fileNameSplit[1]

        if '.jpg' in imagePath or '.png' in imagePath:

            # 记录人员面部特征值
            if id not in feature_dict.keys():
                feature_dict[id] = np.zeros((1, 128))

            # 记录人员处理结果
            if id not in result.keys():
                result[id] = {
                    'name': '',
                    'no_face_list': [],
                    'multiple_face_list': [],
                    'success_face': 0
                    }

            result[id]['name'] = name

            img = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), -1)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            if img.shape[0] * img.shape[1] > 400000:  # 对大图可以进行压缩
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

            # 检测人脸
            bbs = detect_face(img)

            # 如果没有找到人脸 可能图片被旋转了
            if len(bbs) == 0:
                for i in range(3):
                    trans_img = cv2.transpose(img)
                    new_img = cv2.flip(trans_img, 0)
                    bbs = detect_face(new_img)
                    if len(bbs) != 0:
                        img = new_img
                        break

            # 如果没有人脸
            if len(bbs) == 0:
                result[id]['no_face_list'].append(fileName)
                continue

            # 如果存在多张人脸
            if len(bbs) > 1:
                result[id]['multiple_face_list'].append(fileName)
                continue

            bb = bbs[0]
            rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
            shape = sp(img, rec)  # 获取landmark
            face_descriptor = facerec.compute_face_descriptor(img, shape)  # 使用resNet获取128维的人脸特征向量
            face_array = np.array(face_descriptor).reshape((1, 128))  # 转换成numpy中的数据结构

            feature_dict[id] += face_array
            result[id]['success_face'] += 1

            # 控制界面组件输出信息
            if result_tree_view != None:
                count = len(result[id]['no_face_list']) + len(result[id]['multiple_face_list']) + result[id]['success_face']
                result_tree_view.insert('', 0, values=[id, result[id]['name'], f"已处理完成{count}张"])
                result_tree_view.update()

    if result.keys() != feature_dict.keys():
        return None, "面部特征数据与标签数据不符"

    label = []  # 定义空的list存放人脸的标签
    for id in result.keys():
        if len(result[id]['no_face_list']) != 0 or len(result[id]['multiple_face_list']) != 0:
            continue

        if result[id]['success_face'] > 0:
            feature = feature_dict[id] / result[id]['success_face']
            data = np.concatenate((data, feature))  # 保存每个人的人脸特征
            label.append(result[id]['name'])  # 保存标签

    data = data[1:, :]  # 因为data的第一行是128维0向量，所以实际存储的时候从第二行开始
    np.savetxt(featurePath, data, fmt='%f')  # 保存人脸特征向量合成的矩阵到本地
    label_file = open(labelPath, 'w')
    json.dump(label, label_file)  # 使用json保存list到本地
    label_file.close()
    return result, "训练完成"


class Gui:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("录入系统")
        self.root.geometry('500x300+300+100')

        # 顶部的一些细节使用网格
        self.top_frame_view = tk.Frame(self.root, width=10)
        self.top_frame_view.pack(side=tk.TOP, anchor=tk.W, fill=tk.X, expand=True)

        # 选择文件夹文本框
        self.dir_entry_view = tk.Entry(self.top_frame_view, bd=5, width=50)
        self.dir_entry_view.grid(row=0, column=2)

        # 选择文件夹按钮
        self.dir_button_view = tk.Button(self.top_frame_view, text="选择文件夹")
        self.dir_button_view.grid(row=0, column=1)
        self.dir_button_view.bind("<ButtonRelease-1>", self.dir_button_click_even)

        # 开始训练按钮
        self.start_button_view = tk.Button(self.top_frame_view, text="开始训练", height=2)
        self.start_button_view.grid(row=0, column=3)
        self.start_button_view.bind("<ButtonRelease-1>", self.start_button_click_even)

        # 表格
        self.result_tree_view = ttk.Treeview(self.root, columns=('id', 'name', 'result'), show="headings")
        # 设置每列宽度和对齐方式

        self.result_tree_view.column('id', width=100, anchor='center')
        self.result_tree_view.column('name', width=100, anchor='center')
        self.result_tree_view.column('result', width=300, anchor='center')
        # 设置每列表头标题文本
        self.result_tree_view.heading('id', text='序号')
        self.result_tree_view.heading('name', text='姓名')
        self.result_tree_view.heading('result', text='结果(双击查看详细信息)')
        self.result_tree_view.pack(side=tk.BOTTOM, anchor=tk.S, expand=True)
        self.result_tree_view.bind("<Double-1>", self.result_tree_double_even)

        self.result_tree_view.pack()

    # 点击【选择文件夹】按钮事件
    def dir_button_click_even(self, event):
        path = filedialog.askdirectory()
        if path:
            self.dir_entry_view.delete(0, tk.END)
            self.dir_entry_view.insert(0, path)

    # 点击【开始训练】按钮事件
    def start_button_click_even(self, event):
        path = self.dir_entry_view.get()

        if not path:
            messagebox.showwarning("警告", "选择文件夹为空")
            return

        # 禁用训练按钮 防止被多次点击
        self.start_button_view["state"] = "disabled"
        self.start_button_view["text"] = "正在训练中"

        if not os.path.exists("./out"):
            os.mkdir("./out")

        # 清空原来的内容
        for item in self.result_tree_view.get_children():
            self.result_tree_view.delete(item)

        # 获取人脸特征保存到文件中
        result, msg = make_features(path, "./out/face_feature_vec.txt", "./out/label.txt", self.result_tree_view)

        # 清空原来的内容
        for item in self.result_tree_view.get_children():
            self.result_tree_view.delete(item)

        # 判断提取人脸特征是否失败
        if result == None:
            messagebox.showerror("错误", msg)

            # 恢复被禁用的训练按钮
            self.start_button_view["state"] = "normal"
            self.start_button_view["text"] = "开始训练"
            return

        # 将结果展示到界面中
        for id, item in result.items():
            if len(item['no_face_list']) != 0 or len(item['multiple_face_list']) != 0:
                msg = "出现错误:"
                if len(item['no_face_list']) != 0:
                    msg += "无人脸图片" + str(item['no_face_list']) + ' '
                if len(item['multiple_face_list']) != 0:
                    msg += "多张人脸图片" + str(item['multiple_face_list']) + " "

                msg += f"最终成功训练{item['success_face']}张"

                self.result_tree_view.insert('', 0, values=[id, item['name'], msg])
            else:
                self.result_tree_view.insert('', tk.END, values=[id, item['name'], f"训练成功,总共{item['success_face']}张"])

        # 恢复被禁用的训练按钮
        self.start_button_view["state"] = "normal"
        self.start_button_view["text"] = "开始训练"

        messagebox.showinfo("完成", "训练完成\n请将生成的【out/face_feature_vec.txt】和【out/label.txt】放置到识别系统运行路径下")

    def result_tree_double_even(self, even):
        items = self.result_tree_view.selection()
        item_text = self.result_tree_view.item(items[0], "values")
        msg = f'序号:{item_text[0]}\n姓名:{item_text[1]}\n结果:{item_text[2]}'
        messagebox.showinfo("信息", msg)

    def run(self):
        self.root.mainloop()


g = Gui()
g.run()
