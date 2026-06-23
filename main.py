# -*- coding: utf-8 -*-
"""
人脸识别课程设计系统
=====================
功能：人脸检测、人脸录入、模型训练、人脸识别

学号：2023337621129
姓名：朱朝阳

技术栈：
  - OpenCV 4.13 (Haar Cascade人脸检测 + LBPH人脸识别)
  - tkinter (GUI界面)
  - Pillow (图像处理)
  - NumPy (数值计算)

使用方法：
  1. 运行程序 -> 打开摄像头
  2. 【录入人脸】-> 输入姓名 -> 正对摄像头采集人脸
  3. 【训练模型】-> 训练LBPH识别器
  4. 【人脸识别】-> 实时识别人脸并显示姓名
"""

import cv2
import os
import json
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog


class FaceRecognitionApp:
    """人脸识别课程设计主程序"""

    # ============================================================
    #  初始化
    # ============================================================
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别课程设计系统 - 朱朝阳 2023337621129")
        self.root.geometry("900x650")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # ---------- 目录（使用无中文路径，避免OpenCV编码问题）----------
        self.BASE_DIR = r"E:\desk\face_data"
        self.FACE_DIR = os.path.join(self.BASE_DIR, "known_faces")
        self.TRAINER_DIR = os.path.join(self.BASE_DIR, "trainer")

        for d in [self.FACE_DIR, self.TRAINER_DIR]:
            os.makedirs(d, exist_ok=True)

        # ---------- 加载 Haar Cascade 人脸检测器 ----------
        cascade_path = os.path.join(self.BASE_DIR,
                                    "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            # 后备：直接从 OpenCV 包中读取二进制数据
            try:
                import importlib.resources as res
                data = (res.files("cv2.data")
                        .joinpath("haarcascade_frontalface_default.xml")
                        .read_bytes())
                self.face_cascade = cv2.CascadeClassifier()
                if not self.face_cascade.read(cv2.imdecode(
                        np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)):
                    raise RuntimeError("Cascade load failed")
            except Exception:
                messagebox.showerror("错误",
                                     "加载 Haar Cascade 人脸检测器失败！\n"
                                     "请检查 D:\\face_recognition\\data\\ 下是否存在 "
                                     "haarcascade_frontalface_default.xml")
                root.destroy()
                return

        # ---------- LBPH 人脸识别器 ----------
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        # ---------- 标签映射文件 ----------
        self.LABELS_FILE = os.path.join(self.TRAINER_DIR, "labels.json")
        self.MODEL_FILE = os.path.join(self.TRAINER_DIR, "trainer.yml")

        # ---------- 状态变量 ----------
        self.camera = None
        self.is_running = False          # 摄像头是否开启
        self.current_mode = "detection"  # "detection" | "register" | "recognition"
        self.register_name = ""           # 录入时的姓名
        self.register_count = 0           # 已采集的样本数
        self.register_target = 60         # 目标采集数
        self.recog_confidence = 0         # 识别置信度
        self.recog_name = "未知"           # 识别结果

        # ---------- 加载已有标签 ----------
        self.labels = {}      # { label_id: name }
        self.name_to_id = {}  # { name: label_id }
        self.next_label = 0
        self.load_labels()

        # ---------- 若有已训练的模型则加载 ----------
        if os.path.exists(self.MODEL_FILE):
            try:
                self.recognizer.read(self.MODEL_FILE)
                self.model_trained = True
            except Exception:
                self.model_trained = False
        else:
            self.model_trained = False

        # ---------- 构建界面 ----------
        self.init_ui()

        # ---------- 定时刷新 ----------
        self.update_frame()

    # ============================================================
    #  界面构建
    # ============================================================
    def init_ui(self):
        """创建GUI布局"""
        # ---- 主容器 ----
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ===== 左侧：视频画面 =====
        left_frame = ttk.LabelFrame(main_frame, text="视频画面", padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(left_frame, relief=tk.SUNKEN,
                                     background="#000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ===== 右侧：控制面板 =====
        right_frame = ttk.LabelFrame(main_frame, text="控制面板", width=220,
                                     padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)

        # ---- 摄像头控制 ----
        ttk.Label(right_frame, text="摄像头控制",
                  font=("微软雅黑", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.btn_camera = ttk.Button(right_frame, text="打开摄像头",
                                     command=self.toggle_camera)
        self.btn_camera.pack(fill=tk.X, pady=2)

        # ---- 功能模式 ----
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(right_frame, text="功能模式",
                  font=("微软雅黑", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.btn_detect = ttk.Button(right_frame, text="        1. 人脸检测      ",
                                     command=self.set_mode_detection)
        self.btn_detect.pack(fill=tk.X, pady=2)

        self.btn_register = ttk.Button(right_frame, text="        2. 录入人脸      ",
                                       command=self.start_register)
        self.btn_register.pack(fill=tk.X, pady=2)

        self.btn_train = ttk.Button(right_frame, text="        3. 训练模型      ",
                                    command=self.train_model)
        self.btn_train.pack(fill=tk.X, pady=2)

        self.btn_recognize = ttk.Button(right_frame, text="        4. 人脸识别      ",
                                        command=self.set_mode_recognition)
        self.btn_recognize.pack(fill=tk.X, pady=2)

        # ---- 数据管理 ----
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(right_frame, text="数据管理",
                  font=("微软雅黑", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.btn_list = ttk.Button(right_frame, text="查看已录入人脸",
                                   command=self.list_faces)
        self.btn_list.pack(fill=tk.X, pady=2)

        self.btn_clear = ttk.Button(right_frame, text="清空所有数据",
                                    command=self.clear_all_data)
        self.btn_clear.pack(fill=tk.X, pady=2)

        # ---- 退出 ----
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        self.btn_exit = ttk.Button(right_frame, text="退出程序",
                                   command=self.on_close)
        self.btn_exit.pack(fill=tk.X, pady=2)

        # ===== 底部状态栏 =====
        status_frame = ttk.Frame(self.root, padding=(10, 0, 10, 5))
        status_frame.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="就绪 - 请打开摄像头开始使用")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
        status_bar.pack(fill=tk.X)

        # ---- 模式指示灯 ----
        self.mode_var = tk.StringVar(value="模式: 检测")
        mode_label = ttk.Label(status_frame, textvariable=self.mode_var,
                               foreground="#2196F3", font=("微软雅黑", 9, "bold"))
        mode_label.pack(side=tk.RIGHT, padx=(10, 0))

    # ============================================================
    #  摄像头管理
    # ============================================================
    def toggle_camera(self):
        """打开/关闭摄像头"""
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        """打开摄像头"""
        try:
            # CAP_DSHOW 可加速 Windows 下摄像头启动
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("错误", "无法打开摄像头！\n请检查摄像头是否被占用。")
                return
            self.is_running = True
            self.btn_camera.config(text="关闭摄像头")
            self.update_status("摄像头已打开")
        except Exception as e:
            messagebox.showerror("摄像头错误", str(e))

    def stop_camera(self):
        """关闭摄像头"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        self.btn_camera.config(text="打开摄像头")
        self.current_mode = "detection"
        self.mode_var.set("模式: 检测")
        self.update_status("摄像头已关闭")

    def update_frame(self):
        """定时刷新视频帧（由 root.after 驱动）"""
        if self.is_running and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                # 镜像翻转（更像照镜子）
                frame = cv2.flip(frame, 1)
                # 根据当前模式处理
                if self.current_mode == "detection":
                    frame = self.process_detection(frame)
                elif self.current_mode == "recognition":
                    frame = self.process_recognition(frame)
                elif self.current_mode == "register":
                    frame = self.process_register(frame)

                # 转成 tkinter 可显示的格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((560, 440), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

        # 继续定时调度（约 33fps）
        self.root.after(30, self.update_frame)

    # ============================================================
    #  模式切换
    # ============================================================
    def set_mode_detection(self):
        """切换到人脸检测模式"""
        if not self.is_running:
            messagebox.showinfo("提示", "请先打开摄像头")
            return
        self.current_mode = "detection"
        self.register_name = ""
        self.mode_var.set("模式: 人脸检测")
        self.update_status("已切换至 [人脸检测] 模式")

    def set_mode_recognition(self):
        """切换到人脸识别模式"""
        if not self.is_running:
            messagebox.showinfo("提示", "请先打开摄像头")
            return
        if not self.model_trained:
            reply = messagebox.askyesno("提示",
                                        "尚未训练模型！\n是否先录入人脸并训练？")
            if reply:
                self.start_register()
            return
        self.current_mode = "recognition"
        self.mode_var.set("模式: 人脸识别")
        self.update_status("已切换至 [人脸识别] 模式 - 正在识别...")

    # ============================================================
    #  人脸检测（Haar Cascade）
    # ============================================================
    def process_detection(self, frame):
        """人脸检测处理"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        for (x, y, w, h) in faces:
            # 画绿色矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 标注 "Face"
            cv2.putText(frame, "Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示检测结果
        info = f"检测到 {len(faces)} 张人脸"
        self.update_status(info)
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    # ============================================================
    #  人脸录入（注册）
    # ============================================================
    def start_register(self):
        """开始录入人脸"""
        if not self.is_running:
            messagebox.showinfo("提示", "请先打开摄像头")
            return

        # 弹窗输入姓名
        name = simpledialog.askstring("录入人脸", "请输入姓名：",
                                      parent=self.root)
        if not name or name.strip() == "":
            return
        name = name.strip()

        # 创建人员文件夹
        person_dir = os.path.join(self.FACE_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        self.register_name = name
        self.register_count = 0
        self.current_mode = "register"
        self.mode_var.set(f"模式: 录入 ({name})")
        self.update_status(f"正在为 [{name}] 采集人脸样本... 请正对摄像头")

    def process_register(self, frame):
        """人脸录入处理"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            # 画蓝色框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 每5帧保存一张（避免大量相似图片）
            if self.register_count < self.register_target:
                if self.register_count % 5 == 0:
                    face_roi = gray[y:y + h, x:x + w]
                    face_resized = cv2.resize(face_roi, (200, 200))
                    filename = os.path.join(
                        self.FACE_DIR, self.register_name,
                        f"{self.register_name}_{self.register_count:03d}.jpg"
                    )
                    cv2.imwrite(filename, face_resized)

                self.register_count += 1

        # 显示采集进度
        progress = min(self.register_count, self.register_target)
        bar_len = 20
        filled = int(bar_len * progress / self.register_target)
        bar = "#" * filled + "-" * (bar_len - filled)
        info = f"采集进度: [{bar}] {progress}/{self.register_target}"
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"姓名: {self.register_name}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        self.update_status(info)

        # 采集完成
        if self.register_count >= self.register_target:
            self.update_status(
                f"[{self.register_name}] 人脸采集完成！"
                f"共 {self.register_target} 张样本")
            messagebox.showinfo("采集完成",
                                f"[{self.register_name}] 人脸采集完成！\n"
                                f"共采集 {self.register_target} 张样本。\n"
                                "请点击 [训练模型] 完成训练。")
            self.current_mode = "detection"
            self.mode_var.set("模式: 检测")

        return frame

    # ============================================================
    #  模型训练（LBPH）
    # ============================================================
    def train_model(self):
        """训练LBPH人脸识别模型"""
        # 检查是否有训练数据
        persons = [d for d in os.listdir(self.FACE_DIR)
                   if os.path.isdir(os.path.join(self.FACE_DIR, d))]
        if not persons:
            messagebox.showwarning("提示",
                                   "没有找到录入的人脸数据！\n请先点击 [录入人脸]。")
            return

        self.update_status("正在训练模型，请稍候...")
        self.root.update()

        faces = []
        labels = []
        label_id = 0
        label_map = {}

        for person_name in persons:
            person_dir = os.path.join(self.FACE_DIR, person_name)
            for img_name in os.listdir(person_dir):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(person_dir, img_name)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    faces.append(img)
                    labels.append(label_id)
                except Exception:
                    continue
            label_map[label_id] = person_name
            label_id += 1

        if len(faces) == 0:
            messagebox.showerror("错误", "没有有效的训练图像！")
            return

        # 训练 LBPH 识别器
        # LBPH（Local Binary Patterns Histograms）
        # 原理：将人脸分为小区域 -> 计算每个区域的LBP特征 -> 构建直方图 -> 对比直方图相似度
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(faces, np.array(labels))

        # 保存模型
        self.recognizer.write(self.MODEL_FILE)

        # 保存标签映射
        self.labels = label_map
        self.name_to_id = {v: k for k, v in label_map.items()}
        self.next_label = label_id
        self.save_labels()

        self.model_trained = True
        msg = (f"训练完成！共 {len(faces)} 张图片，"
               f"{len(persons)} 人:\n" + ", ".join(persons))
        messagebox.showinfo("训练完成", msg)
        self.update_status(msg)

    # ============================================================
    #  人脸识别（LBPH预测）
    # ============================================================
    def process_recognition(self, frame):
        """人脸识别处理"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (200, 200))

            # LBPH 预测
            # predict() 返回 (label_id, confidence)
            # confidence 越小表示越匹配（距离越小）
            try:
                label_id, confidence = self.recognizer.predict(face_resized)
            except Exception:
                continue

            # 根据置信度区分识别结果
            name = self.labels.get(label_id, "未知")
            if confidence < 60:
                color = (0, 255, 0)       # 绿色 - 匹配良好
                label_text = f"{name}"
            elif confidence < 100:
                color = (0, 255, 255)     # 黄色 - 勉强匹配
                label_text = f"{name}?"
            else:
                color = (0, 0, 255)       # 红色 - 未知
                label_text = "未知"

            # 绘制矩形框和姓名
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"conf:{confidence:.0f}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            self.recog_name = name
            self.recog_confidence = confidence

        info = f"识别到 {len(faces)} 张人脸"
        self.update_status(info)
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    # ============================================================
    #  标签管理
    # ============================================================
    def load_labels(self):
        """从 JSON 加载标签映射"""
        if os.path.exists(self.LABELS_FILE):
            try:
                with open(self.LABELS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.labels = {int(k): v
                               for k, v in data.get("labels", {}).items()}
                self.next_label = data.get("next_label", 0)
                self.name_to_id = {v: k for k, v in self.labels.items()}
            except Exception:
                self.labels = {}
                self.name_to_id = {}
                self.next_label = 0

    def save_labels(self):
        """保存标签映射到 JSON"""
        data = {
            "labels": {str(k): v for k, v in self.labels.items()},
            "next_label": self.next_label
        }
        with open(self.LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ============================================================
    #  辅助功能
    # ============================================================
    def list_faces(self):
        """查看已录入的人脸数据"""
        persons = [d for d in os.listdir(self.FACE_DIR)
                   if os.path.isdir(os.path.join(self.FACE_DIR, d))]
        if not persons:
            messagebox.showinfo("已录入人脸", "暂无已录入的人脸数据")
            return

        info = "已录入的人员:\n"
        for p in persons:
            count = len([f for f in os.listdir(os.path.join(self.FACE_DIR, p))
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            info += f"  - {p} ({count} 张样本)\n"

        if self.model_trained:
            info += f"\n模型已训练，可识别 {len(self.labels)} 人"
        else:
            info += "\n模型尚未训练"

        messagebox.showinfo("已录入人脸", info)

    def clear_all_data(self):
        """清空所有人脸数据和模型"""
        reply = messagebox.askyesno("确认清空",
                                    "确定要清空所有人脸数据和训练模型吗？\n"
                                    "此操作不可撤销！")
        if not reply:
            return

        # 删除人脸图片
        for d in os.listdir(self.FACE_DIR):
            d_path = os.path.join(self.FACE_DIR, d)
            if os.path.isdir(d_path):
                for f in os.listdir(d_path):
                    os.remove(os.path.join(d_path, f))
                os.rmdir(d_path)

        # 删除模型文件
        for f in [self.MODEL_FILE, self.LABELS_FILE]:
            if os.path.exists(f):
                os.remove(f)

        self.labels = {}
        self.name_to_id = {}
        self.next_label = 0
        self.model_trained = False
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        self.update_status("所有数据已清空")
        messagebox.showinfo("完成", "所有数据已清空")

    def update_status(self, msg):
        """更新状态栏"""
        self.status_var.set(msg)

    # ============================================================
    #  退出清理
    # ============================================================
    def on_close(self):
        """程序退出时的清理"""
        if self.is_running:
            self.stop_camera()
        self.root.destroy()


# ================================================================
#  程序入口
# ================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)

    # 设置窗口居中
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    root.mainloop()
