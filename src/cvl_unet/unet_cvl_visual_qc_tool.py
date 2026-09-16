"""
unet_cvl_visual_qc_tool.py
===========================
和你的 cut_line_qc_tool1.py 是同一个思路，界面风格、翻页/点击详情/导出的交互
方式都保持一致，方便你和审稿人对照着看。区别是这个工具看的是U-Net(TDTS训练,
zero-shot)在CVL上的分割结果，而不是图论切割算法的切割线。

关键改进——"水印"而不是"色块"：
    诊断脚本(unet_cvl_diagnose.py)里那版mask可视化，是直接把mask==1/2的像素
    替换成纯红/纯绿色块，会把底下的灰度笔画细节完全盖掉，只能看轮廓形状，看不出
    笔画本身有没有被切穿。这个工具改成半透明叠加(跟你cut_line_qc_tool1.py里
    "显示切割域"用的cyan半透明叠加是同一种技术)：背景灰度笔画始终可见，只是在
    U-Net判给左/右字符的区域上，分别叠一层半透明红/绿"水印"。这样既能看出U-Net
    切到哪，又能同时看清笔画本身有没有被切穿——这正是要说服编辑"53%不是随便糊弄
    出来的数字、是真实可见的失败模式"最需要的那种图。

功能：
    - 网格缩略图 + 半透明红/绿水印叠加(红=U-Net判给左字符，绿=判给右字符)
    - "启用走廊重定位"开关：勾上后用corridor_utils把图像水平对齐到TDTS隐含中心
      x=20再喂给U-Net(margin_frac可调)，方便跟不重定位版本直接对照
      (对应论文里53.05% vs 50.38%那组消融)
    - "运行识别"开关：显示U-Net分割后，标准化左右半图分别识别出的数字，跟真实
      标签对照，右上角绿=正确/红=错误
    - 按数字组合过滤：在过滤框里输入"90"只看真实标签是9+0组合的样本，输入单个
      数字"9"则看含9的所有样本——直接对接我们之前按数字分类统计出的最差组合
      (90/96/86/...)，专门挑失败案例出图
    - 点击缩略图弹出详情：放大的水印叠加图 + 标准化后左右半图 + 各自识别结果
    - "导出当前页为PNG图片"：不依赖截屏，直接用PIL从内存里的像素数组重新拼一张
      高清网格图存盘，可以直接贴到论文/回复信里
    - "导出全部结果CSV"：跑一遍全量数据集，存每张图的预测/正确性/(如果开了
      重定位)位移量

使用方法：
    python unet_cvl_visual_qc_tool.py --input_dir ../../data/cvl_double_char --cnn_model_path ../../models/cvl_cnn_improved.pth --unet_weight_path ../../models/unet_tdts.pt 

依赖：Pillow, numpy, torch, torchvision, tkinter
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import ttk, messagebox
from torchvision import transforms

from torch.utils.data import Dataset
import glob
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from unet_left_right_recog_cvl import UNetLeftRightRecognizerCVL
from unet_recenter_utils import recenter_by_corridor
from recognition_preprocess_fix import standardize_for_recognition


# ============================================================
# 常量：跟 cut_line_qc_tool1.py 保持一致的网格/缩放参数
# ============================================================
COLS = 8
ROWS = 4
SCALE = 4
EXPORT_SCALE = 10          # 导出PNG时用更高的放大倍数，贴论文/回复信更清晰
IMG_W, IMG_H = 40, 32
CELL_PAD = 15
OVERLAY_ALPHA = 0.45       # 半透明水印透明度，跟你原工具"显示切割域"的cyan叠加一致
LEFT_COLOR = np.array([255, 60, 60], dtype=np.float32)   # 红=左字符
RIGHT_COLOR = np.array([60, 200, 60], dtype=np.float32)  # 绿=右字符

class PNGDigitDataset(Dataset):
    """从PNG文件读取数字图片，标签从文件名解析（第一个'-'前的数字）"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, '*.png'))
        if not self.image_paths:
            raise RuntimeError(f"在 {root_dir} 中没有找到PNG图片")
        self.labels = []
        for path in self.image_paths:
            fname = os.path.basename(path)
            label_str = fname.split('-')[0]
            if not label_str.isdigit():
                raise ValueError(f"无法从文件名 {fname} 解析数字标签")
            self.labels.append(int(label_str))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def build_overlay_rgb(gray_uint8, pred_mask, alpha=OVERLAY_ALPHA):
    """半透明"水印"叠加：背景灰度始终保留，只在mask==1/2的区域叠一层半透明色，
    不会像纯色块那样盖住笔画细节。"""
    base_rgb = np.stack([gray_uint8] * 3, axis=-1).astype(np.float32)
    m1 = (pred_mask == 1)
    m2 = (pred_mask == 2)
    base_rgb[m1] = base_rgb[m1] * (1 - alpha) + LEFT_COLOR * alpha
    base_rgb[m2] = base_rgb[m2] * (1 - alpha) + RIGHT_COLOR * alpha
    return np.clip(base_rgb, 0, 255).astype(np.uint8)


def label_to_pair(label):
    """标签统一按两位数字符串处理(比如label=6实际代表'06')，跟
    analyze_cvl_errors_by_class.py里digit_pair的约定保持一致。"""
    s = str(int(label)).zfill(2)
    return s[0], s[1]


class UNetVisualQCTool:
    def __init__(self, root, input_dir, cnn_model_path, unet_weight_path, use_denoise=True):
        self.root = root
        self.root.title(f"U-Net(CVL) 分割水印可视化工具 - {input_dir}")

        transform = transforms.Compose([transforms.ToTensor()])
        self.img_data = PNGDigitDataset(root_dir=input_dir, transform=transform)
        self.n_total = len(self.img_data)
        if self.n_total == 0:
            messagebox.showerror("错误", f"目录中没有找到数据：{input_dir}")
            sys.exit(1)

        print(f"加载识别器 (CNN={cnn_model_path}, U-Net={unet_weight_path}) ...")
        self.recognizer = UNetLeftRightRecognizerCVL(cnn_model_path, unet_weight_path, use_denoise=use_denoise)
        print("加载完成。")

        print("首次扫描全部样本标签（用于过滤功能），可能需要几秒...")
        self.all_labels = self._scan_all_labels()
        print(f"共 {self.n_total} 个样本。")

        self.filtered_indices = list(range(self.n_total))

        self.recentered_mode = tk.BooleanVar(value=False)
        self.margin_frac = tk.DoubleVar(value=0.15)
        self.run_recognition = tk.BooleanVar(value=True)
        self.show_overlay = tk.BooleanVar(value=True)
        self.filter_text = tk.StringVar(value="")

        self.page = 0
        self.page_size = COLS * ROWS

        self._photo_refs = []
        self._cell_boxes = []
        self._cell_results = []
        self._build_ui()
        self._render_page()

    # ---------------- 预扫描标签(用于过滤，不需要跑模型) ----------------
    def _scan_all_labels(self):
        labels_attr = getattr(self.img_data, "labels", None)
        if labels_attr is not None and len(labels_attr) == self.n_total:
            return list(labels_attr)
        # 没有现成的labels属性就老老实实过一遍数据集拿label(不跑模型，很快)
        labels = []
        for i in range(self.n_total):
            _, lbl = self.img_data[i]
            labels.append(lbl)
        return labels

    def _get_filename(self, idx):
        ds = self.img_data
        for attr in ("files", "image_paths", "paths", "png_paths", "filepaths", "samples"):
            lst = getattr(ds, attr, None)
            if lst is not None and idx < len(lst):
                item = lst[idx]
                return os.path.basename(str(item[0] if isinstance(item, (tuple, list)) else item))
        return f"sample_{idx:04d}"

    # ---------------- 过滤 ----------------
    def _apply_filter(self):
        pat = self.filter_text.get().strip()
        if not pat:
            self.filtered_indices = list(range(self.n_total))
        elif pat.isdigit() and len(pat) == 2:
            d1, d2 = pat[0], pat[1]
            self.filtered_indices = [
                i for i, lbl in enumerate(self.all_labels)
                if label_to_pair(lbl) == (d1, d2)
            ]
        elif pat.isdigit() and len(pat) == 1:
            self.filtered_indices = [
                i for i, lbl in enumerate(self.all_labels)
                if pat in label_to_pair(lbl)
            ]
        else:
            messagebox.showwarning("过滤格式错误", "请输入1位数字(含该数字的所有样本)或2位数字(精确组合，如90)")
            return
        self.page = 0
        self._render_page()

    @property
    def total_pages(self):
        return max(1, (len(self.filtered_indices) + self.page_size - 1) // self.page_size)

    # ---------------- UI ----------------
    def _build_ui(self):
        canvas_w = COLS * (IMG_W * SCALE + CELL_PAD) + CELL_PAD
        canvas_h = ROWS * (IMG_H * SCALE + CELL_PAD) + CELL_PAD

        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        self.stat_label = ttk.Label(top, text="", font=("Consolas", 11))
        self.stat_label.pack(side=tk.LEFT, padx=6)
        self.page_label = ttk.Label(top, text="", font=("Consolas", 11))
        self.page_label.pack(side=tk.LEFT, padx=20)
        ttk.Button(top, text="上一页 (←)", command=self.prev_page).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="下一页 (→)", command=self.next_page).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="导出当前页为PNG图片", command=self.export_page_png).pack(side=tk.LEFT, padx=12)
        ttk.Button(top, text="导出全部结果CSV", command=self.export_all_csv).pack(side=tk.LEFT, padx=4)

        opt_frame = ttk.Frame(self.root)
        opt_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        ttk.Checkbutton(opt_frame, text="显示分割水印", variable=self.show_overlay,
                         command=self._render_page).pack(side=tk.LEFT)
        ttk.Checkbutton(opt_frame, text="运行识别", variable=self.run_recognition,
                         command=self._render_page).pack(side=tk.LEFT, padx=(20, 0))
        ttk.Checkbutton(opt_frame, text="启用走廊重定位", variable=self.recentered_mode,
                         command=self._render_page).pack(side=tk.LEFT, padx=(20, 0))
        ttk.Label(opt_frame, text="margin_frac:").pack(side=tk.LEFT, padx=(20, 0))
        ttk.Scale(opt_frame, from_=0.05, to=0.6, variable=self.margin_frac,
                  orient=tk.HORIZONTAL, length=140,
                  command=lambda e: self._on_margin_change()).pack(side=tk.LEFT, padx=4)
        self.margin_val_label = ttk.Label(opt_frame, text="0.15", width=5)
        self.margin_val_label.pack(side=tk.LEFT)

        filter_frame = ttk.Frame(self.root)
        filter_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        ttk.Label(filter_frame, text="按数字过滤 (如 90=只看9+0组合, 9=只看含9的样本，留空=全部):").pack(side=tk.LEFT)
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_text, width=8)
        filter_entry.pack(side=tk.LEFT, padx=6)
        filter_entry.bind("<Return>", lambda e: self._apply_filter())
        ttk.Button(filter_frame, text="应用过滤", command=self._apply_filter).pack(side=tk.LEFT, padx=4)

        legend = ttk.Label(
            self.root,
            text="红色水印=U-Net判给左字符  绿色水印=U-Net判给右字符(半透明，不遮挡笔画)   "
                 "点击缩略图=查看放大图+标准化左右半图详情   右上角标签 绿=识别正确/红=识别错误",
            font=("Consolas", 9), foreground="#555555"
        )
        legend.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

        self.canvas = tk.Canvas(self.root, width=canvas_w, height=canvas_h, bg="#dddddd")
        self.canvas.pack(side=tk.TOP, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._on_cell_click)
        self.root.bind("<Left>", lambda e: self.prev_page())
        self.root.bind("<Right>", lambda e: self.next_page())

    def _on_margin_change(self):
        self.margin_val_label.config(text=f"{self.margin_frac.get():.2f}")
        if self.recentered_mode.get():
            self._render_page()

    # ---------------- 翻页 ----------------
    def prev_page(self):
        if self.page > 0:
            self.page -= 1
            self._render_page()

    def next_page(self):
        if self.page < self.total_pages - 1:
            self.page += 1
            self._render_page()

    def _current_page_indices(self):
        start = self.page * self.page_size
        end = min(start + self.page_size, len(self.filtered_indices))
        return self.filtered_indices[start:end]

    # ---------------- 单张图核心处理 ----------------
    def _process_one(self, idx):
        img, label = self.img_data[idx]
        proc_image = self.recognizer._preprocess(img)

        shift = None
        if self.recentered_mode.get():
            proc_image, shift, _ = recenter_by_corridor(
                proc_image, margin_frac=self.margin_frac.get(), target_center=20.0
            )

        left_img, right_img, pred_mask = self.recognizer._segment_with_unet(proc_image)

        result = {
            "idx": idx,
            "label": label,
            "proc_image": proc_image,
            "pred_mask": pred_mask,
            "shift": shift,
        }

        if self.run_recognition.get():
            left_std = standardize_for_recognition(left_img)
            right_std = standardize_for_recognition(right_img)
            pred_l, conf_l = self.recognizer.recognize_image(left_std)
            pred_r, conf_r = self.recognizer.recognize_image(right_std)
            combined_pred = pred_l * 10 + pred_r
            result.update({
                "left_std": left_std, "right_std": right_std,
                "pred_left": pred_l, "pred_right": pred_r,
                "conf_left": conf_l, "conf_right": conf_r,
                "pred_label": combined_pred,
                "is_correct": (combined_pred == int(label)),
            })

        return result

    # ---------------- 渲染 ----------------
    def _render_page(self):
        self.canvas.delete("all")
        self._photo_refs.clear()
        self._cell_boxes.clear()
        self._cell_results.clear()

        indices = self._current_page_indices()
        cell_w = IMG_W * SCALE
        cell_h = IMG_H * SCALE

        n_correct = 0
        n_recognized = 0

        for pos, idx in enumerate(indices):
            row = pos // COLS
            col = pos % COLS
            x0 = CELL_PAD + col * (cell_w + CELL_PAD)
            y0 = CELL_PAD + row * (cell_h + CELL_PAD)

            result = self._process_one(idx)
            proc_image = result["proc_image"]

            if self.show_overlay.get():
                rgb = build_overlay_rgb(proc_image, result["pred_mask"])
                img_big = Image.fromarray(rgb, mode="RGB").resize((cell_w, cell_h), Image.NEAREST)
            else:
                img_big = Image.fromarray(proc_image, mode="L").convert("RGB").resize((cell_w, cell_h), Image.NEAREST)

            photo = ImageTk.PhotoImage(img_big)
            self._photo_refs.append(photo)
            self.canvas.create_image(x0, y0, image=photo, anchor="nw")

            true_pair = "".join(label_to_pair(result["label"]))
            #cap = f"#{idx} realVal{true_pair}"
            cap =''
            self.canvas.create_text(x0, y0 + cell_h + 2, text=cap, anchor="nw",
                                     font=("Consolas", 7), fill="#333333")

            if "pred_label" in result:
                n_recognized += 1
                is_correct = result["is_correct"]
                if is_correct:
                    n_correct += 1
                bg_color, fg_color = ("#1a5c1a", "#7CFC00") if is_correct else ("#5c1a1a", "#ff5c5c")
                pred_str = f"{result['pred_left']}{result['pred_right']}"
                label_str = f"{pred_str}({true_pair})"
                text_w = 9 * len(label_str) + 6
                self.canvas.create_rectangle(x0 + cell_w - text_w, y0, x0 + cell_w, y0 + 16,
                                              fill=bg_color, outline="")
                self.canvas.create_text(x0 + cell_w - 3, y0 + 2, text=label_str, anchor="ne",
                                         font=("Consolas", 9, "bold"), fill=fg_color)

            self._cell_boxes.append((x0, y0, x0 + cell_w, y0 + cell_h, idx))
            self._cell_results.append(result)

        mode_str = "走廊重定位" if self.recentered_mode.get() else "zero-shot(无重定位)"
        acc_text = f"  本页准确率: {n_correct}/{n_recognized}" if n_recognized > 0 else ""
        self.stat_label.config(
            text=f"过滤后样本数: {len(self.filtered_indices)}   模式: {mode_str}{acc_text}"
        )
        self.page_label.config(text=f"第 {self.page + 1} / {self.total_pages} 页")

    # ---------------- 点击详情 ----------------
    def _on_cell_click(self, event):
        for (x0, y0, x1, y1, idx) in self._cell_boxes:
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                pos = [b[4] for b in self._cell_boxes].index(idx)
                self._show_detail_popup(self._cell_results[pos])
                return

    def _show_detail_popup(self, result):
        popup = tk.Toplevel(self.root)
        popup.title(f"详情 - 样本#{result['idx']}")
        zoom = 8

        true_pair = "".join(label_to_pair(result["label"]))
        info = f"样本#{result['idx']}    真实标签: {true_pair}"
        if result.get("shift") is not None:
            info += f"    重定位平移量: {result['shift']:+d}px"
        ttk.Label(popup, text=info, font=("Consolas", 11, "bold")).pack(pady=6)

        top_frame = ttk.Frame(popup)
        top_frame.pack(padx=20, pady=6)
        ttk.Label(top_frame, text="原图 + U-Net分割水印").pack()
        rgb = build_overlay_rgb(result["proc_image"], result["pred_mask"])
        h, w, _ = rgb.shape
        img_big = Image.fromarray(rgb, mode="RGB").resize((w * zoom, h * zoom), Image.NEAREST)
        photo0 = ImageTk.PhotoImage(img_big)
        lbl0 = ttk.Label(top_frame, image=photo0)
        lbl0.image = photo0
        lbl0.pack()

        if "left_std" in result:
            frame = ttk.Frame(popup)
            frame.pack(padx=20, pady=12)
            for col, (name, arr, pk, ck) in enumerate([
                ("左半(标准化后)", result["left_std"], "pred_left", "conf_left"),
                ("右半(标准化后)", result["right_std"], "pred_right", "conf_right"),
            ]):
                sub = ttk.Frame(frame)
                sub.grid(row=0, column=col, padx=16)
                hh, ww = arr.shape
                im = Image.fromarray(arr).resize((ww * zoom, hh * zoom), Image.NEAREST).convert("RGB")
                photo = ImageTk.PhotoImage(im)
                l = ttk.Label(sub, image=photo)
                l.image = photo
                l.pack()
                ttk.Label(sub, text=f"{name}\n预测: {result[pk]}  置信度: {result[ck]:.2%}",
                          font=("Consolas", 10)).pack(pady=4)
        else:
            ttk.Label(popup, text="(未开启\"运行识别\"，只显示分割结果)",
                      font=("Consolas", 10), foreground="#888888").pack(pady=10)

        ttk.Button(popup, text="关闭", command=popup.destroy).pack(pady=8)

    # ---------------- 导出当前页为PNG(直接贴图用) ----------------
    def export_page_png(self):
        indices = self._current_page_indices()
        if not indices:
            messagebox.showwarning("没有内容", "当前页没有样本可导出")
            return

        cell_w = IMG_W * EXPORT_SCALE
        cell_h = IMG_H * EXPORT_SCALE
        cap_h = 22
        pad = 12
        n_cols = min(COLS, len(indices))
        n_rows = (len(indices) + n_cols - 1) // n_cols

        canvas_w = n_cols * (cell_w + pad) + pad
        canvas_h = n_rows * (cell_h + cap_h + pad) + pad
        canvas_img = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
        draw = ImageDraw.Draw(canvas_img)

        for pos, idx in enumerate(indices):
            result = self._cell_results[pos] if pos < len(self._cell_results) else self._process_one(idx)
            row, col = divmod(pos, n_cols)
            x0 = pad + col * (cell_w + pad)
            y0 = pad + row * (cell_h + cap_h + pad)

            if self.show_overlay.get():
                rgb = build_overlay_rgb(result["proc_image"], result["pred_mask"])
                cell_img = Image.fromarray(rgb, mode="RGB").resize((cell_w, cell_h), Image.NEAREST)
            else:
                cell_img = Image.fromarray(result["proc_image"], mode="L").convert("RGB").resize(
                    (cell_w, cell_h), Image.NEAREST)
            canvas_img.paste(cell_img, (x0, y0))

            true_pair = "".join(label_to_pair(result["label"]))
            if "pred_label" in result:
                pred_str = f"{result['pred_left']}{result['pred_right']}"
                color = (30, 140, 30) if result["is_correct"] else (180, 30, 30)
                cap = f"pred {pred_str} / true {true_pair}"
            else:
                color = (60, 60, 60)
                cap = f"true {true_pair}"
            draw.text((x0 + 2, y0 + cell_h + 3), cap, fill=color)

        mode_tag = "recentered" if self.recentered_mode.get() else "zeroshot"
        out_path = f"unet_cvl_figure_{mode_tag}_page{self.page + 1}.png"
        canvas_img.save(out_path)
        messagebox.showinfo("导出完成", f"已保存: {os.path.abspath(out_path)}")

    # ---------------- 导出全量CSV ----------------
    def export_all_csv(self):
        was_recog = self.run_recognition.get()
        self.run_recognition.set(True)
        mode_tag = "recentered" if self.recentered_mode.get() else "zeroshot"
        out_path = f"unet_cvl_qc_results_{mode_tag}.csv"
        n = len(self.img_data)
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "true_label", "pred_label", "correct", "shift_px"])
            for i in range(n):
                result = self._process_one(i)
                true_pair = "".join(label_to_pair(result["label"]))
                pred_str = f"{result['pred_left']}{result['pred_right']}"
                writer.writerow([i, true_pair, pred_str, result["is_correct"], result.get("shift", "")])
                if i % 200 == 0:
                    print(f"导出进度 {i}/{n}")
        self.run_recognition.set(was_recog)
        messagebox.showinfo("导出完成", f"已保存: {os.path.abspath(out_path)}")


def main():
    parser = argparse.ArgumentParser(description="U-Net(CVL) 分割水印可视化工具")
    parser.add_argument("--input_dir", type=str, required=True, help="CVL PNG图片目录")
    parser.add_argument("--cnn_model_path", type=str, required=True, help="CVL识别CNN权重路径")
    parser.add_argument("--unet_weight_path", type=str, default="./unet_tdts.pt", help="TDTS训练出的U-Net权重路径")
    parser.add_argument("--no_denoise", action="store_true", help="关闭去噪(默认开启)")
    args = parser.parse_args()

    root = tk.Tk()
    app = UNetVisualQCTool(
        root, args.input_dir, args.cnn_model_path, args.unet_weight_path,
        use_denoise=not args.no_denoise
    )
    root.mainloop()


if __name__ == "__main__":
    main()