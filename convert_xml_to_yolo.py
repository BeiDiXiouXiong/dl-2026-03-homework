import os
import xml.etree.ElementTree as ET
import random
import shutil
from tqdm import tqdm

# ===================== 配置 =====================
INPUT_DIR = "raw_data/NEU-DET"
ANNOTATIONS_DIR = os.path.join(INPUT_DIR, "ANNOTATIONS")
IMAGES_DIR = os.path.join(INPUT_DIR, "IMAGES")

OUTPUT_DIR = "dataset"

SPLIT_RATIO = (0.7, 0.2, 0.1)
RANDOM_SEED = 42

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# ===================== 初始化 =====================
random.seed(RANDOM_SEED)

LOG_FILE = "convert_log.txt"
log_fp = open(LOG_FILE, "w", encoding="utf-8")

def log(msg):
    print(msg)
    log_fp.write(msg + "\n")

# ===================== 创建目录 =====================
def make_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# ===================== bbox转换 =====================
def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    xmin, ymin, xmax, ymax = box

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(size[0], xmax)
    ymax = min(size[1], ymax)

    w = xmax - xmin
    h = ymax - ymin

    if w <= 0 or h <= 0:
        return None

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    return (
        x_center * dw,
        y_center * dh,
        w * dw,
        h * dh
    )

# ===================== 解析XML =====================
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text.strip()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()

        if cls_name not in CLASSES:
            log(f"[跳过类别] {cls_name} in {xml_path}")
            continue

        cls_id = CLASSES.index(cls_name)

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        bbox = convert_bbox((width, height), (xmin, ymin, xmax, ymax))

        if bbox is None:
            log(f"[无效bbox] {xml_path}")
            continue

        objects.append((cls_id, bbox))

    return filename, objects

# ===================== 数据划分 =====================
def split_dataset(files):
    random.shuffle(files)
    n = len(files)

    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files

# ===================== 主函数 =====================
def main():
    make_dirs()

    if not os.path.exists(ANNOTATIONS_DIR):
        raise Exception(f"找不到标注目录: {ANNOTATIONS_DIR}")

    if not os.path.exists(IMAGES_DIR):
        raise Exception(f"找不到图片目录: {IMAGES_DIR}")

    xml_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith(".xml")]
    log(f"总XML数量: {len(xml_files)}")

    train_files, val_files, test_files = split_dataset(xml_files)

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split_name, file_list in splits.items():
        log(f"\n处理 {split_name} 集: {len(file_list)}")

        for xml_file in tqdm(file_list):
            xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)

            try:
                filename, objects = parse_xml(xml_path)
            except Exception as e:
                log(f"[解析失败] {xml_file} : {e}")
                continue

            img_src = os.path.join(IMAGES_DIR, filename)

            # 🔥 自动修复 jpg/png 问题
            if not os.path.exists(img_src):
                alt = filename.replace(".jpg", ".png")
                img_src = os.path.join(IMAGES_DIR, alt)

            if not os.path.exists(img_src):
                log(f"[缺失图片] {filename}")
                continue

            img_dst = os.path.join(OUTPUT_DIR, "images", split_name, os.path.basename(img_src))
            shutil.copy(img_src, img_dst)

            label_path = os.path.join(
                OUTPUT_DIR, "labels", split_name,
                os.path.splitext(os.path.basename(img_src))[0] + ".txt"
            )

            with open(label_path, "w") as f:
                if len(objects) == 0:
                    log(f"[空标注] {filename}")

                for cls_id, bbox in objects:
                    f.write(f"{cls_id} {' '.join(map(str, bbox))}\n")

    log("\n转换完成！")

if __name__ == "__main__":
    main()