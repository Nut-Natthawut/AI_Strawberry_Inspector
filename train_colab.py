# ============================================================
# 🍓 Strawberry 3-Class YOLO Training - Google Colab Notebook
# ============================================================
# วิธีใช้:
# 1. เปิด Google Colab (https://colab.research.google.com)
# 2. สร้าง Notebook ใหม่
# 3. Copy แต่ละ cell ด้านล่างไปวางใน Colab (แยกตาม # --- CELL --- )
# 4. เปลี่ยน Runtime เป็น GPU: Runtime > Change runtime type > T4 GPU
# 5. กด Run All (Ctrl+F9)
# 6. รอ ~15-20 นาที → ได้ไฟล์ best.onnx ใน Google Drive!

# --- CELL 1: ติดตั้ง ultralytics ---
# !pip install ultralytics -q

# --- CELL 2: Mount Google Drive ---
# from google.colab import drive
# drive.mount('/content/drive')

# --- CELL 3: ตั้งค่า path + merge classes ---
import os
import shutil
import glob
import yaml

# ===== Path ตรงกับ Drive ของคุณแล้ว =====
DRIVE_DATASET = "/content/drive/MyDrive/AI_ML/Strawberry-DS/Dataset"
OUTPUT_DIR = "/content/strawberry_3class"
# ==================================================

# Mapping 6 classes → 3 classes
# เดิม: 0=Early-Turning, 1=Green, 2=Late-Turning, 3=Red, 4=Turning, 5=White
# ใหม่: 0=Ripe, 1=Turning, 2=Unripe
CLASS_MAP = {
    0: 1,  # Early-Turning → Turning
    1: 2,  # Green → Unripe
    2: 0,  # Late-Turning → Ripe
    3: 0,  # Red → Ripe
    4: 1,  # Turning → Turning
    5: 1,  # White → Turning
}

NEW_CLASSES = ["Ripe", "Turning", "Unripe"]

# สร้างโฟลเดอร์ output
for split in ["train", "valid", "test"]:
    os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

# Copy images + แปลง labels
for split in ["train", "valid", "test"]:
    img_src = f"{DRIVE_DATASET}/{split}/images"
    lbl_src = f"{DRIVE_DATASET}/{split}/labels"

    if not os.path.exists(img_src):
        print(f"⚠️ ไม่เจอ {img_src} — ข้ามไป")
        continue

    # Copy images
    for img in glob.glob(f"{img_src}/*"):
        shutil.copy2(img, f"{OUTPUT_DIR}/{split}/images/")

    # Convert labels (เปลี่ยน class ID)
    if os.path.exists(lbl_src):
        for lbl_file in glob.glob(f"{lbl_src}/*.txt"):
            new_lines = []
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_cls = int(parts[0])
                        new_cls = CLASS_MAP.get(old_cls, old_cls)
                        new_lines.append(f"{new_cls} {' '.join(parts[1:])}")
            
            out_path = f"{OUTPUT_DIR}/{split}/labels/{os.path.basename(lbl_file)}"
            with open(out_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")

    n_img = len(glob.glob(f"{OUTPUT_DIR}/{split}/images/*"))
    n_lbl = len(glob.glob(f"{OUTPUT_DIR}/{split}/labels/*.txt"))
    print(f"✅ {split}: {n_img} images, {n_lbl} labels")

# สร้าง data.yaml
data_yaml = {
    "path": OUTPUT_DIR,
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": len(NEW_CLASSES),
    "names": NEW_CLASSES,
}

yaml_path = f"{OUTPUT_DIR}/data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f)

print(f"\n📄 data.yaml saved: {yaml_path}")
print(f"   Classes: {NEW_CLASSES}")

# --- CELL 4: Train YOLOv11 nano ---
# from ultralytics import YOLO

# model = YOLO("yolo11n.pt")  # nano = เล็กสุด เหมาะกับ browser
# results = model.train(
#     data="/content/strawberry_3class/data.yaml",
#     epochs=150,
#     imgsz=640,
#     batch=16,
#     patience=30,        # early stop ถ้าไม่ดีขึ้น 30 epoch
#     device=0,           # ใช้ GPU
#     project="/content/strawberry_train",
#     name="yolo11n_3class",
# )
# print("✅ Training เสร็จ!")

# --- CELL 5: Export เป็น ONNX ---
# best_model = YOLO("/content/strawberry_train/yolo11n_3class/weights/best.pt")
# best_model.export(format="onnx", imgsz=640)
# print("✅ Export ONNX เสร็จ!")

# --- CELL 6: Copy ไฟล์ไป Drive ---
# import shutil
# 
# onnx_src = "/content/strawberry_train/yolo11n_3class/weights/best.onnx"
# onnx_dst = "/content/drive/MyDrive/ai_ml/strawberryds/best_3class.onnx"
# shutil.copy2(onnx_src, onnx_dst)
# print(f"✅ ONNX saved to: {onnx_dst}")
# print("📥 ดาวน์โหลดจาก Google Drive แล้ววางใน public/models/ ได้เลย!")

# --- CELL 7: ดู confusion matrix ---
# from IPython.display import Image, display
# cm_path = "/content/strawberry_train/yolo11n_3class/confusion_matrix.png"
# if os.path.exists(cm_path):
#     display(Image(filename=cm_path, width=600))
