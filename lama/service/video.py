from pathlib import Path
import os
from natsort import natsorted
from PIL import Image
import numpy as np
import cv2
import torch
import easyocr
from omegaconf import OmegaConf
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_img_to_modulo
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import subprocess

# --- Load model LaMa ---
config_path = 'models/big-lama/config.yaml'
checkpoint_path = 'models/big-lama/models/best.ckpt'

config = OmegaConf.load(config_path)
config.training_model.predict_only = True
config.visualizer.kind = 'noop'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_checkpoint(config, checkpoint_path, strict=False, map_location=device)
model.to(device).eval()

# --- EasyOCR ---
# reader = easyocr.Reader(['ch_sim', 'en'], gpu=torch.cuda.is_available())

def is_dir_not_empty(path):
    return os.path.isdir(path) and any(os.scandir(path))

def split_video_to_frames(input_file) -> None:
    print('process split video to frames')
    output_dir =  Path('frames')
    output_dir.mkdir(parents=True, exist_ok=True)

    # cap = cv2.VideoCapture(str(input_file))
    # frame_num = 0

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     cv2.imwrite(f"{output_dir}/frame_{frame_num:05d}.png", frame)
    #     frame_num += 1

    # cap.release()
    # Tách frame ra thành JPEG chất lượng tốt
    command = [
        'ffmpeg',
        '-i', str(input_file),
        '-qscale:v', '2',
        f'{output_dir}/frame_%05d.jpg'
    ]
    subprocess.run(command, check=True)

# def remove_video_caption(input_file): 
#     cap = cv2.VideoCapture(str(input_file))

#     # Lấy FPS (frames per second)
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     frames_dir = Path('frames')
#     if not is_dir_not_empty(frames_dir):
#         split_video_to_frames(input_file)

#     frames_dir_output = Path('frames_editted')
#     frames_dir_output.mkdir(parents=True, exist_ok=True)

#     # task_list = []
#     for file_name in os.listdir(frames_dir):
#         if file_name.endswith('.png') or file_name.endswith('.jpg'):
#             image = os.path.join(frames_dir, file_name)
#             if not os.path.isfile(image):
#                 continue
            
#             image_output_path = os.path.join(frames_dir_output, file_name)

#             if not os.path.exists(image_output_path):
#                 _process_remove_caption_on_frame(image, image_output_path)

#     return frames_to_video(
#         frame_dir=frames_dir_output,
#         fps=fps,
#     )

def remove_video_caption(input_file): 
    cap = cv2.VideoCapture(str(input_file))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames_dir = Path('frames')
    if not is_dir_not_empty(frames_dir):
        split_video_to_frames(input_file)

    frames_dir_output = Path('frames_editted')
    frames_dir_output.mkdir(parents=True, exist_ok=True)

    # Tạo danh sách các cặp (input_image, output_image)
    tasks = []
    for file_name in os.listdir(frames_dir):
        if file_name.lower().endswith(('.png', '.jpg')):
            input_path = frames_dir / file_name
            output_path = frames_dir_output / file_name
            if not output_path.exists():
                tasks.append((str(input_path), str(output_path)))

    # Chạy xử lý song song
    max_workers = min(8, os.cpu_count()) # type: ignore
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(_process_remove_caption_on_frame, tasks)

    return frames_to_video(frame_dir=frames_dir_output, fps=fps)

def _process_remove_caption_on_frame(image_path_output_tuple):
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=torch.cuda.is_available())

    input_image, output_image = image_path_output_tuple
    print('process image:', str(input_image))

    image_pil = Image.open(input_image).convert("RGB")

    # Tạo mask bằng hàm bạn viết
    mask_pil = generate_mask_from_text(reader, image_pil)

    # Inpaint bằng OpenCV
    image_np = np.array(image_pil)
    mask_np = np.array(mask_pil)
    inpainted_np = cv2.inpaint(image_np, mask_np, inpaintRadius=5, flags=cv2.INPAINT_NS)

    # Chuyển sang PIL để dùng với LaMa
    inpainted_pil = Image.fromarray(inpainted_np)

    final_image = run_lama(reader, inpainted_pil, mask_pil)
    final_image.save(output_image, format="PNG")

    # image_np = cv2.imread(input_image)
    # if image_np is None:
    #     print(f"⚠️ Không đọc được ảnh: {input_image}")
    #     return

    # results = reader.readtext(image_np)
    # mask_dilated = np.zeros(image_np.shape[:2], dtype=np.uint8)

    # for (bbox, text, prob) in results:
    #     pts = np.array(bbox, dtype=np.int32)
    #     cv2.fillPoly(mask_dilated, [pts], 255)

    # inpainted_np = cv2.inpaint(image_np, mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_NS)
    # inpainted_pil = Image.fromarray(inpainted_np)
    # mask_pil = Image.fromarray(mask_dilated)
    # final_image = run_lama(reader, inpainted_pil, mask_pil)
    # final_image.save(output_image, format="PNG")

# def _process_remove_caption_on_frame(input_image, output_image):
#     print('process image: ', str(input_image))
#     image_np = cv2.imread(input_image)
#     # image_np = np.array(input_image)

#     # Dùng OCR để phát hiện caption (text)
#     results = reader.readtext(image_np)

#     # Tạo mask rỗng (đen hoàn toàn)
#     mask_dilated = np.zeros(image_np.shape[:2], dtype=np.uint8)

#     for (bbox, text, prob) in results:
#         # bbox: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
#         pts = np.array(bbox, dtype=np.int32)
#         # Tô vùng chứa chữ lên mask (trắng: 255)
#         cv2.fillPoly(mask_dilated, [pts], 255) # type: ignore

#     # Xoá caption bằng inpainting
#     inpainted_np = cv2.inpaint(image_np, mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_NS)

#     # 5. Dùng LaMa refine lại vùng đó (PIL + mask)
#     inpainted_pil = Image.fromarray(inpainted_np)
#     mask_pil = Image.fromarray(mask_dilated)

#     final_image = run_lama(inpainted_pil, mask_pil)

#     # Trả về file ảnh
#     final_image.save(output_image, format="PNG")

# def frames_to_video(frame_dir, fps=30.0):
#     print('frames to video')

#     # Lấy danh sách file ảnh (chỉ lấy .png, .jpg)
#     frame_files = [f for f in os.listdir(frame_dir) if f.lower().endswith(('.png', '.jpg'))]
#     frame_files = natsorted(frame_files)  # Đảm bảo đúng thứ tự khung hình

#     if not frame_files:
#         print("❌ Không tìm thấy ảnh trong thư mục.")
#         return

#     output_path = Path('video-editted.mp4')

#     # Đọc kích thước ảnh đầu tiên
#     first_frame_path = os.path.join(frame_dir, frame_files[0])
#     frame = cv2.imread(first_frame_path)
#     height, width, _ = frame.shape

#     # Tạo đối tượng VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore # hoặc 'XVID'
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     print(f"▶️ Bắt đầu ghép {len(frame_files)} ảnh thành video...")

#     for file in frame_files:
#         frame_path = os.path.join(frame_dir, file)
#         img = cv2.imread(frame_path)
#         if img is None:
#             print(f"⚠️ Không đọc được ảnh: {file}")
#             continue
#         out.write(img)

#     out.release()
#     return output_path

def frames_to_video_ffmpeg(frame_dir, fps=30):
    frame_dir = Path(frame_dir)
    output_path = Path('video_editted.mp4')

    # Kiểm tra có frame không
    frame_files = [f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not frame_files:
        print("❌ Không tìm thấy ảnh trong thư mục.")
        return None

    # Đảm bảo thứ tự và tên ảnh đúng định dạng ffmpeg (frame_00001.jpg)
    frame_files = natsorted(frame_files)
    first_file = frame_files[0]
    if not first_file.startswith('frame_') or not '%05d' in first_file:
        print("⚠️ FFmpeg yêu cầu tên file theo mẫu: frame_%05d.jpg (frame_00001.jpg, ...)")
        print("💡 Bạn nên đổi tên trước khi dùng FFmpeg.")
        return None

    # Chạy FFmpeg để ghép video
    input_pattern = str(frame_dir / 'frame_%05d.jpg')  # hoặc .png nếu bạn dùng PNG
    command = [
        'ffmpeg',
        '-y',                        # Ghi đè nếu file đã tồn tại
        '-framerate', str(fps),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]

    print(f"▶️ Đang ghép video từ {len(frame_files)} ảnh...")
    subprocess.run(command, check=True)
    print(f"✅ Video đã lưu tại: {output_path}")
    return output_path

# --- Tạo mask tự động từ caption ---
def generate_mask_from_text(reader, image: Image.Image) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    h, w = image_np.shape[:2]
    results = reader.readtext(image_np)
    mask = np.zeros((h, w), dtype=np.uint8)

    for (bbox, text, prob) in results:
        print(f"Detected: {text}, confidence: {prob:.2f}")  # 👈 debug
        if prob < 0.5:
            continue
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    mask_image = Image.fromarray(mask)
    mask_image.save("mask_debug.png")  # 👈 xem thử mask
    return mask_image

# --- Xử lý inpaint bằng LaMa ---
def run_lama(reader, image: Image.Image, mask: Image.Image) -> Image.Image:
    # Convert PIL → numpy
    img = np.array(image.convert("RGB"))  # shape: (H, W, 3)
    msk = np.array(mask.convert("L"))     # shape: (H, W)

    # Convert to (C, H, W)
    img = img.transpose(2, 0, 1)           # (3, H, W)
    msk = np.expand_dims(msk, axis=0)     # (1, H, W)

    # Pad
    img_padded = pad_img_to_modulo(img, 8)
    msk_padded = pad_img_to_modulo(msk, 8)

    # ✅ Convert to torch.Tensor và normalize
    img_tensor = torch.from_numpy(img_padded / 255.0).float().unsqueeze(0).to(device)  # (1, 3, H, W)
    msk_tensor = torch.from_numpy(msk_padded / 255.0).float().unsqueeze(0).to(device)  # (1, 1, H, W)

    batch = {
        'image': img_tensor,
        'mask': msk_tensor,
    }

    with torch.no_grad():
        result = model(batch)['inpainted'][0]  # (3, H, W)
        result = result.permute(1, 2, 0).cpu().numpy()  # → (H, W, 3)
        result = np.clip(result * 255, 0, 255).astype('uint8')

    return Image.fromarray(result)