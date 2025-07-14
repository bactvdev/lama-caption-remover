import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from lama.service.video import remove_video_caption
import uuid

# --- App FastAPI ---
app = FastAPI()

@app.post("/inpaint")
async def inpaint_caption(file: UploadFile = File(...)):
    # Tạo file video tạm
    temp_input = f"temp_input_{uuid.uuid4().hex}.mp4"
    with open(temp_input, "wb") as f:
        contents = await file.read()
        f.write(contents)

    # Xử lý video
    output_path = remove_video_caption(temp_input)

    # Trả file kết quả
    return FileResponse(path=output_path, media_type="video/mp4", filename="output_cleaned.mp4")
