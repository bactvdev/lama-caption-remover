#!/bin/bash

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

mkdir -p lama/models/big-lama/models

# ID của file trên Google Drive (thay bằng ID thật của bạn)
CONFIG_ID="1sw6eLiCJb8ngKCEBczrtJyajDyGbWBVh"
CKPT_ID="1MBjuLv9HphGH1CKlKxwweTdZEXnkAdmB"

# Download config.yaml
if [ ! -f lama/models/big-lama/config.yaml ]; then
    echo "⬇️  Downloading config.yaml..."
    gdown --id $CONFIG_ID -O lama/models/big-lama/config.yaml
else
    echo "✅ config.yaml already exists."
fi

# Download best.ckpt
if [ ! -f lama/models/big-lama/models/best.ckpt ]; then
    echo "⬇️  Downloading best.ckpt..."
    gdown --id $CKPT_ID -O lama/models/big-lama/models/best.ckpt
else
    echo "✅ best.ckpt already exists."
fi

pip install -r lama/requirements.txt

# ======================
# 3. Cài đặt Ngrok CLI
# ======================
if ! command -v ./ngrok &> /dev/null; then
    wget -q -O ngrok.zip https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-stable-linux-amd64.zip
    unzip -o ngrok.zip
else
    echo "✅ Ngrok already installed."
fi

./ngrok config add-authtoken 2zrm3wOOXwW7QdbVajnB1zt1jgc_44gW91d6M8Y1ms9WYtpMs

cd lama

uvicorn main:app --host 0.0.0.0 --port 8000

echo "✅ Done setting up!"

# chmod +x install.sh
# !cd lama-caption-remover && ./ngrok http --url=insect-ideal-unduly.ngrok-free.app 8000 --log=stdout