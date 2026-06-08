"""安装 Tesseract OCR（Windows 版） + 中文语言包。"""
import os, subprocess, sys, tempfile, urllib.request, zipfile, shutil
from pathlib import Path

INSTALL_DIR = Path(r"C:\Program Files\Tesseract-OCR")
ZIP_URL = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.5.0.20251113/tesseract-ocr-w64-setup-5.5.0.20251113.exe"
CHINESE_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/chi_sim.traineddata"

def main():
    # 检查是否已装
    if INSTALL_DIR.exists():
        print("Tesseract 已安装，跳过。")
        return

    print("下载 Tesseract 安装包（约 60MB）...")
    urllib.request.urlretrieve(ZIP_URL, "tesseract_setup.exe")
    print("安装中（静默安装）...")
    subprocess.run(["tesseract_setup.exe", "/S", f"/D={INSTALL_DIR}"], check=True)

    # 添加 PATH
    os.environ["PATH"] += os.pathsep + str(INSTALL_DIR)

    # 下载中文语言包
    print("下载中文语言包 ...")
    tessdata = INSTALL_DIR / "tessdata"
    urllib.request.urlretrieve(CHINESE_URL, str(tessdata / "chi_sim.traineddata"))

    subprocess.run(["tesseract", "--list-langs"], check=True)
    print("完成！")

if __name__ == "__main__":
    main()
