"""
PDF 摄取模块 —— 把 manual/ 下的 PDF 解析、切片、存入 Chroma 向量库。

流程：
  读取 PDF（每页一篇文章）
  → 切成 500 字左右的 chunk
  → 用 text2vec 转成向量
  → 写入 Chroma（本地持久化）

跑一次就行了，后面启动 app 直接加载已有向量库。

支持两种 PDF：
- 普通文本 PDF（如游戏手册）→ pdfplumber 直接提取文字
- 图片型 PDF（如剧本图表截图）→ OCR（需安装 Tesseract）
"""
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .llm import get_embeddings


# ===== OCR 支持（图片型 PDF 用） =====

_OCR_AVAILABLE = False  # 标记 Tesseract 是否可用
_TESSERACT_PATH = None  # tesseract.exe 的路径

# 常见安装位置
_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    str(Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"),
]

def _init_ocr():
    """检测 Tesseract 是否已安装，并配置 pytesseract。"""
    global _OCR_AVAILABLE, _TESSERACT_PATH
    import subprocess
    # 先在 PATH 中找
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, timeout=5)
        _OCR_AVAILABLE = True
        return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # 再在常见安装路径找
    for path in _TESSERACT_CANDIDATES:
        if Path(path).exists():
            _TESSERACT_PATH = path
            _OCR_AVAILABLE = True
            return
    _OCR_AVAILABLE = False

def _ocr_pdf_page(pdf_path: Path, page_index: int) -> str:
    """
    对 PDF 的某一页做 OCR 识别。
    
    用 pypdfium2 把 PDF 页渲染成图片，
    再用 pytesseract 识别图片中的文字。
    """
    import pypdfium2 as pdfium
    import pytesseract

    # 配置 tesseract 路径（如果在非标准位置）
    if _TESSERACT_PATH:
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH

    # PDF 页 → 图片（2x 分辨率，提高识别率）
    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[page_index]
    bitmap = page.render(scale=2)
    img = bitmap.to_pil()
    pdf.close()

    # OCR 识别（中英文混合）
    text = pytesseract.image_to_string(img, lang="chi_sim+eng")
    return text.strip()


# ===== 文本 PDF 解析 =====

def load_pdf(path: Path) -> list[Document]:
    """读取一个文本型 PDF，每页变成一个 Document。"""
    print(f"  解析: {path.name}")
    docs = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": path.name,     # 文件名
                    "page": i,               # 页码
                },
            ))
    print(f"    → {len(docs)} 页")
    return docs


def load_image_pdf(path: Path) -> list[Document]:
    """读取一个图片型 PDF，用 OCR 识别每页文字。"""
    if not _OCR_AVAILABLE:
        print(f"  ⚠ 跳过（未安装 Tesseract）：{path.name}")
        return []

    print(f"  OCR 识别: {path.name}")
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    total = len(pdf)
    pdf.close()

    docs = []
    for i in range(total):
        # 显示进度
        if (i + 1) % 3 == 0 or i == total - 1:
            print(f"    → 第 {i+1}/{total} 页 ...")

        text = _ocr_pdf_page(path, i)
        if text:
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "page": i + 1,
                    "ocr": True,     # 标记是 OCR 识别的
                },
            ))
    print(f"    → OCR 完成，共 {len(docs)} 页有文字")
    return docs


def get_manual_pdfs() -> list[Path]:
    """获取 manual/ 下所有 PDF 文件。"""
    return sorted(config.MANUAL_DIR.rglob("*.pdf"))


def load_all_manuals() -> list[Document]:
    """读取全部 PDF，按类型选择解析方式。"""
    # 启动时检测一次 Tesseract
    _init_ocr()
    if _OCR_AVAILABLE:
        print("[OCR] Tesseract 已就绪，可识别图片型 PDF")
    else:
        print("[OCR] Tesseract 未安装，图片型 PDF 将被跳过")
        print("      安装方法：https://github.com/UB-Mannheim/tesseract/wiki")

    all_docs = []
    for pdf_path in get_manual_pdfs():
        # 判断是文本 PDF 还是图片 PDF
        # 用 pdfplumber 试读第一页，没文字就走 OCR
        has_text = False
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages[:2]:  # 看前两页
                    if (page.extract_text() or "").strip():
                        has_text = True
                        break
        except Exception:
            pass  # 解析失败当图片 PDF 处理

        if has_text:
            # 普通文本 PDF
            all_docs.extend(load_pdf(pdf_path))
        else:
            # 图片型 PDF，走 OCR
            all_docs.extend(load_image_pdf(pdf_path))

    print(f"总计 {len(all_docs)} 页")
    return all_docs


def split_into_chunks(docs: list[Document]) -> list[Document]:
    """
    把"每页一篇文章"切成"每段约 500 字的小块"。
    
    为什么切？
    - 一页可能太长（1000+ 字），LLM 的上下文窗口有限
    - 一段太短也不行，可能信息不完整
    - 500 字 + 80 字重叠是比较稳妥的配置
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"切片: {len(docs)} 页 → {len(chunks)} 个 chunk")
    return chunks


def build_vectorstore_from_scratch():
    """
    从头构建向量库：读取 PDF → 切片 → 向量化 → 写入 Chroma。
    
    这样设计：
    - 不把 Chroma 导入放在文件顶部（因为 pipeline.py 的 _get_vectorstore()
      可能调用这个函数，而 Chroma 不是所有场景都需要）
    - 延迟导入，加快模块加载速度
    """
    from langchain_chroma import Chroma

    print("=" * 40)
    print("构建向量库")
    print("=" * 40)

    # 1. 读取所有 PDF
    print("[1/3] 读取 PDF ...")
    raw_docs = load_all_manuals()

    # 2. 切成 chunk
    print("[2/3] 切片 ...")
    chunks = split_into_chunks(raw_docs)

    # 3. 向量化 + 写入 Chroma
    print("[3/3] 向量化写入 Chroma ...")
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=str(config.VECTORSTORE_DIR),
    )

    print(f"完成！共 {vectorstore._collection.count()} 条向量")
    return vectorstore
