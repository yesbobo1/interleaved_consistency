# x2text.py
import re
import torch
import os
import easyocr
from io import BytesIO
from PIL import Image
import requests
from qwen_omni_utils import process_mm_info
from pathlib import Path
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import warnings
import logging
from transformers.utils import logging as hf_logging
from openai import OpenAI
from dotenv import load_dotenv
from PointLLM.pointllm.eval.PointLLM_chat_local import threed_to_caption

# 关闭所有 warning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 关闭 transformers 的 log
hf_logging.set_verbosity_error()

# 关闭 root logger 的 WARNING 信息
logging.getLogger().setLevel(logging.ERROR)

# ======== 模型加载（video/audio 模态共用） ========
# 然后正常加载模型
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B",
    torch_dtype="auto",
    device_map="auto",
    use_safetensors=True
)
# model.disable_talker()
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")

# 强制关闭音频分支，避免显存爆掉
USE_AUDIO_IN_VIDEO = False

# ======== 各模态接口函数 ========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

reader = easyocr.Reader(['en'], gpu=False)  # GPU 可选

def ocr_extract_text(url):
    try:
        text_list = reader.readtext(url, detail=0)  # 直接传 URL
        return ' '.join(text_list)
    except Exception as e:
        print(f"OCR failed for {url}: {e}")
        return ""

def image_to_text(input_text, model_name="gpt-4o"):
    """
    使用 GPT-4o 对图片进行 dense caption 生成
    input_text: 图片 URL 或 base64
    model_name: GPT 模型名称，默认 gpt-4o
    """
    # 构造提示
    prompt = f"""
    You are a visual captioning assistant. 
    Given the image below, generate a **highly detailed and dense caption**, 
    describing objects, actions, attributes, relationships, and context. 
    Focus on capturing as much visual information as possible.

    [Image URL or Base64]
    {input_text}

    [Response Requirement]
    Return only the caption text, do not include 'user' or 'assistant' prefixes or extra explanations.
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a visual captioning assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        caption = response.choices[0].message.content.strip()
        return caption
    except Exception as e:
        print("Error generating image caption:", e)
        return ""

def document_to_text(input_image_url, model_name="gpt-4o"):
    """
    先用 OCR 提取 document 中的文字，然后用 GPT-4o 生成 dense caption
    """
    # Step 1: OCR 提取文本
    extracted_text = ocr_extract_text(input_image_url)

    print(f"\n{extracted_text}")

    # Step 2: 构造 GPT-4o prompt
    prompt = f"""
    You are a visual and document captioning assistant. 

    Below is a document image and its extracted text. 
    Generate a **dense description** that covers both the visual layout of the image 
    and the textual content extracted. Describe objects, layout, text content, 
    relationships, and context in detail.

    [Image URL]
    {input_image_url}

    [Extracted Text]
    {extracted_text}

    [Response Requirement]
    Return only a dense caption, do not include any system/user markers or extra explanation.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a strict visual captioning assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        caption = response.choices[0].message.content.strip()
        return caption
    except Exception as e:
        print("Error generating dense caption:", e)
        return ""

def code_to_text(code_snippet, model=None):
    """代码模态不需要生成文本"""
    return code_snippet

def video_to_text(input_video_url, model_name="Qwen/Qwen2.5-Omni-3B"):
    """
    使用 Qwen2.5-Omni-7B 对视频进行 dense caption
    支持同时感知视频画面和音频内容
    """
    try:
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a dense captioning assistant. Given a video, describe it in detail, including visual content, actions, objects and context."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": input_video_url},
                ],
            },
        ]

        # 启用视频中的音频
        use_audio_in_video = False

        # 构造输入
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        inputs = inputs.to(model.device).to(model.dtype)

        # 生成 dense caption
        text_ids, _ = model.generate(**inputs, use_audio_in_video=use_audio_in_video)
        captions = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return captions[0] if captions else ""
    except Exception as e:
        print("Error in video_to_text:", e)
        return ""

def audio_to_text(input_audio_url, model_name="Qwen/Qwen2.5-Omni-3B"):
    """
    使用 Qwen2.5-Omni-7B 对音频生成 dense caption（只输出文本）
    """
    try:
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an audio dense captioning assistant. Given an audio clip, provide a highly detailed description, including speech content, speakers, tone, background sounds, and overall context."
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": input_audio_url},
                ],
            },
        ]

        # 构造输入
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        inputs = inputs.to(model.device).to(model.dtype)

        # 推理生成 dense caption（只取 text_ids，忽略 audio）
        text_ids, _ = model.generate(**inputs, use_audio_in_video=False)
        captions = processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        print(captions)
        return captions[0] if captions else ""
    except Exception as e:
        print("Error in audio_to_text:", e)
        return ""

def download_file(url: str, save_dir: str = "/tmp") -> str:
    """
    支持 GitHub/raw URL 下载文件，返回本地文件路径
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        # 已存在文件直接返回
        return save_path

    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

def threed_to_text(input_3d_url: str, model_name: str = "RunsenXu/PointLLM_7B_v1.2", torch_dtype: str = "float16") -> str:
    """
    输入：
        input_3d_url: GitHub/raw URL 或本地路径的 ply/npy 文件
        model_name: PointLLM 模型名称
        torch_dtype: float16/float32/bfloat16
    输出：
        dense caption 字符串
    """
    # 如果是 URL 就先下载到本地
    if input_3d_url.startswith("http://") or input_3d_url.startswith("https://"):
        try:
            input_3d_url = download_file(input_3d_url, save_dir="/tmp/pointclouds")
        except Exception as e:
            print(f"[ERROR] 下载 3D 文件失败: {e}")
            return ""

    # 调用已有函数生成 caption
    try:
        caption = threed_to_caption(input_3d_url, model_name=model_name, torch_dtype=torch_dtype)
        return caption
    except Exception as e:
        print(f"[ERROR] 生成 3D dense caption 失败: {e}")
        return ""

def replace_placeholders(text, code_map=None, fn_map=None, model_map=None, modal_map=None):
    """
    替换文本中的占位符 <codeX>, <imageX>, <documentX>
    替换后格式为 <tag: dense caption>
    """
    if code_map is None:
        code_map = {}
    if modal_map is None:
        modal_map = {}

    def replacer(match):
        tag = match.group(0).strip("<> \n")
        print("Processing tag:", tag)

        # code 模态
        if tag in code_map:
            return f"<{tag}:{code_map[tag]}>"

        # 所有 modal_map 模态
        if tag in modal_map:
            path_or_url = modal_map[tag]
            # 如果有 fn_map 对应的处理函数，则调用
            prefix = re.match(r"[a-zA-Z]+", tag)
            if prefix and fn_map and prefix.group(0) in fn_map:
                fn = fn_map[prefix.group(0)]
                model_name = model_map.get(prefix.group(0)) if model_map else None
                caption = fn(path_or_url, model_name)
                return f"<{tag}:{caption}>"
            # 否则直接返回 modal_map 对应内容
            return f"<{tag}:{path_or_url}>"

        # 默认不替换
        return match.group(0)

    return re.sub(r"<(code|image|document|audio|video|threeD)\d+>", replacer, text)
