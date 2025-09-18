# x2text.py
import re
import torch
import soundfile as sf
import tempfile
import requests
import os
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# ======== 模型加载（image/document 模态共用） ========
model_name = "Qwen/Qwen2.5-Omni-7B"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

USE_AUDIO_IN_VIDEO = False

# ======== GitHub 文件下载辅助函数 ========
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/你的组织/你的仓库/main/"  # 替换为你的实际仓库

def download_from_github(path: str) -> str:
    """
    下载 GitHub 仓库指定路径文件到临时目录，返回本地路径
    """
    url = os.path.join(GITHUB_RAW_BASE, path)
    local_file = tempfile.NamedTemporaryFile(delete=False)
    response = requests.get(url)
    if response.status_code == 200:
        local_file.write(response.content)
        local_file.close()
        return local_file.name
    else:
        raise ValueError(f"Cannot download {url}, status code: {response.status_code}")

# ======== 各模态接口函数 ========
def image_to_text(input_text, model_name=None):
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human capable of perceiving visual inputs and generating text."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": input_text} if input_text.endswith(".mp4") else {"type": "image", "image": input_text}
            ],
        },
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids, _ = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    output_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]

def document_to_text(input_text, model_name=None):
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a document summarization assistant."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": input_text}
            ],
        },
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids, _ = model.generate(**inputs, use_audio_in_video=False)
    output_text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]

def code_to_text(code_snippet, model=None):
    """代码模态不需要生成文本"""
    return code_snippet

# ======== 占位符替换逻辑 ========
def replace_placeholders(text, code_map, fn_map=None, model_map=None):
    """
    text: 原始文本（含 <codeX>/<imageX>/<documentX> 占位符）
    code_map: { "code2": "```cpp ...```" } 从 output.modal 提取
    fn_map, model_map: 可保留用于其他模态
    """

    def replacer(match):
        tag = match.group(0).strip("<> \n")
        print("tag:", tag)

        # 优先替换 codeX
        if tag in code_map:
            return code_map[tag]

        # image/document 等模态
        prefix_match = re.match(r"[a-zA-Z]+", tag)
        if prefix_match and fn_map and prefix_match.group(0) in fn_map:
            prefix = prefix_match.group(0)
            fn = fn_map[prefix]
            model_name = model_map.get(prefix) if model_map else None

            content = code_map.get(tag, tag)  # 默认用 code_map 或 tag
            # 如果内容是 GitHub 路径，则下载
            if isinstance(content, str) and "/" in content:  # 简单判断是路径
                content = download_from_github(content)

            return fn(content, model_name)

        return match.group(0)

    # 只匹配 <codeX> / <imageX> / <documentX>
    return re.sub(r"<(code|image|document)\d+>", replacer, text)
