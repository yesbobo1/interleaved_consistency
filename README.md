# interleaved_consistency
用于5.2连贯性的evaluation
## 运行命令
```
python evaluate_local.py -i "path/to/input.jsonl" -o "path/to/output.jsonl" -d "dapa/path"

```
## dapa/path结构
·dataset
    ·domain
        ·subdomain
            ·video
            ·audio
            ·document
## 生成格式示例
```
{
    "domain": "general_domain",
    "subdomain": "entertainment",
    "id": "501",
    "content": "The track in <audio2:The audio features a mellow instrumental track with a prominent electric guitar leading the melody. The piece begins with a descending riff, followed by arpeggiated chords, hammer-ons, and a slide. The percussion section is minimal, consisting of rimshots and a common time signature, while the bass provides a single note on the first beat of every bar. Minimalist piano chords round out the song, leaving space for the guitar to shine. There are no vocals, making it perfect for a coffee shop or as background music. The key is in E major, with a chord progression centered around that key and a straightforward 4/4 time signature.> features a relaxing atmosphere dominated by an electric guitar with a descending riff, enhanced by subtle percussion and bass elements, making it ideal for a laid-back setting.",
    "modal": {
        "audio2": "The audio features a mellow instrumental track with a prominent electric guitar leading the melody. The piece begins with a descending riff, followed by arpeggiated chords, hammer-ons, and a slide. The percussion section is minimal, consisting of rimshots and a common time signature, while the bass provides a single note on the first beat of every bar. Minimalist piano chords round out the song, leaving space for the guitar to shine. There are no vocals, making it perfect for a coffee shop or as background music. The key is in E major, with a chord progression centered around that key and a straightforward 4/4 time signature."
    },
    "score": {
        "coherence": 5,
        "style_consistency": 5
    }
}
```
## 需要修改的部分
evaluation.py
```
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/liyanlin06/any2any_data/main/"

# --- 定义 subdomain 对应的 URL 模板 ---
SUBDOMAIN_URL_MAP = {
    "default": "{GITHUB_RAW_BASE}{processed_domain}/{subdomain}/{path}",
    "food": "https://raw.githubusercontent.com/Shize-ZHANG/Any2Any-Interleaved-Data-Pipline/main/original_data/{path}",
    "engineering": "https://raw.githubusercontent.com/GuoMinghui07/engineering/main/{path}",
    "computer_science": "{GITHUB_RAW_BASE}natural_science/computer_science/{path}",
    "culture": "https://raw.githubusercontent.com/micky-li-hd/any2any_culture/main/{path}",
}
```
需要改为与当前仓库匹配的路径
