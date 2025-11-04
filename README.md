# interleaved_consistency
用于5.2连贯性的evaluation
## 准备

同时开两个终端并申请GPU节点，注意两个节点的数字必须一样

终端A作为vllm服务器：

```
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-Omni-3B \
--port 8000 \
--dtype bfloat16 \
--max-model-len 32768 \
--enable-log-requests \
--max-num-batched-tokens 8192 \
--max-num-seqs 32 \
--gpu-memory-utilization 0.9 \
--h11-max-incomplete-event-size 200000000 \
--task generate \
--limit-mm-per-prompt '{"video": 1}'
```

终端B作为vllm客户端：

## 运行命令
```
python evaluate_local.py -i "path/to/input.jsonl" -o "path/to/output.jsonl" -d "dapa/path"

```
## dapa/path结构
```
dataset/
├── domain/
│   ├── subdomain/
│   │   ├── video/
│   │   ├── audio/
│   │   └── document/
```
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
    "difficulty_level": 1,
    "score": {
        "coherence": 5,
        "style_consistency": 5
    }
}
```
## 环境配置
如果已经配置过转caption的环境，则不用重新配

environment_1.yaml

如果vllm安装报错，可以直接从如下链接安装

https://github.com/vllm-project/vllm/releases/download/v0.10.2/vllm-0.10.2+cu129-cp38-abi3-manylinux1_x86_64.whl
