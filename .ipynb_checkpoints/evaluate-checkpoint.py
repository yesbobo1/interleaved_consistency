import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 导入 x2text.py 中的函数
from x2text import replace_placeholders, image_to_text, document_to_text, code_to_text

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def evaluate_with_4o(text_content, model="gpt-4o"):
    """Call GPT-4o for evaluation based on coherence and style consistency with emphasized keywords"""
    prompt = f"""
You are a strict evaluation assistant. Evaluate the following text according to the criteria below.

[Evaluation Dimensions & Scoring Guidelines]

1. Coherence (1–5 points)
- 5: Output **highly matches** the input, multimodal references are **accurate and specific**; different modalities **complement** each other **without contradictions**; logic is **rigorous**, structure is **clear**; interleaved tags are **naturally ordered**, **fluent** reading experience.
- 4: Output **mostly matches** the input, references are **generally reasonable**; information is mostly **complementary**, minor **omissions/redundancy**; logic is **generally coherent**, minor **jumps**; tag order **mostly reasonable**.
- 3: Output has **general relation** to input, references are **vague/ambiguous**; some **repetition or missing info**, minor **contradictions**; local logical **jumps**, requiring inference; tag order **somewhat confusing**.
- 2: **Low relevance**, most references **vague or wrong**; multimodal info **repetitive/conflicting**; logic **chaotic**, frequent **contradictions**; tag order **significantly confusing**.
- 1: **Almost irrelevant** to input, references **missing or wrong**; modalities **do not complement**, **contradictory**; logic **collapsed**; tag order **completely disordered**.

2. Style Consistency (1–5 points)
- 5: Style **highly consistent**; multimodal narration **unified**; terminology **fully consistent**; expression and visual style **fully aligned**.
- 4: Style **mostly consistent**, minor **deviations**; terminology **generally consistent**; expression style **mostly unified**.
- 3: **Partially consistent**, noticeable **differences** in narration; terminology **partially mixed**; expression style **partially inconsistent**, affecting readability.
- 2: **Inconsistent style**; multimodal narration **uncoordinated**; terminology **messy**; expression style **clearly conflicting**.
- 1: **Completely inconsistent style**; terminology **wrong or inconsistent**; expression style **extremely uncoordinated**, barely understandable.

[Task Requirement]
Strictly assign a score (1–5) for each dimension. 
Return the result in JSON format only, e.g.:
{{
  "coherence": 4,
  "style_consistency": 5
}}

[Text to Evaluate]
{text_content}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict evaluation assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error calling GPT-4o:", e)
        return None


def stream_json_objects(file_path):
    """
    Sequentially read complete JSON objects from an indented/multiline JSONL file.
    Yield one object at a time.
    """
    buffer = ""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                continue
            buffer += line
            try:
                item = json.loads(buffer)
                yield item
                buffer = ""  # reset buffer for next object
            except json.JSONDecodeError:
                continue


if __name__ == "__main__":
    # Mapping of modal to functions
    fn_map = {
        "image": image_to_text,
        "document": document_to_text,
        "code": code_to_text
    }

    # Mapping of modal to models
    model_map = {
        "image": "qwen2.5-omni-7b",
        "document": "qwen2.5-omni-7b",
        "code": "gpt-4o"
    }

    file_path = "image_test.jsonl"
    output_file = "image_results.jsonl"

    # Clear output file
    open(output_file, "w").close()

    with open(output_file, "a", encoding="utf-8") as fout:
        for item in stream_json_objects(file_path):
            domain = item.get("domain")
            subdomain = item.get("subdomain")
            rec_id = item.get("id")
            output_content = item.get("output", {}).get("content", "")

            # --- 自动生成 code_map ---
            code_map = {}
            for key, value in item.get("output", {}).get("modal", {}).items():
                if key.startswith("code"):
                    # 如果 JSON 里只是路径或标识，可以直接使用 value 或额外读取
                    code_map[key] = value  

            # 替换占位符
            final_text = replace_placeholders(
                output_content,
                code_map=code_map,
                fn_map=fn_map,
                model_map=model_map
            )

            # 调用 GPT-4o 评分
            score = evaluate_with_4o(final_text)

            # 写入 JSONL
            result_obj = {
                "domain": domain,
                "subdomain": subdomain,
                "id": rec_id,
                "final_text": final_text,
                "score": score
            }
            fout.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
            print(f"[Written] domain={domain}, subdomain={subdomain}, id={rec_id}")
