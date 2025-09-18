# test_evaluation.py
from pprint import pprint

# 假设你之前已经定义了 evaluate_with_4o(text_content, model="gpt-4o")
# 并且 generate_optimized_single_tag_samples() 已经定义

def test_samples():
    # 导入优化好的示例文本
    from evaluate import generate_code_samples, evaluate_with_4o

    samples = generate_code_samples()
    results = {}

    print("=== Starting evaluation of single-tag samples ===\n")

    for key, text in samples.items():
        print(f"Evaluating {key}...")
        # 调用 GPT-4o 评估
        score = evaluate_with_4o(text)
        results[key] = score

    print("\n=== Evaluation Results ===")
    pprint(results)

if __name__ == "__main__":
    test_samples()
