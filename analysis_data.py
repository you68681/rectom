import json
from collections import defaultdict

# 改成你的逐样本结果文件路径，不是 summary.json
RESULT_PATH = "/Users/clei1/Downloads/hitom_llm_project/results_hitom_simtom_fixed_qwen_vscode/tomi_simulation_results.jsonl"

stats = defaultdict(lambda: {"correct": 0, "total": 0})

def is_correct(item):
    """
    兼容几种常见字段：
    - correct: true/false
    - is_correct: true/false
    - prediction / predicted_answer 与 answer / gold_answer 比较
    """
    if "correct" in item:
        return bool(item["correct"])

    if "is_correct" in item:
        return bool(item["is_correct"])

    pred = (
        item.get("prediction")
        or item.get("predicted_answer")
        or item.get("model_answer")
        or item.get("final_answer")
    )

    gold = (
        item.get("answer")
        or item.get("gold_answer")
        or item.get("label")
    )

    if pred is None or gold is None:
        raise ValueError(
            f"Cannot determine correctness for sample: {item.get('sample_id', 'unknown')}"
        )

    return str(pred).strip().lower() == str(gold).strip().lower()


with open(RESULT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        item = json.loads(line)

        # 兼容字段名 question_order / order
        order = item.get("question_order", item.get("order"))

        if order is None:
            raise ValueError(
                f"Missing question_order/order in sample: {item.get('sample_id', 'unknown')}"
            )

        order = int(order)

        correct = is_correct(item)

        stats[order]["total"] += 1
        stats[order]["correct"] += int(correct)


print("Accuracy by question order")
print("-" * 40)

for order in range(5):
    total = stats[order]["total"]
    correct = stats[order]["correct"]

    if total == 0:
        acc = 0.0
    else:
        acc = correct / total

    print(
        f"Order {order}: "
        f"{correct}/{total} = {acc:.4f} "
        f"({acc * 100:.2f}%)"
    )