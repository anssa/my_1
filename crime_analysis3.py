import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ===== 模拟数据输入 =====
texts = [
    "今晚老地方交货，带上东西",
    "天气转凉了，多穿点",
    "这次必须干票大的",
    "想结束这种生活了",
    "你联系龙哥了吗，他那边安排好了",
    "龙哥让我们准备“茶叶”",
]

# ===== 关键实体识别 =====
keywords = ["龙哥", "交货", "毒品", "走私", "暴力", "自杀", "茶叶"]
def extract_entities(texts):
    entity_results = []
    for text in texts:
        hits = [kw for kw in keywords if kw in text]
        entity_results.append(", ".join(hits) if hits else "无")
    return entity_results

# ===== 敏感话题聚类（TF-IDF + KMeans） =====
def cluster_texts(texts, n_clusters=2):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X)
    return labels

# ===== 意图风险评分 =====
risk_dict = {
    "交货": 90, "龙哥": 95, "毒品": 98, "茶叶": 85,
    "这次必须干票大的": 99, "想结束这种生活": 80
}
def risk_score(texts):
    scores = []
    for text in texts:
        score = max([risk_dict.get(k, 0) for k in risk_dict if k in text], default=10)
        scores.append(score)
    return scores

# ===== 情感倾向分析（简单词典） =====
emotion_dict = {
    "愤怒": ["干票大的", "交货"],
    "抑郁": ["结束", "生活"],
    "紧张": ["准备", "安排"],
}
def emotion_analysis(texts):
    results = []
    for text in texts:
        for emo, words in emotion_dict.items():
            if any(word in text for word in words):
                results.append(emo)
                break
        else:
            results.append("中性")
    return results

# ===== 汇总结果 =====
df = pd.DataFrame({
    "原始文本": texts,
    "识别实体": extract_entities(texts),
    "话题类别": cluster_texts(texts),
    "风险评分": risk_score(texts),
    "情感倾向": emotion_analysis(texts)
})

# ===== 生成预警可视化 =====
def plot_risk(df):
    plt.figure(figsize=(8, 4))
    plt.barh(df["原始文本"], df["风险评分"], color="red")
    plt.xlabel("风险评分")
    plt.title("文本风险等级图")
    plt.tight_layout()
    plt.show()

# ===== 执行入口 =====
if __name__ == "__main__":
    print(df)
    plot_risk(df)