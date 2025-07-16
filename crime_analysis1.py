import streamlit as st
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import time
import json
from collections import defaultdict
import re
from highlight_text import HighlightText, ax_text

# 初始化模型
nlp = spacy.load("zh_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-full-chinese")
ner_model = pipeline("ner", model="dslim/bert-base-NER")

# 预定义犯罪关键词库
CRIME_KEYWORDS = {
    "毒品": ["白粉", "冰", "叶子", "糖", "货", "东西", "快递", "冰妹", "溜冰", "飞行", "飞行员", "飞叶子", "飞行员",
             "东西"],
    "诈骗": ["转账", "汇款", "银行卡", "密码", "验证码", "中奖", "保证金", "手续费", "返利", "刷单", "投资", "彩票",
             "赌博"],
    "暴力": ["砍", "杀", "打", "教训", "收拾", "废了", "做了", "干掉", "报仇", "算账", "见血", "工具"],
    "赌博": ["赌", "牌", "局", "庄", "下注", "梭哈", "百家乐", "龙虎", "赌资", "抽水", "洗码", "盘口"],
    "走私": ["货", "过关", "水路", "上岸", "清关", "报关", "集装箱", "码头", "渔船", "快艇", "边境", "通道"],
}

# 预定义犯罪暗语映射
EUPHEMISMS = {
    "茶叶": "毒品",
    "钓鱼": "走私",
    "快递": "毒品交易",
    "海鲜": "走私货物",
    "糖": "冰毒",
    "东西": "毒品",
    "货": "毒品",
    "收租": "收赌债",
    "看海": "走私交易",
    "开张": "犯罪活动开始",
    "老地方": "犯罪交易地点"
}


# 初始化数据
def load_sample_data():
    return [
        "龙哥：今晚老地方交货，带够现金",
        "张三：明白，这次货的品质如何？",
        "李四：天气转凉了，多穿点",
        "王五：目标上钩了，已转账5万",
        "龙哥：收到，分你三成",
        "张三：上次的'糖'纯度不够，客户不满意",
        "李四：新'快递'明天到，注意查收",
        "王五：这次必须干票大的",
        "路人甲：周末一起吃饭吗？",
        "龙哥：不守规矩的人要好好教训",
        "张三：警察最近查得严，小心点",
        "李四：想结束这种生活...",
        "王五：海边钓鱼计划取消，改下周",
    ]


# 实体识别
def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            entities.append((ent.text, ent.label_))
    return entities


# 敏感话题分析
def detect_sensitive_topics(text):
    topics = []
    for topic, keywords in CRIME_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            topics.append(topic)
    return topics


# 犯罪意图分析
def analyze_criminal_intent(text):
    # 暗语检测
    for euphemism, meaning in EUPHEMISMS.items():
        if euphemism in text:
            return f"犯罪暗语检测: [{euphemism}] = {meaning}", 85

    # 关键词匹配
    for topic, keywords in CRIME_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return f"关键词触发: {topic}", 90

    # 情感增强分析
    sentiment = sentiment_analyzer(text)[0]
    if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.9:
        return "高负面情绪: " + text, 75

    return "未检测到明显意图", 5


# 情感分析
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']


# 构建关系图谱
def build_relationship_graph(messages):
    G = nx.DiGraph()
    entity_messages = defaultdict(list)

    # 第一遍：收集所有实体
    for msg in messages:
        entities = extract_entities(msg)
        for entity, label in entities:
            G.add_node(entity, type=label)

    # 第二遍：建立实体关系
    for i, msg in enumerate(messages):
        entities = [e[0] for e in extract_entities(msg)]
        for j in range(len(entities)):
            for k in range(j + 1, len(entities)):
                if G.has_edge(entities[j], entities[k]):
                    G[entities[j]][entities[k]]['weight'] += 1
                else:
                    G.add_edge(entities[j], entities[k], weight=1)
                entity_messages[(entities[j], entities[k])].append(msg)

    return G, entity_messages


# 聚类分析
def cluster_messages(messages, n_clusters=4):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(messages)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    clusters = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(messages[i])

    return clusters


# 高亮显示关键片段
def highlight_keywords(text):
    # 高亮实体
    entities = extract_entities(text)
    for entity, label in entities:
        if label == "PERSON":
            text = text.replace(entity,
                                f"<span style='background-color: #ffb6c1; padding: 2px; border-radius: 3px;'>{entity}</span>")
        elif label == "ORG":
            text = text.replace(entity,
                                f"<span style='background-color: #add8e6; padding: 2px; border-radius: 3px;'>{entity}</span>")
        elif label == "GPE":
            text = text.replace(entity,
                                f"<span style='background-color: #98fb98; padding: 2px; border-radius: 3px;'>{entity}</span>")

    # 高亮犯罪关键词
    for topic, keywords in CRIME_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                if topic == "毒品":
                    color = "#ff9999"
                elif topic == "诈骗":
                    color = "#ffcc99"
                elif topic == "暴力":
                    color = "#ff6666"
                elif topic == "赌博":
                    color = "#ccccff"
                else:
                    color = "#99ccff"
                text = text.replace(keyword,
                                    f"<span style='background-color: {color}; padding: 2px; border-radius: 3px; font-weight: bold;'>{keyword}</span>")

    # 高亮犯罪暗语
    for euphemism in EUPHEMISMS.keys():
        if euphemism in text:
            text = text.replace(euphemism,
                                f"<span style='background-color: #ffff99; padding: 2px; border-radius: 3px; font-weight: bold; border: 1px dashed #ff9900;'>{euphemism}</span>")

    return text


# 主应用
def main():
    st.set_page_config(page_title="AI慧眼：涉案文本智能分析系统", layout="wide")
    st.title("🕵️‍♂️ AI慧眼：涉案文本智能分析与预警系统")
    st.caption("基于深度学习的多维度犯罪意图识别平台 | 司法技术赛项解决方案")

    # 初始化会话状态
    if 'messages' not in st.session_state:
        st.session_state.messages = load_sample_data()

    # 侧边栏控制面板
    with st.sidebar:
        st.header("系统控制面板")
        analysis_type = st.selectbox("分析模式", ["实时分析", "批量处理"])
        risk_threshold = st.slider("风险预警阈值", 0, 100, 70)
        st.divider()

        st.subheader("数据管理")
        if st.button("加载示例数据"):
            st.session_state.messages = load_sample_data()
            st.success("示例数据已加载!")

        uploaded_file = st.file_uploader("上传聊天记录", type=["txt", "csv"])
        if uploaded_file:
            if uploaded_file.type == "text/plain":
                st.session_state.messages = uploaded_file.read().decode("utf-8").splitlines()
            st.success(f"已加载 {len(st.session_state.messages)} 条记录")

        st.divider()
        st.subheader("系统信息")
        st.info("""
        **核心功能架构**:
        - 关键实体识别
        - 敏感话题聚类
        - 意图风险评分
        - 情感倾向分析
        """)
        st.markdown("**高亮颜色说明**:")
        st.markdown("- <span style='background-color: #ffb6c1; padding: 2px; border-radius: 3px;'>人名</span>",
                    unsafe_allow_html=True)
        st.markdown("- <span style='background-color: #add8e6; padding: 2px; border-radius: 3px;'>组织</span>",
                    unsafe_allow_html=True)
        st.markdown("- <span style='background-color: #98fb98; padding: 2px; border-radius: 3px;'>地点</span>",
                    unsafe_allow_html=True)
        st.markdown("- <span style='background-color: #ff9999; padding: 2px; border-radius: 3px;'>毒品关键词</span>",
                    unsafe_allow_html=True)
        st.markdown("- <span style='background-color: #ffcc99; padding: 2px; border-radius: 3px;'>诈骗关键词</span>",
                    unsafe_allow_html=True)
        st.markdown("- <span style='background-color: #ff6666; padding: 2px; border-radius: 3px;'>暴力关键词</span>",
                    unsafe_allow_html=True)
        st.markdown(
            "- <span style='background-color: #ffff99; padding: 2px; border-radius: 3px; border: 1px dashed #ff9900;'>犯罪暗语</span>",
            unsafe_allow_html=True)

    # 主界面
    tab1, tab2, tab3, tab4 = st.tabs(["实时分析", "关系图谱", "聚类分析", "预警报告"])

    with tab1:
        st.subheader("实时文本分析")
        input_text = st.text_input("输入文本进行分析", "今晚老地方交货，带够现金")

        if st.button("分析文本", type="primary"):
            with st.spinner("AI分析中..."):
                # 实体识别
                entities = extract_entities(input_text)

                # 敏感话题
                topics = detect_sensitive_topics(input_text)

                # 意图分析
                intent, risk_score = analyze_criminal_intent(input_text)

                # 情感分析
                sentiment, sentiment_score = analyze_sentiment(input_text)

                # 显示结果
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.subheader("🔍 实体识别")
                    if entities:
                        for entity, label in entities:
                            st.info(f"{entity} ({label})")
                    else:
                        st.warning("未识别到实体")

                with col2:
                    st.subheader("📌 敏感话题")
                    if topics:
                        for topic in topics:
                            st.error(f"⚠️ {topic}")
                    else:
                        st.success("未检测到敏感话题")

                with col3:
                    st.subheader("🎯 犯罪意图")
                    progress_color = "red" if risk_score > risk_threshold else "green"
                    st.metric("风险评分", f"{risk_score}%", delta="高风险" if risk_score > risk_threshold else "低风险")
                    st.progress(risk_score / 100, text=f"**{intent}**")

                with col4:
                    st.subheader("😃 情感分析")
                    if sentiment == "NEGATIVE":
                        st.error(f"负面情绪: {sentiment_score:.0%}")
                        st.markdown("<div style='text-align:center; font-size:50px;'>🔥</div>", unsafe_allow_html=True)
                    else:
                        st.success(f"正面情绪: {sentiment_score:.0%}")
                        st.markdown("<div style='text-align:center; font-size:50px;'>😊</div>", unsafe_allow_html=True)

                # 高亮显示输入文本
                st.divider()
                st.subheader("高亮分析结果")
                highlighted_text = highlight_keywords(input_text)
                st.markdown(f"**原始文本**: {highlighted_text}", unsafe_allow_html=True)

                # 显示暗语解释
                detected_euphemisms = []
                for euphemism in EUPHEMISMS.keys():
                    if euphemism in input_text:
                        detected_euphemisms.append(f"**{euphemism}** → {EUPHEMISMS[euphemism]}")

                if detected_euphemisms:
                    st.info("检测到犯罪暗语: " + " | ".join(detected_euphemisms))

    with tab2:
        st.subheader("犯罪网络关系图谱")

        if st.button("生成关系图谱", key="graph_btn"):
            with st.spinner("构建犯罪网络..."):
                G, entity_messages = build_relationship_graph(st.session_state.messages)

                # 计算节点大小（基于度数）
                node_sizes = [G.degree(n) * 500 for n in G.nodes()]

                # 绘制图谱
                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(G, seed=42)
                nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightcoral", alpha=0.9)
                nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color="gray")
                nx.draw_networkx_labels(G, pos, font_size=12, font_family="SimHei")

                plt.title("犯罪网络关系图谱", fontsize=16)
                plt.axis("off")
                st.pyplot(plt)

                # 显示节点详情
                selected_node = st.selectbox("选择节点查看详情", list(G.nodes()))

                if selected_node:
                    st.subheader(f"节点: {selected_node}")

                    # 邻居节点
                    neighbors = list(G.neighbors(selected_node))
                    st.write(f"**关联节点**: {', '.join(neighbors) if neighbors else '无'}")

                    # 相关消息
                    related_msgs = []
                    for neighbor in neighbors:
                        if (selected_node, neighbor) in entity_messages:
                            related_msgs.extend(entity_messages[(selected_node, neighbor)])

                    if related_msgs:
                        st.subheader("相关消息")
                        for msg in set(related_msgs):
                            # 高亮显示消息
                            highlighted_msg = highlight_keywords(msg)
                            st.markdown(f"- {highlighted_msg}", unsafe_allow_html=True)
                    else:
                        st.info("未找到相关消息")

    with tab3:
        st.subheader("敏感话题聚类分析")
        n_clusters = st.slider("聚类数量", 2, 6, 4)

        if st.button("执行聚类分析", key="cluster_btn"):
            with st.spinner("聚类分析中..."):
                clusters = cluster_messages(st.session_state.messages, n_clusters)

                # 显示聚类结果
                for cluster_id, messages in clusters.items():
                    with st.expander(f"话题聚类 #{cluster_id + 1} (共{len(messages)}条)"):
                        for msg in messages:
                            # 高亮关键词
                            highlighted_msg = highlight_keywords(msg)
                            st.markdown(f"- {highlighted_msg}", unsafe_allow_html=True)

    with tab4:
        st.subheader("犯罪预警报告")

        if st.button("生成预警报告", type="primary", key="report_btn"):
            with st.spinner("分析所有消息并生成报告..."):
                # 分析所有消息
                high_risk = []
                entities_count = defaultdict(int)
                topic_count = defaultdict(int)

                for msg in st.session_state.messages:
                    _, risk_score = analyze_criminal_intent(msg)
                    if risk_score > risk_threshold:
                        high_risk.append(msg)

                    # 统计实体
                    for entity, _ in extract_entities(msg):
                        entities_count[entity] += 1

                    # 统计话题
                    topics = detect_sensitive_topics(msg)
                    for topic in topics:
                        topic_count[topic] += 1

                # 显示高风险消息
                st.subheader(f"高风险消息 ({len(high_risk)}条)")
                if high_risk:
                    for msg in high_risk:
                        highlighted_msg = highlight_keywords(msg)
                        st.error(f"⚠️ {highlighted_msg}", icon="🔥")
                else:
                    st.success("未检测到高风险消息")

                # 实体统计
                st.subheader("关键实体统计")
                if entities_count:
                    top_entities = sorted(entities_count.items(), key=lambda x: x[1], reverse=True)[:5]
                    df_entities = pd.DataFrame(top_entities, columns=["实体", "出现次数"])

                    # 添加颜色映射
                    colors = []
                    for entity in df_entities["实体"]:
                        # 根据实体类型分配颜色
                        entity_type = ""
                        for msg in st.session_state.messages:
                            if entity in msg:
                                entities_in_msg = extract_entities(msg)
                                for e, t in entities_in_msg:
                                    if e == entity:
                                        entity_type = t
                                        break
                                if entity_type:
                                    break

                        if entity_type == "PERSON":
                            colors.append("#ffb6c1")
                        elif entity_type == "ORG":
                            colors.append("#add8e6")
                        elif entity_type == "GPE":
                            colors.append("#98fb98")
                        else:
                            colors.append("#cccccc")

                    fig, ax = plt.subplots()
                    bars = ax.bar(df_entities["实体"], df_entities["出现次数"], color=colors)
                    ax.set_title("关键实体出现频次")
                    ax.set_ylabel("出现次数")
                    plt.xticks(rotation=45)

                    # 添加图例
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='#ffb6c1', label='人物'),
                        Patch(facecolor='#add8e6', label='组织'),
                        Patch(facecolor='#98fb98', label='地点')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right')

                    st.pyplot(fig)
                else:
                    st.info("未识别到关键实体")

                # 话题统计
                st.subheader("敏感话题分布")
                if topic_count:
                    # 分配颜色
                    topic_colors = {
                        "毒品": "#ff9999",
                        "诈骗": "#ffcc99",
                        "暴力": "#ff6666",
                        "赌博": "#ccccff",
                        "走私": "#99ccff"
                    }

                    colors = [topic_colors.get(topic, "#dddddd") for topic in topic_count.keys()]

                    fig, ax = plt.subplots()
                    ax.pie(topic_count.values(),
                           labels=topic_count.keys(),
                           colors=colors,
                           autopct="%1.1f%%",
                           startangle=90,
                           wedgeprops={'edgecolor': 'white', 'linewidth': 1})
                    ax.axis("equal")
                    ax.set_title("敏感话题分布")
                    st.pyplot(fig)
                else:
                    st.info("未检测到敏感话题")

                # 生成报告摘要
                st.divider()
                st.subheader("分析报告摘要")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("总消息量", len(st.session_state.messages))
                    st.metric("高风险消息", len(high_risk))

                with col2:
                    st.metric("识别实体数", len(entities_count))
                    st.metric("敏感话题类型", len(topic_count))

                # 风险等级评估
                risk_percentage = len(high_risk) / len(st.session_state.messages) * 100
                risk_level = "高危" if risk_percentage > 20 else "中危" if risk_percentage > 5 else "低危"

                # 根据风险等级设置颜色
                if risk_level == "高危":
                    color = "#ff4b4b"
                elif risk_level == "中危":
                   color = "#ffa700"
                else:
                   color = "#0f9d58"

                st.progress(risk_percentage/ 100, text=f"**整体风险等级: <span style='color:{color};'>{risk_level}</span> ({risk_percentage:.1f}%)**",
                help = f"风险等级评估标准: >20%高危, >5%中危, ≤5%低危")

if __name__ == "__main__":
    main()