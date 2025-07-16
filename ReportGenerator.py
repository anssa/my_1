# # 使用Matplotlib生成预警报告
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
#
#
# class ReportGenerator:
#     def generate_threat_report(self, analysis_results):
#         """生成可视化预警报告"""
#         fig, axs = plt.subplots(2, 2, figsize=(12, 10))
#
#         # 1. 威胁等级仪表盘
#         self._create_gauge(axs[0, 0], analysis_results['threat_score'])
#
#         # 2. 实体关系网络
#         self._create_network(axs[0, 1], analysis_results['entities'])
#
#         # 3. 情感分布图
#         self._create_sentiment(axs[1, 0], analysis_results['sentiments'])
#
#         # 4. 关键词云
#         self._create_wordcloud(axs[1, 1], analysis_results['keywords'])
#
#         plt.tight_layout()
#         return fig
#
#     def _create_gauge(self, ax, score):
#         # 创建仪表盘可视化
#         ax.set_title(f"威胁等级: {score['level']}", fontsize=14)
#         # ... 具体仪表盘绘制代码 ...
#
#     def _create_network(self, ax, entities):
#         # 创建实体关系网络图
#         ax.set_title("核心实体关系", fontsize=14)
#         # ... 网络图绘制代码 ...
#
#     # 其他可视化方法省略...
#
#
# # 示例使用
# generator = ReportGenerator()
# analysis_data = {
#     'threat_score': {'score': 0.87, 'level': '红色预警'},
#     'entities': [('龙哥', 'PER'), ('码头', 'LOC'), ('货', 'DRUG')],
#     'sentiments': {'positive': 12, 'negative': 45, 'neutral': 43},
#     'keywords': [('交易', 23), ('现金', 18), ('今晚', 15)]
# }
#
# report_fig = generator.generate_threat_report(analysis_data)
# report_fig.savefig('threat_report.png')