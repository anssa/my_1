# # 结合规则与深度学习的暗语检测
# import re
# from sentence_transformers import SentenceTransformer
#
#
# class CodeWordDetector:
#     def __init__(self):
#         self.rule_based_patterns = {
#             r"老地方": "MEETING_POINT",
#             r"茶叶|糖果": "DRUG_CODE"
#         }
#         self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#         self.code_db = self._load_code_database()  # 加载犯罪暗语数据库
#
#     def detect_codewords(self, text):
#         """识别文本中的潜在犯罪暗语"""
#         # 规则匹配
#         for pattern, code_type in self.rule_based_patterns.items():
#             if re.search(pattern, text):
#                 yield (pattern, code_type)
#
#         # 语义匹配
#         text_embedding = self.model.encode(text)
#         for codeword, embedding in self.code_db:
#             similarity = cosine_similarity([text_embedding], [embedding])[0][0]
#             if similarity > 0.85:  # 相似度阈值
#                 yield (codeword, "SEMANTIC_CODE")
#
#     def _load_code_database(self):
#         """加载学生标注的犯罪暗语库"""
#         # 示例数据 (实际从数据库加载)
#         return [
#             ("冰糖", model.encode("冰毒")),
#             ("钓鱼", model.encode("走私交易"))
#         ]
#
#
# # 示例使用
# detector = CodeWordDetector()
# message = "天气转凉了多穿点，带些冰糖"
# print(list(detector.detect_codewords(message)))
# # 输出: [('冰糖', 'SEMANTIC_CODE')]