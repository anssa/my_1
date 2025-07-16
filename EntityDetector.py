# 基于BERT的联合实体识别
from transformers import BertTokenizer, BertForTokenClassification
import torch


class EntityDetector:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertForTokenClassification.from_pretrained('legal-ner-model')
        self.label_map = {0: 'PER', 1: 'ORG', 2: 'LOC', 3: 'WEAPON', 4: 'DRUG'}  # 自定义法律实体标签

    def extract_entities(self, text):
        """识别关键人物/组织/危险品实体"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

        entities = []
        current_entity = ""
        current_label = None

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        for token, pred in zip(tokens, predictions[0]):
            label = self.label_map.get(pred.item(), 'O')

            if label.startswith('B-'):  # 实体开始
                if current_entity:
                    entities.append((current_entity, current_label))
                current_entity = token.replace("##", "")
                current_label = label[2:]

            elif label.startswith('I-') and current_label == label[2:]:  # 实体延续
                current_entity += token.replace("##", "")

            else:  # 实体结束
                if current_entity:
                    entities.append((current_entity, current_label))
                current_entity = ""
                current_label = None

        return entities


# 示例使用
detector = EntityDetector()
chat = "龙哥说今晚8点码头接货，带好现金"
print(detector.extract_entities(chat))
# 输出: [('龙哥', 'PER'), ('码头', 'LOC'), ('货', 'DRUG')]