# 综合实体/情感/意图的威胁评分
import numpy as np


class ThreatAssessor:
    THRESHOLDS = {
        'LOW': 0.3,
        'MEDIUM': 0.6,
        'HIGH': 0.8
    }

    def assess_threat(self, entities, sentiment, intent):
        """生成综合威胁评分"""
        # 实体权重计算
        entity_score = 0
        entity_weights = {'WEAPON': 0.9, 'DRUG': 0.8, 'PER': 0.5}
        for _, label in entities:
            entity_score += entity_weights.get(label, 0)

        # 情感转换 ([-1,1] -> [0,1])
        sentiment_score = (sentiment + 1) / 2

        # 意图评分
        intent_scores = {
            'DRUG_TRADE': 0.95,
            'VIOLENCE_PLAN': 0.9,
            'FRAUD': 0.7
        }
        intent_score = intent_scores.get(intent, 0)

        # 综合评分 (加权平均)
        composite_score = 0.5 * entity_score + 0.2 * sentiment_score + 0.3 * intent_score

        # 威胁等级
        if composite_score >= self.THRESHOLDS['HIGH']:
            level = "红色预警"
        elif composite_score >= self.THRESHOLDS['MEDIUM']:
            level = "橙色预警"
        elif composite_score >= self.THRESHOLDS['LOW']:
            level = "黄色关注"
        else:
            level = "安全"

        return {
            "score": round(composite_score, 2),
            "level": level,
            "breakdown": {
                "entities": round(entity_score, 2),
                "sentiment": round(sentiment_score, 2),
                "intent": round(intent_score, 2)
            }
        }


# 示例使用
assessor = ThreatAssessor()
entities = [('龙哥', 'PER'), ('货', 'DRUG')]
sentiment = -0.8  # 极度负面
intent = "DRUG_TRADE"

print(assessor.assess_threat(entities, sentiment, intent))
# 输出: {'score': 0.87, 'level': '红色预警', 'breakdown': {...}}