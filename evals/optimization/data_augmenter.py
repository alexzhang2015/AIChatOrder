"""
数据增强模块

扩充商品语料、生成训练样本、增强意图表达覆盖
"""

import re
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import yaml


@dataclass
class ProductCorpus:
    """商品语料"""
    canonical: str  # 标准名称
    formal_aliases: List[str] = field(default_factory=list)  # 正式别名
    colloquial: List[str] = field(default_factory=list)  # 口语化表达
    internet_slang: List[str] = field(default_factory=list)  # 网络用语
    typos: List[str] = field(default_factory=list)  # 常见错别字
    pinyin: List[str] = field(default_factory=list)  # 拼音变体

    def all_variants(self) -> List[str]:
        """获取所有变体"""
        variants = [self.canonical]
        variants.extend(self.formal_aliases)
        variants.extend(self.colloquial)
        variants.extend(self.internet_slang)
        variants.extend(self.typos)
        variants.extend(self.pinyin)
        return list(set(variants))


@dataclass
class IntentTemplate:
    """意图模板"""
    intent: str
    patterns: List[str]  # 模式模板，支持 {product}, {size} 等占位符
    slot_mappings: Dict[str, str] = field(default_factory=dict)


@dataclass
class AugmentedExample:
    """增强后的训练样本"""
    text: str
    intent: str
    slots: Dict[str, Any]
    source: str  # original, augmented, generated
    augmentation_type: str = ""  # synonym, template, paraphrase
    original_text: str = ""


class DataAugmenter:
    """数据增强器"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.product_corpus: Dict[str, ProductCorpus] = {}
        self.intent_templates: Dict[str, IntentTemplate] = {}
        self.generated_examples: List[AugmentedExample] = []

        # 初始化默认语料
        self._init_default_corpus()
        self._init_default_templates()

    def _init_default_corpus(self):
        """初始化默认商品语料"""
        self.product_corpus = {
            "美式咖啡": ProductCorpus(
                canonical="美式咖啡",
                formal_aliases=["美式", "Americano", "冰美式", "热美式"],
                colloquial=["续命水", "续命咖啡", "续命液", "黑咖啡", "清咖", "苦咖啡"],
                internet_slang=["命续", "提神神器", "打工人标配", "肝帝必备", "黑色血液"],
                typos=["没事咖啡", "美事", "美事咖啡"],
                pinyin=["meishi", "americano"]
            ),
            "拿铁": ProductCorpus(
                canonical="拿铁",
                formal_aliases=["Latte", "咖啡拿铁", "拿铁咖啡", "冰拿铁", "热拿铁"],
                colloquial=["奶咖", "牛奶咖啡", "丝滑咖啡"],
                internet_slang=["奶茶味咖啡", "入门咖啡"],
                typos=["拿贴", "拿帖", "那铁"],
                pinyin=["natie", "latte"]
            ),
            "卡布奇诺": ProductCorpus(
                canonical="卡布奇诺",
                formal_aliases=["Cappuccino", "卡布", "卡布基诺"],
                colloquial=["奶泡咖啡", "意式经典"],
                internet_slang=[],
                typos=["卡不奇诺", "卡布其诺", "卡布奇洛"],
                pinyin=["kabu", "cappuccino"]
            ),
            "摩卡": ProductCorpus(
                canonical="摩卡",
                formal_aliases=["Mocha", "摩卡咖啡", "冰摩卡", "热摩卡"],
                colloquial=["巧克力咖啡", "甜咖啡"],
                internet_slang=["肥宅咖啡", "甜党首选"],
                typos=["磨卡", "末卡"],
                pinyin=["moka", "mocha"]
            ),
            "星冰乐": ProductCorpus(
                canonical="星冰乐",
                formal_aliases=["Frappuccino", "冰沙", "星巴克冰沙"],
                colloquial=["冰冰凉", "冰饮"],
                internet_slang=["肥宅快乐水", "快乐肥宅水", "夏日续命"],
                typos=["星冰了", "星兵乐"],
                pinyin=["xingbingle", "frappuccino"]
            ),
            "馥芮白": ProductCorpus(
                canonical="馥芮白",
                formal_aliases=["Flat White", "澳白", "澳瑞白", "flatwhite"],
                colloquial=["小白咖啡", "澳洲白咖啡"],
                internet_slang=["dirty", "脏咖啡", "网红咖啡"],
                typos=["复瑞白", "馥瑞白", "芙芮白"],
                pinyin=["furuibai", "flatwhite"]
            ),
            "抹茶拿铁": ProductCorpus(
                canonical="抹茶拿铁",
                formal_aliases=["Matcha Latte", "抹茶", "绿茶拿铁"],
                colloquial=["绿色拿铁", "日式抹茶"],
                internet_slang=["绿色续命", "抹茶控必点"],
                typos=["抹差拿铁", "末茶拿铁"],
                pinyin=["mocha", "matchalatte"]
            ),
            "焦糖玛奇朵": ProductCorpus(
                canonical="焦糖玛奇朵",
                formal_aliases=["Caramel Macchiato", "玛奇朵", "焦糖咖啡"],
                colloquial=["甜味咖啡", "焦糖控"],
                internet_slang=["甜妹咖啡", "女生最爱"],
                typos=["焦糖马奇朵", "焦糖玛琪朵"],
                pinyin=["jiaotangmaqiduo", "macchiato"]
            )
        }

    def _init_default_templates(self):
        """初始化默认意图模板"""
        self.intent_templates = {
            "ORDER_NEW": IntentTemplate(
                intent="ORDER_NEW",
                patterns=[
                    "我要一杯{product}",
                    "来一杯{product}",
                    "给我一杯{product}",
                    "帮我点一杯{product}",
                    "要一个{product}",
                    "{product}来一杯",
                    "来个{product}",
                    "一杯{product}",
                    "我想喝{product}",
                    "想要{product}",
                    "{size}{product}",
                    "{temperature}{product}",
                    "一杯{size}{temperature}{product}",
                    "{quantity}杯{product}",
                    "整一杯{product}",
                    "搞一个{product}",
                    "来个{product}呗",
                    "{product}，谢谢",
                    "有{product}吗？来一杯",
                    "点一杯{product}，{temperature}的",
                ],
                slot_mappings={"product": "product_name"}
            ),
            "ORDER_MODIFY": IntentTemplate(
                intent="ORDER_MODIFY",
                patterns=[
                    "换成{size}",
                    "改成{temperature}",
                    "要{sweetness}",
                    "换{milk_type}",
                    "加一份浓缩",
                    "不要{extra}",
                    "少{modifier}",
                    "多{modifier}",
                    "改成{size}{temperature}",
                    "能换成{product}吗",
                    "把{slot}改成{value}",
                    "{modifier}一点",
                ],
                slot_mappings={}
            ),
            "ORDER_CANCEL": IntentTemplate(
                intent="ORDER_CANCEL",
                patterns=[
                    "取消订单",
                    "不要了",
                    "算了",
                    "不点了",
                    "还是不要了",
                    "先不喝了",
                    "取消",
                    "退掉",
                    "我不想要了",
                    "帮我取消一下",
                    "下次再说吧",
                    "改天吧",
                    "突然不想喝了",
                    "算了算了",
                    "不喝了",
                ],
                slot_mappings={}
            ),
            "RECOMMEND": IntentTemplate(
                intent="RECOMMEND",
                patterns=[
                    "有什么推荐的",
                    "推荐一下",
                    "什么好喝",
                    "你们什么最好卖",
                    "有什么新品吗",
                    "哪个比较好喝",
                    "推荐一款{characteristic}的",
                    "有没有{characteristic}的咖啡",
                    "想喝{characteristic}的，推荐一个",
                    "今日特饮是什么",
                    "招牌是什么",
                    "人气最高的是哪个",
                ],
                slot_mappings={}
            ),
            "PRODUCT_INFO": IntentTemplate(
                intent="PRODUCT_INFO",
                patterns=[
                    "{product}多少钱",
                    "{product}价格是多少",
                    "{product}有多少卡路里",
                    "{product}热量高吗",
                    "{product}有什么配料",
                    "{product}成分是什么",
                    "{product}有多大杯",
                    "这个多少钱",
                    "{product}怎么卖",
                ],
                slot_mappings={"product": "product_name"}
            ),
            "CHITCHAT": IntentTemplate(
                intent="CHITCHAT",
                patterns=[
                    "你好",
                    "嗨",
                    "谢谢",
                    "好的",
                    "嗯嗯",
                    "OK",
                    "再见",
                    "拜拜",
                    "今天天气真好",
                    "你是机器人吗",
                ],
                slot_mappings={}
            ),
            "PAYMENT": IntentTemplate(
                intent="PAYMENT",
                patterns=[
                    "我要付款",
                    "怎么支付",
                    "可以微信吗",
                    "支付宝可以吗",
                    "买单",
                    "结账",
                    "多少钱",
                    "有优惠吗",
                    "可以用积分吗",
                ],
                slot_mappings={}
            ),
        }

    def add_product_corpus(self, product_name: str, corpus: ProductCorpus):
        """添加商品语料"""
        self.product_corpus[product_name] = corpus

    def expand_product_aliases(self) -> Dict[str, List[str]]:
        """扩展商品别名映射"""
        alias_map = {}
        for product_name, corpus in self.product_corpus.items():
            for variant in corpus.all_variants():
                if variant.lower() != product_name.lower():
                    alias_map[variant] = product_name
        return alias_map

    def generate_from_templates(
        self,
        num_per_intent: int = 50,
        include_slots: bool = True
    ) -> List[AugmentedExample]:
        """从模板生成训练样本"""
        examples = []
        products = list(self.product_corpus.keys())

        # 槽位值映射表：占位符 -> (可选值列表, 槽位名称或None)
        slot_values = {
            "size": (["中杯", "大杯", "超大杯"], "size"),
            "temperature": (["冰", "热", "温", "去冰", "少冰"], "temperature"),
            "sweetness": (["标准", "半糖", "少糖", "无糖", "三分糖"], "sweetness"),
            "milk_type": (["全脂奶", "脱脂奶", "燕麦奶", "椰奶", "豆奶"], "milk_type"),
            "quantity": (["一", "两", "三", "1", "2", "3"], "quantity"),
            "characteristic": (["不太甜", "提神", "健康", "低卡", "香浓", "清淡"], None),
            "modifier": (["冰", "糖", "奶"], None),
            "extra": (["奶油", "糖", "冰"], None),
            "slot": (["杯型", "温度", "甜度"], None),
            "value": (["中杯", "大杯", "超大杯", "冰", "热", "温", "去冰", "少冰"], None),
        }

        for intent_name, template in self.intent_templates.items():
            generated_count = 0
            seen_texts: Set[str] = set()
            max_iterations = num_per_intent * 100  # 防止无限循环

            for _ in range(max_iterations):
                if generated_count >= num_per_intent:
                    break

                pattern = random.choice(template.patterns)
                text = pattern
                slots = {}

                # 替换产品占位符（特殊处理，需要从语料库获取变体）
                if "{product}" in text:
                    product = random.choice(products)
                    variant = random.choice(self.product_corpus[product].all_variants())
                    text = text.replace("{product}", variant)
                    slots["product_name"] = product

                # 替换其他占位符
                for placeholder, (values, slot_name) in slot_values.items():
                    key = "{" + placeholder + "}"
                    if key in text:
                        value = random.choice(values)
                        text = text.replace(key, value)
                        if slot_name:
                            slots[slot_name] = value

                # 跳过未完全替换的模板或重复文本
                if "{" in text or text in seen_texts:
                    continue

                seen_texts.add(text)
                examples.append(AugmentedExample(
                    text=text,
                    intent=intent_name,
                    slots=slots if include_slots else {},
                    source="generated",
                    augmentation_type="template",
                    original_text=pattern
                ))
                generated_count += 1

        self.generated_examples.extend(examples)
        return examples

    def augment_with_synonyms(
        self,
        examples: List[Dict[str, Any]],
        synonym_ratio: float = 0.3
    ) -> List[AugmentedExample]:
        """使用同义词增强现有样本"""
        augmented = []

        # 同义词映射
        synonyms = {
            "要": ["想要", "需要", "来"],
            "来": ["要", "给我", "帮我点"],
            "一杯": ["一个", "一份", ""],
            "帮我": ["给我", "麻烦", "请"],
            "好喝": ["不错", "好", "推荐"],
            "多少钱": ["价格", "怎么卖", "几块钱"],
            "取消": ["不要了", "算了", "退掉"],
            "换": ["改", "改成", "换成"],
        }

        for example in examples:
            text = example.get("text", "")
            intent = example.get("intent", "")
            slots = example.get("slots", {})

            # 原始样本
            augmented.append(AugmentedExample(
                text=text,
                intent=intent,
                slots=slots,
                source="original"
            ))

            # 同义词替换
            if random.random() < synonym_ratio:
                new_text = text
                for word, syns in synonyms.items():
                    if word in new_text and random.random() < 0.5:
                        new_text = new_text.replace(word, random.choice(syns), 1)

                if new_text != text:
                    augmented.append(AugmentedExample(
                        text=new_text,
                        intent=intent,
                        slots=slots,
                        source="augmented",
                        augmentation_type="synonym",
                        original_text=text
                    ))

        return augmented

    def generate_edge_cases(self, num_cases: int = 50) -> List[AugmentedExample]:
        """生成边界案例（容易混淆的表达）"""
        edge_cases = []

        # 边界案例模板
        edge_templates = [
            # RECOMMEND vs ORDER_NEW
            {
                "text": "有什么推荐的咖啡吗",
                "intent": "RECOMMEND",
                "note": "询问推荐，非下单"
            },
            {
                "text": "推荐的那个给我来一杯",
                "intent": "ORDER_NEW",
                "note": "基于推荐下单"
            },
            {
                "text": "什么比较好喝啊",
                "intent": "RECOMMEND",
                "note": "询问而非下单"
            },
            {
                "text": "好喝的来一杯",
                "intent": "ORDER_NEW",
                "note": "模糊但有下单意图"
            },

            # ORDER_MODIFY vs ORDER_NEW
            {
                "text": "换一杯拿铁",
                "intent": "ORDER_MODIFY",
                "note": "替换当前订单"
            },
            {
                "text": "再来一杯拿铁",
                "intent": "ORDER_NEW",
                "note": "追加新订单"
            },
            {
                "text": "还要一杯",
                "intent": "ORDER_NEW",
                "note": "追加"
            },
            {
                "text": "这个不要了换一个",
                "intent": "ORDER_MODIFY",
                "note": "替换"
            },

            # ORDER_CANCEL 隐式表达
            {
                "text": "算了吧",
                "intent": "ORDER_CANCEL",
                "note": "隐式取消"
            },
            {
                "text": "还是算了",
                "intent": "ORDER_CANCEL",
                "note": "隐式取消"
            },
            {
                "text": "不点了不点了",
                "intent": "ORDER_CANCEL",
                "note": "重复强调"
            },
            {
                "text": "突然不想喝了",
                "intent": "ORDER_CANCEL",
                "note": "间接取消"
            },
            {
                "text": "等会儿再说",
                "intent": "CHITCHAT",
                "note": "延迟，非取消"
            },

            # CHITCHAT vs 其他
            {
                "text": "就这样",
                "intent": "CHITCHAT",
                "note": "确认"
            },
            {
                "text": "可以",
                "intent": "CHITCHAT",
                "note": "确认"
            },
            {
                "text": "这个怎么样",
                "intent": "PRODUCT_INFO",
                "note": "询问产品"
            },
        ]

        for template in edge_templates:
            edge_cases.append(AugmentedExample(
                text=template["text"],
                intent=template["intent"],
                slots={},
                source="generated",
                augmentation_type="edge_case",
                original_text=template.get("note", "")
            ))

        # 生成更多变体
        products = list(self.product_corpus.keys())
        for _ in range(num_cases - len(edge_templates)):
            product = random.choice(products)
            corpus = self.product_corpus[product]
            variant = random.choice(corpus.all_variants())

            # 随机选择边界模式
            patterns = [
                (f"有{variant}吗", "PRODUCT_INFO"),  # 可能是询问也可能是点单
                (f"{variant}还有吗", "PRODUCT_INFO"),
                (f"来个{variant}试试", "ORDER_NEW"),
                (f"{variant}怎么样", "PRODUCT_INFO"),
            ]

            pattern, intent = random.choice(patterns)
            edge_cases.append(AugmentedExample(
                text=pattern,
                intent=intent,
                slots={"product_name": product} if product in pattern or variant in pattern else {},
                source="generated",
                augmentation_type="edge_case"
            ))

        return edge_cases

    def export_training_data(
        self,
        output_path: str,
        format: str = "jsonl"
    ) -> str:
        """导出训练数据"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with open(output_file, "w", encoding="utf-8") as f:
                for example in self.generated_examples:
                    line = json.dumps(asdict(example), ensure_ascii=False)
                    f.write(line + "\n")

        elif format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                data = [asdict(e) for e in self.generated_examples]
                json.dump(data, f, ensure_ascii=False, indent=2)

        elif format == "yaml":
            with open(output_file, "w", encoding="utf-8") as f:
                data = [asdict(e) for e in self.generated_examples]
                yaml.dump(data, f, allow_unicode=True)

        return str(output_file)

    def export_embedding_data(
        self,
        output_path: str,
        include_metadata: bool = True
    ) -> str:
        """导出用于 Embedding 的数据"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        embedding_data = []

        # 1. 商品知识
        for product_name, corpus in self.product_corpus.items():
            # 标准名称
            embedding_data.append({
                "id": f"product_{self._hash(product_name)}",
                "text": product_name,
                "metadata": {
                    "type": "product",
                    "product_name": product_name,
                    "is_canonical": True
                }
            })

            # 别名
            for alias in corpus.all_variants():
                if alias != product_name:
                    embedding_data.append({
                        "id": f"alias_{self._hash(alias)}",
                        "text": f"{alias}是{product_name}",
                        "metadata": {
                            "type": "product_alias",
                            "product_name": product_name,
                            "alias": alias
                        }
                    })

        # 2. 意图示例
        for example in self.generated_examples:
            embedding_data.append({
                "id": f"intent_{self._hash(example.text)}",
                "text": example.text,
                "metadata": {
                    "type": "intent_example",
                    "intent": example.intent,
                    "slots": example.slots,
                    "source": example.source
                }
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(embedding_data, f, ensure_ascii=False, indent=2)

        return str(output_file)

    def _hash(self, text: str) -> str:
        """生成短哈希"""
        return hashlib.md5(text.encode()).hexdigest()[:8]

    def generate_statistics(self) -> Dict[str, Any]:
        """生成统计信息"""
        stats = {
            "total_examples": len(self.generated_examples),
            "by_intent": {},
            "by_source": {},
            "by_augmentation_type": {},
            "product_corpus_size": len(self.product_corpus),
            "total_product_variants": sum(
                len(c.all_variants()) for c in self.product_corpus.values()
            )
        }

        for example in self.generated_examples:
            # 按意图统计
            stats["by_intent"][example.intent] = \
                stats["by_intent"].get(example.intent, 0) + 1

            # 按来源统计
            stats["by_source"][example.source] = \
                stats["by_source"].get(example.source, 0) + 1

            # 按增强类型统计
            if example.augmentation_type:
                stats["by_augmentation_type"][example.augmentation_type] = \
                    stats["by_augmentation_type"].get(example.augmentation_type, 0) + 1

        return stats


def main():
    """测试数据增强"""
    augmenter = DataAugmenter()

    # 1. 生成模板样本
    print("生成模板样本...")
    template_examples = augmenter.generate_from_templates(num_per_intent=30)
    print(f"  生成 {len(template_examples)} 个模板样本")

    # 2. 生成边界案例
    print("生成边界案例...")
    edge_cases = augmenter.generate_edge_cases(num_cases=30)
    augmenter.generated_examples.extend(edge_cases)
    print(f"  生成 {len(edge_cases)} 个边界案例")

    # 3. 统计信息
    stats = augmenter.generate_statistics()
    print("\n统计信息:")
    print(f"  总样本数: {stats['total_examples']}")
    print(f"  商品语料数: {stats['product_corpus_size']}")
    print(f"  商品变体总数: {stats['total_product_variants']}")
    print("\n  按意图分布:")
    for intent, count in sorted(stats["by_intent"].items()):
        print(f"    {intent}: {count}")

    # 4. 导出数据
    output_dir = Path("evals/optimization/augmented_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_file = augmenter.export_training_data(
        output_dir / "training_examples.jsonl"
    )
    print(f"\n训练数据已导出: {training_file}")

    embedding_file = augmenter.export_embedding_data(
        output_dir / "embedding_data.json"
    )
    print(f"Embedding 数据已导出: {embedding_file}")

    # 5. 显示示例
    print("\n示例样本:")
    for example in random.sample(augmenter.generated_examples, min(5, len(augmenter.generated_examples))):
        print(f"  [{example.intent}] {example.text}")
        if example.slots:
            print(f"    slots: {example.slots}")


if __name__ == "__main__":
    main()
