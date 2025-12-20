import polars as pl 
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tqdm import tqdm
import re


@dataclass
class AtKMetric(ABC):
    k: int

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def full_name(self) -> str:
        return f"{self.name}@{self.k}"

    @abstractmethod
    def __call__(self, ground_truth_col: str = "ground_truth", recommendations_col: str = "recommendations") -> pl.Expr:
        raise NotImplementedError


@dataclass
class NdcgAtK(AtKMetric):
    @property
    def name(self) -> str:
        return "NDCG"

    def __call__(self, ground_truth_col: str = "ground_truth", recommendations_col: str = "recommendations") -> pl.Expr:
        return pl.struct([ground_truth_col, recommendations_col]).map_elements(
            lambda x: self._calculate_ndcg(x[ground_truth_col], x[recommendations_col], self.k),
            return_dtype=pl.Float64
        ).alias(self.full_name)
    
    def _calculate_ndcg(self, gt_items: list, predicted: list, k: int) -> float:
        relevance = [1 if x in predicted[:k] else 0 for x in gt_items]
        dcg = sum((rel) / np.log2(idx + 2) for idx, rel in enumerate(relevance[:k]))
        ideal_relevance = sorted(relevance, reverse=True)[:k]
        ideal_dcg = sum((rel) / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


@dataclass  
class PrecisionAtK(AtKMetric):
    @property
    def name(self) -> str:
        return "Precision"

    def __call__(self, ground_truth_col: str = "ground_truth", recommendations_col: str = "recommendations") -> pl.Expr:
        return (
            pl.col(ground_truth_col)
            .list.set_intersection(pl.col(recommendations_col).list.head(self.k))
            .list.len()
            .truediv(self.k)
            .alias(self.full_name)
        )


@dataclass
class RecallAtK(AtKMetric):
    @property
    def name(self) -> str:
        return "Recall"

    def __call__(self, ground_truth_col: str = "ground_truth", recommendations_col: str = "recommendations") -> pl.Expr:
        return (
            pl.col(ground_truth_col)
            .list.set_intersection(pl.col(recommendations_col).list.head(self.k))
            .list.len()
            .truediv(pl.col(ground_truth_col).list.len())
            .alias(self.full_name)
        )


@dataclass
class HitRateAtK(AtKMetric):
    @property
    def name(self) -> str:
        return "HitRate"

    def __call__(self, ground_truth_col: str = "ground_truth", recommendations_col: str = "recommendations") -> pl.Expr:
        return (
            pl.col(ground_truth_col)
            .list.set_intersection(pl.col(recommendations_col).list.head(self.k))
            .list.len()
            .gt(0)
            .cast(pl.UInt8)
            .alias(self.full_name)
        )


def evaluate_recommender(
    df: pl.DataFrame,
    metrics: list[AtKMetric],
    ground_truth_col: str = "ground_truth", 
    recommendations_col: str = "recommendations"
) -> dict[str, float]:
    exprs = [metric(ground_truth_col=ground_truth_col, recommendations_col=recommendations_col) 
             for metric in metrics]
    return df.select(exprs).mean().to_dicts()[0]


##################################################################################################################################


RECS_COUNT_PATTERN = re.compile(r"Количество хороших рекомендаций:\s*(\d+)/\d+")
RATING_PATTERN = re.compile(r"Оценка:\s*(\d+\.?\d*)/5")
INDIVIDUAL_RATINGS_PATTERN = re.compile(r"Общая оценка:\s*([-\d]+)")
LLM = ...


def generate_prompt(basket: list[str], recommendations: list[str], prompt_type: str = "rating_scale") -> str:
    if prompt_type == "basic":
        
        return f"""Подумай и определи, какие из рекомендаций к корзине являются хорошими. 
Хорошие рекомендации — это рекомендации, которые связаны с товарами в корзине с точки зрения:
1. Частоты совместного употребления.
2. Типичных покупок.
3. Разнообразия - из группы одинаковых товаров разрешено выбрать только один. Если товар уже есть в корзине, такой же товар из рекомендаций нельзя считать подходящим.

Формат ответа:
- Количество хороших рекомендаций: X/{len(recommendations)}
- Хорошие рекомендации: [item1; item2; ...]
- Обоснование: Почему выбранные рекомендации хорошие, а остальные - нет.

Корзина: [{"; ".join([item.strip() for item in basket])}]. Рекомендации: [{"; ".join([item.strip() for item in recommendations])}]. 
"""
        
    elif prompt_type == "rating_scale":
        
        return f"""Оцени качество рекомендаций к корзине покупателя по шкале от 0 до 5.
Ориентируйся на следующие критерии: 
1. Способствуют ли рекомендации увеличению корзины? 
2. Есть ли логичные "сопутствующие товары"? Уместны ли они для типичного покупателя с таким набором? 
3. Нет ли избыточных или повторяющихся предложений (например, несколько видов одного и того же овоща без необходимости)? 
4. Рекомендации должны быть разнообразными и полезными.

Формат ответа:
- Оценка: X/5
- Обоснование: Укажи сильные и слабые стороны.

Корзина: [{"; ".join([item.strip() for item in basket])}]. Рекомендации: [{"; ".join([item.strip() for item in recommendations])}].
"""
    elif prompt_type == "individual_ratings":
        
        return f"""Оцени каждую рекомендацию индивидуально по шкале от -1 до 2:
-1 - вообще не вписывающаяся, очень плохая рекомендация.
 0 - случайная, нейтральная рекомендация.
 1 - в принципе подходящая рекомендация.
 2 - очень хорошая, подходящая рекомендация.
Хорошие рекомендации — это рекомендации, которые связаны с товарами в корзине с точки зрения:
1. Частоты совместного употребления.
2. Типичных покупок.
3. Разнообразия - из группы одинаковых товаров разрешено выбрать только один. Если товар уже есть в корзине, такой же товар из рекомендаций нельзя считать подходящим.

Формат ответа:
- Общая оценка: X  # Сумма баллов по всем рекомендациям
- Детальные оценки: [item1: оценка; item2: оценка; ...]
- Обоснование: Кратко объясни поставленные оценки.

Корзина: [{"; ".join([item.strip() for item in basket])}]. Рекомендации: [{"; ".join([item.strip() for item in recommendations])}].
"""
        
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def parse_response(model_response: str, prompt_type: str, recommendations_count: int) -> float:
    global RECS_COUNT_PATTERN, RATING_PATTERN, INDIVIDUAL_RATINGS_PATTERN

    if prompt_type == "basic":
        match = RECS_COUNT_PATTERN.search(model_response)
        if match:
            return float(match.group(1)) / recommendations_count  # normalize to 0-1
        return np.nan
    
    elif prompt_type == "rating_scale":
        match = RATING_PATTERN.search(model_response)
        if match:
            return float(match.group(1)) / 5.0  # normalize to 0-1
        return np.nan
    
    elif prompt_type == "individual_ratings":
        match = INDIVIDUAL_RATINGS_PATTERN.search(model_response)
        if match:
            total_score = float(match.group(1))
            max_possible_score = recommendations_count * 2
            return total_score / max_possible_score  # normalize to 0-1
        return np.nan
    
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def _evaluate_recommender_by_llm(
    baskets: list[list[str]],
    recommendations: list[list[str]],
    prompt_type: str = "basic",
    num_attempts_per_basket: int = 3,
    verbose: bool = False,
) -> list[float]:
    global LLM

    all_scores = []
    
    for i in tqdm(range(len(baskets)), desc=f"Evaluating recommendations ({prompt_type})"):
        attempt_scores = []
        
        for attempt in range(num_attempts_per_basket):
            response_text = LLM.invoke(
                input=generate_prompt(baskets[i], recommendations[i], prompt_type)
            ).content
            
            if verbose:
                print(f"attempt {attempt + 1} for basket {i}: {response_text}")
                
            score = parse_response(response_text, prompt_type, len(recommendations[i]))
            attempt_scores.append(score)
            
        all_scores.append(np.nanmean(attempt_scores))
        
    return all_scores


def evaluate_recommender_by_llm(
    df: pl.DataFrame,
    items: pl.DataFrame,
    recommendations_col: str = "item_id_target",
    k: int = 10,
    prompt_type: str = "rating_scale",  # basic, rating_scale, individual_ratings
    num_attempts_per_basket: int = 3,
    verbose: bool = False
) -> tuple[np.array, float]:
    
    baskets = (
        df.select("order_id", "item_id_for_pred").explode("item_id_for_pred")
        .join(items, left_on="item_id_for_pred", right_on="item_id", how="left")
        .group_by("order_id", maintain_order=True).agg(pl.col("name"))["name"].to_list()
    )
    
    recommendations = (
        df.select("order_id", recommendations_col).explode(recommendations_col)
        .join(items, left_on=recommendations_col, right_on="item_id", how="left")
        .group_by("order_id", maintain_order=True).agg(pl.col("name").head(k))["name"].to_list()
    )

    llmscores = np.array(
        _evaluate_recommender_by_llm(baskets, recommendations, prompt_type, num_attempts_per_basket, verbose), 
        dtype=np.float32
    )

    return np.nanmean(llmscores), np.isnan(llmscores).mean()
