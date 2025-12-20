import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import json
from pathlib import Path
from tqdm import tqdm


model_name = "t-tech/T-lite-it-1.0"

TOKENIZER = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    trust_remote_code=True
)
MODEL = AutoModelForCausalLM.from_pretrained(
    model_name,
    pad_token_id=TOKENIZER.pad_token_id,
    device_map="auto",
    trust_remote_code=True
).eval()

ITEMS_PATTERN = re.compile(r"\[\[?(.*)\]")

BAD_WORDS_IDS = [
    TOKENIZER.encode(bad_word, add_special_tokens=False)
    for bad_word in ["или", "например"]
]


def generate_prompt(basket: list[str], n_new_items: int = 10) -> str:
    prompt = f"""Корзина покупателя: [{"; ".join([item.strip() for item in basket])}].
Изучи корзину покупателя, подумай и дополнительно предложи {n_new_items} товаров, которые будут ему полезны и интересны и которые он с высокой вероятностью купит. Новые товары должны быть разнообразными и обязательно должны быть логически связаны с имеющимися.
Товары необходимо описать без указания товарного бренда и граммовки - например, "Куриная грудка" хорошо, "Куриная грудка Петелинка 300г" плохо. Нельзя повторять товары из корзины покупателя.

Формат ответа:
- Товары: [item1; item2; ...]
- Объяснение: ...
"""
    return prompt.strip()


def get_items(model_response: str) -> list[str]:
    try:
        items = ITEMS_PATTERN.search(model_response).group(1).split(";")
        return [] if not len(items) else [it.strip() for it in items]
    except (TypeError, AttributeError):
        return []


def generate_recommendations(
    baskets: list[list[str]],
    n_new_items: list[int] | int | None = None,
    max_new_tokens: int = 300,
    batch_size: int = 64,
    verbose: bool = False,
    checkpoint_dir: Path | None = None,
) -> list[list[str] | None]:
    global TOKENIZER, MODEL

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
         
    all_items = [None] * len(baskets)

    for i in tqdm(range(0, len(baskets), batch_size), desc="Generating recommendations"):
        batch_indices = list(range(i, min(i + batch_size, len(baskets))))
        batch_baskets = [baskets[j] for j in batch_indices]
        batch_n_new_items = (
            [n_new_items[j] for j in batch_indices]
            if isinstance(n_new_items, list)
            else [n_new_items] * len(batch_indices)
        )

        prompts = [
            TOKENIZER.apply_chat_template(
                [
                    {"role": "system", "content": "Ты T-lite, виртуальный ассистент в Т-Технологии. Твоя задача - быть полезным диалоговым ассистентом."},
                    {"role": "user", "content": generate_prompt(basket, n_new_items)}
                ],
                tokenize=False,
                add_generation_prompt=True
            ) + "\n- Товары: ["
            for basket, n_new_items in zip(batch_baskets, batch_n_new_items)
        ]

        inputs = TOKENIZER(prompts, return_tensors="pt", padding=True, max_length=512, truncation=True, return_token_type_ids=False)
        input_ids = inputs["input_ids"].to(MODEL.device)
        attention_mask = inputs["attention_mask"].to(MODEL.device)

        with torch.no_grad():
            outputs = MODEL.generate(
                tokenizer=TOKENIZER,
                input_ids=input_ids,
                attention_mask=attention_mask,
                bad_words_ids=BAD_WORDS_IDS,
                max_new_tokens=max_new_tokens,
                temperature=0.4,
                top_p=0.8,
                top_k=70,
                repetition_penalty=1.05,
                do_sample=True,
                stop_strings=None if verbose else "\n- "
            )

        for batch_idx, (idx, output) in enumerate(zip(batch_indices, outputs)):
            input_length = input_ids.shape[1] if batch_idx == 0 else input_ids[batch_idx].shape[0]
            generated = output[input_length:]
            response_text = "[" + TOKENIZER.decode(generated, skip_special_tokens=True)
            if verbose:
                print(f"Batch index {batch_idx}, Response: {response_text}")
            items = get_items(response_text)
            if len(items):
                items = [it for it in items if it not in baskets[idx]]
            all_items[idx] = items

        if checkpoint_dir is not None:
            with open(checkpoint_dir / f"batch_{i // batch_size}.json", "w", encoding="utf-8") as f:
                json.dump({"all_items": all_items}, f, ensure_ascii=False, indent=2)

    return all_items
