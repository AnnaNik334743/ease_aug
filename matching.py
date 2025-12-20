import re
import json
import pickle
import numpy as np
import polars as pl
import pymorphy2
from pathlib import Path
from functools import lru_cache
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import fasttext
import fasttext.util
from tqdm import tqdm


class BM25Matcher:
    def __init__(self, vectorizer_kwargs=None, bm25_k1=1.5, bm25_b=0.75):
        self.morph = pymorphy2.MorphAnalyzer()
        
        self.vectorizer_kwargs = vectorizer_kwargs or {}
        self.vectorizer = CountVectorizer(**self.vectorizer_kwargs)
        self.vectors = None
        
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.doc_lengths = None
        self.avg_doc_length = None
        self.idf = None
        
        self.id2index = None
        self.index2id = None
        
        self._fitted = False

    @lru_cache(maxsize=10000)
    def _lemmatize_word(self, word: str) -> str:
        return self.morph.parse(word)[0].normal_form

    def _preprocess_text(self, text: str) -> str:
        words = re.sub(r"[^a-zа-я]", " ", text.lower().replace("ё", "е")).split()
        normalized_words = [self._lemmatize_word(word) for word in words]
        return " ".join(normalized_words)
    
    def _compute_bm25_weights(self, tf_matrix):
        tf_matrix_csr = tf_matrix.tocsr()
        
        self.doc_lengths = np.array(tf_matrix_csr.sum(axis=1)).flatten()
        self.avg_doc_length = self.doc_lengths.mean()
        
        n_docs = tf_matrix_csr.shape[0]
        df = np.array((tf_matrix_csr > 0).sum(axis=0)).flatten()
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        
        bm25_data = []
        bm25_indices = []
        bm25_indptr = [0]
        
        for i in range(tf_matrix_csr.shape[0]):
            doc_length = self.doc_lengths[i]
            length_norm = self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_length / self.avg_doc_length)
            
            start, end = tf_matrix_csr.indptr[i], tf_matrix_csr.indptr[i+1]
            
            for j in range(start, end):
                term_idx = tf_matrix_csr.indices[j]
                tf_val = tf_matrix_csr.data[j]
                
                numerator = self.idf[term_idx] * tf_val * (self.bm25_k1 + 1)
                denominator = tf_val + length_norm
                bm25_val = numerator / denominator
                
                bm25_data.append(bm25_val)
                bm25_indices.append(term_idx)
            
            bm25_indptr.append(len(bm25_data))
        
        bm25_matrix = sparse.csr_matrix(
            (bm25_data, bm25_indices, bm25_indptr),
            shape=tf_matrix_csr.shape
        )
        
        return bm25_matrix
    
    def fit(self, catalog: pl.DataFrame, id_col: str = "item_id", text_col: str = "name"):
        texts = catalog[text_col].to_list()
        ids = catalog[id_col].to_list()
        
        self.id2index = {}
        processed_texts = []
        
        for idx, (text, item_id) in enumerate(zip(texts, ids)):
            preprocessed_text = self._preprocess_text(str(text))
            processed_texts.append(preprocessed_text)
            self.id2index[str(item_id)] = idx
        
        n_items = len(ids)
        self.index2id = np.empty(n_items, dtype=object)
        for item_id, idx in self.id2index.items():
            self.index2id[idx] = item_id
 
        tf_matrix = self.vectorizer.fit_transform(processed_texts)
        
        self.vectors = self._compute_bm25_weights(tf_matrix)
        self._fitted = True
    
    def _compute_query_bm25(self, query_vector):
        query_csr = query_vector.tocsr()
        
        n_docs = self.vectors.shape[0]
        scores = np.zeros(n_docs)
        
        query_indices = query_csr.indices
        query_data = query_csr.data
        
        for query_idx, query_tf_val in zip(query_indices, query_data):
            term_bm25_scores = self.vectors[:, query_idx]
            scores += term_bm25_scores.toarray().flatten() * query_tf_val
            
        return scores
    
    def match(self, queries, top_k=10):
        if not self._fitted:
            raise ValueError("The model is not fitted yet. Call fit() first.")
        
        if isinstance(queries, str):
            queries = [queries]

        if len(queries) == 0:
            return []
        
        preprocessed_queries = [self._preprocess_text(str(query)) for query in queries]
        queries_vectorized = self.vectorizer.transform(preprocessed_queries)
        
        results = []
        
        for query_vec in queries_vectorized:
            scores = self._compute_query_bm25(query_vec)
    
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

            results.append(self.index2id[top_indices])
        
        return results
    
    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        
        if not self._fitted:
            raise ValueError("The model is not fitted yet. Call fit() first.")
        
        sparse.save_npz(Path(path) / "vectors.npz", self.vectors)
        
        np.savez_compressed(
            Path(path) / "bm25_params.npz",
            doc_lengths=self.doc_lengths,
            avg_doc_length=self.avg_doc_length,
            idf=self.idf,
            index2id=self.index2id
        )
        
        with open(Path(path) / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        config = {
            "vectorizer_kwargs": self.vectorizer_kwargs,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "fitted": self._fitted
        }
        with open(Path(path) / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str):
        path = Path(path)
        
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        model = cls(
            vectorizer_kwargs=config["vectorizer_kwargs"],
            bm25_k1=config.get("bm25_k1", 1.5),
            bm25_b=config.get("bm25_b", 0.75)
        )

        model.vectors = sparse.load_npz(path / "vectors.npz").tocsr()
        
        bm25_params = np.load(path / "bm25_params.npz", allow_pickle=True)
        model.doc_lengths = bm25_params["doc_lengths"]
        model.avg_doc_length = bm25_params["avg_doc_length"]
        model.idf = bm25_params["idf"]
        model.index2id = bm25_params["index2id"]
        
        model.id2index = {str(item_id): idx for idx, item_id in enumerate(model.index2id)}
        
        with open(path / "vectorizer.pkl", "rb") as f:
            model.vectorizer = pickle.load(f)
        
        model._fitted = config["fitted"]
        
        return model


class FastTextMatcher:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.vectors = None
        
        self.id2index = None
        self.index2id = None
        
        self._fitted = False
        self._load_fasttext_model()

    def _load_fasttext_model(self):
        if self.model_path and Path(self.model_path).exists():
            self.model = fasttext.load_model(self.model_path)
        else:
            try:
                self.model = fasttext.load_model("cc.ru.300.bin")
            except:
                fasttext.util.download_model("ru", if_exists="ignore")
                self.model = fasttext.load_model("cc.ru.300.bin")

    @staticmethod
    def _preprocess_text(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"""[\[\]\'\"\{\}\t\n\r]""", " ", text)).strip()

    def _text_to_vector(self, text: str) -> np.ndarray:
        return self.model.get_sentence_vector(self._preprocess_text(text))

    def fit(self, catalog: pl.DataFrame, id_col: str = "item_id", text_col: str = "name"):
        texts = catalog[text_col].to_list()
        ids = catalog[id_col].to_list()
        
        self.id2index = {}
        vectors = []
        
        for idx, (text, item_id) in enumerate(zip(texts, ids)):
            vector = self._text_to_vector(str(text))
            vectors.append(vector)
            self.id2index[str(item_id)] = idx
        
        n_items = len(ids)
        self.index2id = np.empty(n_items, dtype=object)
        for item_id, idx in self.id2index.items():
            self.index2id[idx] = item_id
        
        vectors = np.array(vectors)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid zero division
        self.vectors = vectors / norms
        
        self._fitted = True

    def match(self, queries, top_k=10):
        if not self._fitted:
            raise ValueError("The model is not fitted yet. Call fit() first.")
        
        if isinstance(queries, str):
            queries = [queries]

        if len(queries) == 0:
            return []
        
        query_vectors = np.array([self._text_to_vector(str(query)) for query in queries])
    
        query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        query_norms[query_norms == 0] = 1
        query_vectors_normalized = query_vectors / query_norms
    
        similarities = np.dot(query_vectors_normalized, self.vectors.T)

        top_indices = np.argpartition(-similarities, top_k, axis=1)[:, :top_k]
        rows = np.arange(similarities.shape[0])[:, None]
        top_scores_for_sorting = similarities[rows, top_indices]
        sorted_indices = np.argsort(-top_scores_for_sorting, axis=1)
        top_indices_sorted = top_indices[rows, sorted_indices]

        return [self.index2id[top_indices] for top_indices in top_indices_sorted]
    
    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        
        if not self._fitted:
            raise ValueError("The model is not fitted yet. Call fit() first.")
        
        np.save(Path(path) / "vectors.npy", self.vectors)
        
        np.savez_compressed(
            Path(path) / "indices.npz",
            index2id=self.index2id
        )
        
        config = {
            "model_path": self.model_path,
            "fitted": self._fitted
        }
        
        import json
        with open(Path(path) / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str):
        path = Path(path)
        
        import json
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        model = cls(
            model_path=config["model_path"],
        )
        
        model.vectors = np.load(path / "vectors.npy")
        
        indices = np.load(path / "indices.npz", allow_pickle=True)
        model.index2id = indices["index2id"]
        model.id2index = {str(item_id): idx for idx, item_id in enumerate(model.index2id)}
        
        model._fitted = config["fitted"]
        
        return model


def match_llm_recommendations(
    baskets: list[list[str]],
    llm_generated_recommendations: list[list[str]],
    items: pl.DataFrame,  # must be present in ranker train
    matcher_retriever,
    matcher_ranker,
    top_k_retriever: int = 10,
) -> list[list[str]]:
    llm_generated_items = []

    for i, llm_generated_recommendation in enumerate(tqdm(llm_generated_recommendations, desc="Matching to catalog")):
        all_topn_matched = matcher_retriever.match(llm_generated_recommendation, top_k=top_k_retriever)
    
        llm_generated_basket_items = []
        for topn_matched in all_topn_matched:
            top_matched = matcher_ranker.get_best_item(input_basket=baskets[i], list_of_items=topn_matched)
            llm_generated_basket_items.append(top_matched)
    
        llm_generated_items.append(llm_generated_basket_items)

    return llm_generated_items


def get_augmented_orders(
    rare_items_augmented: pl.DataFrame,  # contains item_id, llm_recommendations
    items: pl.DataFrame,  # must be present in ranker train
    matcher_retriever,
    matcher_ranker,
    top_k_retriever: int = 10,
) -> pl.DataFrame:
    
    baskets = [[item] for item in rare_items_augmented["item_id"]]
    llm_generated_recommendations = rare_items_augmented["llm_recommendations"].to_list()
    llm_generated_items = match_llm_recommendations(
        baskets, llm_generated_recommendations, items, 
        matcher_retriever=matcher_retriever, matcher_ranker=matcher_ranker, top_k_retriever=top_k_retriever
    )

    order_ids = []
    item_ids = []
    for i, llm_generated_item in enumerate(llm_generated_items):
        order_ids.append(-i)
        item_ids.append(baskets[i] + llm_generated_item)
        
    aug_orders = pl.DataFrame({"order_id": order_ids, "item_id": item_ids})

    return aug_orders
