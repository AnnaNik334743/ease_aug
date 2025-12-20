import os
import json
import polars as pl
import numpy as np
from tqdm.notebook import tqdm
import random as random_module


class Random:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.all_items = None
        self.trained = False
        
    def fit(self, df: pl.DataFrame, basket_col="order_id", item_col="item_id", rating_col=None):
        np.random.seed(self.seed)
        random_module.seed(self.seed)
        self.all_items = df[item_col].unique().to_list()
        self.trained = True
        
    def predict(self, df: pl.DataFrame, basket_with_items_for_pred_col="item_id_for_pred", topn: int = 10):
        if not self.trained:
            raise ValueError("Model must be trained before calling predict.")
            
        recommendations = []
        for _ in tqdm(range(len(df)), desc="Generating random recommendations"):
            recs = np.random.choice(self.all_items, size=topn, replace=False).tolist()
            recommendations.append(recs)
            
        return recommendations
    
    def get_best_item(self, input_basket: list[str], list_of_items: list[str]) -> str:
        if not self.trained:
            raise ValueError("Model must be trained before calling predict.")
            
        return random_module.choice(list_of_items)
    
    def save(self, path: str):
        if not self.trained:
            raise ValueError("No model to save. Fit the model first.")
            
        os.makedirs(path, exist_ok=True)
        
        items_df = pl.DataFrame({"item_id": self.all_items})
        items_df.write_parquet(os.path.join(path, "items.pq"))
        
        config = {
            "seed": self.seed,
            "trained": self.trained
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path: str):
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
            
        model = cls(seed=config["seed"])
        model.trained = config["trained"]
        
        items_df = pl.read_parquet(os.path.join(path, "items.pq"))
        model.all_items = items_df["item_id"].to_list()
        
        return model


class TopPopular:
    def __init__(self):
        self.popularity = None
        self.item_col = None
        self.trained = False
        
    def fit(self, df: pl.DataFrame, basket_col="order_id", item_col="item_id", rating_col=None):
        self.item_col = item_col
        
        self.popularity = (
            df[item_col]
            .value_counts()
            .rename({"count": "score"})
            .sort("score", descending=True)
        )
        self.trained = True
        
    def predict(self, df: pl.DataFrame, basket_with_items_for_pred_col="item_id_for_pred", topn: int = 10):
        if not self.trained:
            raise ValueError("Model must be trained before calling predict.")
            
        top_items = (
            self.popularity
            .head(topn)[self.item_col]
            .to_list()
        )
        
        recommendations = [top_items] * len(df)
        
        return recommendations
    
    def get_best_item(self, input_basket: list[str], list_of_items: list[str]) -> str:
        if not self.trained:
            raise ValueError("Model must be trained before calling predict.")
            
        candidate_popularity = (
            self.popularity
            .filter(pl.col(self.item_col).is_in(list_of_items))
            .sort("score", descending=True)
        )
        
        if len(candidate_popularity) == 0:
            raise ValueError("No candidate items found in training data")
            
        return candidate_popularity[self.item_col][0]
    
    def save(self, path: str):
        if not self.trained:
            raise ValueError("No model to save. Fit the model first.")
            
        os.makedirs(path, exist_ok=True)
    
        self.popularity.write_parquet(os.path.join(path, "popularity.pq"))
        metadata = pl.DataFrame({
            "item_col": [self.item_col],
            "trained": [self.trained]
        })
        metadata.write_parquet(os.path.join(path, "metadata.pq"))
    
    @classmethod
    def load(cls, path: str):
        model = cls()
        model.popularity = pl.read_parquet(os.path.join(path, "popularity.pq"))
        metadata = pl.read_parquet(os.path.join(path, "metadata.pq"))
        model.item_col = metadata["item_col"][0]
        model.trained = metadata["trained"][0]
        return model
        

class EASE():
    def __init__(self, regularization: float = 1200.0):
        self.regularization = regularization
        self.basket2index = None 
        self.item2index = None
        self.index2item = None
        self.bi_matrix = None
        self.weight = None
        self.trained = False

    def _get_basket_item_matrix(self, df: pl.DataFrame, basket_col="order_id", 
                               item_col="item_id", 
                               rating_col=None):
        basket_ids = df[basket_col].unique()
        item_ids = df[item_col].unique()

        basket2index = {bid: idx for idx, bid in enumerate(basket_ids)}
        item2index = {iid: idx for idx, iid in enumerate(item_ids)}

        num_baskets = len(basket2index)
        num_items = len(item2index)
        bi_matrix = np.zeros((num_baskets, num_items))

        for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Building basket-item matrix"):
            bidx = basket2index[row[basket_col]]
            iidx = item2index[row[item_col]]
            bi_matrix[bidx, iidx] = row[rating_col] if rating_col else 1  # binary or weighted

        return basket2index, item2index, bi_matrix

    def fit(self, df: pl.DataFrame, basket_col="order_id", item_col="item_id", rating_col=None):
        self.basket2index, self.item2index, self.bi_matrix = self._get_basket_item_matrix(
            df, basket_col, item_col, rating_col
        )
        self.index2item = [""] * len(self.item2index)
        for item, idx in self.item2index.items():
            self.index2item[idx] = item
        self.index2item = np.array(self.index2item)

        # compute Gram matrix
        gram_matrix = self.bi_matrix.T @ self.bi_matrix
        diag_indices = np.diag_indices(gram_matrix.shape[0])
        gram_matrix[diag_indices] += self.regularization
        gram_matrix_inv = np.linalg.inv(gram_matrix)
        
        # compute weight matrix
        self.weight = -gram_matrix_inv / np.diag(gram_matrix_inv)[:, None]
        np.fill_diagonal(self.weight, 0.0)
    
        self.trained = True

    def predict(self, df: pl.DataFrame, basket_with_items_for_pred_col="item_id_for_pred", topn: int = 10, batch_size: int = 10_000):
        """
        Predict top-N items for provided baskets (in form of list[item]).
        Each row in df should have a list of item_ids under `basket_with_items_for_pred_col`.
        """
        if not self.trained:
            raise ValueError("Model must be trained before calling predict.")

        baskets = []
        for row in tqdm(df.iter_rows(named=True), total=df.shape[0], desc="Preparing baskets"):
            basket = np.zeros(len(self.item2index))
            curr_basket_items = [
                self.item2index[iid] 
                for iid in row[basket_with_items_for_pred_col] 
                if iid in self.item2index
            ]
            basket[curr_basket_items] = 1
            baskets.append(basket)
        baskets = np.array(baskets)  # (num_baskets, num_items)

        batch_basket_predictions = []
        for i in tqdm(range(0, len(baskets), batch_size), desc="Predicting by batch"):
            batch = baskets[i:i+batch_size]
            batch_pred = batch @ self.weight  # (batch_size, num_items)
            batch_basket_predictions.append(batch_pred)
        basket_predictions = np.vstack(batch_basket_predictions)

        # mask out already purchased items
        basket_predictions[baskets > 0] = float("-inf")

        # get top-N items
        top_items = np.argpartition(-basket_predictions, topn, axis=1)[:, :topn]
        top_items = np.take_along_axis(
            top_items,
            np.argsort(-np.take_along_axis(basket_predictions, top_items, axis=1), axis=1), 
            axis=1
        )

        # map indices back to original item IDs
        recommendations = [
            self.index2item[top_row]
            for top_row in top_items
        ]

        return recommendations

    def get_best_item(self, input_basket: list[str], list_of_items: list[str]) -> str:
        if not self.trained:
            raise ValueError("Model must be trained before calling predict.")
    
        items_to_score = [self.item2index[iid] for iid in list_of_items]
    
        converted_basket = np.zeros(len(self.item2index))
        for item in input_basket:
            converted_basket[self.item2index[item]] = 1
        return list_of_items[np.argsort(-(converted_basket @ self.weight[:, items_to_score]))[0]]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "basket2index.json"), "w") as f:
            json.dump({str(k): int(v) for k, v in self.basket2index.items()}, f)

        with open(os.path.join(path, "item2index.json"), "w") as f:
            json.dump({str(k): int(v) for k, v in self.item2index.items()}, f)

        np.savez(os.path.join(path, "index2item.npz"), data=self.index2item)
        np.savez(os.path.join(path, "weight.npz"), data=self.weight)

        config = {
            "regularization": self.regularization,
            "trained": self.trained
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str):
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        model = cls(regularization=config["regularization"])
        model.trained = config["trained"]

        with open(os.path.join(path, "basket2index.json"), "r") as f:
            model.basket2index = {k: int(v) for k, v in json.load(f).items()}

        with open(os.path.join(path, "item2index.json"), "r") as f:
            model.item2index = {k: int(v) for k, v in json.load(f).items()}

        model.index2item = np.load(os.path.join(path, "index2item.npz"))["data"]
        model.weight = np.load(os.path.join(path, "weight.npz"))["data"]

        return model