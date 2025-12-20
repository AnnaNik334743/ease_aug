import polars as pl
import datetime as dt
import numpy as np
from tqdm import tqdm


class OrdersDataLoader:
    def __init__(self):
        self.items = None
    
    def load_items_metadata(self) -> pl.DataFrame:
        ...

        print(f"loaded {self.items.shape[0]} items")
        return self.items
    
    def load_orders(
        self, 
        date_from: dt.date, 
        date_to: dt.date, 
        item_ids: list[str] | None = None
    ) -> pl.DataFrame:
        ...
        
        print(f"loaded {orders.shape[0]} orders")
        return orders

    @staticmethod
    def train_val_test_split(
        orders: pl.DataFrame, 
        train_proportion: float = 0.7, 
        val_proportion: float = 0.1, 
        test_proportion: float = 0.2,
        k_core_min: int = 2,
        k_core_max: int = 50,
        min_orders_per_good: int = 1,
        keep_only_warm_goods: bool = False
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        assert train_proportion + val_proportion + test_proportion == 1
        
        orders = (
            orders
            .join(
                (
                    orders    
                    .group_by("item_id")
                    .agg(num_orders=pl.col("order_id").n_unique())
                    .filter(pl.col("num_orders") >= min_orders_per_good)
                    .select("item_id")
                ),
                how="inner",
                on="item_id",
            )
            .group_by("order_id").agg(
                num_goods=pl.col("item_id").n_unique(), 
                item_id=pl.col("item_id").unique(), 
                ts=pl.col("ts").mean()
            )
            .filter((pl.col("num_goods") > k_core_min) & (pl.col("num_goods") < k_core_max))
            .sort("ts")
            .select("order_id", "item_id")
        )
        
        orders_len = len(orders)
        train_size = int(train_proportion * orders_len)
        val_size = int(val_proportion * orders_len)
        test_size = orders_len - train_size - val_size
        
        train_orders = orders[:train_size]
        val_orders = orders[train_size:train_size + val_size]
        test_orders = orders[train_size + val_size:]
        
        if keep_only_warm_goods:
            warm_goods = train_orders.explode("item_id")["item_id"].unique().to_list()
            
            val_orders = (
                val_orders.explode("item_id")
                .filter(pl.col("item_id").is_in(warm_goods))
                .group_by("order_id").agg(
                    num_goods=pl.col("item_id").n_unique(), 
                    item_id=pl.col("item_id").unique()
                )
                .filter((pl.col("num_goods") > k_core_min))
                .select("order_id", "item_id")
            )
            
            test_orders = (
                test_orders.explode("item_id")
                .filter(pl.col("item_id").is_in(warm_goods))
                .group_by("order_id").agg(
                    num_goods=pl.col("item_id").n_unique(), 
                    item_id=pl.col("item_id").unique()
                )
                .filter((pl.col("num_goods") > k_core_min))
                .select("order_id", "item_id")
            )
        
        print(f"train size = {len(train_orders)} ({(len(train_orders) / (len(train_orders) + len(val_orders) + len(test_orders))):.2f}%),",
              f"val size = {len(val_orders)} ({(len(val_orders) / (len(train_orders) + len(val_orders) + len(test_orders))):.2f}%),",
              f"test size = {len(test_orders)} ({(len(test_orders) / (len(train_orders) + len(val_orders) + len(test_orders))):.2f}%)")

        return train_orders, val_orders, test_orders

    @staticmethod
    def distribute_orders_to_pred_and_target(
        orders: pl.DataFrame, 
        masking_prob: float = 0.2
    ) -> pl.DataFrame:
        new_order_id = []
        new_item_id_for_pred = []
        new_item_id_target = []
        initial_item_id = []
        
        for order_id, item_id_arr in tqdm(orders.iter_rows(), total=orders.shape[0]):
            if len(item_id_arr) < 2:
                # see k_core_min in train_val_test_split
                continue
        
            item_id_arr = np.array(item_id_arr)
            masking_probs = np.random.uniform(size=len(item_id_arr))
            mask = (masking_probs < masking_prob)
        
            if sum(mask) == 0:
                # at least one has to go to target
                mask[np.argmin(masking_probs)] = True
        
            if sum(~mask) == 0:
                # at least one has to stay for prediction
                # (stronger restriction than for target)
                mask[np.argmax(masking_probs)] = False 
        
            new_order_id.append(order_id)
            new_item_id_for_pred.append(item_id_arr[~mask])
            new_item_id_target.append(item_id_arr[mask])
            initial_item_id.append(item_id_arr)
        
        return pl.DataFrame({
            "order_id": new_order_id, 
            "item_id": initial_item_id,
            "item_id_for_pred": new_item_id_for_pred, 
            "item_id_target": new_item_id_target,
        })

    @staticmethod
    def leave_only_rare_items(
        orders_df_to_extract_rare_items: pl.DataFrame, 
        orders_dfs_to_leave_rare_items: list[pl.DataFrame], 
        rare_items_count_threshold: int = 3,  # occures less than
        return_items_counts: bool = True
    ) -> tuple[set[str], list[pl.DataFrame], pl.DataFrame] | tuple[set[str], list[pl.DataFrame]]:
        item_counts = (
            orders_df_to_extract_rare_items
            .explode("item_id")["item_id"]
            .value_counts()
        )        
        rare_items = set(item_counts.filter(pl.col("count") < rare_items_count_threshold)["item_id"].to_list())
        print(f"got {len(rare_items)} rare items")

        orders_dfs_to_leave_rare_items_resulting = []
        
        for i, orders_df in enumerate(orders_dfs_to_leave_rare_items):
            
            orders_df_with_rare_items = set(
                orders_df.explode("item_id").filter(pl.col("item_id").is_in(rare_items))["order_id"].to_list()
            )
            orders_rare = orders_df.filter(pl.col("order_id").is_in(orders_df_with_rare_items))
            
            item_id_for_pred = []
            item_id_target = []
            
            for row in tqdm(orders_rare.iter_rows(named=True), total=len(orders_rare), desc=f"orders_df {i}"):
                item_id = row["item_id"]
                curr_item_id_for_pred = []
                curr_item_id_target = []
                
                for iid in item_id:
                    if iid in rare_items:
                        curr_item_id_for_pred.append(iid)
                    else:
                        curr_item_id_target.append(iid)
                        
                item_id_for_pred.append(curr_item_id_for_pred)
                item_id_target.append(curr_item_id_target)
                
            orders_rare = orders_rare.with_columns(
                item_id_for_pred=pl.Series(item_id_for_pred),
                item_id_target=pl.Series(item_id_target)
            )

            orders_dfs_to_leave_rare_items_resulting.append(orders_rare)
            print(f"orders_df {i} size after leaving rare items only = {len(orders_rare)}")

        if return_items_counts:
            return rare_items, orders_dfs_to_leave_rare_items_resulting, item_counts
        return rare_items, orders_dfs_to_leave_rare_items_resulting

    def cast_item_ids_to_names(self, orders: pl.DataFrame):
        orders_casted = orders.clone()
        for col in orders_casted.columns:
            if col == "order_id":
                continue
            orders_casted = orders_casted.with_columns(
                (
                    orders_casted.select("order_id", col).explode(col)
                    .join(self.items, left_on=col, right_on="item_id", how="left")
                    .group_by("order_id", maintain_order=True).agg(pl.col("name").alias(col))
                )[col]
            )
        return orders_casted
