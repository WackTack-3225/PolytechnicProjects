from kedro.pipeline import Pipeline, node

from .nodes import preprocessing_indiv, merge_data, preprocess_all, data_processing


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocessing_indiv,
                inputs=["customers", "items", "payment", 
                        "review", "order", "product"],
                outputs=["customers_df", "items_df", "payment_df", 
                        "review_df", "order_df", "product_df"],
                name="preprocess_individual_nodes",
            ),
            node(
                func=merge_data,
                inputs=["customers_df", "items_df", "payment_df", 
                        "review_df", "order_df", "product_df"],
                outputs="merged_df",
                name="merging_dataset_node",
            ),
            node(
                func=preprocess_all,
                inputs="merged_df",
                outputs="cleaned_df",
                name="feature_engineering_node",
            ),
            node(
                func=data_processing,
                inputs="cleaned_df",
                outputs="model_input_table",
                name="normalization_node",
            ),
        ]
    )
