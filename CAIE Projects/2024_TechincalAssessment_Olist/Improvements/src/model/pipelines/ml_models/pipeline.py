from kedro.pipeline import Pipeline, node

from .nodes import split_data, train_base_model, validate_model, fine_tuning, validate_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:data_split", 
                        "params:data_scaling"],
                outputs=["X_train_resampled", "y_train_resampled", 
                         "X_test", "y_test", 
                         "X_val", "y_val",
                         "step_1"],
                name="split_data_node",
            ),
            node(
                func=train_base_model,
                inputs=["X_train_resampled", "y_train_resampled", 
                        "params:ML_model", "step_1"],
                outputs=["regressor", "step_2"],
                name="train_base_node",
            ),
            node(
                func=validate_model,
                inputs=["regressor", "X_val", "y_val", "step_2"],
                outputs=["metrics", "step_3"],
                name="validate_base_model_node",
            ),
            node(
                func=fine_tuning,
                inputs=["X_train_resampled", "y_train_resampled", 
                        "params:ML_model", "params:ML_Finetuning", 
                        "step_3"],
                outputs=["regressor_fine", "step_4"],
                name="fine_tuning_node",
            ),
            node(
                func=validate_model,
                inputs=["regressor_fine", "X_val", "y_val", "step_4"],
                outputs=["metrics_fine", "step_5"],
                name="validation_finetuned_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "regressor_fine", 
                        "X_test", "y_test", "step_5"],
                outputs="metrics_final",
                name="evaluation_node",
            ),
        ]
    )