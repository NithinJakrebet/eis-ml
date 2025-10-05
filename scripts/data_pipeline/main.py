from .test_train_split import test_train_split
from feature_engineering.main import build_model_input

def load_and_prepare_data(
    data_folder=None, 
    frequency_selection=None, 
    include_action_vector=True, 
    cycle_range=None
):
    df_train, df_test = test_train_split(data_folder=data_folder, cycle_range=cycle_range)
    
    X_train, y_train = build_model_input(
        df_train, 
        frequency_selection=frequency_selection, 
        include_action_vector=include_action_vector
    )
    
    X_test, y_test = build_model_input(
        df_test, 
        frequency_selection=frequency_selection, 
        include_action_vector=include_action_vector
    )
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"Train capacity range: {y_train.min():.1f} - {y_train.max():.1f} mAh")
    print(f"Test capacity range: {y_test.min():.1f} - {y_test.max():.1f} mAh")
    
    return X_train, X_test, y_train, y_test


