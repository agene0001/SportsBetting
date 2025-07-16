# ml/model_trainer.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler


def _calculate_implied_probability(moneyline):
    """Converts American odds to implied probability."""
    if moneyline is None or pd.isna(moneyline):
        return None
    moneyline = float(moneyline)
    if moneyline > 0:
        return 100 / (moneyline + 100)
    else:
        return (-moneyline) / (-moneyline + 100)

def _create_nfl_features(df):
    """
    Performs ADVANCED feature engineering for NFL passing yards.
    This version removes the unstable 'rank' feature.
    """
    print("  Creating advanced NFL features...")

    # --- 1. Clean and Prepare Data ---
    numeric_cols = [
        'passing_yards', 'passing_attempts', 'vegas_spread',
        'vegas_total', 'home_moneyline', 'away_moneyline'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['passing_yards', 'passing_attempts'])
    df = df.sort_values(by=['game_id', 'player_name']).reset_index(drop=True)

    # --- 2. Game Context Features ---
    print("    - Engineering game context features (odds, home/away)...")
    df['team_moneyline'] = df.apply(lambda row: row['home_moneyline'] if row['is_home_game'] == 1 else row['away_moneyline'], axis=1)
    df['opponent_moneyline'] = df.apply(lambda row: row['away_moneyline'] if row['is_home_game'] == 1 else row['home_moneyline'], axis=1)
    df['team_win_prob'] = df['team_moneyline'].apply(_calculate_implied_probability)
    df['team_spread'] = df.apply(lambda row: row['vegas_spread'] if row['is_home_game'] == 1 else -row['vegas_spread'], axis=1)

    # --- 3. Player's Own Rolling Performance Features ---
    print("    - Engineering player rolling performance features...")
    gb_player = df.groupby('player_name')
    df['player_rolling_avg_pass_yds'] = gb_player['passing_yards'].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df['player_rolling_avg_pass_attempts'] = gb_player['passing_attempts'].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
    df['player_rolling_std_pass_yds'] = gb_player['passing_yards'].transform(lambda x: x.rolling(window=5, min_periods=1).std().shift(1))

    # --- 4. Opponent's Defensive Performance Features ---
    print("    - Engineering opponent defensive performance features...")
    opponent_yards_allowed = df.groupby(['opponent', 'game_id'])['passing_yards'].sum().reset_index()
    opponent_yards_allowed = opponent_yards_allowed.rename(columns={'opponent': 'team', 'passing_yards': 'pass_yds_allowed'})
    opponent_yards_allowed = opponent_yards_allowed.sort_values(by=['game_id', 'team'])
    gb_opponent = opponent_yards_allowed.groupby('team')['pass_yds_allowed']
    opponent_yards_allowed['opponent_rolling_avg_pass_yds_allowed'] = gb_opponent.transform(lambda x: x.rolling(window=8, min_periods=1).mean().shift(1))

    df = pd.merge(
        df,
        opponent_yards_allowed[['team', 'game_id', 'opponent_rolling_avg_pass_yds_allowed']], # REMOVED rank from merge
        left_on=['opponent', 'game_id'],
        right_on=['team', 'game_id'],
        how='left'
    )

    # --- 5. Finalize for Modeling ---
    TARGET = 'passing_yards'
    # UPDATED FEATURES LIST
    FEATURES = [
        'is_home_game', 'vegas_total', 'team_win_prob', 'team_spread',
        'player_rolling_avg_pass_yds', 'player_rolling_avg_pass_attempts', 'player_rolling_std_pass_yds',
        'opponent_rolling_avg_pass_yds_allowed', # REMOVED: 'opponent_pass_def_rank'
    ]
    df = df.dropna(subset=FEATURES)
    # Fill any remaining NaNs (e.g., from std on single-game players) with the column median
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    if df.empty:
        return None, None, None, None

    X = df[FEATURES]
    y = df[TARGET]
    print(f"  Feature engineering complete. Dataset has {len(df)} samples with {len(FEATURES)} features.")
    return X, y, df, FEATURES


# In ml/model_trainer.py
# Make sure to import pandas as pd and numpy as np at the top of the file
# In ml/model_trainer.py

def _create_lineup_aggregate_features(lineup_df):
    """
    Takes a DataFrame of individual batter stats for a lineup and aggregates them
    into a set of powerful features without using handedness.

    Args:
        lineup_df (pd.DataFrame): A DataFrame where each row is a batter in the lineup
                                  and columns contain their rolling stats (e.g., K, AB, BB, HR).

    Returns:
        A dictionary of aggregated features for the entire lineup.
    """
    if lineup_df.empty or 'hitting_at_bats' not in lineup_df.columns or lineup_df['hitting_at_bats'].sum() == 0:
        # Return a dictionary of empty features if lineup is invalid
        return {
            'lineup_avg_k_rate': None, 'lineup_avg_walk_rate': None,
            'lineup_avg_hr_per_ab': None, 'lineup_high_k_batters_count': None,
            'lineup_low_k_batters_count': None, 'lineup_power_threats_count': None,
            'lineup_std_k_rate': None
        }

    # --- Calculate rates for each batter on the fly ---
    # Use an epsilon to avoid division by zero
    epsilon = 1e-6
    lineup_df['k_rate'] = lineup_df['hitting_strikeouts'] / (lineup_df['hitting_at_bats'] + epsilon)
    lineup_df['walk_rate'] = lineup_df['hitting_bases_on_balls'] / (lineup_df['hitting_at_bats'] + epsilon)
    lineup_df['hr_per_ab'] = lineup_df['hitting_home_runs'] / (lineup_df['hitting_at_bats'] + epsilon)

    # --- Aggregate the entire lineup into single features ---
    features = {}

    # 1. Overall Lineup Averages (The "personality" of the lineup)
    features['lineup_avg_k_rate'] = lineup_df['k_rate'].mean()
    features['lineup_avg_walk_rate'] = lineup_df['walk_rate'].mean()
    features['lineup_avg_hr_per_ab'] = lineup_df['hr_per_ab'].mean()

    # 2. Quantify Strengths and Weaknesses (More telling than just an average)
    # How many batters are "easy" strikeout targets? (e.g., K-Rate > 25%)
    features['lineup_high_k_batters_count'] = lineup_df[lineup_df['k_rate'] > 0.25].shape[0]
    # How many batters are "tough outs" with great discipline? (e.g., K-Rate < 15%)
    features['lineup_low_k_batters_count'] = lineup_df[lineup_df['k_rate'] < 0.15].shape[0]
    # How many true power threats are there? (e.g., 1 HR every 20 ABs)
    features['lineup_power_threats_count'] = lineup_df[lineup_df['hr_per_ab'] > 0.05].shape[0]

    # 3. Standard Deviation (Is it a consistent lineup or "stars and scrubs"?)
    features['lineup_std_k_rate'] = lineup_df['k_rate'].std()

    return features

# In ml/model_trainer.py

def _create_pitcher_strikeouts_features(df, db_manager, start_year, end_year):
    """
    Performs ADVANCED feature engineering for MLB pitcher strikeouts.
    *** V6: Includes data type standardization and debugging to ensure correct data merging. ***
    """
    print("  Creating V6 features for Pitcher Strikeouts (Standardized & Debug)...")

    # --- 1. Clean and Prepare Data ---
    numeric_cols = [
        'pitcher_strikeouts', 'pitcher_innings_pitched', 'pitcher_bases_on_balls', 'pitcher_whip',
        'pitcher_earned_run_average', 'pitcher_hits', 'pitcher_home_runs',
        'hitting_strikeouts', 'hitting_at_bats', 'hitting_bases_on_balls',
        'vegas_total', 'home_moneyline', 'away_moneyline'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['pitcher_strikeouts', 'pitcher_innings_pitched'])
    df = df[df['pitcher_innings_pitched'] > 0].copy()
    df = df.sort_values(by=['game_id', 'player_name']).reset_index(drop=True)

    # --- 2. Engineer Pitcher's Own Efficiency Metrics ---
    print("    - Engineering pitcher efficiency metrics (K/IP, K/BB, WHIP)...")
    df['player_k_per_inning'] = df['pitcher_strikeouts'] / df['pitcher_innings_pitched']
    df['player_k_to_bb_ratio'] = df['pitcher_strikeouts'] / (df['pitcher_bases_on_balls'] + 1e-6)

    # --- 3. Engineer Pitcher's Rolling Performance ---
    print("    - Engineering pitcher rolling performance features...")
    gb_player = df.groupby('player_name')

    df['player_ewma_k_L5'] = gb_player['pitcher_strikeouts'].transform(lambda x: x.ewm(span=5, adjust=False).mean().shift(1))
    df['player_rolling_avg_k_L5'] = gb_player['pitcher_strikeouts'].transform(lambda x: x.rolling(window=5, min_periods=2).mean().shift(1))
    df['player_rolling_k_per_inning_L5'] = gb_player['player_k_per_inning'].transform(lambda x: x.rolling(window=5, min_periods=2).mean().shift(1))
    df['player_rolling_k_to_bb_L10'] = gb_player['player_k_to_bb_ratio'].transform(lambda x: x.rolling(window=10, min_periods=3).mean().shift(1))
    df['player_rolling_whip_L5'] = gb_player['pitcher_whip'].transform(lambda x: x.rolling(window=5, min_periods=2).mean().shift(1))
    df['player_rolling_era_L5'] = gb_player['pitcher_earned_run_average'].transform(lambda x: x.rolling(window=5, min_periods=2).mean().shift(1))
    df['player_rolling_std_k_L10'] = gb_player['pitcher_strikeouts'].transform(lambda x: x.rolling(window=10, min_periods=3).std().shift(1))

    # --- 4. Engineer OPPONENT features (OPTIMIZED APPROACH) ---
    print("    - Performing bulk fetch of all historical batting data...")
    all_batting_stats_df = db_manager.fetch_all_batting_stats_for_seasons(start_year, end_year)

    if all_batting_stats_df.empty:
        print("  WARNING: Bulk fetch returned no batting data. Opponent features will be empty.")
        lineup_feature_names = [
            'lineup_avg_k_rate', 'lineup_avg_walk_rate', 'lineup_avg_hr_per_ab',
            'lineup_high_k_batters_count', 'lineup_low_k_batters_count',
            'lineup_power_threats_count', 'lineup_std_k_rate'
        ]
        for col in lineup_feature_names:
            df[col] = np.nan
    else:
        # --- FIX: Standardize data types to ensure keys match ---
        print("    - Standardizing data types for lookup keys...")

        # Clean keys in the main DataFrame
        df['game_id'] = pd.to_numeric(df['game_id'], errors='coerce')
        df['opponent_team_id'] = pd.to_numeric(df['opponent_team_id'], errors='coerce')
        df = df.dropna(subset=['game_id', 'opponent_team_id'])
        df['opponent_team_id'] = df['opponent_team_id'].astype(int)

        # Clean keys in the batting data lookup table
        all_batting_stats_df['source_game_id'] = pd.to_numeric(all_batting_stats_df['source_game_id'], errors='coerce')
        all_batting_stats_df['team_id'] = pd.to_numeric(all_batting_stats_df['team_id'], errors='coerce')
        all_batting_stats_df = all_batting_stats_df.dropna(subset=['source_game_id', 'team_id'])
        all_batting_stats_df['source_game_id'] = all_batting_stats_df['source_game_id'].astype(int)
        all_batting_stats_df['team_id'] = all_batting_stats_df['team_id'].astype(int)

        print("    - Pre-grouping lineup data for fast lookups...")
        lineup_groups = dict(tuple(all_batting_stats_df.groupby(['source_game_id', 'team_id'])))

        print("    - Generating lineup features from in-memory data...")

        def get_lineup_features_for_game(game_row):
            # The keys are now guaranteed to be integers and will match correctly
            lookup_key = (game_row['game_id'], game_row['opponent_team_id'])
            lineup_df = lineup_groups.get(lookup_key, pd.DataFrame())
            return pd.Series(_create_lineup_aggregate_features(lineup_df))

        lineup_features_df = df.apply(get_lineup_features_for_game, axis=1)

        # Diagnostic print to confirm the fix
        successful_lookups = lineup_features_df['lineup_avg_k_rate'].notna().sum()
        print(f"    - DIAGNOSTIC: Successfully found and processed {successful_lookups} lineups out of {len(df)} games.")

        df = pd.concat([df, lineup_features_df], axis=1)

    # --- 5. Finalize for Modeling ---
    df['strikeout_deviation'] = df['pitcher_strikeouts'] - df['player_rolling_avg_k_L5']
    TARGET = 'pitcher_strikeouts'

    FEATURES = [
        'player_rolling_avg_k_L5', 'player_ewma_k_L5', 'player_rolling_k_per_inning_L5',
        'player_rolling_whip_L5', 'player_rolling_std_k_L10', 'player_rolling_era_L5',
        'is_home_game',
        'lineup_avg_k_rate', 'lineup_avg_walk_rate', 'lineup_avg_hr_per_ab',
        'lineup_high_k_batters_count', 'lineup_low_k_batters_count',
        'lineup_power_threats_count', 'lineup_std_k_rate'
    ]

    df = df.dropna(subset=FEATURES + [TARGET])

    if df.empty:
        print("ERROR: DataFrame is empty after creating features and dropping NaNs. Check data quality or feature logic.")
        return None, None, None, None

    X = df[FEATURES]
    y = df[TARGET]

    return X, y, df, FEATURES
# ... (keep all your existing imports and functions) ...

def run_hyperparameter_tuning(X_train, y_train, model_name,quantile):
    """
    Performs hyperparameter tuning using GridSearchCV with time-series cross-validation.

    Args:
        X_train (pd.DataFrame): The training feature data.
        y_train (pd.Series): The training target data.
        model_name (str): The name of the model for logging purposes.
        quantile (int): quantile to train on
    Returns:
        An already-trained XGBoost model with the best found hyperparameters.
    """
    print(f"\n--- Starting Hyperparameter Tuning for {model_name} with GridSearchCV ---")

    # 1. Define the grid of hyperparameters to search.
    # Start with a smaller grid to make it faster, then you can expand it.
    param_grid = {
        'n_estimators': [100, 250, 500, 750],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [1, 1.5, 2, 3],
        'gamma': [0, 0.1, 0.5, 1],
        'min_child_weight': [1, 5, 10, 15]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    xgb_model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile, random_state=42, n_jobs=-1)

    # --- Use RandomizedSearchCV instead of GridSearchCV ---
    grid_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid, # Note: param_distributions, not param_grid
        n_iter=100, # Number of parameter settings that are sampled. 100 is a good starting point.
        scoring='neg_mean_absolute_error', # Changed scoring (see next section)
        cv=tscv,
        verbose=1,
        random_state=42, # for reproducibility of the random search
        n_jobs=-1
    )

    # 5. Run the search on the training data.
    print(f"  Running grid search on {len(X_train)} training samples... This may take a while.")
    grid_search.fit(X_train, y_train)

    # 6. Report the best findings and return the best model.
    print("\n--- Tuning Complete ---")
    print(f"  Best Hyperparameters Found: {grid_search.best_params_}")

    # GridSearchCV automatically retrains the best model on the entire training set (X_train, y_train)
    # so we can directly return the best_estimator_.
    return grid_search.best_estimator_

def train_model(db_manager, sport_name, start_year, end_year, feature_func, data_filter_col, model_name, model_type='xgb'):
    """
    Generic model training function with feature scaling.
    """
    print(f"\n--- Starting Model Training for {model_name} ({model_type.upper()}) ---")
    print(f"Fetching {sport_name} data from {start_year} to {end_year} seasons...")

    all_seasons_df = [db_manager.fetch_player_game_logs_for_season(sport_name, year) for year in range(start_year, end_year + 1)]
    all_seasons_df = [df for df in all_seasons_df if not df.empty]
    if not all_seasons_df:
        print("No data found for the specified years. Aborting training.")
        return

    full_df = pd.concat(all_seasons_df, ignore_index=True)
    full_df[data_filter_col] = pd.to_numeric(full_df[data_filter_col], errors='coerce').fillna(0)
    filtered_df = full_df[full_df[data_filter_col] > 0].copy()

    if filtered_df.empty:
        print(f"No data found after filtering on '{data_filter_col}'. Aborting training.")
        return

    X, y, processed_df, features_list = feature_func(filtered_df, db_manager, start_year, end_year)
    if X is None or X.empty or y.empty:
        print("Could not create features or features/target are empty. Aborting training.")
        return

    # Time-based split
    if 'season' not in X.columns and 'season' in processed_df.columns:
        X_with_season = X.join(processed_df.loc[X.index, 'season'])
    else:
        X_with_season = X.copy()

    train_season_end = end_year - 1
    train_indices = X_with_season[X_with_season['season'] <= train_season_end].index
    test_indices = X_with_season[X_with_season['season'] > train_season_end].index

    if len(train_indices) == 0 or len(test_indices) == 0:
        print(f"ERROR: Not enough seasonal data to perform a train/test split. Aborting.")
        return

    X_train_raw = X_with_season.loc[train_indices, features_list]
    y_train = y.loc[train_indices]
    X_test_raw = X_with_season.loc[test_indices, features_list]
    y_test = y.loc[test_indices]

    # --- MEDIAN IMPUTATION (before scaling) ---
    print("  Calculating medians from the TRAINING set...")
    feature_medians = X_train_raw.median().to_dict()

    print("  Imputing NaNs in train and test sets using training medians...")
    X_train_imputed = X_train_raw.fillna(feature_medians)
    X_test_imputed = X_test_raw.fillna(feature_medians)

    if X_train_imputed.isnull().sum().sum() > 0 or X_test_imputed.isnull().sum().sum() > 0:
        print("WARNING: NaNs still exist after median imputation. Check for non-numeric features.")

    # --- NEW: FEATURE SCALING ---
    print("  Scaling features using StandardScaler...")
    scaler = StandardScaler()

    # Fit the scaler ONLY on the training data to learn the mean and standard deviation
    # Then transform the training data
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    # Apply the SAME transformation (using the mean/std from training) to the test data
    X_test_scaled = scaler.transform(X_test_imputed)

    # The scaler returns a numpy array, so convert it back to a DataFrame to keep column names
    X_train = pd.DataFrame(X_train_scaled, index=X_train_imputed.index, columns=features_list)
    X_test = pd.DataFrame(X_test_scaled, index=X_test_imputed.index, columns=features_list)
    # --- END NEW SECTION ---

    print(f"  Training on {len(X_train)} samples (Seasons <= {train_season_end})")
    print(f"  Testing on {len(X_test)} samples (Seasons > {train_season_end})")

    print(f"  Training {model_type.upper()} model...")
    # Pass the scaled data to the tuning function
    quantiles = [0.25, 0.50, 0.75]
    trained_models = {} # Use a dictionary to hold the models

    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')

    for q in quantiles:
        print(f"\n===== Training for Quantile: {q} =====")
        # Train a model specifically for this quantile
        model = run_hyperparameter_tuning(X_train, y_train, model_name, quantile=q)
        trained_models[q] = model

        # Save the model to a UNIQUE path based on its quantile
        model_path = f'trained_models/{model_name}_{model_type}_q{int(q*100)}_model.pkl'
        joblib.dump(model, model_path)
        print(f"  Model for quantile {q} saved to {model_path}")

    # --- SAVE OTHER ARTIFACTS (only need to do this once) ---
    features_path = f'trained_models/{model_name}_{model_type}_features.pkl'
    medians_path = f'trained_models/{model_name}_{model_type}_medians.pkl'
    scaler_path = f'trained_models/{model_name}_{model_type}_scaler.pkl'

    joblib.dump(features_list, features_path)
    joblib.dump(feature_medians, medians_path)
    joblib.dump(scaler, scaler_path)

    print("\n--- All training artifacts saved successfully. ---")
    print(f"  Features list saved to {features_path}")
    print(f"  Feature medians saved to {medians_path}")
    print(f"  Scaler saved to {scaler_path}")

    # --- OPTIONAL: Evaluate the final models ---
    print("\n--- Evaluating Quantile Models on Test Set ---")
    for q, model in trained_models.items():
        predictions = model.predict(X_test)
        # For quantile regression, a simple RMSE isn't the best metric, but it gives a general idea.
        # A better metric is the pinball loss, but we can stick to RMSE for now.
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"  RMSE for quantile {q} model: {rmse:.2f}")
def train_passing_yards_model(db_manager, start_year=2020, end_year=2023, model_choice='xgb'):
    """Trains the NFL passing yards model."""
    train_model(
        db_manager,
        sport_name='NFL',
        start_year=start_year,
        end_year=end_year,
        feature_func=_create_nfl_features,
        data_filter_col='passing_attempts',
        model_name='nfl_passing_yards',
        model_type=model_choice
    )

def train_pitcher_strikeouts_model(db_manager, start_year=2021, end_year=2023, model_choice='xgb'):
    """Trains the MLB pitcher strikeouts model."""
    train_model(
        db_manager,
        sport_name='MLB',
        start_year=start_year,
        end_year=end_year,
        feature_func=_create_pitcher_strikeouts_features,
        data_filter_col='pitcher_innings_pitched',
        model_name='mlb_pitcher_strikeouts',
        model_type=model_choice
    )