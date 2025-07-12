# ml/model_trainer.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import joblib

from database.db_manager import DatabaseManager

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

def _create_pitcher_strikeouts_features(df):
    """
    Performs ADVANCED feature engineering for MLB pitcher strikeouts using a rich dataset.
    This version focuses on efficiency, control, and specific opponent matchup metrics.

    KEY DATA ASSUMPTIONS (based on your schema):
    - df contains 'pitcher_strikeouts', 'pitcher_innings_pitched', 'pitcher_bases_on_balls', 'pitcher_whip'
    - df contains opponent batting stats: 'hitting_strikeouts', 'hitting_at_bats', 'hitting_bases_on_balls'
    """
    print("  Creating V2 features for Pitcher Strikeouts (Efficiency & Matchup)...")

    # --- 1. Clean and Prepare Data ---
    # Convert all potential numeric columns to numbers, coercing errors
    # This list now includes all the new stats from your schema
    numeric_cols = [
        # Pitcher Stats
        'pitcher_strikeouts', 'pitcher_innings_pitched', 'pitcher_bases_on_balls', 'pitcher_whip',
        'pitcher_earned_run_average', 'pitcher_hits', 'pitcher_home_runs',
        # Hitting Stats
        'hitting_strikeouts', 'hitting_at_bats', 'hitting_bases_on_balls',
        # Vegas
        'vegas_total', 'home_moneyline', 'away_moneyline'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure we have the necessary data to calculate rates, drop games with missing key info
    df = df.dropna(subset=['pitcher_strikeouts', 'pitcher_innings_pitched'])
    df = df[df['pitcher_innings_pitched'] > 0].copy() # Avoid division by zero
    df = df.sort_values(by=['game_id', 'player_name']).reset_index(drop=True)

    # --- 2. Engineer Pitcher's Own Efficiency Metrics ---
    print("    - Engineering pitcher efficiency metrics (K/IP, K/BB, WHIP)...")
    # Using a small epsilon to avoid division by zero for pitchers with 0 walks
    df['player_k_per_inning'] = df['pitcher_strikeouts'] / df['pitcher_innings_pitched']
    df['player_k_to_bb_ratio'] = df['pitcher_strikeouts'] / (df['pitcher_bases_on_balls'] + 1e-6)

    # --- 3. Engineer Pitcher's Rolling Performance ---
    print("    - Engineering pitcher rolling performance features...")
    gb_player = df.groupby('player_name')

    # Baseline for our target variable
    df['player_rolling_avg_k_L5'] = gb_player['pitcher_strikeouts'].transform(lambda x: x.rolling(window=5, min_periods=2).mean().shift(1))

    # Rolling efficiency and control metrics
    df['player_rolling_k_per_inning_L5'] = gb_player['player_k_per_inning'].transform(lambda x: x.rolling(window=5, min_periods=2).mean().shift(1))
    df['player_rolling_k_to_bb_L10'] = gb_player['player_k_to_bb_ratio'].transform(lambda x: x.rolling(window=10, min_periods=3).mean().shift(1))
    df['player_rolling_whip_L5'] = gb_player['pitcher_whip'].transform(lambda x: x.rolling(window=5, min_periods=2).mean().shift(1))
    df['player_rolling_era_L5'] = gb_player['pitcher_earned_run_average'].transform(lambda x: x.rolling(window=5, min_periods=2).mean().shift(1))
    df['matchup_k_potential'] = df['player_rolling_k_per_inning_L5'] * df['opponent_rolling_k_rate']
    df['player_rolling_std_k_L10'] = gb_player['pitcher_strikeouts'].transform(lambda x: x.rolling(window=10, min_periods=3).std().shift(1))

    # --- 4. Engineer Opponent's True Offensive Profile ---
    print("    - Engineering opponent offensive profile (K-Rate, Walk-Rate)...")
    # First, calculate team-level offensive stats for each game
    team_offense_per_game = df.groupby(['opponent', 'game_id']).agg(
        team_k_total=('hitting_strikeouts', 'sum'), # CORRECT: Use what batters actually did
        team_bb_total=('hitting_bases_on_balls', 'sum'), # CORRECT: Use what batters actually did
        team_ab_total=('hitting_at_bats', 'sum')
    ).reset_index()

    # Calculate rates
    team_offense_per_game['team_k_rate'] = team_offense_per_game['team_k_total'] / (team_offense_per_game['team_ab_total'] + 1e-6)
    team_offense_per_game['team_bb_rate'] = team_offense_per_game['team_bb_total'] / (team_offense_per_game['team_ab_total'] + 1e-6)

    # Now, calculate the rolling average of these rates for the opponent
    team_offense_per_game = team_offense_per_game.sort_values(by=['game_id', 'opponent'])
    gb_opponent = team_offense_per_game.groupby('opponent')

    team_offense_per_game['opponent_rolling_k_rate'] = gb_opponent['team_k_rate'].transform(lambda x: x.rolling(window=15, min_periods=5).mean().shift(1))
    team_offense_per_game['opponent_rolling_bb_rate'] = gb_opponent['team_bb_rate'].transform(lambda x: x.rolling(window=15, min_periods=5).mean().shift(1))

    # Merge these advanced opponent stats back into the main dataframe
    df = pd.merge(
        df,
        team_offense_per_game[['opponent', 'game_id', 'opponent_rolling_k_rate', 'opponent_rolling_bb_rate']],
        on=['opponent', 'game_id'],
        how='left'
    )

    # --- 5. Finalize for Modeling ---

    # !!! NEW TARGET VARIABLE REMAINS THE SAME: DEVIATION FROM BASELINE !!!
    df['strikeout_deviation'] = df['pitcher_strikeouts'] - df['player_rolling_avg_k_L5']
    TARGET = 'strikeout_deviation'

    # !!! THE NEW, MORE POWERFUL FEATURE SET !!!
    FEATURES = [
        'is_home_game',
        'player_rolling_avg_k_L5',        # Player's baseline K performance
        'player_rolling_k_per_inning_L5', # Player's K efficiency
        'player_rolling_whip_L5',         # Player's overall effectiveness
        'player_rolling_era_L5',          # Player's overall quality
        'opponent_rolling_k_rate',        # Matchup: Opponent's TRUE K weakness
        'opponent_rolling_bb_rate',       # Matchup: Opponent's TRUE plate discipline
        'matchup_k_potential',
        'player_rolling_std_k_L10',
    ]

    # Drop rows where we can't calculate the target or key features
    df = df.dropna(subset=[TARGET] + FEATURES)

    # Fill any remaining NaNs with the column median (robust to outliers)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

    if df.empty:
        return None, None, None, None

    X = df[FEATURES]
    y = df[TARGET]

    print(f"  V2 Feature engineering complete. Target: '{TARGET}'. Dataset has {len(df)} samples with {len(FEATURES)} features.")

    return X, y, df, FEATURES

def train_model(db_manager, sport_name, start_year, end_year, feature_func, data_filter_col, model_name, model_type='xgb'):
    """Generic model training function."""
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

    X, y, processed_df, features_list = feature_func(filtered_df)
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

    # Add robustness check for train/test split
    if len(train_indices) == 0 or len(test_indices) == 0:
        print(f"ERROR: Not enough seasonal data to perform a train/test split.")
        print(f"Training requires data from seasons <= {train_season_end}.")
        print(f"Testing requires data from seasons > {train_season_end}.")
        print(f"Available seasons in dataset: {sorted(X_with_season['season'].unique())}")
        print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}. Aborting.")
        return

    X_train_raw = X_with_season.loc[train_indices, features_list]
    y_train = y.loc[train_indices]
    X_test_raw = X_with_season.loc[test_indices, features_list]
    y_test = y.loc[test_indices]
    # --- NEW: MEDIAN CALCULATION AND IMPUTATION ---
    print("  Calculating medians from the TRAINING set...")
    # Calculate medians ONLY from the training data to prevent data leakage
    feature_medians = X_train_raw.median().to_dict()

    print("  Imputing NaNs in train and test sets using training medians...")
    X_train = X_train_raw.fillna(feature_medians)
    X_test = X_test_raw.fillna(feature_medians)

    # Check if any NaNs remain after filling (should not happen if all features are numeric)
    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        print("WARNING: NaNs still exist after median imputation. Check for non-numeric features.")
        print("NaNs in training set:\n", X_train.isnull().sum())
        print("NaNs in test set:\n", X_test.isnull().sum())
    # --- END NEW SECTION ---
    print(f"  Training on {len(X_train)} samples (Seasons <= {train_season_end})")
    print(f"  Testing on {len(X_test)} samples (Seasons > {train_season_end})")

    print(f"  Training {model_type.upper()} model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"  Model training complete. Test Set RMSE: {rmse:.2f}")

    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
    model_path = f'trained_models/{model_name}_{model_type}_model.pkl'
    features_path = f'trained_models/{model_name}_{model_type}_features.pkl'
    medians_path = f'trained_models/{model_name}_{model_type}_medians.pkl' # New file path

    joblib.dump(model, model_path)
    joblib.dump(features_list, features_path)
    joblib.dump(feature_medians, medians_path) # Save the dictionary

    print(f"  Model saved to {model_path}")
    print(f"  Features list saved to {features_path}")
    print(f"  Feature medians saved to {medians_path}") # Log the save

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