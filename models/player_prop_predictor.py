# models/player_prop_predictor.py
import pandas as pd
import joblib
import os
# We only need this function from the trainer now
from ml.model_trainer import _calculate_implied_probability, _create_lineup_aggregate_features


class PlayerPropPredictor:
    def __init__(self, db_manager):
        self.db = db_manager

        # --- REFACTORED PROPERTIES ---
        # Properties for NFL (remains a single model system)
        self.nfl_model, self.nfl_features, self.nfl_medians, self.nfl_scaler = None, None, None, None

        # Properties for MLB (now holds multiple quantile models)
        self.mlb_models = {}  # A dictionary to hold the q25, q50, q75 models
        self.mlb_features, self.mlb_medians, self.mlb_scaler = None, None, None

        # --- LOADING LOGIC ---
        # Load the standard NFL model assets
        self._load_nfl_assets()
        # Load the new MLB quantile model assets
        self._load_mlb_quantile_assets()

    def _load_nfl_assets(self):
        """Loads all assets for the standard NFL regression model."""
        model_name = 'nfl_passing_yards'
        print(f"--- Loading assets for {model_name} ---")
        base_path = 'trained_models/nfl_passing_yards_xgb'
        model_path = f"{base_path}_model.pkl"
        features_path = f"{base_path}_features.pkl"
        medians_path = f"{base_path}_medians.pkl"
        scaler_path = f"{base_path}_scaler.pkl"

        paths = [model_path, features_path, medians_path, scaler_path]
        if all(os.path.exists(p) for p in paths):
            self.nfl_model = joblib.load(model_path)
            self.nfl_features = joblib.load(features_path)
            self.nfl_medians = joblib.load(medians_path)
            self.nfl_scaler = joblib.load(scaler_path)
            print(f"{model_name.upper()} assets loaded successfully.")
        else:
            print(f"WARNING: One or more assets not found for {model_name}. Please train it first.")

    def _load_mlb_quantile_assets(self):
        """Loads all assets for the MLB quantile regression system."""
        model_name = 'mlb_pitcher_strikeouts'
        print(f"--- Loading QUANTILE assets for {model_name} ---")
        base_path = 'trained_models/mlb_pitcher_strikeouts_xgb'

        # Define paths for all required files
        model_paths = {
            q: f"{base_path}_q{int(q*100)}_model.pkl" for q in [0.25, 0.50, 0.75]
        }
        features_path = f"{base_path}_features.pkl"
        medians_path = f"{base_path}_medians.pkl"
        scaler_path = f"{base_path}_scaler.pkl"

        # Check if all files exist before attempting to load
        all_paths = list(model_paths.values()) + [features_path, medians_path, scaler_path]
        if not all(os.path.exists(p) for p in all_paths):
            print(f"WARNING: One or more assets not found for the {model_name} quantile system. Please train it first.")
            for p in all_paths:
                if not os.path.exists(p):
                    print(f"  Missing file: {p}")
            return

        # Load all assets if they exist
        try:
            for q, path in model_paths.items():
                self.mlb_models[q] = joblib.load(path)
            self.mlb_features = joblib.load(features_path)
            self.mlb_medians = joblib.load(medians_path)
            self.mlb_scaler = joblib.load(scaler_path)
            print(f"{model_name.upper()} QUANTILE assets loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading MLB assets: {e}")

    def _get_nfl_prediction_features(self, player_name, opponent_name, game_context):
        """Generates features for a single NFL prediction efficiently and correctly."""
        features = {}
        # Fetch recent games for the player
        player_history = self.db.fetch_recent_games_for_player(player_name, 'NFL', n=10)
        if player_history.empty:
            return None, f"No recent game history found for player '{player_name}'."
        player_history = player_history.sort_values(by='game_id').reset_index(drop=True)

        # Fetch recent games for the opponent's defense
        opponent_history = self.db.fetch_recent_games_for_team_as_opponent(opponent_name, 'NFL', n=40)
        if opponent_history.empty:
            return None, f"No recent defensive history for opponent '{opponent_name}'."
        opponent_history = opponent_history.sort_values(by='game_id').reset_index(drop=True)

        # Player features
        features['player_rolling_avg_pass_yds'] = player_history['passing_yards'].rolling(window=5, min_periods=1).mean().iloc[-1]
        features['player_rolling_avg_pass_attempts'] = player_history['passing_attempts'].rolling(window=5, min_periods=1).mean().iloc[-1]
        features['player_rolling_std_pass_yds'] = player_history['passing_yards'].rolling(window=5, min_periods=1).std()

        # Opponent features
        opponent_yards_allowed = opponent_history.groupby('game_id')['passing_yards'].sum()
        features['opponent_rolling_avg_pass_yds_allowed'] = opponent_yards_allowed.rolling(window=8, min_periods=1).mean().iloc[-1]

        # Game context features (assuming they are not in self.nfl_features if you removed them)
        features.update(game_context)

        final_df = pd.DataFrame([features])
        # Use the FULL feature list from the loaded file for ordering and selection
        return final_df[self.nfl_features], None

    def _get_mlb_prediction_features(self, player_name, opponent_name, game_context):
        """Generates features for a single MLB prediction."""
        # --- 1. Fetch historical data ---
        player_history = self.db.fetch_recent_pitching_logs_for_player(player_name, n=20)
        if player_history.empty or len(player_history) < 2:
            return None, f"Not enough game history for player '{player_name}'."

        player_history = player_history.sort_values(by='game_id').reset_index(drop=True)
        pitching_games = player_history[player_history['pitcher_innings_pitched'].astype(float) > 0].copy()
        if pitching_games.empty or len(pitching_games) < 2:
            return None, f"Not enough recent PITCHING history for '{player_name}'."

        # --- 2. Engineer Pitcher's Own Metrics (Features) ---
        pitcher_features = {
            'player_ewma_k_L5': pitching_games['pitcher_strikeouts'].ewm(span=5, adjust=False).mean().iloc[-1],
            'player_rolling_avg_k_L5': pitching_games['pitcher_strikeouts'].rolling(window=5, min_periods=2).mean().iloc[-1],
            'player_rolling_k_per_inning_L5': (pitching_games['pitcher_strikeouts'] / pitching_games['pitcher_innings_pitched']).rolling(window=5, min_periods=2).mean().iloc[-1],
            'player_rolling_whip_L5': pitching_games['pitcher_whip'].rolling(window=5, min_periods=2).mean().iloc[-1],
            'player_rolling_era_L5': pitching_games['pitcher_earned_run_average'].rolling(window=5, min_periods=2).mean().iloc[-1],
            'player_rolling_std_k_L10': pitching_games['pitcher_strikeouts'].rolling(window=10, min_periods=3).std().iloc[-1]
        }

        # --- 3. Engineer Opponent features ---
        opponent_history = self.db.fetch_recent_games_for_team_as_opponent(opponent_name, 'MLB', n=100)
        if opponent_history.empty:
            return None, f"No recent offensive history for opponent '{opponent_name}'."
        proxy_lineup = self.db.fetch_most_recent_lineup_for_team(opponent_name, 'MLB')
        if not proxy_lineup:
            return None, f"Could not generate a proxy lineup for '{opponent_name}'."
        lineup_stats_df = self.db.fetch_recent_stats_for_players(proxy_lineup)
        if lineup_stats_df.empty:
            return None, "Could not fetch stats for the generated proxy lineup."
        lineup_features = _create_lineup_aggregate_features(lineup_stats_df)

        # --- 4. Combine all features ---
        final_features = {}
        final_features.update(pitcher_features)
        final_features.update(lineup_features)
        final_features.update(game_context) # Add is_home_game, etc.

        final_df = pd.DataFrame([final_features])
        # Use the FULL feature list from the loaded file for ordering and selection
        return final_df[self.mlb_features], None

    def predict_pitcher_strikeouts_range(self, player_name, opponent_name, game_context, prizepicks_line):
        """
        Uses the three quantile models to predict a range and identify a betting signal.
        """
        print(f"\n--- Predicting MLB Pitcher Strikeouts RANGE for {player_name} vs {opponent_name} ---")

        # 1. Check if all models and assets are loaded
        if not self.mlb_models or len(self.mlb_models) < 3 or not self.mlb_scaler:
            return "Error: MLB Quantile Models or assets are not fully loaded.", None

        # 2. Get the feature DataFrame
        features_df, err = self._get_mlb_prediction_features(player_name, opponent_name, game_context)
        if err:
            return f"Could not generate features: {err}", None

        # 3. Apply medians for missing values and scale the features
        features_df_imputed = features_df.fillna(self.mlb_medians)
        features_scaled = self.mlb_scaler.transform(features_df_imputed[self.mlb_features])

        # 4. Get a prediction from EACH specialized quantile model
        pred_q25 = self.mlb_models[0.25].predict(features_scaled)[0]
        pred_q50 = self.mlb_models[0.50].predict(features_scaled)[0]
        pred_q75 = self.mlb_models[0.75].predict(features_scaled)[0]

        # 5. Apply the betting logic based on the "Zone of Uncertainty"
        print(f"\nPrizePicks Line: {prizepicks_line}")
        print(f"Model's Plausible Range: {pred_q25:.2f} (Low) - {pred_q75:.2f} (High)")
        print(f"Model's Median Prediction: {pred_q50:.2f}")

        choice = "NO BET"
        edge = 0

        if prizepicks_line < pred_q25:
            choice = "OVER"
            edge = pred_q25 - prizepicks_line
            print(f"  => BETTING SIGNAL: Strong OVER (Edge: {edge:.2f})")
        elif prizepicks_line > pred_q75:
            choice = "UNDER"
            edge = prizepicks_line - pred_q75
            print(f"  => BETTING SIGNAL: Strong UNDER (Edge: {edge:.2f})")
        else:
            print("  => NO BET: Line is within the model's plausible range.")

        return choice, edge

    def predict_passing_yards(self, player_name, opponent_name, game_context):
        """Predicts NFL passing yards using the single regression model."""
        print(f"\n--- Predicting NFL Passing Yards for {player_name} vs {opponent_name} ---")
        if not self.nfl_model or not self.nfl_scaler:
            return "Error: NFL Model or Scaler is not loaded.", None

        # Generate, impute, and scale features
        features_df, err = self._get_nfl_prediction_features(player_name, opponent_name, game_context)
        if err:
            return f"Could not generate features: {err}", None
        features_df_imputed = features_df.fillna(self.nfl_medians)
        features_scaled = self.nfl_scaler.transform(features_df_imputed[self.nfl_features])

        # Predict the single value
        prediction = self.nfl_model.predict(features_scaled)[0]

        print(f"  => Model Prediction: {prediction:.1f} passing yards.")
        # We compare this prediction to the sportsbook line to find the edge
        return None, prediction