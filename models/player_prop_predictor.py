# models/player_prop_predictor.py
import pandas as pd
import joblib
import os
# We only need this function from the trainer now
from ml.model_trainer import _calculate_implied_probability

class PlayerPropPredictor:
    def __init__(self, db_manager):
        self.db = db_manager
        # Initialize properties
        self.nfl_model, self.nfl_features, self.nfl_medians = None, None, None
        self.mlb_model, self.mlb_features, self.mlb_medians = None, None, None

        # Load NFL Model and its assets
        self._load_model_assets(
            'nfl', 'nfl_passing_yards',
            'trained_models/nfl_passing_yards_xgb_model.pkl',
            'trained_models/nfl_passing_yards_xgb_features.pkl',
            'trained_models/nfl_passing_yards_xgb_medians.pkl'
        )

        # Load MLB Model and its assets
        self._load_model_assets(
            'mlb', 'mlb_pitcher_strikeouts',
            'trained_models/mlb_pitcher_strikeouts_xgb_model.pkl',
            'trained_models/mlb_pitcher_strikeouts_xgb_features.pkl',
            'trained_models/mlb_pitcher_strikeouts_xgb_medians.pkl'
        )

    def _load_model_assets(self, sport_attr, model_name, model_path, features_path, medians_path):
        """Helper function to load model, features, and medians."""
        print(f"--- Loading assets for {model_name} ---")
        if os.path.exists(model_path) and os.path.exists(features_path) and os.path.exists(medians_path):
            setattr(self, f"{sport_attr}_model", joblib.load(model_path))
            setattr(self, f"{sport_attr}_features", joblib.load(features_path))
            setattr(self, f"{sport_attr}_medians", joblib.load(medians_path))
            print(f"{model_name.upper()} assets loaded successfully.")
        else:
            print(f"WARNING: One or more assets not found for {model_name}. Please train it first.")
            print(f"  Model Check: {'Found' if os.path.exists(model_path) else 'Missing'}")
            print(f"  Features Check: {'Found' if os.path.exists(features_path) else 'Missing'}")
            print(f"  Medians Check: {'Found' if os.path.exists(medians_path) else 'Missing'}")

    def _get_nfl_prediction_features(self, player_name, opponent_name, game_context):
        """
        Generates features for a single NFL prediction efficiently and correctly.
        """
        # 1. Fetch recent games for the player
        player_history = self.db.fetch_recent_games_for_player(player_name, 'NFL', n=10)
        if player_history.empty:
            return None, f"No recent game history found for player '{player_name}'."
        # Sort chronologically to ensure rolling stats are correct
        player_history = player_history.sort_values(by='game_id').reset_index(drop=True)

        # 2. Fetch recent games for the opponent's defense
        opponent_history = self.db.fetch_recent_games_for_team_as_opponent(opponent_name, 'NFL', n=40)
        if opponent_history.empty:
            return None, f"No recent defensive history found for opponent '{opponent_name}'."
        opponent_history = opponent_history.sort_values(by='game_id').reset_index(drop=True)

        # 3. Calculate features based on this targeted data
        features = {}

        # Player features - .iloc[-1] gets the last value, which is the stat up to the most recent game.
        # This correctly mimics the .shift(1) used in batch training.
        features['player_rolling_avg_pass_yds'] = player_history['passing_yards'].rolling(window=5, min_periods=1).mean().iloc[-1]
        features['player_rolling_avg_pass_attempts'] = player_history['passing_attempts'].rolling(window=5, min_periods=1).mean().iloc[-1]
        features['player_rolling_std_pass_yds'] = player_history['passing_yards'].rolling(window=5, min_periods=1).std().iloc[-1]

        # Opponent features
        opponent_yards_allowed = opponent_history.groupby('game_id')['passing_yards'].sum()
        features['opponent_rolling_avg_pass_yds_allowed'] = opponent_yards_allowed.rolling(window=8, min_periods=1).mean().iloc[-1]

        # 4. Add game context features
        features.update({
            'is_home_game': game_context['is_home_game'],
            'vegas_total': game_context['vegas_total'],
            'team_win_prob': _calculate_implied_probability(game_context['team_moneyline']),
            'team_spread': game_context['team_spread'],
        })

        # 5. Create final DataFrame, ensuring correct column order and handling NaNs
        final_df = pd.DataFrame([features])

        # Fill NaN for std dev if player has only 1 game of history
        # A more robust solution would use pre-calculated medians from the training set.
        final_df = final_df.fillna(0)

        # Ensure the columns are in the exact order the model was trained on
        return final_df[self.nfl_features], None

    # In models/player_prop_predictor.py

    # ... (imports and PlayerPropPredictor class definition) ...

# In models/player_prop_predictor.py

# ... (class definition and other functions) ...

    def _get_mlb_prediction_features(self, player_name, opponent_name, game_context):
        """
        Generates features for a single MLB prediction, mirroring the V3 training logic.
        *** FIX: Creates all required V3 features and aligns with raw K prediction. ***
        """
        # --- 1. Fetch historical data ---
        player_history = self.db.fetch_recent_games_for_player(player_name, 'MLB', n=20)
        if player_history.empty or len(player_history) < 2:
            return None, f"Not enough game history for player '{player_name}'."
        required_pitching_cols = [
            'pitcher_innings_pitched', 'pitcher_strikeouts', 'pitcher_bases_on_balls',
            'pitcher_whip', 'pitcher_earned_run_average'
        ]
        # Ensure all required columns exist, adding them with NA if they don't.
        for col in required_pitching_cols:
            if col not in player_history.columns:
                player_history[col] = pd.NA

        # --- End of new block ---

        # Sort and filter the data AFTER ensuring columns exist.
        player_history = player_history.sort_values(by='game_id').reset_index(drop=True)

        # This filter now works safely without a KeyError.
        # It will create an empty DataFrame if the player has no pitching games, which is handled next.
        pitching_games = player_history[player_history['pitcher_innings_pitched'] > 0].copy()

        if pitching_games.empty or len(pitching_games) < 2:
            return None, f"Not enough recent PITCHING history for '{player_name}' to make a prediction."

        # --- All subsequent calculations use 'pitching_games' DataFrame ---
        opponent_history = self.db.fetch_recent_games_for_team_as_opponent(opponent_name, 'MLB', n=100)
        if opponent_history.empty:
            return None, f"No recent offensive history for opponent '{opponent_name}'."
        opponent_history = opponent_history.sort_values(by='game_id').reset_index(drop=True)
        player_history = player_history.sort_values(by='game_id').reset_index(drop=True)
        player_history = player_history[player_history['pitcher_innings_pitched'] > 0].copy()

        opponent_history = self.db.fetch_recent_games_for_team_as_opponent(opponent_name, 'MLB', n=100)
        if opponent_history.empty:
            return None, f"No recent offensive history for opponent '{opponent_name}'."
        opponent_history = opponent_history.sort_values(by='game_id').reset_index(drop=True)

        # --- 2. Engineer Pitcher's Own Metrics (Features) ---
        features = {}

        # Calculate base efficiency metrics on the fly
        player_history['player_k_per_inning'] = player_history['pitcher_strikeouts'] / player_history['pitcher_innings_pitched']

        # --- FIX: Calculate all required rolling features ---
        # .iloc[-1] gets the last value, correctly simulating the .shift(1) from training
        features['player_rolling_avg_k_L5'] = player_history['pitcher_strikeouts'].rolling(window=5, min_periods=2).mean().iloc[-1]
        features['player_rolling_k_per_inning_L5'] = player_history['player_k_per_inning'].rolling(window=5, min_periods=2).mean().iloc[-1]
        features['player_rolling_whip_L5'] = player_history['pitcher_whip'].rolling(window=5, min_periods=2).mean().iloc[-1]
        # Check if 'pitcher_earned_run_average' is available before calculating the feature
        if 'pitcher_earned_run_average' in player_history.columns:
            features['player_rolling_era_L5'] = player_history['pitcher_earned_run_average'].rolling(window=5, min_periods=2).mean().iloc[-1]
        else:
            # Handle case where ERA data is missing for the player
            features['player_rolling_era_L5'] = None # Will be filled by median later

        # --- 3. Engineer Opponent's TRUE Offensive Profile ---
        team_offense_per_game = opponent_history.groupby('game_id').agg(
            team_k_total=('hitting_strikeouts', 'sum'),
            team_bb_total=('hitting_bases_on_balls', 'sum'),
            team_ab_total=('hitting_at_bats', 'sum')
        ).reset_index()

        team_offense_per_game['team_k_rate'] = team_offense_per_game['team_k_total'] / (team_offense_per_game['team_ab_total'] + 1e-6)
        team_offense_per_game['team_bb_rate'] = team_offense_per_game['team_bb_total'] / (team_offense_per_game['team_ab_total'] + 1e-6)

        features['opponent_rolling_k_rate'] = team_offense_per_game['team_k_rate'].rolling(window=15, min_periods=5).mean().iloc[-1]
        features['opponent_rolling_bb_rate'] = team_offense_per_game['team_bb_rate'].rolling(window=15, min_periods=5).mean().iloc[-1]

        # --- 4. Add Game Context Features ---
        features['is_home_game'] = game_context['is_home_game']

        # --- 5. Create final DataFrame, fill NaNs, and ensure column order ---
        final_df = pd.DataFrame([features])

        if self.mlb_medians:
            print("  Applying pre-calculated medians to fill NaNs...")
            final_df = final_df.fillna(self.mlb_medians)
        else:
            print("  WARNING: Median values not available. Falling back to fillna(0).")
            final_df = final_df.fillna(0)

        # Return the feature dataframe aligned to the model's expectations.
        return final_df[self.mlb_features], None

    def predict_pitcher_strikeouts(self, player_name, opponent_name, game_context):
        print(f"\n--- Predicting MLB Pitcher Strikeouts for {player_name} vs {opponent_name} ---")
        if not self.mlb_model:
            return "Error: MLB Model is not loaded.", None

        # --- FIX: Updated logic to match V3 model ---
        # Get the feature DataFrame. We no longer need baseline_avg.
        features_df, err = self._get_mlb_prediction_features(player_name, opponent_name, game_context)
        if err:
            return f"Could not generate features: {err}", None

        print(f"  Generated Features: {features_df.to_dict('records')[0]}")

        # The new model directly predicts the strikeout count.
        prediction = self.mlb_model.predict(features_df)[0]

        # Ensure prediction isn't negative
        final_prediction = max(0, prediction)

        print(f"  => Final Model Prediction: {final_prediction:.1f} strikeouts.")
        return None, final_prediction

    def predict_passing_yards(self, player_name, opponent_name, game_context):
        print(f"\n--- Predicting NFL Passing Yards for {player_name} vs {opponent_name} ---")
        if not self.nfl_model:
            return "Error: NFL Model is not loaded.", None

        # Call the new, efficient feature generation function
        features_df, err = self._get_nfl_prediction_features(player_name, opponent_name, game_context)
        if err:
            return f"Could not generate features: {err}", None

        print(f"  Generated Features: {features_df.to_dict('records')[0]}")
        prediction = self.nfl_model.predict(features_df)[0]
        print(f"  => Model Prediction: {prediction:.1f} passing yards.")
        return None, prediction