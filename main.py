# main.py
import traceback
from concurrent.futures import ThreadPoolExecutor

from config import DB_SETTINGS
from database.db_manager import DatabaseManager
from data_collection.yahoo_scraper import YAHOOScraper

# --- UPDATED IMPORTS ---
from ml.model_trainer import train_passing_yards_model, train_pitcher_strikeouts_model
from models.player_prop_predictor import PlayerPropPredictor

team_threads=1 # Using 4 threads for faster scraping

def initialize_database(db_manager):
    """Function to create tables and populate initial data."""
    print("--- Initializing Database ---")
    db_manager.create_tables()

    db_manager.insert_sport('NFL')
    db_manager.insert_sport('NBA')
    db_manager.insert_sport('NHL')
    db_manager.insert_sport('MLB')

    print("--- Database Initialized ---\n")


def run_historical_scrape():
    """Manages the user flow for scraping historical data."""
    VALID_SPORTS = {'nfl', 'nba', 'nhl', 'mlb'}
    print("\n--- Scrape Historical Player & Team Stats ---")

    while True:
        sport_choice = input(f"Enter the sport to scrape ({', '.join(VALID_SPORTS)}): ").lower()
        if sport_choice in VALID_SPORTS: break
        else: print(f"Invalid sport. Please choose one of the following: {', '.join(VALID_SPORTS)}")

    while True:
        try:
            start_year = int(input(f"Enter the start year for {sport_choice.upper()} (e.g., 2020): "))
            end_year = int(input(f"Enter the end year for {sport_choice.upper()} (e.g., 2023): "))
            if 1900 < start_year <= end_year <= 2030: break
            else: print("Invalid year range. Please ensure start year is before or same as end year.")
        except ValueError: print("Invalid input. Please enter valid years.")

    with DatabaseManager(DB_SETTINGS) as db:
        if not db.conn: return
        scraper = YAHOOScraper(sport_choice, db)
        team_data = scraper.discover_teams()
        if not team_data:
            print("Could not discover any teams. Aborting scrape.")
            return

        print(f"\n--- Stage 2: Processing {len(team_data)} teams in parallel ---")
        with ThreadPoolExecutor(max_workers=team_threads) as executor:
            futures = {
                executor.submit(scraper.process_one_team, full_name, info, start_year, end_year): full_name
                for full_name, info in team_data.items()
            }
            for future in futures:
                team_name = futures[future]
                future.result() # Wait for thread to finish
                print(f"--- Team processing thread for '{team_name}' has finished. ---")
    print("\nAll teams processed. Scrape complete.")


def run_model_training(db_manager):
    """Handles the model training submenu."""
    print("\n--- Train a Model ---")
    print("1. Train NFL Passing Yards Model")
    print("2. Train MLB Pitcher Strikeouts Model")
    print("3. Back to Main Menu")
    choice = input("Enter your choice: ").strip()

    if choice == '1':
        # You can make start/end years dynamic if you wish
        train_passing_yards_model(db_manager, start_year=2020, end_year=2025)
    elif choice == '2':
        train_pitcher_strikeouts_model(db_manager, start_year=2020, end_year=2025)
    else:
        return

def run_prediction(db_manager):
    """Handles the prediction submenu."""
    print("\n--- Make a Prediction ---")
    print("1. Predict NFL Player Passing Yards")
    print("2. Predict MLB Pitcher Strikeouts")
    print("3. Back to Main Menu")
    choice = input("Enter your choice: ").strip()

    try:
        predictor = PlayerPropPredictor(db_manager)

        if choice == '1':
            if not predictor.nfl_model:
                print("\nERROR: NFL model not found. Please train it first (Main Menu Option 3).")
                return

            print("\n--- Enter Details for NFL Passing Yards Prediction ---")
            player_name = input("Enter player's full name (e.g., 'Patrick Mahomes'): ")
            opponent_name = input("Enter opponent's full name (e.g., 'Buffalo Bills'): ")
            season = int(input("Enter the current season year (e.g., 2023): "))

            # Collect game context needed by the model
            is_home_input = input(f"Is {player_name}'s team the home team? (yes/no): ").lower()
            is_home_game = 1 if is_home_input == 'yes' else 0
            vegas_total = float(input("Enter the game total (over/under, e.g., 48.5): "))
            team_moneyline = int(input(f"Enter the moneyline for {player_name}'s team (e.g., -150 or +120): "))
            team_spread = float(input(f"Enter the point spread for {player_name}'s team (e.g., -7.5 or +3.0): "))

            game_context = {
                'is_home_game': is_home_game,
                'vegas_total': vegas_total,
                'team_moneyline': team_moneyline,
                'team_spread': team_spread
            }
            err, prediction = predictor.predict_pitcher_strikeouts(player_name, opponent_name, game_context)

            if err:
                # If an error string was returned, print it!
                print(f"\nAN ERROR OCCURRED: {err}\n")
            elif prediction is not None:
                # If there was no error and we got a prediction, print the result.
                print(f"\n=> FINAL PREDICTION: {prediction:.1f} Strikeouts\n")
            else:
                # Fallback for any other unexpected case
                print("\nCould not generate a prediction for an unknown reason.\n")

        elif choice == '2':
            if not predictor.mlb_model:
                print("\nERROR: MLB model not found. Please train it first (Main Menu Option 3).")
                return

            print("\n--- Enter Details for MLB Pitcher Strikeouts Prediction ---")
            player_name = input("Enter pitcher's full name (e.g., 'Gerrit Cole'): ")
            opponent_name = input("Enter opponent's full name (e.g., 'Boston Red Sox'): ")
            season = int(input("Enter the current season year (e.g., 2023): "))

            # Collect game context
            is_home_input = input(f"Is {player_name}'s team the home team? (yes/no): ").lower()
            is_home_game = 1 if is_home_input == 'yes' else 0
            vegas_total = float(input("Enter the game total (over/under, e.g., 8.5): "))
            team_moneyline = int(input(f"Enter the moneyline for {player_name}'s team (e.g., -180 or +150): "))
            team_spread = float(input(f"Enter the run line for {player_name}'s team (e.g., -1.5 or +1.5): "))

            game_context = {
                'is_home_game': is_home_game,
                'vegas_total': vegas_total,
                'team_moneyline': team_moneyline,
                'team_spread': team_spread
            }
            predictor.predict_pitcher_strikeouts(player_name, opponent_name, game_context)

        else:
            return

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()

def main_menu():
    """Displays the main menu and handles user input."""
    while True:
        print("\n" + "="*25)
        print("    Sports AI Main Menu")
        print("="*25)
        print("1. Initialize Database (Run this first!)")
        print("2. Scrape Historical Stats")
        print("3. Train Prediction Models")
        print("4. Make a Prediction")
        print("5. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            with DatabaseManager(DB_SETTINGS) as db:
                if db.conn:
                    initialize_database(db)
        elif choice == '2':
            run_historical_scrape()
        elif choice == '3':
            with DatabaseManager(DB_SETTINGS) as db:
                if db.conn:
                    run_model_training(db)
        elif choice == '4':
            with DatabaseManager(DB_SETTINGS) as db:
                if db.conn:
                    run_prediction(db)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()