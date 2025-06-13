# main.py
import time
import traceback
from playwright.sync_api import sync_playwright, Error as PlaywrightError

from config import DB_SETTINGS
from database.db_manager import DatabaseManager
# This class has been updated
from data_collection.yahoo_scraper import YAHOOScraper

# ... (initialize_database function is unchanged) ...
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
    # ... (user input section is unchanged) ...
    VALID_SPORTS = {'nfl', 'nba', 'nhl','mlb'}

    print("\n--- Scrape Historical Player & Team Stats ---")

    while True:
        sport_choice = input(f"Enter the sport to scrape ({', '.join(VALID_SPORTS)}): ").lower()
        if sport_choice in VALID_SPORTS:
            break
        else:
            print(f"Invalid sport. Please choose one of the following: {', '.join(VALID_SPORTS)}")

    while True:
        try:
            start_year = int(input(f"Enter the start year for {sport_choice.upper()} (e.g., 2020): "))
            end_year = int(input(f"Enter the end year for {sport_choice.upper()} (e.g., 2023): "))
            if 1900 < start_year <= end_year <= 2030:
                break
            else:
                print("Invalid year range. Please ensure start year is before or same as end year.")
        except ValueError:
            print("Invalid input. Please enter valid years.")

    db = DatabaseManager(DB_SETTINGS)
    if not db.conn:
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        print("Playwright browser launched.")

        # --- UPDATED ---
        # Create ONE scraper instance, passing the chosen sport to select the strategy
        scraper = YAHOOScraper(sport_choice, None, db)

        # === Stage 1: Discovering all teams ===
        print("\n--- Stage 1: Discovering all teams ---")
        page = None
        team_data = None
        try:
            page = browser.new_page()
            scraper.page = page
            # _discover_teams no longer needs the sport_choice argument
            team_data = scraper.discover_teams()
        except PlaywrightError as e:
            print(f"FATAL: Could not complete team discovery. Aborting scrape. Error: {e}")
        finally:
            if page: page.close()

        if not team_data:
            browser.close()
            db.close()
            return

        # === Stage 2: Processing each team ===
        print(f"\n--- Stage 2: Processing {len(team_data)} teams individually ---")
        for full_name, info in team_data.items():
            max_retries = 2
            for attempt in range(max_retries):
                page = None
                try:
                    print(f"\n{'='*25}\nProcessing Team: {full_name} (Attempt {attempt + 1}/{max_retries})\n{'='*25}")
                    page = browser.new_page()

                    scraper.page = page
                    # process_one_team no longer needs the sport_choice argument
                    scraper.process_one_team(full_name, info, start_year, end_year)

                    print(f"--- Successfully completed processing for {full_name} ---")
                    break
                except (PlaywrightError, ValueError) as e: # Catch strategy errors too
                    print(f"  - ATTEMPT {attempt + 1} FAILED for team {full_name} with Error.")
                    if attempt < max_retries - 1:
                        print(f"  - Error: {e}\n  - Retrying...")
                        time.sleep(5)
                    else:
                        print(f"  - MAX RETRIES REACHED for {full_name}. Moving to the next team.")
                        traceback.print_exc()
                finally:
                    if page: page.close()

        print("\nAll teams processed. Browser closing. Scrape complete.")
        browser.close()
        db.close()

# ... (main_menu is unchanged) ...
def main_menu():
    """Displays the main menu and handles user input."""
    while True:
        print("\n--- Sports AI Main Menu ---")
        print("1. Initialize Database (Run this first!)")
        print("2. Scrape Historical Stats")
        print("3. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            with DatabaseManager(DB_SETTINGS) as db:
                if db.conn:
                    initialize_database(db)
        elif choice == '2':
            run_historical_scrape()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()