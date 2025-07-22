# database/db_manager.py
import psycopg2
import json
import pandas as pd

class DatabaseManager:
    def __init__(self, db_settings):
        try:
            self.conn = psycopg2.connect(**db_settings)
            print("Database connection successful.")
        except psycopg2.OperationalError as e:
            print(f"Could not connect to the database: {e}")
            self.conn = None

    def execute_query(self, query, params=None, fetch=None):
        if not self.conn:
            print("Cannot execute query: No database connection.")
            return None
        result = None
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                if fetch == 'one':
                    result = cur.fetchone()
                elif fetch == 'all':
                    result = cur.fetchall()
            self.conn.commit()
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            if self.conn:
                self.conn.rollback()
            return None
        return result

    # --- NEW CENTRALIZED HELPER FUNCTION ---
    def _clean_and_rename_game_log_df(self, df):
        """
        A centralized helper to apply all column cleaning and renaming rules.
        This ensures consistency between training and prediction data.
        """
        if df.empty:
            return df

        # General renaming for consistency (e.g., 'passing_passing_yards' -> 'passing_yards')
        df = df.rename(columns=lambda c: c.replace('passing_passing_', 'passing_'))
        df = df.rename(columns=lambda c: c.replace('rushing_rushing_', 'rushing_'))
        df = df.rename(columns=lambda c: c.replace('receiving_receiving_', 'receiving_'))
        df = df.rename(columns=lambda c: c.replace('pitching_', 'pitcher_'))
        df = df.rename(columns=lambda c: c.replace('batting_', 'hitting_'))

        # Specific, targeted renames for long column names to make them model-friendly
        rename_map = {
            'pitcher_walks_plus_hits_per_inning_pitched': 'pitcher_whip',
            # Add any other specific renames here in the future
        }
        df = df.rename(columns=rename_map)
        return df

    def create_tables(self):
        """Creates all necessary tables with the new array-based schema."""
        print("Creating/verifying database tables with array-based schema...")

        sports_table = """
                       CREATE TABLE IF NOT EXISTS sports (
                                                             sport_id SERIAL PRIMARY KEY, name VARCHAR(50) UNIQUE NOT NULL
                       );"""

        teams_table = """
                      CREATE TABLE IF NOT EXISTS teams (
                                                           team_id SERIAL PRIMARY KEY, sport_id INTEGER NOT NULL REFERENCES sports(sport_id),
                                                           name VARCHAR(100) UNIQUE NOT NULL, yahoo_slug VARCHAR(100) UNIQUE
                      );"""

        players_table = """
                        CREATE TABLE IF NOT EXISTS players (
                                                               player_id SERIAL PRIMARY KEY, full_name VARCHAR(150) NOT NULL,
                                                               source_id VARCHAR(50) NOT NULL, source VARCHAR(50) NOT NULL,
                                                               sport_id INTEGER NOT NULL REFERENCES sports(sport_id), UNIQUE (source_id, source)
                        );"""

        stat_definitions_table = """
                                 CREATE TABLE IF NOT EXISTS stat_definitions (
                                                                                 stat_definition_id SERIAL PRIMARY KEY,
                                                                                 sport_id INTEGER NOT NULL REFERENCES sports(sport_id),
                                                                                 category_name VARCHAR(100) NOT NULL,
                                                                                 stat_names TEXT[] NOT NULL,
                                                                                 stat_abbreviations TEXT[] NOT NULL,
                                                                                 UNIQUE(sport_id, category_name)
                                 );"""

        player_game_stats_table = """
                                  CREATE TABLE IF NOT EXISTS player_game_stats (
                                                                                   player_game_stat_id SERIAL PRIMARY KEY,
                                                                                   player_id INTEGER NOT NULL REFERENCES players(player_id),
                                                                                   team_id INTEGER NOT NULL REFERENCES teams(team_id),
                                                                                   opponent_team_id INTEGER NOT NULL REFERENCES teams(team_id),
                                                                                   season INTEGER NOT NULL,
                                                                                   source_game_id VARCHAR(50),
                                                                                   stat_definition_id INTEGER NOT NULL REFERENCES stat_definitions(stat_definition_id),
                                                                                   stat_values TEXT[] NOT NULL,
                                                                                   last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                   UNIQUE(player_id, source_game_id, stat_definition_id)
                                  );"""
        game_details_table = """
                             CREATE TABLE IF NOT EXISTS game_details (
                                                                         game_detail_id SERIAL PRIMARY KEY,
                                                                         source_game_id VARCHAR(50) UNIQUE NOT NULL,
                                                                         sport_id INTEGER NOT NULL REFERENCES sports(sport_id),
                                                                         home_team_id INTEGER NOT NULL REFERENCES teams(team_id),
                                                                         away_team_id INTEGER NOT NULL REFERENCES teams(team_id),
                                                                         vegas_spread FLOAT,
                                                                         vegas_total FLOAT,
                                                                         home_moneyline INTEGER,
                                                                         away_moneyline INTEGER,
                                                                         team_stats_json JSONB,
                                                                         last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                             ); """


        self.execute_query(sports_table)
        self.execute_query(teams_table)
        self.execute_query(players_table)
        self.execute_query(stat_definitions_table)
        self.execute_query(player_game_stats_table)
        self.execute_query(game_details_table)
        print("Array-based schema tables created/verified successfully.")

    def get_or_create_stat_definition(self, sport_id, category_name, stat_names, stat_abbreviations):
        """Finds or creates a stat definition based on the category name. Returns its ID."""
        query_find = "SELECT stat_definition_id FROM stat_definitions WHERE sport_id = %s AND category_name = %s;"
        definition = self.execute_query(query_find, (sport_id, category_name), fetch='one')
        if definition:
            return definition[0]
        else:
            query_insert = """
                           INSERT INTO stat_definitions (sport_id, category_name, stat_names, stat_abbreviations)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (sport_id, category_name) DO NOTHING
                           RETURNING stat_definition_id; """
            new_def_id = self.execute_query(query_insert, (sport_id, category_name, stat_names, stat_abbreviations), fetch='one')
            if new_def_id:
                return new_def_id[0]
            else:
                return self.execute_query(query_find, (sport_id, category_name), fetch='one')[0]

    def insert_player_game_stats(self, player_id, team_id, opponent_team_id, season, source_game_id, stat_def_id, stat_values):
        """Inserts a player's game stats using an array column (e.g., TEXT[])."""
        query = """
                INSERT INTO player_game_stats (
                    player_id, team_id, opponent_team_id, season, source_game_id, stat_definition_id, stat_values
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_id, source_game_id, stat_definition_id) DO UPDATE SET
                                                                                          stat_values = EXCLUDED.stat_values,
                                                                                          last_updated = CURRENT_TIMESTAMP; """
        params = (player_id, team_id, opponent_team_id, season, source_game_id, stat_def_id, stat_values)
        self.execute_query(query, params)
    # In database/db_manager.py

    # In database/db_manager.py

    def fetch_recent_pitching_logs_for_player(self, player_name, n=15):
        """
        Fetches the last N PITCHING-ONLY games for a specific player.
        This is a specialized version to avoid issues with two-way players.
        """
        query = """
                SELECT p.full_name as player_name, pgs.source_game_id as game_id, pgs.season,
                       sd.category_name, sd.stat_names, pgs.stat_values
                FROM player_game_stats pgs
                         JOIN players p ON pgs.player_id = p.player_id
                         JOIN sports s ON p.sport_id = s.sport_id
                         JOIN stat_definitions sd ON pgs.stat_definition_id = sd.stat_definition_id
                WHERE p.full_name ILIKE %s
                  AND s.name ILIKE 'MLB'
                  AND sd.category_name = 'Pitching' -- <<< --- THE CRUCIAL ADDITION
                ORDER BY pgs.source_game_id DESC
                LIMIT %s; """
        raw_data = self.execute_query(query, params=(f'%{player_name}%', n), fetch='all')
        if not raw_data:
            return pd.DataFrame()

        processed_data = []
        for row in raw_data:
            (name, game_id, season, cat_name, stat_names, stat_values) = row
            stat_dict = {'player_name': name, 'game_id': game_id, 'season': season}
            for i, stat_name in enumerate(stat_names):
                clean_name = f"{cat_name.lower().replace(' ', '_')}_{stat_name.lower().replace(' ', '_')}"
                stat_dict[clean_name] = pd.to_numeric(stat_values[i], errors='coerce')
            processed_data.append(stat_dict)

        df = pd.DataFrame(processed_data)
        df = self._clean_and_rename_game_log_df(df)
        return df

    def insert_player_game_stats_from_dict(self, player_id, data, season, source_game_id):
        """
        Takes a dictionary of player data prepared by the scraper, unpacks it,
        and saves it to the database. This acts as a helper method.

        Args:
            player_id (int): The player's unique database ID.
            data (dict): A dictionary containing team_id, opponent_id, stat_def_id, and a sub-dict of stats.
            season (int): The season year.
            source_game_id (str): The game's unique ID.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # 1. Unpack the main keys from the data dictionary
            team_id = data['team_id']
            opponent_team_id = data['opponent_id']
            stat_def_id = data['stat_def_id']
            stats_dict = data.get('stats', {}) # Use .get() for safety

            # 2. To insert the data, we need the stat values in the correct order.
            #    We fetch the canonical order from the stat_definitions table.
            query = "SELECT stat_abbreviations FROM stat_definitions WHERE stat_definition_id = %s;"
            result = self.execute_query(query, (stat_def_id,), fetch='one')

            if not result:
                print(f"  - ERROR: Could not find stat definition for ID {stat_def_id}. Cannot save player stats.")
                return False

            ordered_abbreviations = result[0]

            # 3. Build the list of stat values in the correct order.
            #    Use .get(abbr, '0') to handle cases where a player might be missing a stat (e.g., a DH with no fielding stats).
            stat_values = [stats_dict.get(abbr, '0') for abbr in ordered_abbreviations]

            # 4. Now, call the existing, low-level insertion method with the correctly formatted data.
            self.insert_player_game_stats(
                player_id=player_id,
                team_id=team_id,
                opponent_team_id=opponent_team_id,
                season=season,
                source_game_id=source_game_id,
                stat_def_id=stat_def_id,
                stat_values=stat_values
            )
            return True

        except (KeyError, TypeError) as e:
            print(f"  - ERROR: Data dictionary was missing a required key for player {player_id} in game {source_game_id}. Details: {e}")
            return False
    def fetch_player_game_logs_for_season(self, sport_name, season):
        """
        Fetches all player game logs for a given sport and season, joining
        all necessary tables to create a comprehensive DataFrame.
        *** UPDATED to include opponent_team_id ***
        """
        print(f"Fetching game logs for {sport_name} season {season}...")
        query = """
                SELECT
                    p.full_name,
                    p.source_id as player_source_id,
                    t.name AS team_name,
                    opp.name AS opponent_team_name,
                    pgs.opponent_team_id, -- <<< --- ADD THIS LINE
                    pgs.season,
                    pgs.source_game_id,
                    sd.category_name,
                    sd.stat_names,
                    pgs.stat_values,
                    gd.vegas_spread,
                    gd.vegas_total,
                    gd.home_moneyline,
                    gd.away_moneyline,
                    CASE WHEN pgs.team_id = gd.home_team_id THEN 1 ELSE 0 END as is_home_game
                FROM player_game_stats pgs
                         JOIN players p ON pgs.player_id = p.player_id
                         JOIN teams t ON pgs.team_id = t.team_id
                         JOIN teams opp ON pgs.opponent_team_id = opp.team_id
                         JOIN stat_definitions sd ON pgs.stat_definition_id = sd.stat_definition_id
                         JOIN sports s ON p.sport_id = s.sport_id
                         LEFT JOIN game_details gd ON pgs.source_game_id = gd.source_game_id
                WHERE UPPER(s.name) = UPPER(%s) AND pgs.season = %s;
                """
        params = (sport_name, season)
        raw_data = self.execute_query(query, params=params, fetch='all')

        if not raw_data:
            print(f"No data found for {sport_name} season {season}.")
            return pd.DataFrame()

        processed_data = []
        for row in raw_data:
            # The new opponent_team_id will be in the row tuple
            (full_name, player_source_id, team_name, opponent_team_name, opponent_team_id,
             season_val, game_id, category_name, stat_names, stat_values,
             vegas_spread, vegas_total, home_ml, away_ml, is_home_game) = row
            for i, name in enumerate(stat_names):
                clean_name = f"{category_name.lower().replace(' ', '_')}_{name.lower().replace(' ', '_')}"
                try:
                    stat_record = {
                        'player_name': full_name, 'team': team_name, 'opponent': opponent_team_name,
                        'opponent_team_id': opponent_team_id, # <<< --- ADD THIS
                        'season': season_val, 'game_id': game_id, 'vegas_spread': vegas_spread,
                        'vegas_total': vegas_total, 'home_moneyline': home_ml, 'away_moneyline': away_ml,
                        'is_home_game': is_home_game, 'stat_name': clean_name,
                        'stat_value': pd.to_numeric(stat_values[i], errors='coerce')
                    }
                    processed_data.append(stat_record)
                except (ValueError, IndexError):
                    continue

        if not processed_data:
            print("Data fetched but could not be processed into records.")
            return pd.DataFrame()

        df = pd.DataFrame(processed_data)
        try:
            # Add opponent_team_id to the list of columns to pivot on
            index_cols = [
                'player_name', 'team', 'opponent', 'opponent_team_id', 'season', 'game_id', # <<<--- ADD THIS
                'vegas_spread', 'vegas_total', 'home_moneyline', 'away_moneyline', 'is_home_game'
            ]
            final_df = df.pivot_table(
                index=index_cols, columns='stat_name', values='stat_value', aggfunc='first'
            ).reset_index()
            final_df.columns.name = None
        except Exception as e:
            print(f"An error occurred during pivoting: {e}")
            return pd.DataFrame()

        final_df = self._clean_and_rename_game_log_df(final_df)

        print(f"Successfully fetched and processed {len(final_df)} game records.")
        return final_df


    def fetch_batting_stats_for_game(self, source_game_id, team_id):
        """
        Fetches all the raw batting stats for every player on a specific team
        in a specific game, returning a clean DataFrame.

        Args:
            source_game_id (str): The unique identifier for the game.
            team_id (int): The unique identifier for the team whose batters we want.

        Returns:
            pd.DataFrame: A DataFrame where each row is a player's batting stats
                          for the game, with columns like 'hitting_at_bats', 'hitting_strikeouts', etc.
                          Returns an empty DataFrame if no data is found.
        """
        query = """
                SELECT
                    sd.stat_names,
                    pgs.stat_values
                FROM player_game_stats pgs
                         JOIN stat_definitions sd ON pgs.stat_definition_id = sd.stat_definition_id
                WHERE pgs.source_game_id = %s
                  AND pgs.team_id = %s
                  AND sd.category_name = 'Batting'; \
                """
        params = (str(source_game_id), team_id)

        # Execute the query to get a list of tuples, e.g., [ (['AB', 'H'], ['4', '2']), ... ]
        raw_data = self.execute_query(query, params=params, fetch='all')

        if not raw_data:
            # If the query returns no players for that team/game, return an empty frame.
            return pd.DataFrame()

        processed_lineup = []
        # Each 'player_row' represents one player's performance in the game.
        for player_row in raw_data:
            stat_names = player_row[0]  # This is the list of stat names, e.g., ['At Bats', 'Runs', ...]
            stat_values = player_row[1] # This is the list of stat values, e.g., ['4', '1', ...]

            # Create a dictionary to hold the stats for this single player
            player_stats_dict = {}

            # Use zip to pair each stat name with its corresponding value
            for name, value in zip(stat_names, stat_values):
                # Clean the stat name to be model-friendly, e.g., 'At Bats' -> 'hitting_at_bats'
                # This matches the format from your _clean_and_rename_game_log_df helper.
                clean_name = f"hitting_{name.lower().replace(' ', '_')}"

                # Convert the stat value from a string to a number.
                # 'coerce' will turn any errors (like empty strings) into NaN (Not a Number),
                # which is what pandas and scikit-learn expect for missing data.
                numeric_value = pd.to_numeric(value, errors='coerce')

                player_stats_dict[clean_name] = numeric_value

            # Add the fully processed dictionary for this player to our list
            processed_lineup.append(player_stats_dict)

        # Convert the list of dictionaries into a final, clean DataFrame.
        lineup_df = pd.DataFrame(processed_lineup)

        # The columns from your stat_definitions table might include long names.
        # We can reuse the centralized cleaner to ensure consistency.
        # For batting, it might not change much, but it's excellent practice.
        final_df = self._clean_and_rename_game_log_df(lineup_df)

        return final_df
# In database/db_manager.py

    def fetch_all_batting_stats_for_seasons(self, start_year, end_year):
        """
        *** V2: REWRITTEN to process data robustly, matching the single-game fetcher. ***
        Fetches ALL batting stats for ALL players across a range of seasons.
        """
        print("  Executing bulk fetch for all batting stats (V2)...")
        query = """
                SELECT
                    pgs.source_game_id,
                    pgs.team_id,
                    p.full_name, -- Use player name to distinguish rows
                    sd.stat_names,
                    pgs.stat_values
                FROM player_game_stats pgs
                         JOIN stat_definitions sd ON pgs.stat_definition_id = sd.stat_definition_id
                         JOIN players p ON pgs.player_id = p.player_id -- Join players table
                WHERE sd.category_name = 'Batting'
                  AND pgs.season BETWEEN %s AND %s; \
                """
        params = (start_year, end_year)
        raw_data = self.execute_query(query, params=params, fetch='all')

        if not raw_data:
            return pd.DataFrame()

        processed_records = []
        # This logic now mirrors the robust single-game fetcher
        for row in raw_data:
            game_id, team_id, player_name, stat_names, stat_values = row

            player_stats_dict = {
                'source_game_id': game_id,
                'team_id': team_id,
                'player_name': player_name # Identify each player
            }
            for name, value in zip(stat_names, stat_values):
                clean_name = f"hitting_{name.lower().replace(' ', '_')}"
                player_stats_dict[clean_name] = pd.to_numeric(value, errors='coerce')

            processed_records.append(player_stats_dict)

        if not processed_records:
            return pd.DataFrame()

        # No pivot needed! The logic creates the correct structure directly.
        final_df = pd.DataFrame(processed_records)

        final_df = self._clean_and_rename_game_log_df(final_df)

        print(f"  Bulk fetch complete. Loaded {len(final_df)} individual batting performances.")
        return final_df
    def fetch_recent_games_for_player(self, player_name, sport_name, n=15):
        """
        Fetches the last N games for a specific player to calculate rolling stats.
        Uses the centralized cleaner for consistent output.
        """
        query = """
                SELECT p.full_name as player_name, pgs.source_game_id as game_id, pgs.season,
                       sd.category_name, sd.stat_names, pgs.stat_values
                FROM player_game_stats pgs
                         JOIN players p ON pgs.player_id = p.player_id
                         JOIN sports s ON p.sport_id = s.sport_id
                         JOIN stat_definitions sd ON pgs.stat_definition_id = sd.stat_definition_id
                WHERE p.full_name ILIKE %s AND s.name ILIKE %s
                ORDER BY pgs.source_game_id DESC
                LIMIT %s; """
        raw_data = self.execute_query(query, params=(f'%{player_name}%', sport_name, n), fetch='all')
        if not raw_data:
            return pd.DataFrame()

        processed_data = []
        for row in raw_data:
            (name, game_id, season, cat_name, stat_names, stat_values) = row
            stat_dict = {'player_name': name, 'game_id': game_id, 'season': season}
            for i, stat_name in enumerate(stat_names):
                clean_name = f"{cat_name.lower().replace(' ', '_')}_{stat_name.lower().replace(' ', '_')}"
                stat_dict[clean_name] = pd.to_numeric(stat_values[i], errors='coerce')
            processed_data.append(stat_dict)

        df = pd.DataFrame(processed_data)
        # Use the centralized helper to ensure consistent column names
        df = self._clean_and_rename_game_log_df(df)
        return df

    def fetch_most_recent_lineup_for_team(self, team_name, sport_name, num_games=15):
        """
        Finds the most frequent 9 batters for a team over their last N games
        to serve as a "proxy lineup" for predictions.

        Args:
            team_name (str): The name of the team.
            sport_name (str): The sport (e.g., 'MLB').
            num_games (int): The number of recent games to look back on.

        Returns:
            list: A list of the 9 player names who appeared most frequently.
        """
        print(f"  Fetching proxy lineup for {team_name} based on last {num_games} games...")

        # First, get the most recent N game IDs for the team
        game_id_query = """
                        SELECT DISTINCT pgs.source_game_id
                        FROM player_game_stats pgs
                                 JOIN teams t ON pgs.team_id = t.team_id
                        WHERE t.name ILIKE %s
                        ORDER BY pgs.source_game_id DESC
                        LIMIT %s; \
                        """
        game_ids_raw = self.execute_query(game_id_query, params=(team_name, num_games), fetch='all')
        if not game_ids_raw:
            return []
        game_ids = [row[0] for row in game_ids_raw]

        # Now, find which players appeared most often in those games
        player_counts_query = """
                              SELECT p.full_name, COUNT(DISTINCT pgs.source_game_id) as games_played
                              FROM player_game_stats pgs
                                       JOIN players p ON pgs.player_id = p.player_id
                                       JOIN stat_definitions sd ON pgs.stat_definition_id = sd.stat_definition_id
                              WHERE pgs.source_game_id = ANY(%s)
                                AND sd.category_name = 'Batting'
                              GROUP BY p.full_name
                              ORDER BY games_played DESC
                              LIMIT 9; -- Get the top 9 most frequent players \
                              """
        player_names_raw = self.execute_query(player_counts_query, params=(game_ids,), fetch='all')

        if not player_names_raw:
            return []

        return [row[0] for row in player_names_raw]
    def fetch_recent_stats_for_players(self, player_names_list):
        """
        Takes a list of player names and fetches their aggregated batting stats
        over a recent period (e.g., this season or all available data).

        Args:
            player_names_list (list): A list of strings, e.g., ['Mookie Betts', 'Freddie Freeman'].

        Returns:
            pd.DataFrame: A DataFrame where each row is a player from the list
                          and the columns are their summed stats ('hitting_at_bats', etc.).
        """
        if not player_names_list:
            return pd.DataFrame()

        query = """
                SELECT
                    p.full_name,
                    sd.stat_names,
                    pgs.stat_values
                FROM player_game_stats pgs
                         JOIN players p ON pgs.player_id = p.player_id
                         JOIN stat_definitions sd ON pgs.stat_definition_id = sd.stat_definition_id
                WHERE sd.category_name = 'Batting'
                  AND p.full_name = ANY(%s); -- Use = ANY() to find all players in the list \
                """
        params = (player_names_list,)
        raw_data = self.execute_query(query, params=params, fetch='all')

        if not raw_data:
            return pd.DataFrame()

        processed_records = []
        for player_name, stat_names, stat_values in raw_data:
            stat_dict = {'player_name': player_name}
            for name, value in zip(stat_names, stat_values):
                clean_name = f"hitting_{name.lower().replace(' ', '_')}"
                stat_dict[clean_name] = pd.to_numeric(value, errors='coerce')
            processed_records.append(stat_dict)

        player_df = pd.DataFrame(processed_records)

        # Aggregate all game logs for each player into a single row of stats
        # Group by player and sum their stats
        aggregated_df = player_df.groupby('player_name').sum().reset_index()

        return aggregated_df
    def fetch_recent_games_for_team_as_opponent(self, team_name, sport_name, n=50):
        """
        Fetches recent games where a team was the opponent.
        Uses the centralized cleaner for consistent output.
        """
        query = """
                SELECT opp.name as opponent_name, pgs.source_game_id as game_id, pgs.season,
                       sd.category_name, sd.stat_names, pgs.stat_values
                FROM player_game_stats pgs
                         JOIN teams opp ON pgs.opponent_team_id = opp.team_id
                         JOIN sports s ON opp.sport_id = s.sport_id
                         JOIN stat_definitions sd ON pgs.stat_definition_id = sd.stat_definition_id
                WHERE opp.name ILIKE %s AND s.name ILIKE %s
                ORDER BY pgs.source_game_id DESC
                LIMIT %s; """
        raw_data = self.execute_query(query, params=(f'%{team_name}%', sport_name, n), fetch='all')
        if not raw_data:
            return pd.DataFrame()

        processed_data = []
        for row in raw_data:
            (name, game_id, season, cat_name, stat_names, stat_values) = row
            stat_dict = {'opponent_name': name, 'game_id': game_id, 'season': season}
            for i, stat_name in enumerate(stat_names):
                clean_name = f"{cat_name.lower().replace(' ', '_')}_{stat_name.lower().replace(' ', '_')}"
                stat_dict[clean_name] = pd.to_numeric(stat_values[i], errors='coerce')
            processed_data.append(stat_dict)

        df = pd.DataFrame(processed_data)
        # Use the centralized helper to ensure consistent column names
        df = self._clean_and_rename_game_log_df(df)
        return df

    def get_or_create_team(self, sport_id, team_name, yahoo_slug):
        query_find = "SELECT team_id FROM teams WHERE yahoo_slug = %s AND sport_id = %s;"
        team = self.execute_query(query_find, (yahoo_slug, sport_id), fetch='one')
        if team: return team[0]
        else:
            query_insert = "INSERT INTO teams (sport_id, name, yahoo_slug) VALUES (%s, %s, %s) ON CONFLICT (name) DO UPDATE SET yahoo_slug = EXCLUDED.yahoo_slug RETURNING team_id;"
            new_team_id = self.execute_query(query_insert, (sport_id, team_name, yahoo_slug), fetch='one')
            return new_team_id[0]

    def get_or_create_player(self, sport_id, full_name, source_id, source='yahoo'):
        query_find = "SELECT player_id FROM players WHERE source_id = %s AND source = %s;"
        player = self.execute_query(query_find, (source_id, source), fetch='one')
        if player: return player[0]
        else:
            query_insert = "INSERT INTO players (sport_id, full_name, source_id, source) VALUES (%s, %s, %s, %s) RETURNING player_id;"
            new_player_id = self.execute_query(query_insert, (sport_id, full_name, source_id, source), fetch='one')
            return new_player_id[0]

    def insert_sport(self, sport_name):
        query = "INSERT INTO sports (name) VALUES (%s) ON CONFLICT (name) DO NOTHING;"
        self.execute_query(query, (sport_name,))
        print(f"Ensured sport '{sport_name}' exists.")

    def insert_game_details(self, source_game_id, sport_id, home_team_id, away_team_id, spread, total,home_ml, away_ml, team_stats):
        team_stats_json = json.dumps(team_stats) if team_stats else None
        query = """
                INSERT INTO game_details (
                    source_game_id, sport_id, home_team_id, away_team_id,
                    vegas_spread, vegas_total, home_moneyline, away_moneyline, team_stats_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s)
                ON CONFLICT (source_game_id) DO UPDATE SET
                                                           home_team_id = EXCLUDED.home_team_id,
                                                           away_team_id = EXCLUDED.away_team_id,
                                                           vegas_spread = EXCLUDED.vegas_spread,
                                                           vegas_total = EXCLUDED.vegas_total,
                                                           home_moneyline = EXCLUDED.home_moneyline,
                                                           away_moneyline = EXCLUDED.away_moneyline,
                                                           team_stats_json = EXCLUDED.team_stats_json,
                                                           last_updated = CURRENT_TIMESTAMP; """
        params = (source_game_id, sport_id, home_team_id, away_team_id, spread, total,home_ml, away_ml, team_stats_json)
        self.execute_query(query, params)

    def close(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

