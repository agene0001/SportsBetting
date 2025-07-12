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

    def fetch_player_game_logs_for_season(self, sport_name, season):
        """
        Fetches all player game logs for a given sport and season, joining
        all necessary tables to create a comprehensive DataFrame.
        """
        print(f"Fetching game logs for {sport_name} season {season}...")
        query = """
                SELECT
                    p.full_name,
                    p.source_id as player_source_id,
                    t.name AS team_name,
                    opp.name AS opponent_team_name,
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
            (full_name, player_source_id, team_name, opponent_team_name, season_val, game_id,
             category_name, stat_names, stat_values,
             vegas_spread, vegas_total, home_ml, away_ml, is_home_game) = row
            for i, name in enumerate(stat_names):
                clean_name = f"{category_name.lower().replace(' ', '_')}_{name.lower().replace(' ', '_')}"
                try:
                    stat_record = {
                        'player_name': full_name, 'team': team_name, 'opponent': opponent_team_name,
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
            index_cols = [
                'player_name', 'team', 'opponent', 'season', 'game_id',
                'vegas_spread', 'vegas_total', 'home_moneyline', 'away_moneyline', 'is_home_game'
            ]
            final_df = df.pivot_table(
                index=index_cols, columns='stat_name', values='stat_value', aggfunc='first'
            ).reset_index()
            final_df.columns.name = None
        except Exception as e:
            print(f"An error occurred during pivoting: {e}")
            return pd.DataFrame()

        # Use the centralized helper to ensure consistent column names
        final_df = self._clean_and_rename_game_log_df(final_df)

        print(f"Successfully fetched and processed {len(final_df)} game records.")
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