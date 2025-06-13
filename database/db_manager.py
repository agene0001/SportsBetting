# database/db_manager.py
import psycopg2
import json

class DatabaseManager:
    def __init__(self, db_settings):
        # ... __init__ is unchanged ...
        try:
            self.conn = psycopg2.connect(**db_settings)
            print("Database connection successful.")
        except psycopg2.OperationalError as e:
            print(f"Could not connect to the database: {e}")
            self.conn = None

    def execute_query(self, query, params=None, fetch=None):
        # ... execute_query is unchanged ...
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

    def create_tables(self):
        """Creates all necessary tables with the new array-based schema."""
        print("Creating/verifying database tables with array-based schema...")

        # ... (sports_table, teams_table, players_table are unchanged) ...
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
                                                                                 category_name VARCHAR(100) NOT NULL, -- e.g., 'Rushing', 'Skater Stats'
                                                                                 stat_names TEXT[] NOT NULL,         -- e.g., ARRAY['Attempts', 'Yards', 'Touchdowns']
                                                                                 stat_abbreviations TEXT[] NOT NULL, -- e.g., ARRAY['Att', 'Yds', 'TD']
                                                                                 UNIQUE(sport_id, category_name)
                                 );"""

        player_game_stats_table = """
                                  CREATE TABLE IF NOT EXISTS player_game_stats (
                                                                                   player_game_stat_id SERIAL PRIMARY KEY,
                                                                                   player_id INTEGER NOT NULL REFERENCES players(player_id),
                                                                                   team_id INTEGER NOT NULL REFERENCES teams(team_id),
                                                                                   opponent_team_id INTEGER NOT NULL REFERENCES teams(team_id),
                                                                                   season INTEGER NOT NULL,
                                                                                   season_type VARCHAR(50) NOT NULL,
                                                                                   source_game_id VARCHAR(50), -- Yahoo's game ID from URL
                                                                                   stat_definition_id INTEGER NOT NULL REFERENCES stat_definitions(stat_definition_id),
                                                                                   stat_values TEXT[] NOT NULL, -- The list of stats, e.g., ARRAY['15', '75', '1']
                                                                                   last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                   UNIQUE(player_id, source_game_id, stat_definition_id)
                                  );"""

        # --- NEW TABLE FOR SEASON STATS ---
        player_season_stats_table = """
                                    CREATE TABLE IF NOT EXISTS player_season_stats (
                                                                                       player_season_stat_id SERIAL PRIMARY KEY,
                                                                                       player_id INTEGER NOT NULL REFERENCES players(player_id),
                                                                                       team_id INTEGER NOT NULL REFERENCES teams(team_id),
                                                                                       season INTEGER NOT NULL,
                                                                                       stat_definition_id INTEGER NOT NULL REFERENCES stat_definitions(stat_definition_id),
                                                                                       stat_values TEXT[] NOT NULL,
                                                                                       last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                       UNIQUE(player_id, team_id, season, stat_definition_id)
                                    );"""

        # Drop old JSONB tables if they exist
        self.execute_query("DROP TABLE IF EXISTS player_game_stats CASCADE;")
        self.execute_query("DROP TABLE IF EXISTS team_season_stats CASCADE;")

        self.execute_query(sports_table)
        self.execute_query(teams_table)
        self.execute_query(players_table)
        self.execute_query(stat_definitions_table)
        self.execute_query(player_game_stats_table)
        self.execute_query(player_season_stats_table) # Create the new table
        print("Array-based schema tables created/verified successfully.")
    # --- NEW HELPER METHODS ---

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
                           RETURNING stat_definition_id; \
                           """
            new_def_id = self.execute_query(query_insert, (sport_id, category_name, stat_names, stat_abbreviations), fetch='one')
            if new_def_id:
                return new_def_id[0]
            else:
                return self.execute_query(query_find, (sport_id, category_name), fetch='one')[0]

    def insert_player_game_stats(self, player_id, team_id, opponent_team_id, season, season_type, source_game_id, stat_def_id, stat_values):
        """Inserts a player's game stats using the new array-based schema."""
        query = """
                INSERT INTO player_game_stats (player_id, team_id, opponent_team_id, season, season_type, source_game_id, stat_definition_id, stat_values)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_id, source_game_id, stat_definition_id) DO UPDATE SET
                                                                                          stat_values = EXCLUDED.stat_values,
                                                                                          last_updated = CURRENT_TIMESTAMP; \
                """
        self.execute_query(query, (player_id, team_id, opponent_team_id, season, season_type, source_game_id, stat_def_id, stat_values))

    # --- Methods to keep ---
    def get_or_create_team(self, sport_id, team_name, yahoo_slug):
        # ... this method is unchanged ...
        query_find = "SELECT team_id FROM teams WHERE yahoo_slug = %s AND sport_id = %s;"
        team = self.execute_query(query_find, (yahoo_slug, sport_id), fetch='one')
        if team: return team[0]
        else:
            query_insert = "INSERT INTO teams (sport_id, name, yahoo_slug) VALUES (%s, %s, %s) ON CONFLICT (name) DO UPDATE SET yahoo_slug = EXCLUDED.yahoo_slug RETURNING team_id;"
            new_team_id = self.execute_query(query_insert, (sport_id, team_name, yahoo_slug), fetch='one')
            return new_team_id[0]

    def get_or_create_player(self, sport_id, full_name, source_id, source='yahoo'):
        # ... this method is unchanged ...
        query_find = "SELECT player_id FROM players WHERE source_id = %s AND source = %s;"
        player = self.execute_query(query_find, (source_id, source), fetch='one')
        if player: return player[0]
        else:
            query_insert = "INSERT INTO players (sport_id, full_name, source_id, source) VALUES (%s, %s, %s, %s) RETURNING player_id;"
            new_player_id = self.execute_query(query_insert, (sport_id, full_name, source_id, source), fetch='one')
            return new_player_id[0]
    # --- NEW METHOD FOR SEASON STATS ---
    def insert_player_season_stats(self, player_id, team_id, season, stat_def_id, stat_values):
        """Inserts a player's season stats using the array-based schema."""
        query = """
                INSERT INTO player_season_stats (player_id, team_id, season, stat_definition_id, stat_values)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (player_id, team_id, season, stat_definition_id) DO UPDATE SET
                                                                                           stat_values = EXCLUDED.stat_values,
                                                                                           last_updated = CURRENT_TIMESTAMP; \
                """
        self.execute_query(query, (player_id, team_id, season, stat_def_id, stat_values))
    def insert_sport(self, sport_name):
        # ... this method is unchanged ...
        query = "INSERT INTO sports (name) VALUES (%s) ON CONFLICT (name) DO NOTHING;"
        self.execute_query(query, (sport_name,))
        print(f"Ensured sport '{sport_name}' exists.")

    def close(self):
        if self.conn: self.conn.close(); print("Database connection closed.")

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()