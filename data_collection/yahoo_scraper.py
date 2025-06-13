# data_collection/yahoo_scraper.py
import time
import json
import re
from playwright.sync_api import Error as PlaywrightError

# Import the new strategies
from .scraping_strategies import NFLStrategy, NHLStrategy, NBAStrategy

class YAHOOScraper:
    def __init__(self, sport_name, page, db_manager):
        self.page = page
        self.db = db_manager
        self.sport_name = sport_name
        self.sport_id = None
        self.YAHOO_BASE_URL = "https://sports.yahoo.com"
        self.scraped_games = set()

        # FACTORY: Select the correct strategy based on the sport name
        if sport_name == 'nfl':
            self.strategy = NFLStrategy()
        elif sport_name == 'nhl':
            self.strategy = NHLStrategy()
        elif sport_name == 'nba':
            self.strategy = NBAStrategy()
        else:
            raise ValueError(f"Unsupported sport: '{sport_name}'. No strategy available.")

        # Delegate map creation to the strategy
        self.display_to_full_name_map, self.full_name_to_slug_map = self.strategy.get_display_to_full_name_map()

    def discover_teams(self):
        # This method no longer needs sport_name, as it's part of the instance
        print(f"  Discovering teams and building URL slug map from {self.YAHOO_BASE_URL}/{self.sport_name}/teams/")
        teams_data = {}
        try:
            self.page.goto(f"{self.YAHOO_BASE_URL}/{self.sport_name}/teams/", wait_until="domcontentloaded", timeout=60000)
            team_info_containers = self.page.locator("div#team-info").all()
            print(f"    - Found {len(team_info_containers)} team containers to process.")
            for container in team_info_containers:
                link_element = container.locator("a._ys_lbjwi2")
                href = link_element.get_attribute('href')
                display_name = link_element.text_content().strip()

                full_name = self.display_to_full_name_map.get(display_name)
                if not full_name:
                    print(f"    - Could not map display name '{display_name}' to a full team name. Skipping.")
                    continue

                slug = self.full_name_to_slug_map.get(full_name)
                if not all([href, slug]):
                    print(f"    - Incomplete data for display name '{display_name}'. Skipping.")
                    continue

                teams_data[full_name] = {'url': href, 'slug': slug}
            return teams_data
        except PlaywrightError as e:
            print(f"  - Could not load the Yahoo teams page: {e}")
            return None
    # --- REVISED GENERIC PARSER (AGAIN) ---
    def _parse_stats_table(self, stat_block_locator):
        """
        Parses a standard Yahoo stats block. It's now more flexible.
        """
        try:
            # The category name is always in an H3 within the block
            category_name = stat_block_locator.locator("h3").text_content(timeout=5000).strip()

            # This method will now parse ALL tables within the block and return
            # a list of parsed table data.
            all_tables_data = []

            tables_in_block = stat_block_locator.locator("table").all()

            for table_locator in tables_in_block:
                header_cells = table_locator.locator('thead th').all()
                if not header_cells: continue

                header_full_names, header_abbreviations = [], []
                for th in header_cells[1:]:
                    title_div = th.locator('div[title]')
                    full_name = title_div.get_attribute('title') if title_div.count() > 0 else th.text_content().strip()
                    abbreviation = th.text_content().strip()
                    header_full_names.append(full_name or abbreviation)
                    header_abbreviations.append(abbreviation)
                headers = {'names': header_full_names, 'abbreviations': header_abbreviations}

                player_data = []
                player_rows = table_locator.locator('tbody tr').all()
                for row in player_rows:
                    player_name_th = row.locator('th a')
                    player_name_td = row.locator('td:first-child a')
                    is_in_th = player_name_th.count() > 0
                    is_in_td = player_name_td.count() > 0

                    if is_in_th: player_link = player_name_th
                    elif is_in_td: player_link = player_name_td
                    else: continue

                    player_name = player_link.text_content().strip()
                    if "TOTAL" in player_name.upper(): continue

                    player_url = player_link.get_attribute('href')
                    yahoo_id_match = re.search(r'(\d+)$', player_url)
                    if not yahoo_id_match: continue
                    yahoo_id = yahoo_id_match.group(1)

                    cols = row.locator('td').all()
                    start_index = 1 if is_in_td else 0
                    stat_values = [col.text_content().strip() for col in cols[start_index:]]
                    player_data.append({'name': player_name, 'id': yahoo_id, 'stats': stat_values})

                # Add the parsed data for this specific table to our list
                all_tables_data.append({'headers': headers, 'players': player_data})

            return category_name, all_tables_data
        except PlaywrightError as e:
            print(f"        - Error in generic table parser: {e}")
            return None, None
    def process_one_team(self, full_name, team_info, start_year, end_year):
        """
        :param full_name:
        :param team_info:
        :param start_year:
        :param end_year:
        :return:
        """
        if not self.sport_id:
            sport_info = self.db.execute_query("SELECT sport_id FROM sports WHERE name = %s", (self.sport_name.upper(),), fetch='one')
            self.sport_id = sport_info[0]

        team_db_id = self.db.get_or_create_team(self.sport_id, full_name, team_info['slug'])
        if not team_db_id: return

        for year in range(end_year, start_year - 1, -1):
            # Renamed from _scrape_yahoo_nfl_team_for_year
            self._scrape_team_for_year(year, team_db_id, full_name, team_info)


    # <select class="_ys_7jlmej _ys_pfo100 _ys_1xjsazm"><option value="2025" selected="">2025</option><option value="2024">2024</option><option value="2023">2023</option><option value="2022">2022</option><option value="2021">2021</option><option value="2020">2020</option><option value="2019">2019</option><option value="2018">2018</option><option value="2017">2017</option><option value="2016">2016</option><option value="2015">2015</option><option value="2014">2014</option><option value="2013">2013</option></select>
    def _scrape_team_for_year(self, year, team_db_id, team_name, team_info):
        print(f"    - Scraping data for {team_name} for season: {year}")

        stats_url = f"https://sports.yahoo.com{team_info['url']}stats/"
        self.page.goto(stats_url, wait_until="domcontentloaded", timeout=60000)
        try:
            season_selected = False
            target_year = str(year)

            # Let's first check if the correct year is already selected to avoid unnecessary actions
            # This is a good practice for efficiency.
            try:
                current_selection = self.page.locator('select[data-tst="season-dropdown"] option[selected]').get_attribute('value')
                if current_selection == target_year:
                    print(f"      - Season {year} is already selected. No action needed.")
                    season_selected = True
            except PlaywrightError:
                # If we can't find it, that's fine, we'll try to select it anyway.
                pass


            if not season_selected:
                # --- NEW LOGIC TO TRY MULTIPLE SELECTORS AND WAIT CORRECTLY ---
                print(f"      - Attempting to select season {year}...")
                try:
                    # Use a single block to attempt selection and then wait.
                    primary_selector = 'select[data-tst="season-dropdown"]'
                    alternate_selector = 'select._ys_7jlmej'

                    # Use a Promise-based approach to handle the selection action
                    # This ensures we are ready to listen for the network events right after the action.
                    with self.page.expect_response(lambda response: "sports.yahoo.com" in response.url, timeout=20000) as response_info:
                        # Try the primary selector first
                        try:
                            self.page.select_option(primary_selector, value=target_year, timeout=5000)
                            print(f"      - Selected season {year} (using primary selector).")
                        except PlaywrightError:
                            print("        - Primary season dropdown selector not found. Trying alternate...")
                            # If primary fails, try the alternate
                            self.page.select_option(alternate_selector, value=target_year, timeout=5000)
                            print(f"      - Selected season {year} (using alternate selector).")

                    # The `with` block automatically waits for a response.
                    # Now, we add an extra wait for the DOM to be re-rendered.
                    self.page.wait_for_load_state('networkidle', timeout=15000)
                    print(f"      - Network is idle. DOM should be updated for {year}.")

                    season_selected = True

                except PlaywrightError as e:
                    print(f"      - Could not find or select year {year} on stats page. Skipping. Error: {e}")
                except Exception as e:
                    # The expect_response might time out, which is a different exception.
                    print(f"      - Timed out waiting for a network response after selecting {year}. The page may not have updated. Error: {e}")


            # If a season was successfully selected (or was already selected), parse the stats
            if season_selected:
                # --- DEBUGGING STEP (Optional but Recommended) ---
                # You can uncomment this during testing to visually confirm the page has changed.
                # self.page.screenshot(path=f'debug_screenshot_{year}.png')

                self.page.wait_for_selector('div.ys-stats-table-wrapper', timeout=10000) # Wait for the table wrapper to be present
                self._parse_yahoo_team_season_stats(year, team_db_id)

        except Exception as e:
            # A general catch-all for any other unexpected errors during this process
            print(f"      - An unexpected error occurred during season stats processing for {year}: {e}")
        finally:
            try:
                schedule_url = f"https://sports.yahoo.com{team_info['url']}schedule/?scheduleType=list&season={year}"
                print(f"      - Navigating directly to schedule list view: {schedule_url}")
                self.page.goto(schedule_url, wait_until="domcontentloaded", timeout=60000)
                print(f"      - On schedule page for season {year}.")
                self.page.wait_for_timeout(2000)

                box_score_games = self._parse_yahoo_schedule_for_box_scores()
                if not box_score_games:
                    print(f"        - No completed games found for {year}.")
                    return # Use return here to stop processing for this year if no games are found

                print(f"        - Found {len(box_score_games)} completed games.")
                for game in box_score_games:
                    if game['game_id'] in self.scraped_games:
                        print(f"        - Skipping already parsed game vs {game['opponent_display_name']} (ID: {game['game_id']})")
                        continue

                    opponent_full_name = self.display_to_full_name_map.get(game['opponent_display_name'])
                    if not opponent_full_name:
                        print(f"        - Could not map opponent display name '{game['opponent_display_name']}'. Skipping game.")
                        continue

                    # This line caused the error in a previous attempt and needs to be correct.
                    # It relies on the full_name_to_slug_map being available from your strategy.
                    opponent_slug = self.full_name_to_slug_map.get(opponent_full_name)
                    if not opponent_slug:
                        print(f"        - Could not map opponent '{opponent_full_name}' to a slug. Skipping.")
                        continue

                    opponent_team_id = self.db.get_or_create_team(self.sport_id, opponent_full_name, opponent_slug)
                    print(f"        -> Scraping Box Score vs {opponent_full_name}")

                    # --- *** THIS IS THE ONLY CHANGE TO YOUR ORIGINAL LOGIC *** ---
                    # Call the correct parser based on the sport.
                    if self.sport_name == 'nba':
                        # Call the new, specialized parser for NBA pages
                        self._parse_nba_box_score(game['url'], year, team_db_id, opponent_team_id, team_name, opponent_full_name)
                    else:
                        # Call your original, default parser for NFL, NHL, and MLB
                        self._parse_yahoo_box_score(game['url'], year, team_db_id, opponent_team_id, team_name, opponent_full_name)
                    # --- END OF CHANGE ---

                    self.scraped_games.add(game['game_id'])
                    time.sleep(2) # Politeness delay

            except PlaywrightError as e:
                print(f"      - An error occurred during schedule processing. Skipping game stats for {year}. Error: {e}")
    def _parse_yahoo_team_season_stats(self, year, team_id):
        print("        - Parsing season-level player stats...")
        try:
            stat_wrappers = self.page.locator('div.ys-stats-table-wrapper').all()

            if not stat_wrappers:
                print("        - New layout not found, checking for old layout...")
                stat_wrappers = self.page.locator('#team-stats-player-tables section > div > div > table').all()

                # If after checking both, we still have nothing, then exit.
            if not stat_wrappers:
                print("        - No season stat wrappers found on page for either layout.")
                time.sleep(1000)
                return


            saved_categories = 0
            for wrapper in stat_wrappers:
                category, tables_data = self._parse_stats_table(wrapper)

                # Season stats pages only have one table per wrapper
                if not all([category, tables_data]) or not tables_data[0]['players']:
                    print(f"        - Skipping stat block '{category or 'Unknown'}' due to no player data.")
                    continue

                table_info = tables_data[0]
                headers = table_info['headers']
                players = table_info['players']

                stat_def_id = self.db.get_or_create_stat_definition(
                    self.sport_id, category, headers['names'], headers['abbreviations']
                )
                if not stat_def_id: continue

                for player in players:
                    player_db_id = self.db.get_or_create_player(self.sport_id, player['name'], player['id'])
                    if not player_db_id: continue

                    self.db.insert_player_season_stats(
                        player_id=player_db_id, team_id=team_id, season=year,
                        stat_def_id=stat_def_id, stat_values=player['stats']
                    )
                saved_categories += 1

            if saved_categories > 0:
                print(f"        - Successfully parsed and saved season stats for {saved_categories} categories.")
        except PlaywrightError as e:
            print(f"        - An error occurred while processing season stats: {e}")
    def _parse_yahoo_schedule_for_box_scores(self):
        games = []
        try:
            rows = self.page.locator("table.latest-results-table tbody tr").all()
            for row in rows:
                row_text = row.text_content()
                if "Bye Week" in row_text or "(Preseason)" in row_text: continue

                box_score_link_element = row.locator('td:first-child a')
                if box_score_link_element.count() == 0: continue

                # --- FIXED ---
                # Use a more specific selector to avoid grabbing extra "series score" spans in playoffs.
                opponent_name_element = row.locator('td:nth-child(3) > span > span:first-child')
                if opponent_name_element.count() == 0: continue # Skip if element not found

                href = box_score_link_element.get_attribute('href')
                game_id_match = re.search(r'-(\d{10,})/?$', href)

                if game_id_match:
                    game_id = game_id_match.group(1)
                    # The new selector is more precise, so the complex regex is less critical,
                    # but it's good to keep for cleaning up any remaining score text.
                    opponent_display_name = re.sub(r'\s*\d+-\d+(-\d+)?\s*$', '', opponent_name_element.text_content()).strip()
                    games.append({'opponent_display_name': opponent_display_name, 'url': href, 'game_id': game_id})
        except PlaywrightError as e:
            # The error will still print here if something goes wrong, which is good for debugging.
            print(f"        - An error occurred while parsing the schedule table: {e}")
        return games
    # --- REFACTORED game stats parser (to use the new return format) ---
    # Paste this entire method into your class, replacing the old one.
    def _parse_yahoo_box_score(self, box_score_url, year, main_team_db_id, opponent_team_id, main_team_name, opponent_team_name):
        print(f"           - Parsing {self.YAHOO_BASE_URL}{box_score_url}")
        try:
            # Increased timeout and used 'domcontentloaded' as the page is JS-heavy
            self.page.goto(f"{self.YAHOO_BASE_URL}{box_score_url}", wait_until="domcontentloaded", timeout=90000)
            # Wait for the main stats container to be present to ensure the page has loaded
            match_stats_container = self.page.locator("div.match-stats")
            match_stats_container.wait_for(timeout=30000)
        except PlaywrightError as e:
            print(f"           - CRITICAL Error navigating to or loading box score page: {e}")
            return

        source_game_id_match = re.search(r'-(\d{10,})/?$', box_score_url)
        if not source_game_id_match:
            print("           - CRITICAL WARNING: Could not extract Yahoo Game ID from URL. Skipping game.")
            return
        source_game_id = source_game_id_match.group(1)

        players_saved_this_game = set()
        try:
            # --- Team Name and ID Mapping ---
            # This selector targets the specific container for the team headers
            team_header_container = match_stats_container.locator("div.D\\(f\\).Mx\\(-10px\\).Jc\\(sb\\).Pt\\(16px\\)")
            team_headers = team_header_container.locator("a").all()
            if len(team_headers) != 2:
                print(f"           - Could not find the two team headers. Aborting parse for this game.")
                return

            left_team_name = team_headers[0].text_content().strip()
            column_to_team_id = {}
            # Simple check to see which team is in the left column
            if main_team_name in left_team_name:
                column_to_team_id[0] = main_team_db_id
                column_to_team_id[1] = opponent_team_id
            else:
                column_to_team_id[0] = opponent_team_id
                column_to_team_id[1] = main_team_db_id

            # --- Stat Block Parsing ---
            # This is the key change: Find all direct child divs of 'div.match-stats' that contain a table.
            # This is much more robust than relying on the fragile utility classes.
            stat_blocks = match_stats_container.locator("div:has(table.W\\(100\\%\\))").all()

            for block in stat_blocks:
                tables_in_block = block.locator("table").all()
                if len(tables_in_block) != 2:
                    # This block might not be a standard stat block, skip it.
                    continue

                # --- Category and Header Parsing ---
                # Get category from the FIRST th in the FIRST table's header
                first_table_headers = tables_in_block[0].locator("thead th").all()
                if not first_table_headers:
                    continue

                category = first_table_headers[0].text_content().strip()
                # Get the stat names and their abbreviations (from the 'title' attribute)
                header_names = [th.get_attribute('title')  for th in first_table_headers[1:]]
                header_abbreviations = [th.text_content().strip() or '' for th in first_table_headers[1:]]

                # Defensive check for empty category
                if not category:
                    print("           - WARNING: Found a stat block with no category name. Skipping.")
                    continue

                stat_def_id = self.db.get_or_create_stat_definition(
                    self.sport_id, category, header_names, header_abbreviations
                )
                if not stat_def_id:
                    print(f"           - WARNING: Could not get/create stat definition for category '{category}'.")
                    continue

                # Process each of the two tables in the block
                for i, table in enumerate(tables_in_block):
                    team_id = column_to_team_id[i]
                    opponent_id = column_to_team_id[1 - i]

                    player_rows = table.locator("tbody tr").all()
                    for row in player_rows:
                        player_anchor = row.locator("th a").first
                        player_name = player_anchor.text_content().strip()
                        player_url = player_anchor.get_attribute('href')

                        player_source_id_match = re.search(r'/(\d+)', player_url or '')
                        player_source_id = player_source_id_match.group(1) if player_source_id_match else None

                        if not all([player_name, player_source_id]):
                            continue

                        player_db_id = self.db.get_or_create_player(self.sport_id, player_name, player_source_id)
                        if player_db_id:
                            players_saved_this_game.add(player_db_id)

                            # Get all stat values for the current player row
                            stat_values = [td.text_content().strip() for td in row.locator("td").all()]

                            self.db.insert_player_game_stats(
                                player_id=player_db_id, team_id=team_id, opponent_team_id=opponent_id,
                                season=year, season_type="Regular", source_game_id=source_game_id,
                                stat_def_id=stat_def_id, stat_values=stat_values
                            )

            if players_saved_this_game:
                print(f"           - Success: Parsed and saved stats for {len(players_saved_this_game)} unique players.")
            else:
                print(f"           - CRITICAL WARNING: Finished parsing, but ZERO players were saved for this game.")

        except PlaywrightError as e:
            print(f"           - CRITICAL Error parsing box score page: {e}")
        except Exception as e:
            # Catch other potential errors during parsing
            import traceback
            print(f"           - An unexpected error occurred: {e}")
            traceback.print_exc()
    def _parse_nba_box_score(self, box_score_url, year, main_team_db_id, opponent_team_id, main_team_name, opponent_team_name):
        """
        SPECIALIZED PARSER for the unique NBA box score page layout.
        """
        print(f"           - Parsing (NBA SPECIFIC) {self.YAHOO_BASE_URL}{box_score_url}")
        try:
            self.page.goto(f"{self.YAHOO_BASE_URL}{box_score_url}", wait_until="domcontentloaded", timeout=60000)
            # The main container for NBA stats is div.player-stats
            main_container = self.page.locator("div.player-stats").first
            main_container.wait_for(timeout=30000)
        except PlaywrightError as e:
            print(f"           - CRITICAL Error navigating to or loading NBA box score page: {e}")
            return

        source_game_id_match = re.search(r'-(\d{10,})/?$', box_score_url)
        if not source_game_id_match:
            print("           - CRITICAL WARNING: Could not extract Yahoo Game ID. Skipping game.")
            return
        source_game_id = source_game_id_match.group(1)

        players_saved_this_game = set()
        try:
            # --- Team Name and ID Mapping for NBA ---
            team_headers = main_container.locator("div.ys-section-header h3 span.Va\\(m\\)").all_text_contents()
            if len(team_headers) < 2:
                print(f"           - Could not find the two NBA team headers. Aborting parse for this game.")
                return

            column_to_team_id = {}
            # The main team name might contain the city name from the header (e.g. "Boston Celtics" contains "Boston")
            if any(h in main_team_name for h in team_headers[0].split()):
                column_to_team_id[0] = main_team_db_id
                column_to_team_id[1] = opponent_team_id
            else:
                column_to_team_id[0] = opponent_team_id
                column_to_team_id[1] = main_team_db_id

            # --- Stat Block Parsing for NBA ---
            all_team_stat_blocks = main_container.locator("div.Mt\\(21px\\)").all()
            if len(all_team_stat_blocks) < 2: return

            for i, team_block in enumerate(all_team_stat_blocks):
                team_id = column_to_team_id[i]
                opponent_id = column_to_team_id[1 - i]

                stat_tables = team_block.locator("table").all()
                for table in stat_tables:
                    header_cells = table.locator("thead th").all()
                    if not header_cells: continue

                    category = header_cells[0].text_content().strip()
                    header_names = [th.get_attribute('title') for th in header_cells[1:]]
                    header_abbreviations = [th.text_content().strip() for th in header_cells[1:]]

                    stat_def_id = self.db.get_or_create_stat_definition(
                        self.sport_id, category, header_names, header_abbreviations
                    )
                    if not stat_def_id: continue

                    for row in table.locator("tbody tr").all():
                        player_anchor = row.locator("th a").first
                        if player_anchor.count() == 0: continue

                        player_name = player_anchor.text_content().strip()
                        player_url = player_anchor.get_attribute('href')
                        player_source_id_match = re.search(r'/(\d+)', player_url or '')
                        if not player_source_id_match: continue

                        player_db_id = self.db.get_or_create_player(self.sport_id, player_name, player_source_id_match.group(1))
                        if not player_db_id: continue

                        players_saved_this_game.add(player_db_id)
                        stat_values = [td.text_content().strip() for td in row.locator("td").all()]

                        self.db.insert_player_game_stats(
                            player_id=player_db_id, team_id=team_id, opponent_team_id=opponent_id,
                            season=year, season_type="Regular", source_game_id=source_game_id,
                            stat_def_id=stat_def_id, stat_values=stat_values
                        )

            if players_saved_this_game:
                print(f"           - Success: Parsed and saved stats for {len(players_saved_this_game)} unique players.")
            else:
                print(f"           - WARNING: Finished parsing NBA game, but ZERO players were saved.")
        except Exception as e:
            import traceback
            print(f"           - An unexpected error occurred in the NBA parser: {e}")
            traceback.print_exc()