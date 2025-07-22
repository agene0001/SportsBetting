import re
import time
import traceback
from playwright.sync_api import Error as PlaywrightError, Page, BrowserContext, sync_playwright, expect

# --- NEW IMPORTS ---
import threading
from concurrent.futures import ThreadPoolExecutor

# Import the new strategies
from .scraping_strategies import NFLStrategy, NHLStrategy, NBAStrategy, MLBStrategy

# You can adjust this, but 2 is a safe starting point to avoid getting blocked.
game_threads = 2

class YAHOOScraper:
    def __init__(self, sport_name, db_manager):
        """
        Initializes the scraper with a specific sport, database manager,
        and a shared shutdown event for graceful termination.
        """
        self.db = db_manager
        self.sport_name = sport_name
        self.sport_id = None
        self.YAHOO_BASE_URL = "https://sports.yahoo.com"

        # --- THREAD-SAFE GAME TRACKING ---
        self.scraped_games = set()
        self.scraped_games_lock = threading.Lock()

        # FACTORY: Select the correct strategy based on sport name
        if sport_name == 'nfl':
            self.strategy = NFLStrategy()
        elif sport_name == 'nhl':
            self.strategy = NHLStrategy()
        elif sport_name == 'nba':
            self.strategy = NBAStrategy()
        elif sport_name == 'mlb':
            self.strategy = MLBStrategy()
        else:
            raise ValueError(f"Unsupported sport: '{sport_name}'. No strategy available.")

        self.display_to_full_name_map, self.full_name_to_slug_map = self.strategy.get_display_to_full_name_map()

    def discover_teams(self):
        """
        Connects to Yahoo, discovers all teams for the sport, and returns their data.
        This method is self-contained and manages its own browser lifecycle.
        """
        print(f"  Discovering teams and building URL slug map from {self.YAHOO_BASE_URL}/{self.sport_name}/teams/")
        teams_data = {}
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(bypass_csp=True,user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',locale='en-US', timezone_id='America/New_York')
            try:
                page.goto(f"{self.YAHOO_BASE_URL}/{self.sport_name}/teams/", wait_until="load", timeout=60000)
                team_info_containers = page.locator("div#team-info").all()
                print(f"    - Found {len(team_info_containers)} team containers to process.")

                for container in team_info_containers:
                    link_element = container.locator("a._ys_lbjwi2")
                    href = link_element.get_attribute('href')
                    display_name = link_element.text_content().strip()

                    full_name = self.display_to_full_name_map.get(display_name)
                    if not full_name: continue

                    slug = self.full_name_to_slug_map.get(full_name)
                    if not all([href, slug]): continue

                    teams_data[full_name] = {'url': href, 'slug': slug}
                return teams_data
            except PlaywrightError as e:
                print(f"  - CRITICAL FAILURE during team discovery: {e}")
                return None
            finally:
                print("  Team discovery browser closed.")

    def process_one_team(self, full_name, team_info, start_year, end_year):
        """
        Worker function to process all seasons for a single team. This is called
        by the ThreadPoolExecutor and creates its own browser to ensure thread safety.
        """
        max_retries = 2
        for attempt in range(max_retries):
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(bypass_csp=True,user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',locale='en-US', timezone_id='America/New_York')
                try:
                    print(f"\n{'='*25}\nProcessing Team: {full_name} (Thread: {threading.get_ident()}, Attempt: {attempt + 1})\n{'='*25}")

                    if not self.sport_id:
                        sport_info = self.db.execute_query("SELECT sport_id FROM sports WHERE name = %s", (self.sport_name.upper(),), fetch='one')
                        if sport_info: self.sport_id = sport_info[0]

                    team_db_id = self.db.get_or_create_team(self.sport_id, full_name, team_info['slug'])
                    if not team_db_id: return

                    for year in range(end_year, start_year - 1, -1):
                        base_schedule_url = f"https://sports.yahoo.com{team_info['url']}schedule/"
                        if "?" in base_schedule_url:
                            schedule_url = f"{base_schedule_url}&scheduleType=list&season={year}"
                        else:
                            schedule_url = f"{base_schedule_url}?scheduleType=list&season={year}"

                        print(f"      - [{full_name}] Navigating to schedule for season {year}: {schedule_url}")
                        page.goto(schedule_url, wait_until="domcontentloaded", timeout=60000)

                        print(f"      - [{full_name}] Waiting for schedule content to load for {year}...")

                        primary_schedule_container_selector = "div[data-testid='sched-filter-results']" # Preferred, modern table view
                        secondary_schedule_container_selector = "div#schedule-list" # Fallback, list view

                        parsing_mode = None
                        try:
                            print(f"        Trying primary selector: {primary_schedule_container_selector}")
                            page.wait_for_selector(primary_schedule_container_selector, timeout=15000) # Shorter timeout for the first try
                            print(f"        Primary schedule container found ('{primary_schedule_container_selector}').")
                            parsing_mode = 'table_view'
                        except PlaywrightError:
                            print(f"        Primary selector timed out. Trying secondary selector: {secondary_schedule_container_selector}")
                            try:
                                page.wait_for_selector(secondary_schedule_container_selector, timeout=15000)
                                print(f"        Secondary schedule container found ('{secondary_schedule_container_selector}').")
                                parsing_mode = 'list_view'
                            except PlaywrightError:
                                print(f"        Both primary and secondary schedule selectors failed for {full_name} - {year}.")
                                page.screenshot(path=f"debug_schedule_load_failure_{full_name}_{year}.png")
                                print(f"        Screenshot saved to debug_schedule_load_failure_{full_name}_{year}.png")
                                continue # Skip to next year or team

                        if not parsing_mode:
                            # Should not happen if one of the try-except blocks succeeded or continued
                            print(f"        Could not determine parsing mode for {full_name} - {year}.")
                            continue

                        # Pass the main team's full name for opponent disambiguation in list_view
                        box_score_games = self._parse_yahoo_schedule_for_box_scores(page, parsing_mode, full_name)

                        if not box_score_games:
                            print(f"        - [{full_name}] No completed games found for {year} (mode: {parsing_mode}).")
                            continue

                        print(f"        - [{full_name}] Found {len(box_score_games)} games for {year}. Scraping in parallel...")

                        with ThreadPoolExecutor(max_workers=game_threads) as game_executor:
                            futures = [
                                game_executor.submit(self._scrape_one_box_score_worker, game, year, team_db_id, full_name)
                                for game in box_score_games
                            ]
                            for future in futures:
                                future.result()
                    break
                except (PlaywrightError, ValueError) as e:
                    print(f"  - ATTEMPT FAILED for team {full_name} with error: {e}")
                    if attempt < max_retries - 1:
                        print("  - Retrying...")
                        time.sleep(5)
                    else:
                        print(f"  - MAX RETRIES REACHED for {full_name}. This team failed.")
                        traceback.print_exc()
                finally:
                    page.close()
                    browser.close()

    def _scrape_one_box_score_worker(self, game, year, team_db_id, team_name):
        """
        Self-contained worker to scrape a single box score.
        Now includes its own error handling to prevent the thread from crashing.
        """
        game_id = game['game_id']
        with self.scraped_games_lock:
            if game_id in self.scraped_games:
                # print(f"        -> Skipping already processed game {game_id}")
                return True # Return True for a successful skip

        opponent_full_name = self.display_to_full_name_map.get(game['opponent_display_name'])
        if not opponent_full_name:
            return True # Not an error, just no mapping for this opponent
        opponent_slug = self.full_name_to_slug_map.get(opponent_full_name)
        if not opponent_slug:
            return True # Not an error

        opponent_team_id = self.db.get_or_create_team(self.sport_id, opponent_full_name, opponent_slug)

        print(f"        -> Scraping Box Score vs {opponent_full_name} (ID: {game_id})")

        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context()
            page = context.new_page()

            # --- THE FIX IS HERE: Wrap the parsing logic in a try/except block ---
            try:
                href_from_schedule = game['url']
                if href_from_schedule.startswith("http://") or href_from_schedule.startswith("https://"):
                    full_url = href_from_schedule
                else:
                    # Ensure leading slashes are handled correctly if it's a relative path
                    if href_from_schedule.startswith("/"):
                        full_url = f"{self.YAHOO_BASE_URL}{href_from_schedule}"
                    else:
                        full_url = f"{self.YAHOO_BASE_URL}/{href_from_schedule}"

                print(f"           Navigating to box score: {full_url}") # Add this print for debugging
                page.goto(full_url, wait_until="domcontentloaded", timeout=60000)

                # This is where your _parse_odds_information is called
                # It will correctly return None if it fails all retries.
                odds_data = self._parse_odds_information(page)
                team_stats_data = self._parse_team_stats(page)

                if odds_data:
                    home_team_odds_name, away_team_odds_name, home_ml, away_ml, spread, total = odds_data
                    # Determine home/away based on the names returned from the odds table
                    if team_name in home_team_odds_name:
                        home_team_id, away_team_id = team_db_id, opponent_team_id
                    else:
                        home_team_id, away_team_id = opponent_team_id, team_db_id

                    self.db.insert_game_details(
                        game_id, self.sport_id, home_team_id, away_team_id,
                        spread, total, home_ml, away_ml, team_stats_data
                    )
                    print(f"           - Success: Saved game details for game {game_id}.")
                else:
                    # This is a graceful failure. The function returned None as designed.
                    print(f"           - Warning: Could not parse odds info for game {game_id}.")

                # Now parse player stats
                self._parse_player_stats(page, team_db_id, opponent_team_id, team_name, game_id, year)

                with self.scraped_games_lock:
                    self.scraped_games.add(game_id)

                return True # Indicate success for this worker task

            except Exception as e:
                # This is a broader catch for any unexpected errors (like navigation failure)
                print(f"        -! UNEXPECTED WORKER FAILURE for game {game_id}. Error: {e}")
                traceback.print_exc()
                return False # Indicate failure for this worker task

            finally:
                # This block will always run, ensuring the browser is closed.
                page.close()
                context.close()
                browser.close()
    def _parse_nba_box_score(self, page: Page, box_score_url, year, main_team_db_id, opponent_team_id, main_team_name, opponent_team_name):
        try:
            page.goto(f"{self.YAHOO_BASE_URL}{box_score_url}", wait_until="domcontentloaded", timeout=60000)
            main_container = page.locator("div.player-stats").first
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
            team_headers = main_container.locator("div.ys-section-header h3 span.Va\\(m\\)").all_text_contents()
            if len(team_headers) < 2:
                print(f"           - Could not find the two NBA team headers. Aborting parse for this game.")
                return

            column_to_team_id = {}
            if any(h in main_team_name for h in team_headers[0].split()):
                column_to_team_id[0] = main_team_db_id
                column_to_team_id[1] = opponent_team_id
            else:
                column_to_team_id[0] = opponent_team_id
                column_to_team_id[1] = main_team_db_id

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

                    stat_def_id = self.db.get_or_create_stat_definition(self.sport_id, category, header_names, header_abbreviations)
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
                            season=year, source_game_id=source_game_id,
                            stat_def_id=stat_def_id, stat_values=stat_values
                        )

            if players_saved_this_game:
                print(f"           - Success [Thread-{threading.get_ident()}]: Parsed and saved stats for {len(players_saved_this_game)} players in game {source_game_id}.")
            else:
                print(f"           - WARNING [Thread-{threading.get_ident()}]: Finished parsing NBA game, but ZERO players were saved for game {source_game_id}.")
        except Exception as e:
            print(f"           - An unexpected error occurred in the NBA parser: {e}")
            traceback.print_exc()
    # In YAHOOScraper class:
    def _parse_yahoo_schedule_for_box_scores(self, page: Page, parsing_mode: str, current_team_full_name: str):
        """
        Parses a team's schedule page to find links to completed game box scores.
        Handles different page structures based on parsing_mode.
        'current_team_full_name' is the name of the team whose schedule is being parsed.
        """
        games = []
        print(f"        Parsing schedule with mode: {parsing_mode}")

        if parsing_mode == 'table_view':
            # --- Logic for the TABLE structure (similar to your previous refined version) ---
            schedule_table_rows_selector = "div[data-testid='sched-filter-results'] table tbody tr"
            try:
                print(f"        - Locating rows with table selector: {schedule_table_rows_selector}")
                rows = page.locator(schedule_table_rows_selector).all()
                if not rows:
                    # Fallback if the primary table selector changed slightly
                    fallback_table_selector = "table.latest-results-table tbody tr"
                    print(f"        - Primary table selector found no rows. Trying fallback: {fallback_table_selector}")
                    rows = page.locator(fallback_table_selector).all()

                print(f"        - Found {len(rows)} potential game rows in table view.")
                for i, row in enumerate(rows):
                    row_text_content = row.text_content()
                    print(f"          Processing table row {i+1}: {row_text_content[:100]}...")

                    if "Bye Week" in row_text_content or \
                            "(Preseason)" in row_text_content or \
                            any(status in row_text_content for status in ["Postponed", "Canceled", "Suspended", "TBD"]):
                        print(f"            Skipping row (status/type): {row_text_content[:50]}")
                        continue

                    # Link finding (adjust as needed from your previous robust version)
                    box_score_link = row.locator('td a[href*="/gamelog/"], td a[href*="/recap?"], td a[href*="/boxscore"], td:first-child a[href], td:nth-child(2) a[href]').first
                    if not box_score_link.is_visible():
                        print(f"        - Warning: No visible box score link in table row: {row_text_content[:100]}")
                        continue

                    href = box_score_link.get_attribute('href')
                    if not href or not re.search(r'/([a-z0-9-]+-\d{8,})/?$', href): # More general game ID pattern
                        print(f"        - Warning: Link href doesn't look like a box score: {href}. Skipping row.")
                        continue

                    game_id_match = re.search(r'-(\d{8,})/?$', href)
                    if not game_id_match: # Should be caught by previous check, but good to be safe
                        print(f"        - Warning: Could not extract game_id from href: {href}")
                        continue
                    game_id = game_id_match.group(1)

                    # Opponent name finding for table view
                    opponent_cell_candidate_selectors = [
                        "td:nth-child(2) span[data-tst='opponent-name']",
                        "td:nth-child(3) span[data-tst='opponent-name']", # Sometimes it's the 3rd child
                        "td:nth-child(2) a > span",
                        "td:nth-child(3) a > span",
                        "td:nth-child(3) > span > span:first-child", # Your original
                        "td:has-text('@') span",
                        "td:has-text('vs') span"
                    ]
                    opponent_display_name = "Unknown Opponent"
                    for opp_sel in opponent_cell_candidate_selectors:
                        opponent_name_element = row.locator(opp_sel).first
                        if opponent_name_element.is_visible():
                            name_text = opponent_name_element.text_content().strip()
                            name_text = re.sub(r'\s*\d+-\d+(-\d+)?\s*$', '', name_text).strip()
                            name_text = re.sub(r'\s+\d+$', '', name_text).strip()
                            if name_text and len(name_text) > 1 : # Avoid single letters or empty strings
                                opponent_display_name = name_text
                                break

                    print(f"          Found game (table): ID={game_id}, Opponent={opponent_display_name}, URL={href}")
                    games.append({'opponent_display_name': opponent_display_name, 'url': href, 'game_id': game_id})

            except PlaywrightError as e:
                print(f"        - An error occurred while parsing schedule table view: {e}")
                page.screenshot(path=f"debug_schedule_table_error_{page.title().replace(' ','_')}.png")
                print(f"        - Screenshot saved to debug_schedule_table_error_{page.title().replace(' ','_')}.png")

        elif parsing_mode == 'list_view':
            # --- Logic for the div#schedule-list structure ---
            # Each game is an <li> with class e.g. "_ys_sjaa1b _ys_1lru6g2"
            # The link is <a class="_ys_1vd099q">
            # Team names are in <span class="_ys_y2r9ts">

            list_item_selector = "div#schedule-list li"
            try:
                print(f"        - Locating items with list selector: {list_item_selector}")
                list_items = page.locator(list_item_selector).all()
                print(f"        - Found {len(list_items)} potential game list items.")

                for i, item in enumerate(list_items):
                    item_text_content = item.text_content()
                    # print(f"          Processing list item {i+1}: {item_text_content[:150]}...")

                    # Check for game status like Ppd (Postponed), Canceled, etc.
                    # These might be within a span with id="game-state" or similar
                    game_state_element = item.locator("div#game-state span, span").first # Check for specific status spans
                    game_state_text = ""
                    if game_state_element.is_visible():
                        game_state_text = game_state_element.text_content().strip().upper()

                    if any(status in game_state_text for status in ["PPD", "POSTPONED", "CANCELED", "CANCELLED", "SUSPENDED", "TBD"]):
                        print(f"            Skipping list item (status: {game_state_text}): {item_text_content[:50]}")
                        continue

                    # Check if it's a "Bye Week" or "Preseason" if such text exists directly in item
                    if "Bye Week" in item_text_content or "(Preseason)" in item_text_content:
                        print(f"            Skipping list item (type): {item_text_content[:50]}")
                        continue

                    link_element = item.locator("a").first
                    if not link_element.is_visible():
                        print(f"        - Warning: No visible link element in list item: {item_text_content[:100]}")
                        continue

                    href = link_element.get_attribute('href')
                    if not href or not re.search(r'/([a-z0-9-]+-\d{8,})/?$', href):
                        print(f"        - Warning: Link href in list item doesn't look like a box score: {href}. Skipping item.")
                        continue

                    game_id_match = re.search(r'-(\d{8,})/?$', href)
                    if not game_id_match:
                        print(f"        - Warning: Could not extract game_id from list item href: {href}")
                        continue
                    game_id = game_id_match.group(1)

                    # Extract both team names
                    team_name_elements = item.locator("span._ys_y2r9ts").all_text_contents()
                    # Clean the names: " Baltimore Orioles BAL " -> "Baltimore Orioles"
                    cleaned_team_names = [re.sub(r'\s+[A-Z]{2,3}\s*$', '', name).strip() for name in team_name_elements]

                    opponent_display_name = "Unknown Opponent"
                    if len(cleaned_team_names) == 2:
                        # We need to identify which of the two is the opponent.
                        # current_team_full_name is the team whose schedule page we are on.
                        if current_team_full_name.lower() in cleaned_team_names[0].lower():
                            opponent_display_name = cleaned_team_names[1]
                        elif current_team_full_name.lower() in cleaned_team_names[1].lower():
                            opponent_display_name = cleaned_team_names[0]
                        else:
                            # This case means current_team_full_name didn't match either,
                            # which could happen if team names are slightly different.
                            # As a fallback, assume the second team is the opponent if the first contains "vs" or "@",
                            # or just pick one if context isn't clear. For now, let's be cautious.
                            print(f"          Warning: Could not reliably determine opponent for {current_team_full_name} from names: {team_name_elements}")
                            # Heuristic: if one team name contains "vs" or "@", the other is the main team.
                            # This part of the HTML is complex, so we're taking the first one that's NOT the current team.
                            if current_team_full_name not in team_name_elements[0]:
                                opponent_display_name = cleaned_team_names[0]
                            else:
                                opponent_display_name = cleaned_team_names[1]


                    elif len(cleaned_team_names) == 1: # Could happen if it's a "vs TeamX" format somewhere else
                        opponent_display_name = cleaned_team_names[0]
                    else:
                        print(f"          Warning: Found {len(cleaned_team_names)} team names in list item. Expected 2. Names: {team_name_elements}")
                        # Attempt to find opponent name via other common patterns within the list item
                        opponent_search_elements = item.locator("div._ys_tx32sl span._ys_y2r9ts").all()
                        potential_opponents = []
                        for el in opponent_search_elements:
                            name_text = el.text_content().strip()
                            name_text = re.sub(r'\s+[A-Z]{2,3}\s*$', '', name_text).strip()
                            if name_text and name_text.lower() not in current_team_full_name.lower() and len(name_text) > 1:
                                potential_opponents.append(name_text)
                        if potential_opponents:
                            opponent_display_name = potential_opponents[0] # Take the first non-current team name found
                        else:
                            print(f"          Could not identify opponent name in list item: {item_text_content[:100]}")
                            continue # Skip if no opponent found

                    # Final check if game is still marked as "To Be Determined" or similar after score parsing
                    if "TBD" in item.locator("._ys_1yxp8my").text_content().upper(): # Check game date/time area
                        print(f"            Skipping list item (still TBD): {item_text_content[:50]}")
                        continue
                    if not item.locator("div._ys_1gl9ke6").all_text_contents(): # No scores present
                        print(f"            Skipping list item (no scores found, likely not played): {item_text_content[:50]}")
                        continue

                    print(f"          Found game (list): ID={game_id}, Opponent={opponent_display_name}, URL={href}")
                    games.append({'opponent_display_name': opponent_display_name, 'url': href, 'game_id': game_id})

            except PlaywrightError as e:
                print(f"        - An error occurred while parsing schedule list view: {e}")
                page.screenshot(path=f"debug_schedule_list_error_{page.title().replace(' ','_')}.png")
                print(f"        - Screenshot saved to debug_schedule_list_error_{page.title().replace(' ','_')}.png")

        else:
            print(f"        - Unknown parsing_mode: {parsing_mode}")

        if not games:
            print(f"        - No games extracted in mode '{parsing_mode}'. Dumping page HTML for review.")
            try:
                with open(f"debug_no_games_{parsing_mode}_{page.title().replace(' ','_')[:50]}.html", "w", encoding="utf-8") as f:
                    f.write(page.content())
            except Exception as dump_err:
                print(f"          Failed to dump HTML: {dump_err}")
        return games
    # --- MAIN PARSING SUPERVISOR ---
    def _parse_yahoo_box_score(self, page: Page, box_score_url, year, main_team_db_id, opponent_team_id, main_team_name, opponent_team_name):
        """
        Controller for scraping a single box score page. It navigates and then
        calls helper functions to parse different sections of the page.
        """
        try:
            full_url = f"{self.YAHOO_BASE_URL}{box_score_url}"
            page.goto(full_url, wait_until="domcontentloaded", timeout=90000)
            team_headers_locator = page.locator("div.match-stats")
            team_headers_locator1 = page.locator("div.match-stats .D\\(f\\).Jc\\(sb\\) a").nth(1)

            # Add fallbacks for specific stat tables
            passing_header_locator = page.locator("div.match-stats h4:has-text('PASSING')")
            batting_header_locator = page.locator("div.match-stats h4:has-text('BATTING')")

            # Combine them into a single super-locator
            # This tells Playwright to wait for ANY of these to become visible.
            reliable_content_locator = team_headers_locator.or_(team_headers_locator1).or_(passing_header_locator).or_(batting_header_locator)

            # Now, wait for the first element found by any of these strategies to be visible.
            reliable_content_locator.first.wait_for(state="visible", timeout=30000)

            # We wait for the first element matching ANY of these strategies to be visible.

            print(f"           - Box score content loaded.")
            page.locator("div.match-stats").wait_for(state="visible", timeout=30000)
        except PlaywrightError as e:
            print(f"           - CRITICAL Error navigating to or loading main page container: {e}")
            return

        source_game_id_match = re.search(r'-(\d{8,})/?$', box_score_url)
        if not source_game_id_match: return
        source_game_id = source_game_id_match.group(1)

        odds_data = self._parse_odds_information(page)
        team_stats_data = self._parse_team_stats(page)

        if odds_data:
            home_team_odds_name, away_team_odds_name, home_ml, away_ml, spread, total = odds_data
            if main_team_name in home_team_odds_name:
                home_team_id, away_team_id = main_team_db_id, opponent_team_id
            else:
                home_team_id, away_team_id = opponent_team_id, main_team_db_id

            self.db.insert_game_details(
                source_game_id, self.sport_id, home_team_id, away_team_id,
                spread, total, home_ml, away_ml, team_stats_data
            )
            print(f"           - Success: Saved game details for game {source_game_id}.")
        else:
            print(f"           - Warning: Could not parse odds info for game {source_game_id}.")

        self._parse_player_stats(page, main_team_db_id, opponent_team_id, main_team_name, source_game_id,year)

    # --- HELPER PARSERS ---

    def _parse_odds_information(self, page: Page):
        """
        Parses odds with a retry mechanism. If an attempt fails, it will
        reload the page and try again up to a specified number of times.
        It handles both completed and upcoming game pages and provides detailed debugging.
        """
        max_attempts = 3
        retry_delay_seconds = 3  # Wait 3 seconds between retries

        for attempt in range(max_attempts):
            try:
                print(f"--- Attempt {attempt + 1} of {max_attempts} ---")

                # --- STEP 1: Click the Odds tab ---
                odds_button = page.locator('button[data-tst="game_odds"]')
                # Use a longer timeout on the first, most crucial interaction

                expect(odds_button).to_be_visible(timeout=15000)
                odds_button.click()

                # --- STEP 2: Wait for the main container ---
                betsheet = page.locator("div#betsheet").first
                expect(betsheet).to_be_visible(timeout=10000)

                # --- STEP 3: Check for the "No Bets Available" state (for completed games) ---
                no_bets_container = betsheet.locator("div.empty-odds")
                try:
                    # Use a short timeout here. If this element exists, we want to know quickly.
                    expect(no_bets_container).to_be_visible(timeout=3000)
                    print("INFO: 'No Bets Available' container found. This game is likely complete.")
                except Exception:
                    # If the "no_bets_container" is NOT found, we proceed to parse the odds.
                    print("INFO: 'No Bets Available' not found, proceeding to parse odds table...")

                # --- STEP 4: Wait for a stable element in the odds table ---
                home_team_container = betsheet.locator("div.sixpack-home-team").first
                expect(home_team_container).to_be_visible(timeout=10000)

                # --- STEP 5: Parse all data ---
                away_row = betsheet.locator("tr:has(div.sixpack-away-team)").first
                home_row = betsheet.locator("tr:has(div.sixpack-home-team)").first
                away_team_name = away_row.locator("span.Fw\\(600\\)").first.text_content().strip()
                home_team_name = home_row.locator("span.Fw\\(600\\)").first.text_content().strip()

                home_ml_text = home_row.locator("td").nth(1).inner_text()
                away_ml_text = away_row.locator("td").nth(1).inner_text()
                home_spread_text = home_row.locator("td").nth(2).inner_text()
                total_text = home_row.locator("td").nth(3).inner_text()

                # --- STEP 6: Use regex to reliably extract numbers ---
                home_ml_match = re.search(r'([+\-–−]\d+)', home_ml_text)
                away_ml_match = re.search(r'([+\-–−]\d+)', away_ml_text)
                spread_match = re.search(r'([+\-–−]\d+\.?\d*)', home_spread_text)
                total_match = re.search(r'[OUou]\s*(\d+\.?\d*)', total_text)

                if not all([home_ml_match, away_ml_match, spread_match, total_match]):
                    raise ValueError("Failed to find all odds values with regex after content loaded.")

                home_moneyline = int(home_ml_match.group(1).replace('–', '-').replace('−', '-'))
                away_moneyline = int(away_ml_match.group(1).replace('–', '-').replace('−', '-'))
                spread = float(spread_match.group(1).replace('–', '-').replace('−', '-'))
                total = float(total_match.group(1))

                # --- SUCCESS! ---
                # If we get here, everything worked. Return the data and exit the function.
                print(f"Success on attempt {attempt + 1}!")
                return home_team_name, away_team_name, home_moneyline, away_moneyline, spread, total

            except Exception as e:
                # --- HANDLE FAILED ATTEMPT ---
                print(f"Error on attempt {attempt + 1} of {max_attempts}: {type(e).__name__}")

                # If this was not the last attempt, reload and wait before trying again.
                if attempt < max_attempts - 1:
                    print(f"Reloading page and retrying in {retry_delay_seconds} seconds...")
                    try:
                        page.reload(wait_until="domcontentloaded", timeout=20000)
                        time.sleep(retry_delay_seconds)
                    except Exception as reload_error:
                        print(f"Page reload failed: {reload_error}. Aborting.")
                        break # Exit the loop if reload fails
                else:
                    # This was the final attempt. Print the detailed failure message.
                    # print(f"\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"!!!!!!   ALL {max_attempts} ATTEMPTS FAILED. SEE DETAILS.   !!!!!!")
                    # print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    # print(f"URL at time of failure: {page.url}")
                    # print(f"FINAL ERROR TYPE: {type(e).__name__}")
                    # print(f"FINAL ERROR DETAILS: {e}")
                    #
                    # screenshot_path = "final_debug_odds_failure.png"
                    # html_path = "final_debug_odds_failure.html"
                    # page.screenshot(path=screenshot_path, full_page=True)
                    # with open(html_path, "w", encoding="utf-8") as f:
                    #     f.write(page.content())
                    #
                    # print(f"\n--- DEBUG ARTIFACTS SAVED ---")
                    # print(f"Screenshot of the page saved to: '{screenshot_path}'")
                    # print(f"Full HTML of the page saved to: '{html_path}'")
                    # print(f"--- Please inspect these files to see what the scraper saw. ---\n\n\n")

                    return None

        # This line is reached only if the loop completes without a successful return,
        # which shouldn't happen with the logic above, but is a safe fallback.
        return None
    def _parse_team_stats(self, page: Page):
        """Parses team stats after explicitly navigating to the team stats tab."""
        team_stats = {}
        try:
            team_stats_button = page.locator('button[data-tst="teamcomparison"]')
            team_stats_button.wait_for(state="visible", timeout=10000)
            team_stats_button.click()

            stats_container = page.locator("div.ys-boxscore-teamstats").first
            stats_container.wait_for(state="visible", timeout=10000)

            stat_rows = stats_container.locator("tbody tr").all()
            for row in stat_rows:
                cols = row.locator("td").all()
                if len(cols) != 3: continue
                away_val_text = cols[0].text_content().strip()
                stat_name = cols[1].text_content().strip()
                home_val_text = cols[2].text_content().strip()
                # Create a nested structure for clarity
                team_stats[stat_name] = {'home': home_val_text, 'away': away_val_text}
            return team_stats
        except Exception as e:
            print(f"           - Warning: Could not parse team stats table. Error: {e}")
            return None
    def _parse_summary_dl(self, dl_element):
        """
        Parses a <dl> element to extract summary data like Pitches-Strikes.
        Returns a dictionary mapping player names (as found in the text) to their stats.
        e.g., {'K Gausman': {'pitches_thrown': 96}, 'T Nance': {...}}
        """
        summary_data = {}
        if dl_element.count() == 0:
            return summary_data

        all_descriptions = dl_element.locator('dd').all()
        for dd in all_descriptions:
            line_text = dd.text_content().strip()
            data_part = line_text.split(' - ')[-1]
            player_chunks = data_part.split(',')

            if "Pitches-strikes" in line_text:
                for chunk in player_chunks:
                    match = re.search(r"([\w.\s'-]+)\s+(\d+)-(\d+)", chunk)
                    if match:
                        # Get the name exactly as it appears in the summary
                        player_name_in_summary = match.group(1).strip()
                        summary_data.setdefault(player_name_in_summary, {})['pitches_thrown'] = int(match.group(2))
                        summary_data.setdefault(player_name_in_summary, {})['strikes_thrown'] = int(match.group(3))
            elif "Batters faced" in line_text:
                for chunk in player_chunks:
                    match = re.search(r"([\w.\s'-]+)\s+(\d+)", chunk)
                    if match:
                        player_name_in_summary = match.group(1).strip()
                        summary_data.setdefault(player_name_in_summary, {})['batters_faced'] = int(match.group(2))
        return summary_data

    # In data_collection/yahoo_scraper.py

    def _parse_player_stats(self, page: Page, main_team_db_id, opponent_team_id, main_team_name, source_game_id,year):
        """
        Parses player stats with a flexible approach.
        --- V8 (Definitive): Uses positional selection to isolate the correct parent container. ---
        """
        try:
            # --- FLEXIBLE TAB CLICKING (No changes needed) ---
            try:
                player_stats_button = page.locator('button[data-tst="stats"], button[data-tst="matchstats"]').first
                print(f"           - Checking for a 'Stats' button...")
                player_stats_button.wait_for(state="visible", timeout=3000)
                print(f"           - 'Stats' button found. Clicking it to ensure stats are visible.")
                player_stats_button.click()
            except PlaywrightError:
                print(f"           - No 'Stats' button found. Assuming stats are already the default view.")

            # --- MAIN CONTENT PARSING (No changes needed) ---
            match_stats_container = page.locator("div.match-stats")
            print("           - Waiting for the main stats container to load...")
            match_stats_container.wait_for(state="visible", timeout=100000)
            print("           - Main stats container is visible. Proceeding to parse data.")

            # --- THE DEFINITIVE POSITIONAL LOCATOR ---
            # 1. Find all potential rows. Based on the HTML, the class combo "D(f) Jc(sb)" is common.
            all_rows = match_stats_container.locator(".D\\(f\\).Jc\\(sb\\)").all()

            # 2. The row we want is the one that contains the stat columns (divs with class Va(t)).
            # We will loop through the rows and find the first one that fits this description.
            stats_row_container = None
            for row in all_rows:
                # Check if this row has at least two `div` children with the `Va(t)` class.
                if row.locator("> div.Va\\(t\\)").count() >= 2:
                    stats_row_container = row
                    break # We found the correct row, so we stop looking.

            if not stats_row_container:
                print(f"           - CRITICAL ERROR: Could not find the main row container for stats in game {source_game_id}.")
                return

            # 3. Now, get the two columns from *within the correct row container*.
            team_column_containers = stats_row_container.locator("> div.Va\\(t\\)").all()

            if len(team_column_containers) != 2:
                print(f"           - ERROR: Expected 2 team column containers, but found {len(team_column_containers)} for game {source_game_id}.")
                return

            print(f"           - Found exactly 2 team stat columns. Parsing each...")
            all_players_data = {}

            # The rest of your proven parsing logic can now proceed safely.
            for i, team_column in enumerate(team_column_containers):
                try:
                    team_name_in_header = team_column.locator("a").first.text_content(timeout=1000).strip()
                except PlaywrightError:
                    print(f"           - WARNING: Could not find a team name link in column {i+1}. Skipping column.")
                    continue

                if main_team_name in team_name_in_header:
                    team_id = main_team_db_id; opponent_id = opponent_team_id
                else:
                    team_id = opponent_team_id; opponent_id = main_team_db_id

                all_stat_tables = team_column.locator("table").all()
                if not all_stat_tables:
                    print(f"           - WARNING: No stat tables found within column for '{team_name_in_header}'.")
                    continue

                for table in all_stat_tables:
                    dl_element = table.locator("xpath=./parent::div/following-sibling::div/dl").first
                    summary_data = self._parse_summary_dl(dl_element)
                    header_info = table.locator("thead th").all()
                    if not header_info: continue

                    category = header_info[0].text_content().strip()
                    if not category: continue

                    header_abbreviations = [th.text_content().strip() for th in header_info[1:]]
                    header_names = [th.get_attribute('title') or abbr for abbr, th in zip(header_abbreviations, header_info[1:])]
                    stat_def_id = self.db.get_or_create_stat_definition(self.sport_id, category, header_names, header_abbreviations)
                    if not stat_def_id: continue

                    for row in table.locator("tbody tr").all():
                        player_anchor = row.locator("th a").first
                        if not player_anchor.is_visible(): continue
                        player_name_raw = player_anchor.text_content().strip()
                        if "TOTAL" in player_name_raw.upper(): continue
                        player_name = re.sub(r'^[•\s]+', '', player_name_raw).strip()
                        player_url = player_anchor.get_attribute('href')
                        player_source_id_match = re.search(r'/(\d+)', player_url or '')
                        if not player_source_id_match: continue
                        player_db_id = self.db.get_or_create_player(self.sport_id, player_name, player_source_id_match.group(1))
                        if not player_db_id: continue

                        all_players_data.setdefault(player_db_id, {})
                        all_players_data[player_db_id].update({'team_id': team_id, 'opponent_id': opponent_id, 'stat_def_id': stat_def_id})
                        all_players_data[player_db_id].setdefault('stats', {})
                        stat_values = [td.text_content().strip() for td in row.locator("td").all()]
                        for header_key, value in zip(header_abbreviations, stat_values):
                            all_players_data[player_db_id]['stats'][header_key] = value

                        if category.upper() == 'PITCHING':
                            for summary_name, summary_stats in summary_data.items():
                                if summary_name in player_name_raw:
                                    all_players_data[player_db_id]['stats'].update(summary_stats)
                                    break

            if not all_players_data:
                print(f"           - WARNING: No player data was successfully parsed for game {source_game_id}.")
                return

            players_saved_count = 0
            for player_db_id, data in all_players_data.items():
                success = self.db.insert_player_game_stats_from_dict(player_id=player_db_id, data=data, season=year, source_game_id=source_game_id)
                if success: players_saved_count += 1

            if players_saved_count > 0:
                print(f"           - Success: Saved merged stats for {players_saved_count} players in game {source_game_id}.")
            else:
                print(f"           - WARNING: Found player data but failed to save any to the database for game {source_game_id}.")

        except Exception as e:
            print(f"           - CRITICAL Error during player stat parsing for game {source_game_id}: {e}")
            traceback.print_exc()