from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException, StaleElementReferenceException
from openpyxl import Workbook
import time
import os # For file path handling

class SportScraper:
    def __init__(self, driver):
        self.driver = driver
        self.workbook = Workbook()
        self.current_sheet = self.workbook.active
        self.current_sheet.title = "Athlete Stats" # Default sheet title
        self.sport_name = None # To store the selected sport name

    def _wait_for_element(self, by, value, timeout=10):
        """Helper to wait for an element to be present."""
        return WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )

    def _wait_for_elements(self, by, value, timeout=10):
        """Helper to wait for multiple elements to be present."""
        return WebDriverWait(self.driver, timeout).until(
            EC.presence_of_all_elements_located((by, value))
        )

    def select_sport(self):
        """Allows the user to select a sport."""
        valid_sports = ['nba', 'nfl', 'nhl'] # Extend as needed
        while True:
            sport_input = input(f"Enter the sport you'd like to scrape ({'/'.join(valid_sports)}): ").strip().lower()
            if sport_input in valid_sports:
                self.sport_name = sport_input
                print(f"Navigating to ESPN {self.sport_name.upper()} stats page...")
                try:
                    self.driver.get(f"https://www.espn.com/{self.sport_name}/stats")
                    self._wait_for_element(by='css selector', value=".statistics__player-stats")
                    print(f"Successfully loaded {self.sport_name.upper()} stats page.")
                    return True
                except TimeoutException:
                    print("Error: Page took too long to load. Please check your internet connection or the URL.")
                    return False
                except WebDriverException as e:
                    print(f"A browser error occurred: {e}. Please ensure Chrome is up to date.")
                    return False
            else:
                print(f"Invalid sport. Please choose from {', '.join(valid_sports)}.")

    def get_stats(self):
        if not self.sport_name:
            print("Please select a sport first using option 1.")
            return

        try:
            player_stats_section = self._wait_for_element(by='css selector', value=".statistics__player-stats")

            print("\nAvailable Stats Categories:")
            # Use more robust selector for category links
            category_links = self._wait_for_elements(by='css selector', value=".statistics__player-stats .AnchorLink")
            visible_category_links = [link for link in category_links if link.text.strip()]
            for i, link in enumerate(visible_category_links):
                print(f"  [{i+1}] {link.text}")
            print(f"  [all] Scrape all available stats")

            while True:
                choice = input("Enter the number of the category to scrape, 'all', or 'back' to return: ").strip().lower()
                if choice == 'back':
                    return
                elif choice == 'all':
                    print("Scraping all default visible stats without drilling down into a specific player category...")
                    break
                elif choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(visible_category_links):
                        selected_category = visible_category_links[idx]
                        print(f"Attempting to click on '{selected_category.text}'...")
                        try:
                            selected_category.click()
                            print("Category selected. Loading full list...")
                            # Wait for the "view all" button to appear, indicating the category loaded
                            self._wait_for_element(by="xpath", value='//a[contains(@class, "loadMore__link")]', timeout=10)
                            break
                        except TimeoutException:
                            print("Failed to click category or the 'View All' button did not appear. Trying again.")
                            visible_category_links = self._wait_for_elements(by='css selector', value=".statistics__player-stats .AnchorLink") # Re-fetch links
                            continue
                        except (NoSuchElementException, StaleElementReferenceException) as e:
                            print(f"Error clicking element: {e}. Maybe the page structure changed?")
                            return
                    else:
                        print("Invalid category number.")
                else:
                    print("Invalid input. Please enter a number, 'all', or 'back'.")

            # Click 'View All' / 'Load More' until no more buttons
            print("Expanding player list...")
            try:
                while True:
                    view_all_button = self._wait_for_element(
                        by="xpath", value='//a[contains(@class, "loadMore__link")]',
                        timeout=3
                    )
                    self.driver.execute_script("arguments[0].click();", view_all_button)
                    time.sleep(0.7)
            except TimeoutException:
                print('Full player list loaded (or no more "view all" button found).')
            except Exception as e:
                print(f"An unexpected error occurred while expanding list: {e}")

            # --- START: NEW DATA EXTRACTION LOGIC ---
            print("Extracting data...")

            # Clear previous sheet data before writing new data
            self.current_sheet.delete_rows(1, self.current_sheet.max_row)

            # Locate the two tables after all data is loaded
            try:
                tables = self._wait_for_elements(by='tag name', value='table', timeout=15)
                if len(tables) < 2:
                    print(f"Error: Expected 2 data tables but found {len(tables)}. Cannot proceed.")
                    return
                tplayers = tables[0]
                tstats = tables[1]
            except TimeoutException:
                print("Error: Could not find the two main data tables after loading. Page structure may have changed.")
                return

            # Get headers from both tables
            player_headers = tplayers.find_element(by='tag name', value='thead').find_elements(by='tag name', value='th')
            stat_headers = tstats.find_element(by='tag name', value='thead').find_elements(by='tag name', value='th')
            all_headers = player_headers + stat_headers

            # Write combined headers to the spreadsheet
            for col_idx, header_cell in enumerate(all_headers):
                # Use chr(65) for 'A', chr(66) for 'B', etc.
                col_letter = chr(col_idx + 65)
                self.current_sheet[f'{col_letter}1'] = header_cell.text
            print(f"Wrote {len(all_headers)} headers to the spreadsheet.")

            # Get all data rows from both table bodies
            player_rows = tplayers.find_element(by='tag name', value='tbody').find_elements(by='tag name', value='tr')
            stat_rows = tstats.find_element(by='tag name', value='tbody').find_elements(by='tag name', value='tr')

            # Sanity check: ensure we have the same number of rows for players and stats
            if len(player_rows) != len(stat_rows):
                print(f"Warning: Mismatch in row count! Players: {len(player_rows)}, Stats: {len(stat_rows)}. Data may be misaligned.")

            num_rows = min(len(player_rows), len(stat_rows)) # Use the smaller number to avoid errors
            print(f"Processing {num_rows} player rows...")

            # Loop through each row index
            for row_idx in range(num_rows):
                # Get all cells from the current player row and stat row
                player_cells = player_rows[row_idx].find_elements(by='tag name', value='td')
                stat_cells = stat_rows[row_idx].find_elements(by='tag name', value='td')

                # Combine the cells into a single list for the full data row
                all_cells = player_cells + stat_cells

                # Write each cell's text to the corresponding column in the spreadsheet
                for col_idx, cell in enumerate(all_cells):
                    col_letter = chr(col_idx + 65)
                    # +2 because spreadsheet rows are 1-based and we have a header row
                    self.current_sheet[f'{col_letter}{row_idx + 2}'] = cell.text

            # --- END: NEW DATA EXTRACTION LOGIC ---

            print('Data extraction complete.')
            self.save_sheet()

        except TimeoutException as e:
            print(f"Error: A required element timed out. Page structure may have changed or network is slow: {e}")
        except NoSuchElementException as e:
            print(f"Error: A required element was not found. Page structure might have changed: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during stat retrieval: {e}")

    def save_sheet(self):
        if not self.sport_name:
            print("No sport selected to save data for.")
            return

        default_filename = f"{self.sport_name.lower()}_{self.current_sheet.title.replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        filename = input(f"Enter filename to save (default: {default_filename}): ").strip()
        if not filename:
            filename = default_filename
        elif not filename.lower().endswith('.xlsx'):
            filename += '.xlsx'

        # Give the sheet a meaningful name based on the selected category
        try:
            # Let's try to get the active category name to name the sheet
            active_category = self.driver.find_element(by='css selector', value='div.pl3.active').text
            self.current_sheet.title = active_category
        except:
            self.current_sheet.title = "Player_Stats" # Fallback title

        try:
            self.workbook.save(filename=filename)
            print(f"Data saved successfully to {os.path.abspath(filename)}")
        except Exception as e:
            print(f"Error saving workbook: {e}")

    def close_driver(self):
        print("Closing browser...")
        self.driver.quit()

# Main execution loop
if __name__ == "__main__":
    driver = None
    try:
        from selenium.webdriver.chrome.options import Options
        chrome_options = Options()
        # Uncomment the next line to run Chrome without a visible browser window (good for performance)
        # chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--log-level=3')

        driver = webdriver.Chrome(options=chrome_options)
        scraper = SportScraper(driver)

        while True:
            print("\n--- Sports Scraper Menu ---")
            print("1. Select Sport & Load Page")
            print("2. Scrape Player Stats (current sport)")
            print("3. Exit")

            choice = input("Enter your choice: ").strip()

            if choice == '1':
                scraper.select_sport()
            elif choice == '2':
                scraper.get_stats()
            elif choice == '3':
                print("Exiting scraper. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    except WebDriverException as e:
        print(f"Failed to initialize WebDriver: {e}")
        print("Please ensure your Chrome browser and ChromeDriver are compatible and installed correctly.")
        print("You can download ChromeDriver from: https://chromedriver.chromium.org/downloads")
    except Exception as e:
        print(f"An unhandled error occurred in the main program: {e}")
    finally:
        if driver:
            scraper.close_driver()