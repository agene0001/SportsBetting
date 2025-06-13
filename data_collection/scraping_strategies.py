# data_collection/scraping_strategies.py
import abc

class SportScrapingStrategy(abc.ABC):
    """
    Abstract Base Class for a sport-specific scraping strategy.
    The primary responsibility is to provide the display_name -> full_name map.
    The scraper will dynamically discover the URL slugs.
    """
    @abc.abstractmethod
    def get_display_to_full_name_map(self):
        """Returns the display_name -> full_name map."""
        pass

class NFLStrategy(SportScrapingStrategy):
    """Scraping strategy for NFL."""
    def get_display_to_full_name_map(self):
        display_to_full= {
            "Buffalo": "Buffalo Bills", "Miami": "Miami Dolphins", "New England": "New England Patriots", "NY Jets": "New York Jets",
            "Denver": "Denver Broncos", "Kansas City": "Kansas City Chiefs", "LA Chargers": "Los Angeles Chargers", "Las Vegas": "Las Vegas Raiders",
            "Baltimore": "Baltimore Ravens", "Cincinnati": "Cincinnati Bengals", "Cleveland": "Cleveland Browns", "Pittsburgh": "Pittsburgh Steelers",
            "Houston": "Houston Texans", "Indianapolis": "Indianapolis Colts", "Jacksonville": "Jacksonville Jaguars", "Tennessee": "Tennessee Titans",
            "Dallas": "Dallas Cowboys", "NY Giants": "New York Giants", "Philadelphia": "Philadelphia Eagles", "Washington": "Washington Commanders",
            "Arizona": "Arizona Cardinals", "LA Rams": "Los Angeles Rams", "San Francisco": "San Francisco 49ers", "Seattle": "Seattle Seahawks",
            "Chicago": "Chicago Bears", "Detroit": "Detroit Lions", "Green Bay": "Green Bay Packers", "Minnesota": "Minnesota Vikings",
            "Atlanta": "Atlanta Falcons", "Carolina": "Carolina Panthers", "New Orleans": "New Orleans Saints", "Tampa Bay": "Tampa Bay Buccaneers"
        }
        full_to_slug = {full: full.lower().replace(" ", "-") for full in display_to_full.values()}
        return display_to_full, full_to_slug
class NHLStrategy(SportScrapingStrategy):
    """Scraping strategy for NHL."""
    def get_display_to_full_name_map(self):
        display_to_full= {
            "Boston": "Boston Bruins", "Buffalo": "Buffalo Sabres", "Detroit": "Detroit Red Wings", "Florida": "Florida Panthers",
            "Montreal": "Montreal Canadiens", "Ottawa": "Ottawa Senators", "Tampa Bay": "Tampa Bay Lightning", "Toronto": "Toronto Maple Leafs",
            "Carolina": "Carolina Hurricanes", "Columbus": "Columbus Blue Jackets", "New Jersey": "New Jersey Devils", "NY Islanders": "New York Islanders",
            "NY Rangers": "New York Rangers", "Philadelphia": "Philadelphia Flyers", "Pittsburgh": "Pittsburgh Penguins", "Washington": "Washington Capitals",
            "Utah": "Utah Hockey Club", "Chicago": "Chicago Blackhawks", "Colorado": "Colorado Avalanche", "Dallas": "Dallas Stars",
            "Minnesota": "Minnesota Wild", "Nashville": "Nashville Predators", "St. Louis": "St. Louis Blues", "Winnipeg": "Winnipeg Jets",
            "Anaheim": "Anaheim Ducks", "Calgary": "Calgary Flames", "Edmonton": "Edmonton Oilers", "Los Angeles": "Los Angeles Kings",
            "San Jose": "San Jose Sharks", "Seattle": "Seattle Kraken", "Vancouver": "Vancouver Canucks", "Vegas": "Vegas Golden Knights"
        }
        full_to_slug = {full: full.lower().replace(" ", "-") for full in display_to_full.values()}
        return display_to_full, full_to_slug

class NBAStrategy(SportScrapingStrategy):
    """Scraping strategy for NBA."""
    def get_display_to_full_name_map(self):
        display_to_full= {
            "Boston": "Boston Celtics", "Brooklyn": "Brooklyn Nets", "New York": "New York Knicks", "Philadelphia": "Philadelphia 76ers", "Toronto": "Toronto Raptors",
            "Chicago": "Chicago Bulls", "Cleveland": "Cleveland Cavaliers", "Detroit": "Detroit Pistons", "Indiana": "Indiana Pacers", "Milwaukee": "Milwaukee Bucks",
            "Atlanta": "Atlanta Hawks", "Charlotte": "Charlotte Hornets", "Miami": "Miami Heat", "Orlando": "Orlando Magic", "Washington": "Washington Wizards",
            "Golden State": "Golden State Warriors", "LA Clippers": "LA Clippers", "LA Lakers": "LA Lakers", "Phoenix": "Phoenix Suns", "Sacramento": "Sacramento Kings",
            "Dallas": "Dallas Mavericks", "Houston": "Houston Rockets", "Memphis": "Memphis Grizzlies", "New Orleans": "New Orleans Pelicans", "San Antonio": "San Antonio Spurs",
            "Denver": "Denver Nuggets", "Minnesota": "Minnesota Timberwolves", "Oklahoma City": "Oklahoma City Thunder", "Portland": "Portland Trail Blazers", "Utah": "Utah Jazz"
        }
        full_to_slug = {full: full.lower().replace(" ", "-") for full in display_to_full.values()}
        return display_to_full, full_to_slug

class MLBStrategy(SportScrapingStrategy):
    """Scraping strategy for MLB."""
    def get_display_to_full_name_map(self):
        display_to_full= {
            "Baltimore": "Baltimore Orioles", "Boston": "Boston Red Sox", "NY Yankees": "New York Yankees", "Tampa Bay": "Tampa Bay Rays", "Toronto": "Toronto Blue Jays",
            "Chi White Sox": "Chicago White Sox", "Cleveland": "Cleveland Guardians", "Detroit": "Detroit Tigers", "Kansas City": "Kansas City Royals", "Minnesota": "Minnesota Twins",
            "Houston": "Houston Astros", "LA Angels": "Los Angeles Angels", "Athletics": "Oakland Athletics", "Seattle": "Seattle Mariners", "Texas": "Texas Rangers",
            "Atlanta": "Atlanta Braves", "Miami": "Miami Marlins", "NY Mets": "New York Mets", "Philadelphia": "Philadelphia Phillies", "Washington": "Washington Nationals",
            "Chi Cubs": "Chicago Cubs", "Cincinnati": "Cincinnati Reds", "Milwaukee": "Milwaukee Brewers", "Pittsburgh": "Pittsburgh Pirates", "St. Louis": "St. Louis Cardinals",
            "Arizona": "Arizona Diamondbacks", "Colorado": "Colorado Rockies", "LA Dodgers": "Los Angeles Dodgers", "San Diego": "San Diego Padres", "San Francisco": "San Francisco Giants"
        }
        full_to_slug = {full: full.lower().replace(" ", "-") for full in display_to_full.values()}
        return display_to_full, full_to_slug