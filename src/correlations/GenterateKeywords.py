import random
from pathlib import Path
import pandas as pd

# --- Configuration ---
TARGET_COUNT = 3500
MIN_SAMPLE_SIZE_FOR_COMPLEX_TEMPLATES = 20
OUTPUT_FILENAME = "keywords.csv"

# --- Seed Categories and Words ---
SEED_CATEGORIES = {
    "tech_company": [
        "Apple", "Microsoft", "Google", "Amazon", "Nvidia", "Tesla", "Meta", "Netflix",
        "AMD", "Intel", "Salesforce", "Oracle", "Adobe", "IBM", "Shopify", "Uber", "Lyft",
        "Airbnb", "Spotify", "Zoom", "Palantir", "Snowflake", "Cloudflare", "CrowdStrike",
        "Block", "PayPal", "Cisco", "Broadcom", "Qualcomm", "Texas Instruments", "Micron",
        "Dell", "HP", "Pinterest", "Twitter", "Reddit", "ASML", "ARM", "TSMC",
        "Samsung Electronics", "Xiaomi", "Unity", "Atlassian", "ServiceNow", "Workday",
        "Etsy", "eBay", "Alibaba", "JD.com", "Baidu", "MercadoLibre", "Sea Limited", "Snapchat"
    ],
    "israeli_company": [
        "Wix", "Monday.com", "Mobileye", "Teva Pharmaceutical", "Check Point Software",
        "CyberArk", "Nice Ltd.", "SolarEdge", "Elbit Systems", "Payoneer", "Fiverr"
    ],
    "finance_company": [
        "JPMorgan Chase", "Bank of America", "Goldman Sachs", "Morgan Stanley", "Citigroup",
        "Wells Fargo", "Visa", "Mastercard", "American Express", "BlackRock", "Berkshire Hathaway",
        "Charles Schwab"
    ],
    "health_company": [
        "Johnson & Johnson", "Pfizer", "Moderna", "Eli Lilly", "UnitedHealth", "Merck", "AbbVie",
        "Novartis", "AstraZeneca", "GSK", "Sanofi", "Roche", "Bristol-Myers Squibb", "Amgen"
    ],
    "retail_company": [
        "Walmart", "Costco", "Home Depot", "Nike", "Starbucks", "McDonald's", "Coca-Cola", "PepsiCo",
        "Lululemon", "Target", "Best Buy", "Lowe's", "CVS Health", "Walgreens", "Kroger",
        "Zara", "H&M", "Uniqlo", "IKEA"
    ],
    "tech_product": [
        "iPhone 16", "iPhone Pro Max", "MacBook Pro M4", "Apple Vision Pro", "Apple Watch", "Windows 11",
        "Microsoft Surface", "Xbox Series X", "Google Pixel 9", "Android 15", "Samsung Galaxy S25",
        "Playstation 5 Pro", "Tesla Model Y", "Cybertruck", "Nvidia RTX 5090",
        "Copilot", "ChatGPT-5", "OpenAI Sora", "Google Gemini", "AWS", "Azure", "Google Cloud",
        "Meta Quest 3", "Fitbit", "Garmin Watch", "GoPro", "Kindle", "Roomba", "Nest Thermostat",
        "Ring Doorbell", "Sonos Speaker", "Peloton", "Dyson Vacuum", "AirPods Pro", "Galaxy Buds"
    ],
    "business_leader": [
        "Elon Musk", "Tim Cook", "Satya Nadella", "Jensen Huang", "Mark Zuckerberg",
        "Sundar Pichai", "Jeff Bezos", "Warren Buffett", "Jerome Powell", "Jamie Dimon",
        "Larry Fink", "Mary Barra", "Sheryl Sandberg", "Reed Hastings", "Marc Benioff",
        "Andy Jassy", "Lisa Su"
    ],
    "israeli_leader": [
        "Benjamin Netanyahu", "Yair Lapid", "Benny Gantz", "Isaac Herzog",
        "Naftali Bennett", "Gideon Sa'ar", "Mansour Abbas", "Amir Ohana", "Yuli Edelstein",
        "Nitzan Horowitz", "Ayman Odeh", "Itamar Ben-Gvir"
    ],
    "us_political_figure": [
        "Ron DeSantis", "Donald Trump", "Joe Biden", "Kamala Harris", "Nancy Pelosi", "Chuck Schumer",
        "Mitch McConnell", "Kevin McCarthy", "Alexandria Ocasio-Cortez", "Bernie Sanders",
        "Elizabeth Warren", "Ted Cruz", "Marco Rubio", "Gavin Newsom"
    ],
    "geopolitics_israel_places": [
        "Israel", "Palestine", "Gaza", "West Bank", "Jerusalem",
        "Golan Heights", "Sinai Peninsula"
    ],
    "geopolitics_israel_conflicts": [
        "Hamas", "Hezbollah", "IDF", "Iron Dome", "Settlements",
        "Peace talks", "Ceasefire", "Intifada", "Security fence",
        "BDS movement", "October 7 attacks", "Nova music festival massacre"
    ],
    "geopolitics_global_places": [
        "Ukraine", "Russia", "China", "Taiwan", "US", "Iran", "NATO", "Red Sea", "Middle East", "North Korea",
        "Syria", "Saudi Arabia", "India", "Pakistan", "Afghanistan", "Iraq", "Yemen", "Sudan", "Ethiopia"
    ],
    "economics": [
        "inflation", "recession", "interest rates", "unemployment", "GDP", "CPI report",
        "consumer confidence", "housing market", "stock market", "bond yields", "fed meeting",
        "CHIPS Act", "Inflation Reduction Act", "supply chain", "trade deficit",
        "quantitative easing", "fiscal policy", "monetary policy", "labor market", "wage growth",
        "corporate earnings", "market volatility", "economic stimulus", "global trade", "tariffs",
        "energy prices", "commodity prices", "cryptocurrency market", "financial regulation",
        "venture capital", "emerging markets", "central banks", "foreign exchange", "real estate market",
        "mortgage rates", "consumer spending", "savings rate", "business investment", "industrial production",
        "retail sales", "income inequality", "gold prices", "oil prices", "dollar index", "euro index"
    ],
    "social_esg": [
        "lgbtq rights", "gender equality", "pride month", "climate change", "carbon emissions",
        "renewable energy", "ESG investing", "AI ethics", "veteran support", "me too movement",
        "Black Lives Matter", "cancel culture", "body positivity", "quiet quitting", "sustainable living",
        "fast fashion", "mental health awareness", "workplace diversity", "inclusion initiatives",
        "corporate social responsibility", "ethical sourcing", "fair trade", "greenwashing",
        "social justice", "human rights", "data privacy", "digital detox", "mindfulness",
        "work-life balance", "remote work culture", "gig economy", "universal basic income",
        "living wage", "corporate governance", "stakeholder capitalism", "circular economy",
        "eco-friendly products", "sustainable agriculture", "clean energy transition", "carbon footprint",
        "biodiversity conservation", "ocean pollution", "deforestation", "water scarcity", "air quality",
        "climate resilience", "environmental activism", "youth climate movement",
        "sustainable finance", "impact investing", "green bonds", "social entrepreneurship"
    ],
    "culture_event": [
        "Super Bowl", "Oscars", "Grammys", "World Cup", "Olympics", "Coachella", "Met Gala",
        "Barbie movie", "Oppenheimer movie", "burning man", "fashion week"
    ],
    "culture_person": [
        "Taylor Swift", "Beyonce", "Sydney Sweeney", "Rihanna", "Drake", "Adele", "Kendrick Lamar",
        "Billie Eilish", "Harry Styles", "The Weeknd", "Olivia Rodrigo", "Bad Bunny", "Dua Lipa",
        "Lady Gaga", "Kylie Jenner", "Kim Kardashian", "Kanye West", "Travis Scott", "Cardi B",
        "Megan Thee Stallion", "Doja Cat", "Lizzo", "Zendaya", "Tom Holland", "Margot Robbie",
        "Leonardo DiCaprio", "Will Smith", "Joe Rogan", "MrBeast", "PewDiePie", "Logan Paul", "Jake Paul",
        "Andrew Tate", "Greta Thunberg", "Malala Yousafzai", "Oprah Winfrey",
        "Ellen DeGeneres", "George Clooney", "Ryan Reynolds", "Dwayne Johnson"
    ],
    "health_lifestyle": [
        "weight loss drug", "Ozempic", "Wegovy", "plant-based food", "lab grown meat", "keto diet",
        "intermittent fasting", "mental health", "anxiety", "depression", "mindfulness", "yoga",
        "meditation", "home workout", "sleep tracking", "wearable fitness", "smartwatch health features",
        "immune system", "gut health", "probiotics", "virtual doctor visit", "genetic testing",
        "vegan", "vegetarian", "gluten-free", "dairy-free", "organic food", "superfoods", "anti-aging"
    ],
    "general_trends": [
        "supply chain", "remote work", "cybersecurity", "data breach", "metaverse", "web3", "crypto",
        "bitcoin", "ethereum", "electric vehicle", "sustainable fashion", "influencer marketing",
        "social media trends", "digital transformation", "5G technology", "blockchain", "NFTs",
        "AI advancements", "machine learning", "automation", "robotics", "quantum computing",
        "augmented reality", "virtual reality", "big data", "smart home", "digital currency", "fintech",
        "e-sports", "gaming industry", "streaming services", "online education"
    ]
}

# --- Keyword Templates ---
TEMPLATES = [
    # Template, Primary Category, Placeholder Key
    ("{company} stock price", "finance_company", "company"),
    ("{company} earnings report", "finance_company", "company"),
    ("{company} news", "general_trends", "company"),
    ("{company} controversy", "general_trends", "company"),
    ("buy {company} shares", "finance_company", "company"),
    ("{company} layoffs", "economics", "company"),
    ("new {product} release date", "tech_product", "product"),
    ("{product} review", "tech_product", "product"),
    ("{product} price", "economics", "product"),
    ("{product} vs competitor", "tech_product", "product"),
    ("{leader} tweet", "business_leader", "leader"),
    ("{leader} interview", "business_leader", "leader"),
    ("{geopolitical_place} conflict", "geopolitics_global_places", "geopolitical_place"),
    ("{geopolitical_place} news", "geopolitics_global_places", "geopolitical_place"),
    ("war in {geopolitical_place}", "geopolitics_global_places", "geopolitical_place"),
    ("US aid to {geopolitical_place}", "geopolitics_global_places", "geopolitical_place"),
    ("{economic_indicator} forecast", "economics", "economic_indicator"),
    ("impact of {economic_indicator}", "economics", "economic_indicator"),
    ("{social_theme} controversy", "social_esg", "social_theme"),
    ("{social_theme} stocks", "finance_company", "social_theme"),
    ("{cultural_event} ratings", "culture_event", "cultural_event"),
    ("{person} new album", "culture_person", "person"),
    ("{generic_concept} trends", "general_trends", "generic_concept"),
    ("investing in {generic_concept}", "finance_company", "generic_concept"),
]

COMPLEX_TEMPLATES = [
    # Template, Combined Category, Placeholder 1, Placeholder 2
    ("{company} vs {company}", "finance_company", "company", "company"),
    ("{product} sales in {geopolitical_place}", "economics", "product", "geopolitical_place"),
    ("{leader} on {social_theme}", "social_esg", "leader", "social_theme"),
    ("{company} response to {geopolitical_topic}", "geopolitics", "company", "geopolitical_topic"),
]


def generate_keywords():
    """Generates a large, unique, and categorized list of meaningful keywords."""
    keyword_data = []

    # 1. Add seeds directly
    for category, seeds in SEED_CATEGORIES.items():
        for seed in seeds:
            keyword_data.append({"keyword": seed, "category": category})

    # 2. Prepare placeholder lists
    all_companies = (
            SEED_CATEGORIES["tech_company"] + SEED_CATEGORIES["israeli_company"] +
            SEED_CATEGORIES["finance_company"] + SEED_CATEGORIES["health_company"] +
            SEED_CATEGORIES["retail_company"]
    )
    all_leaders = SEED_CATEGORIES["business_leader"] + SEED_CATEGORIES["israeli_leader"] + SEED_CATEGORIES[
        "us_political_figure"]
    geopolitics_places = SEED_CATEGORIES["geopolitics_israel_places"] + SEED_CATEGORIES[
        "geopolitics_global_places"]
    geopolitics_all = geopolitics_places + SEED_CATEGORIES["geopolitics_israel_conflicts"]

    placeholder_map = {
        "company": all_companies,
        "product": SEED_CATEGORIES["tech_product"],
        "leader": all_leaders,
        "geopolitical_place": geopolitics_places,
        "geopolitical_topic": geopolitics_all,
        "economic_indicator": SEED_CATEGORIES["economics"],
        "social_theme": SEED_CATEGORIES["social_esg"],
        "cultural_event": SEED_CATEGORIES["culture_event"],
        "person": SEED_CATEGORIES["culture_person"],
        "generic_concept": SEED_CATEGORIES["general_trends"] + SEED_CATEGORIES["health_lifestyle"],
    }

    # 3. Simple templates
    for template, category, placeholder_key in TEMPLATES:
        for seed in placeholder_map[placeholder_key]:
            keyword = template.replace("{" + placeholder_key + "}", seed)
            keyword_data.append({"keyword": keyword, "category": category})

    # 4. Complex templates
    for template, category, p_key1, p_key2 in COMPLEX_TEMPLATES:
        list1 = placeholder_map[p_key1]
        list2 = placeholder_map[p_key2]
        #  Use samples to generate fewer combinations
        sample_size1 = min(MIN_SAMPLE_SIZE_FOR_COMPLEX_TEMPLATES, len(list1))
        sample_size2 = min(MIN_SAMPLE_SIZE_FOR_COMPLEX_TEMPLATES, len(list2))
        for seed1 in random.sample(list1, k=sample_size1):
            for seed2 in random.sample(list2, k=sample_size2):
                if seed1 == seed2: continue
                keyword = template.replace("{" + p_key1 + "}", seed1).replace("{" + p_key2 + "}", seed2)
                keyword_data.append({"keyword": keyword, "category": category})

    # 5. Finalize
    df = pd.DataFrame(keyword_data)
    df.drop_duplicates(subset="keyword", inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)

    # 6. Write to CSV file in the project's root directory
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
    except NameError:
        # Fallback for environments where __file__ is not defined (like notebooks)
        project_root = Path.cwd()

    output_path = project_root / OUTPUT_FILENAME
    df.to_csv(output_path, index=False)
    print(f"Successfully generated and saved {len(df)} keywords to:\n'{output_path}'")


if __name__ == "__main__":
    generate_keywords()
