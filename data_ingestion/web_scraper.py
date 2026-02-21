import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()

WEB_LINK = os.getenv('WEB_LINK')

class Web_Parser:
    def __init__(self):
        # -------------------------------------------------
        # CONFIG
        # -------------------------------------------------

        HEADERS = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
            }

        # -------------------------------------------------
        # STEP 1: Get All Google Drive Links
        # -------------------------------------------------

        print("🔍 Fetching drive links...")

        response = requests.get(WEB_LINK, headers=HEADERS)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        self.drive_links = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "drive.google.com" in href:
                self.drive_links.append(href)

        print(f"✅ Found {len(self.drive_links)} Drive links")

        if not self.drive_links:
            raise Exception("No Drive links found!")
