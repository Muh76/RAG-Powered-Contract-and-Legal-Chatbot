#!/usr/bin/env python3
"""
Comprehensive UK Legislation Scraper for legislation.gov.uk
Creates JSON files in the same format as existing legislation files
"""

import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
import time
import re

BASE_URL = "https://www.legislation.gov.uk"
DATA_DIR = Path("data/uk_legislation")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Legislation to fetch with their URLs
LEGISLATION_TO_FETCH = {
    "health_and_safety_at_work_act_1974": {
        "title": "Health and Safety at Work Act 1974",
        "url": f"{BASE_URL}/ukpga/1974/37/contents",
        "act_id": "ukpga/1974/37"
    },
    "working_time_regulations_1998": {
        "title": "Working Time Regulations 1998",
        "url": f"{BASE_URL}/ukdsi/1998/9780110656577/contents",
        "act_id": "ukdsi/1998/9780110656577"
    },
    "national_minimum_wage_act_1998": {
        "title": "National Minimum Wage Act 1998",
        "url": f"{BASE_URL}/ukpga/1998/39/contents",
        "act_id": "ukpga/1998/39"
    }
}


def scrape_legislation_content(act_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Scrape legislation content from legislation.gov.uk"""
    try:
        print(f"   Fetching: {act_info['url']}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(act_info['url'], headers=headers, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to extract main content
            content_parts = []
            
            # Look for main content area
            main_content = soup.find('div', class_='LegText') or soup.find('div', id='content') or soup.find('main')
            
            if main_content:
                # Extract text content, preserving structure
                for element in main_content.find_all(['p', 'div', 'section']):
                    text = element.get_text(strip=True)
                    if text and len(text) > 20:  # Filter very short text
                        content_parts.append(text)
                
                content = '\n\n'.join(content_parts)
            else:
                # Fallback: get all text
                content = soup.get_text(separator='\n\n', strip=True)
            
            # Clean up content
            content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excessive newlines
            content = content.strip()
            
            if len(content) > 1000:
                return {
                    "title": act_info["title"],
                    "url": act_info['url'],
                    "content": content,
                    "sections": [],
                    "content_length": len(content)
                }
            else:
                print(f"   ⚠️ Content too short ({len(content)} chars) - may need manual download")
                return None
                
        else:
            print(f"   ⚠️ HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def create_legislation_json(act_data: Dict[str, Any], filename: str):
    """Create JSON file in UK legislation format"""
    output_path = DATA_DIR / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(act_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created: {output_path}")
    print(f"   Content length: {len(act_data.get('content', ''))} characters")


def main():
    """Scrape all missing legislation"""
    print("📥 Scraping Missing UK Legislation...")
    print("=" * 60)
    print("\n⚠️ Note: Web scraping may not capture all content perfectly.")
    print("   For complete accuracy, consider manual download from legislation.gov.uk\n")
    
    for filename, info in LEGISLATION_TO_FETCH.items():
        print(f"\n📄 Processing: {info['title']}")
        act_data = scrape_legislation_content(info)
        
        if act_data:
            create_legislation_json(act_data, f"{filename}.json")
        else:
            # Create placeholder file with instructions
            placeholder = {
                "title": info["title"],
                "url": info['url'],
                "content": f"PLACEHOLDER: Please download complete content from {info['url']}",
                "sections": [],
                "content_length": 0,
                "note": "This is a placeholder. Download complete content manually."
            }
            create_legislation_json(placeholder, f"{filename}.json")
            print(f"   💡 Manual download required from: {info['url']}")
        
        time.sleep(2)  # Be polite to the server
    
    print("\n" + "=" * 60)
    print("✅ Scraping complete!")
    print("\n📝 Next steps:")
    print("   1. Review created JSON files")
    print("   2. If placeholders exist, manually download from URLs above")
    print("   3. Update JSON files with complete content")
    print("   4. Run: python scripts/ingest_data.py")


if __name__ == "__main__":
    main()

