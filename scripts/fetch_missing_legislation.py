#!/usr/bin/env python3
"""
Script to fetch missing UK legislation from legislation.gov.uk API
"""

import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional
import time

BASE_URL = "https://www.legislation.gov.uk"
DATA_DIR = Path("data/uk_legislation")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Missing legislation to fetch
MISSING_LEGISLATION = {
    "health_and_safety_at_work_act_1974": {
        "act_id": "ukpga/1974/37",
        "title": "Health and Safety at Work Act 1974",
        "url": f"{BASE_URL}/ukpga/1974/37"
    },
    "working_time_regulations_1998": {
        "act_id": "ukdsi/1998/9780110656577",
        "title": "Working Time Regulations 1998",
        "url": f"{BASE_URL}/ukdsi/1998/9780110656577"
    },
    "national_minimum_wage_act_1998": {
        "act_id": "ukpga/1998/39",
        "title": "National Minimum Wage Act 1998",
        "url": f"{BASE_URL}/ukpga/1998/39"
    }
}


def fetch_legislation_content(act_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Fetch legislation content from UK Legislation website"""
    try:
        url = act_info['url']
        print(f"   Fetching from: {url}")
        
        # Try to get the content page
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # Parse HTML to extract content (simplified - may need BeautifulSoup for complex pages)
            content = response.text
            
            # Extract text content (basic extraction)
            # Note: This is a simplified approach - real implementation might need web scraping
            # For now, we'll create a placeholder structure
            
            return {
                "title": act_info["title"],
                "url": url,
                "content": content[:50000] if len(content) > 50000 else content,  # Limit content
                "note": "This is a placeholder - full content needs proper web scraping"
            }
        else:
            print(f"   ⚠️ HTTP {response.status_code} - may need manual download")
            return None
            
    except Exception as e:
        print(f"   ❌ Error fetching {act_info['title']}: {e}")
        return None


def create_legislation_json(act_data: Dict[str, Any], filename: str, title: str):
    """Create JSON file in UK legislation format"""
    # Extract content
    content = act_data.get('content', '')
    
    legislation_json = {
        "title": title,
        "url": act_data.get('url', ''),
        "content": content,
        "sections": [],
        "content_length": len(content)
    }
    
    # Save to file
    output_path = DATA_DIR / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(legislation_json, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created: {output_path}")
    print(f"   Content length: {len(content)} characters")


def main():
    """Fetch all missing legislation"""
    print("📥 Fetching Missing UK Legislation...")
    print("=" * 60)
    print("\n⚠️ Note: This script creates placeholder files.")
    print("   For complete content, you may need to:")
    print("   1. Manually download from legislation.gov.uk")
    print("   2. Use a web scraping tool with BeautifulSoup")
    print("   3. Use the legislation.gov.uk API if available\n")
    
    for filename, info in MISSING_LEGISLATION.items():
        print(f"\n📄 Fetching: {info['title']}")
        act_data = fetch_legislation_content(info)
        
        if act_data:
            create_legislation_json(
                act_data,
                f"{filename}.json",
                info['title']
            )
        else:
            print(f"   ❌ Failed to fetch {info['title']}")
            print(f"   💡 Manual download required from:")
            print(f"      {info['url']}")
        
        time.sleep(1)  # Be polite to the server
    
    print("\n" + "=" * 60)
    print("✅ Fetch script complete!")
    print("\n📝 Next steps:")
    print("   1. Review the created JSON files")
    print("   2. If content is incomplete, manually download from legislation.gov.uk")
    print("   3. Update JSON files with complete content")
    print("   4. Run ingestion script: python scripts/ingest_data.py")


if __name__ == "__main__":
    main()

