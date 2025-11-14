#!/usr/bin/env python3
"""
Download Real UK Legislation Data
"""

import requests
import json
import time
from pathlib import Path
from bs4 import BeautifulSoup
import re

def download_uk_legislation():
    """Download real UK legislation from legislation.gov.uk"""
    
    print("⚖️ Downloading Real UK Legislation Data...")
    print("=" * 60)
    
    # Create data directory
    data_dir = Path("data/uk_legislation")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # UK Legislation URLs
    legislation_urls = {
        "sale_of_goods_act": {
            "title": "Sale of Goods Act 1979",
            "url": "https://www.legislation.gov.uk/ukpga/1979/54/contents",
            "sections": [
                "Implied terms about title",
                "Implied terms about quality", 
                "Implied terms about fitness for purpose",
                "Remedies for breach of contract",
                "Rights of unpaid seller"
            ]
        },
        "employment_rights_act": {
            "title": "Employment Rights Act 1996",
            "url": "https://www.legislation.gov.uk/ukpga/1996/18/contents",
            "sections": [
                "Right to written statement of employment particulars",
                "Right to itemised pay statement",
                "Right to minimum notice",
                "Right to redundancy payment",
                "Protection from unfair dismissal"
            ]
        },
        "equality_act": {
            "title": "Equality Act 2010",
            "url": "https://www.legislation.gov.uk/ukpga/2010/15/contents",
            "sections": [
                "Protected characteristics",
                "Prohibited conduct",
                "Discrimination",
                "Harassment",
                "Victimisation"
            ]
        }
    }
    
    downloaded_data = {}
    
    for act_key, act_info in legislation_urls.items():
        print(f"\n Downloading {act_info['title']}...")
        
        try:
            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(act_info['url'], headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract the main content
            content_div = soup.find('div', {'class': 'LegContents'})
            if not content_div:
                content_div = soup.find('div', {'id': 'content'})
            
            if content_div:
                # Get the text content
                text_content = content_div.get_text(separator=' ', strip=True)
                
                # Clean up the text
                text_content = re.sub(r'\s+', ' ', text_content)
                text_content = text_content.strip()
                
                downloaded_data[act_key] = {
                    "title": act_info['title'],
                    "url": act_info['url'],
                    "content": text_content,
                    "sections": act_info['sections'],
                    "content_length": len(text_content)
                }
                
                print(f"   ✅ Downloaded {len(text_content)} characters")
                
                # Save to file
                output_file = data_dir / f"{act_key}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(downloaded_data[act_key], f, indent=2, ensure_ascii=False)
                
                print(f"   💾 Saved to: {output_file}")
                
            else:
                print(f"   ❌ Could not find content for {act_info['title']}")
                
        except Exception as e:
            print(f"   ❌ Error downloading {act_info['title']}: {e}")
            continue
        
        # Be respectful - add delay between requests
        time.sleep(2)
    
    print(f"\n✅ UK Legislation Download Complete!")
    print(f"📁 Data saved to: {data_dir}")
    
    return downloaded_data

if __name__ == "__main__":
    download_uk_legislation()


