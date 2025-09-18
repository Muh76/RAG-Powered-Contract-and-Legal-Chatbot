# Legal Chatbot - Setup Scripts

## Data Download Scripts

### CUAD Dataset Download
```python
# scripts/setup/download_cuad.py
import requests
import zipfile
import os

def download_cuad_dataset():
    """Download CUAD dataset for contract analysis"""
    url = "https://zenodo.org/record/4599830/files/CUAD_v1.zip"
    output_dir = "data/raw/cuad"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading CUAD dataset...")
    response = requests.get(url, stream=True)
    
    with open(f"{output_dir}/cuad.zip", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("Extracting CUAD dataset...")
    with zipfile.ZipFile(f"{output_dir}/cuad.zip", 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print("CUAD dataset downloaded successfully!")

if __name__ == "__main__":
    download_cuad_dataset()
```

### UK Legislation Download
```python
# scripts/setup/download_legislation.py
import requests
import json
import os

def download_uk_legislation():
    """Download UK legislation from Legislation.gov.uk API"""
    base_url = "https://www.legislation.gov.uk/api/v1"
    output_dir = "data/raw/uk_legislation"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download key UK acts
    acts = [
        "ukpga/1979/54",  # Sale of Goods Act 1979
        "ukpga/1990/18",  # Companies Act 1990
        "ukpga/1996/18",  # Employment Rights Act 1996
        "ukpga/1998/29",  # Data Protection Act 1998
        "ukpga/2006/46",  # Companies Act 2006
        "ukpga/2010/15",  # Equality Act 2010
        "ukpga/2018/12",  # Data Protection Act 2018
    ]
    
    for act in acts:
        try:
            print(f"Downloading {act}...")
            response = requests.get(f"{base_url}/{act}")
            response.raise_for_status()
            
            with open(f"{output_dir}/{act.replace('/', '_')}.json", "w") as f:
                json.dump(response.json(), f, indent=2)
                
        except Exception as e:
            print(f"Error downloading {act}: {e}")
    
    print("UK legislation download completed!")

if __name__ == "__main__":
    download_uk_legislation()
```

### Database Setup
```python
# scripts/setup/setup_database.py
import os
import psycopg2
from sqlalchemy import create_engine
from app.core.config import settings

def setup_database():
    """Set up PostgreSQL database with pgvector extension"""
    try:
        # Create database connection
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="password",
            database="postgres"
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create database
        cursor.execute("CREATE DATABASE legal_chatbot;")
        print("Database 'legal_chatbot' created successfully!")
        
        # Connect to new database
        conn.close()
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="password",
            database="legal_chatbot"
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("pgvector extension enabled!")
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                title VARCHAR(255),
                content TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id),
                chunk_text TEXT,
                embedding vector(384),
                chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        print("Database tables created successfully!")
        
    except Exception as e:
        print(f"Database setup error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    setup_database()
```

### Seed Data Script
```python
# scripts/setup/seed_data.py
import json
import os
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor

def seed_sample_data():
    """Seed database with sample legal documents"""
    
    # Sample legal documents
    sample_docs = [
        {
            "title": "Sale of Goods Act 1979 - Section 12",
            "content": "In a contract of sale, unless the circumstances of the contract are such as to show a different intention, there is an implied condition on the part of the seller that in the case of a sale he has a right to sell the goods, and in the case of an agreement to sell he will have a right to sell the goods at the time when the property is to pass.",
            "metadata": {"act": "Sale of Goods Act 1979", "section": "12", "jurisdiction": "UK"}
        },
        {
            "title": "Companies Act 2006 - Section 172",
            "content": "A director of a company must act in the way he considers, in good faith, would be most likely to promote the success of the company for the benefit of its members as a whole, and in doing so have regard to the matters set out in subsection (2).",
            "metadata": {"act": "Companies Act 2006", "section": "172", "jurisdiction": "UK"}
        }
    ]
    
    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="password",
            database="legal_chatbot"
        )
        cursor = conn.cursor()
        
        for doc in sample_docs:
            # Insert document
            cursor.execute("""
                INSERT INTO documents (title, content, metadata)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (doc["title"], doc["content"], json.dumps(doc["metadata"])))
            
            doc_id = cursor.fetchone()[0]
            
            # Create embeddings for chunks
            chunks = [doc["content"][i:i+500] for i in range(0, len(doc["content"]), 400)]
            
            for i, chunk in enumerate(chunks):
                embedding = model.encode(chunk).tolist()
                
                cursor.execute("""
                    INSERT INTO embeddings (document_id, chunk_text, embedding, chunk_index)
                    VALUES (%s, %s, %s, %s);
                """, (doc_id, chunk, embedding, i))
        
        conn.commit()
        print("Sample data seeded successfully!")
        
    except Exception as e:
        print(f"Seeding error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    seed_sample_data()
```
