import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time
import os

def fetch_arxiv_papers(query, max_results=100):
    url = f'http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={max_results}'
    print(f"Fetching {max_results} papers for query: {query}")
    
    response = urllib.request.urlopen(url)
    data = response.read().decode('utf-8')
    root = ET.fromstring(data)
    
    # Arxiv API XML namespaces
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    papers = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
        summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
        published = entry.find('atom:published', ns).text
        year = published[:4] if published else ""
        
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns).text
            authors.append(name)
            
        doc_id = entry.find('atom:id', ns).text.split('/')[-1]
        
        papers.append({
            "document_id": doc_id,
            "title": title,
            "body": summary,
            "authors": authors,
            "year": year,
            "source": "arXiv"
        })
        
    return papers

if __name__ == "__main__":
    queries = [
        "information retrieval",
        "distributed computing",
        "machine learning",
        "neural networks training",
        "database index"
    ]
    
    all_papers = []
    seen_ids = set()
    
    for q in queries:
        papers = fetch_arxiv_papers(q, max_results=60)
        for p in papers:
            if p['document_id'] not in seen_ids:
                all_papers.append(p)
                seen_ids.add(p['document_id'])
        time.sleep(3) # Polite crawling delay
        
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'arxiv_corpus.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(all_papers)} papers to {output_path}")
