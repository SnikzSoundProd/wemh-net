import sys
import os
import urllib.request
import json
import xml.etree.ElementTree as ET

def fetch_arxiv(query):
    # If URL is provided, extract ID
    if "arxiv.org" in query:
        arxiv_id = query.split("/")[-1].replace(".pdf", "")
    else:
        arxiv_id = query

    print(f"Fetching metadata for Arxiv ID: {arxiv_id}")
    url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    
    try:
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
        root = ET.fromstring(data)
        
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', namespace)
        
        if not entry:
            print("Error: Paper not found.")
            sys.exit(1)
            
        title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ')
        
        # We save it to a local workspace
        workspace = os.path.join(os.path.dirname(__file__), 'workspace')
        os.makedirs(workspace, exist_ok=True)
        
        out_path = os.path.join(workspace, f"paper_{arxiv_id.replace('.', '_')}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"TITLE: {title}\n")
            f.write(f"ARXIV ID: {arxiv_id}\n\n")
            f.write(f"SUMMARY (ABSTRACT):\n{summary}\n")
            
        print(f"Success! Extracted abstract saved to: {out_path}")
        print("Note: For full PDF parsing, you would integrate PyMuPDF (fitz) or nougat here.")
        
    except Exception as e:
        print(f"Failed to fetch Arxiv data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch.py <arxiv_url_or_id>")
        sys.exit(1)
        
    fetch_arxiv(sys.argv[1])
