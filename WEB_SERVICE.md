# Citation Assistant Web Service

Remote access to your Citation Assistant via web interface and REST API.

## Quick Start

### Option 1: Manual Start (Testing)

```bash
cd /home/david/projects/citation_assistant
bash start_server.sh
```

Access at: `http://YOUR_SERVER_IP:8000`

Press `Ctrl+C` to stop.

### Option 2: Systemd Service (Auto-start on boot)

```bash
# Copy service file
sudo cp citation-assistant.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable citation-assistant
sudo systemctl start citation-assistant

# Check status
sudo systemctl status citation-assistant

# View logs
sudo journalctl -u citation-assistant -f
```

## Accessing the Service

### Web Interface

Open in browser: `http://YOUR_SERVER_IP:8000`

Features:
- **Search Papers**: Search your EndNote library
- **Summarize Research**: Get AI summaries on topics
- **Suggest Citations**: Upload or paste manuscript text
- **Update Index**: Trigger re-indexing of new papers

### API Documentation

Interactive API docs: `http://YOUR_SERVER_IP:8000/docs`

### API Endpoints

**Health Check**
```bash
curl http://YOUR_SERVER_IP:8000/api/health
```

**Get Statistics**
```bash
curl http://YOUR_SERVER_IP:8000/api/stats
```

**Search Papers**
```bash
curl -X POST http://YOUR_SERVER_IP:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning genomics", "n_results": 10}'
```

**Summarize Research**
```bash
curl -X POST http://YOUR_SERVER_IP:8000/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"query": "microbiome analysis", "n_papers": 5}'
```

**Suggest Citations (Text)**
```bash
curl -X POST http://YOUR_SERVER_IP:8000/api/suggest/text \
  -F "text=Your manuscript text here..." \
  -F "n_suggestions=3"
```

**Suggest Citations (File)**
```bash
curl -X POST http://YOUR_SERVER_IP:8000/api/suggest \
  -F "manuscript=@my_manuscript.txt" \
  -F "n_suggestions=3"
```

**Trigger Indexing**
```bash
curl -X POST http://YOUR_SERVER_IP:8000/api/index
```

## Remote Access Setup

### Local Network Access

The server binds to `0.0.0.0:8000` by default, accessible from any device on your network.

Find your server IP:
```bash
hostname -I
```

Access from another computer: `http://YOUR_SERVER_IP:8000`

### Port Forwarding (External Access)

If you want to access from outside your network:

1. **Router Configuration**:
   - Forward port 8000 to your server's local IP
   - Use your public IP or setup dynamic DNS

2. **Firewall**:
   ```bash
   sudo ufw allow 8000/tcp
   ```

3. **Security Warning**: For production, use:
   - HTTPS (SSL/TLS)
   - Authentication
   - Reverse proxy (nginx/apache)
   - VPN access instead of public exposure

### SSH Tunnel (Secure Remote Access)

For secure access without exposing ports:

```bash
# From remote computer
ssh -L 8000:localhost:8000 david@YOUR_SERVER_IP

# Then access http://localhost:8000 in your browser
```

## Python API Client Example

```python
import requests

API_URL = "http://YOUR_SERVER_IP:8000/api"

# Search for papers
response = requests.post(
    f"{API_URL}/search",
    json={"query": "CRISPR gene editing", "n_results": 5}
)
papers = response.json()['papers']

for paper in papers:
    print(f"{paper['filename']}: {paper['similarity']:.2%}")

# Get citation suggestions
with open("manuscript.txt", "r") as f:
    manuscript_text = f.read()

response = requests.post(
    f"{API_URL}/suggest/text",
    data={"text": manuscript_text, "n_suggestions": 3}
)
suggestions = response.json()['suggestions']

for suggestion in suggestions:
    print(f"\\nStatement: {suggestion['statement']}")
    for paper in suggestion['suggested_papers']:
        print(f"  - {paper['filename']} ({paper['similarity']:.2%})")
```

## Configuration

### Change Port

Edit `server.py` and change:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # Change 8000 to your port
```

Or use environment variable:
```bash
PORT=9000 python server.py
```

### Performance Tuning

For multiple concurrent users, increase workers:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

## Monitoring

### Check Server Status
```bash
# If using systemd
sudo systemctl status citation-assistant

# Manual check
curl http://localhost:8000/api/health
```

### View Logs
```bash
# Systemd
sudo journalctl -u citation-assistant -f

# Manual run logs
# Check terminal output
```

### Resource Usage
```bash
# GPU usage
nvidia-smi

# CPU/Memory
htop
```

## Troubleshooting

**Server won't start:**
```bash
# Check if port is in use
sudo lsof -i:8000

# Check conda environment
conda activate rag
python server.py
```

**"Collection not found" error:**
```bash
# Run indexing first
cd /home/david/projects/citation_assistant
python cite.py index
```

**Slow responses:**
- First query after startup is slower (model loading)
- Large manuscripts take longer to analyze
- LLM inference can take 30-60 seconds

**Out of memory:**
- Reduce number of concurrent workers
- Check GPU memory: `nvidia-smi`
- Restart server

## Security Considerations

**For Local/Testing Use:**
- Current setup is fine for local network
- No authentication required

**For Production/External Access:**

1. **Add Authentication**:
   ```python
   from fastapi.security import HTTPBasic
   # Add auth to endpoints
   ```

2. **Use HTTPS**:
   ```bash
   # Setup nginx reverse proxy with SSL
   sudo apt install nginx certbot
   ```

3. **Rate Limiting**:
   ```python
   from slowapi import Limiter
   # Add rate limits
   ```

4. **VPN Access**:
   - Setup WireGuard or OpenVPN
   - Access server through VPN only

## Integration Examples

### Jupyter Notebook
```python
import requests

class CitationClient:
    def __init__(self, api_url="http://localhost:8000/api"):
        self.api_url = api_url

    def search(self, query, n=10):
        r = requests.post(f"{self.api_url}/search",
                         json={"query": query, "n_results": n})
        return r.json()['papers']

    def suggest(self, text, n=3):
        r = requests.post(f"{self.api_url}/suggest/text",
                         data={"text": text, "n_suggestions": n})
        return r.json()['suggestions']

# Usage
client = CitationClient()
papers = client.search("microbiome dysbiosis")
```

### R Script
```r
library(httr)
library(jsonlite)

search_papers <- function(query, n = 10) {
  response <- POST(
    "http://localhost:8000/api/search",
    body = list(query = query, n_results = n),
    encode = "json"
  )
  fromJSON(content(response, "text"))$papers
}

# Usage
papers <- search_papers("GWAS studies", n = 5)
```

## Stopping the Service

**Manual:**
```bash
# Press Ctrl+C in terminal
```

**Systemd:**
```bash
sudo systemctl stop citation-assistant
```

**Disable auto-start:**
```bash
sudo systemctl disable citation-assistant
```

## Updates and Maintenance

**Update index after adding papers:**
- Via web interface: Click "Update Index" tab
- Via API: `curl -X POST http://localhost:8000/api/index`
- Via CLI: `python cite.py index`

**Restart service after code changes:**
```bash
sudo systemctl restart citation-assistant
```

**View current stats:**
- Via web interface: Statistics shown at top
- Via API: `curl http://localhost:8000/api/stats`
