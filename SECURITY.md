# Citation Assistant - Security Guide

## 🔒 Rock-Solid Security Implementation

Your Citation Assistant now has enterprise-grade security with:
- JWT (JSON Web Token) authentication
- Password hashing with bcrypt
- Trusted host middleware
- Session management
- Protected API endpoints

## Quick Start (Secure Setup)

### 1. Create Your First User

```bash
cd /home/david/projects/citation_assistant
mamba activate rag

# Interactive user creation
python manage_users.py

# Or quick command-line creation
python manage_users.py create admin YOUR_STRONG_PASSWORD your@email.com
```

**Password Requirements:**
- Minimum 8 characters
- Use a strong, unique password
- Consider using a password manager

### 2. Start the Secure Server

```bash
# Start manually
python server_secure.py

# Or use systemd (update service file first)
sudo cp citation-assistant-secure.service /etc/systemd/system/
sudo systemctl enable citation-assistant-secure
sudo systemctl start citation-assistant-secure
```

### 3. Access the Secure Interface

Open browser: `http://192.168.1.163:8000`

Login with your credentials.

## Security Features

### 1. JWT Authentication

**How it works:**
1. User logs in with username/password
2. Server validates credentials
3. Server issues JWT token (24-hour expiry)
4. Client includes token in all API requests
5. Server validates token for each request

**Token Storage:**
- Stored in browser localStorage
- Automatically included in requests
- Expires after 24 hours
- Must re-login after expiry

### 2. Password Security

**Hashing:**
- Uses bcrypt (industry standard)
- Salted and hashed (never stored in plain text)
- Computationally expensive (prevents brute force)

**Password File:**
- `/home/david/projects/citation_assistant/configs/users.json`
- Contains only hashed passwords
- Secure file permissions: `chmod 600 configs/users.json`

### 3. Trusted Host Middleware

**Allowed Hosts:**
- localhost / 127.0.0.1
- 192.168.1.163 (your server)
- 192.168.1.* (your local network)

Blocks requests from unauthorized hosts.

### 4. CORS Protection

**Allowed Origins:**
- http://192.168.1.163:8000
- http://localhost:8000

Prevents cross-site request forgery.

## User Management

### Create User (Interactive)

```bash
python manage_users.py
# Choose: 1. Create user
# Enter details when prompted
```

### Create User (Command Line)

```bash
python manage_users.py create USERNAME PASSWORD [EMAIL]

# Example:
python manage_users.py create alice SecurePass123 alice@example.com
```

### List Users

```bash
python manage_users.py
# Choose: 2. List users
```

### Change Password

```bash
python manage_users.py
# Choose: 4. Change password
# Enter username and new password
```

### Delete User

```bash
python manage_users.py
# Choose: 3. Delete user
# Enter username to delete
```

## API Usage with Authentication

### Get Access Token

```bash
curl -X POST http://192.168.1.163:8000/api/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=YOUR_PASSWORD"
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Use Token in Requests

```bash
TOKEN="your_access_token_here"

# Search papers
curl -X POST http://192.168.1.163:8000/api/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "CRISPR", "n_results": 5}'
```

### Python Client Example

```python
import requests

class SecureCitationClient:
    def __init__(self, base_url, username, password):
        self.base_url = base_url
        self.token = self._login(username, password)

    def _login(self, username, password):
        """Get access token"""
        response = requests.post(
            f"{self.base_url}/api/token",
            data={"username": username, "password": password}
        )
        response.raise_for_status()
        return response.json()["access_token"]

    def _headers(self):
        """Get auth headers"""
        return {"Authorization": f"Bearer {self.token}"}

    def search(self, query, n=10):
        """Search papers"""
        response = requests.post(
            f"{self.base_url}/api/search",
            headers=self._headers(),
            json={"query": query, "n_results": n}
        )
        response.raise_for_status()
        return response.json()["papers"]

    def suggest(self, text, n=3):
        """Get citation suggestions"""
        response = requests.post(
            f"{self.base_url}/api/suggest/text",
            headers=self._headers(),
            data={"text": text, "n_suggestions": n}
        )
        response.raise_for_status()
        return response.json()["suggestions"]

# Usage
client = SecureCitationClient(
    "http://192.168.1.163:8000",
    "admin",
    "YOUR_PASSWORD"
)

papers = client.search("microbiome")
for paper in papers:
    print(f"{paper['filename']}: {paper['similarity']:.2%}")
```

## Security Best Practices

### 1. Strong Passwords

✅ **DO:**
- Use passwords ≥ 12 characters
- Mix uppercase, lowercase, numbers, symbols
- Use unique passwords per user
- Use a password manager

❌ **DON'T:**
- Use common passwords
- Reuse passwords
- Share passwords
- Write passwords in code

### 2. File Permissions

```bash
# Secure user file
chmod 600 /home/david/projects/citation_assistant/configs/users.json

# Verify
ls -la /home/david/projects/citation_assistant/configs/
```

### 3. Network Security

**Local Network Only (Recommended):**
- Access only from devices on 192.168.1.x network
- No port forwarding needed
- No external exposure

**For External Access (Advanced):**
1. **Use VPN (Most Secure)**
   - Setup WireGuard/OpenVPN
   - Access server through VPN tunnel
   - No public exposure

2. **SSH Tunnel (Secure)**
   ```bash
   ssh -L 8000:localhost:8000 david@192.168.1.163
   # Access http://localhost:8000 on your machine
   ```

3. **Reverse Proxy with HTTPS (Production)**
   ```bash
   # Install nginx and certbot
   sudo apt install nginx certbot python3-certbot-nginx

   # Configure nginx
   sudo nano /etc/nginx/sites-available/citation-assistant

   # Get SSL certificate
   sudo certbot --nginx -d your-domain.com
   ```

### 4. Firewall Configuration

```bash
# Allow only from local network
sudo ufw allow from 192.168.1.0/24 to any port 8000

# Or specific IPs
sudo ufw allow from 192.168.1.100 to any port 8000
sudo ufw allow from 192.168.1.101 to any port 8000

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

### 5. Regular Maintenance

**Update passwords regularly:**
```bash
python manage_users.py
# Choose: 4. Change password
```

**Review users:**
```bash
python manage_users.py
# Choose: 2. List users
# Delete unused accounts
```

**Monitor logs:**
```bash
# If using systemd
sudo journalctl -u citation-assistant-secure -f --since "1 hour ago"
```

### 6. Token Management

**Token Expiration:**
- Tokens expire after 24 hours
- Users must re-login
- No persistent "remember me" (more secure)

**Revoke Access:**
- Change user password
- Delete user account
- Restart server (invalidates all tokens)

## Advanced Security (Optional)

### 1. Environment Variables for Secrets

```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set environment variable
export JWT_SECRET_KEY="your_generated_secret_here"

# Add to ~/.bashrc for persistence
echo 'export JWT_SECRET_KEY="your_secret"' >> ~/.bashrc
```

### 2. Rate Limiting

To prevent abuse, add rate limiting:

```bash
pip install slowapi

# In server_secure.py, add:
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add to endpoints:
@limiter.limit("10/minute")
async def search_papers(...):
    ...
```

### 3. HTTPS (Production)

```nginx
# /etc/nginx/sites-available/citation-assistant
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

**"Invalid credentials" error:**
- Check username/password
- Ensure user exists: `python manage_users.py`
- Verify password was entered correctly

**"Session expired" error:**
- Token expired (24 hours)
- Log in again
- Browser will prompt for login

**"Unauthorized" error:**
- Token missing or invalid
- Clear browser cache/localStorage
- Log in again

**Can't access from other devices:**
- Check firewall: `sudo ufw status`
- Verify server is running: `curl http://192.168.1.163:8000/api/health`
- Check allowed hosts in `server_secure.py`

**Forgot password:**
```bash
# Reset password as admin
python manage_users.py
# Choose: 4. Change password
```

## Security Checklist

- [x] JWT authentication enabled
- [x] Passwords hashed with bcrypt
- [x] Trusted host middleware active
- [x] CORS protection configured
- [x] User management system in place
- [ ] Secure users.json file permissions (`chmod 600`)
- [ ] Strong passwords for all users
- [ ] Firewall rules configured
- [ ] Regular password updates
- [ ] Unused accounts deleted
- [ ] HTTPS enabled (if external access)
- [ ] VPN/SSH tunnel (if remote access)

## Support

**Check server status:**
```bash
curl http://192.168.1.163:8000/api/health
```

**View API documentation:**
Open: `http://192.168.1.163:8000/docs`

**Test authentication:**
```bash
# Get token
curl -X POST http://192.168.1.163:8000/api/token \
  -d "username=admin&password=YOUR_PASSWORD"

# Should return: {"access_token": "...", "token_type": "bearer"}
```

Your Citation Assistant is now **production-ready and secure**! 🔒
