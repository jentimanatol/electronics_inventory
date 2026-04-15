# Electronics Inventory + QR + Private Login

This version keeps the app private for a single user.

## What was added
- single-user login page
- session-based authentication
- logout button
- all inventory pages require login
- protected media route for photos and QR images
- QR scans redirect to login first when logged out, then open the item after login

## Environment variables for Railway
Set these in Railway:

```text
APP_BASE_URL=https://electronicsinventory-production.up.railway.app
ADMIN_USERNAME=your_username
ADMIN_PASSWORD=choose_a_strong_password
SECRET_KEY=generate_a_long_random_secret
```

### Better option: store a password hash instead of plain text
You can use `ADMIN_PASSWORD_HASH` instead of `ADMIN_PASSWORD`.
The supported format is:

```text
pbkdf2_sha256$ITERATIONS$SALT$HEX_DIGEST
```

Example Python snippet to generate one locally:

```python
import hashlib, secrets
password = "your_password_here"
iterations = 260000
salt = secrets.token_hex(16)
digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations).hex()
print(f"pbkdf2_sha256${iterations}${salt}${digest}")
```

Then set:

```text
ADMIN_USERNAME=your_username
ADMIN_PASSWORD_HASH=<generated value>
SECRET_KEY=<long random secret>
```

## Important deployment notes
1. Keep the Railway volume mounted to `/app/uploads`
2. Redeploy after pushing this version
3. Photos and QR codes are now served through `/media/...` only after login

## Main routes
- `/login` — login page
- `/` — inventory dashboard (private)
- `/items/{id}` — item page (private)
- `/labels` — printable labels (private)
- `/health` — public health check

## Local run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ADMIN_USERNAME=admin
export ADMIN_PASSWORD=change_me
export SECRET_KEY=replace_me
uvicorn app.main:app --reload
```
