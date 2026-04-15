# Electronics Inventory (Private, Single User)

FastAPI app for a private electronics inventory with:
- single-user login
- duplicate detection
- editable items
- QR generation per item
- printable labels page
- protected photos and QR images
- public `/health` endpoint for Railway

## Railway variables
Set these in Railway:

- `APP_BASE_URL=https://your-domain.up.railway.app`
- `ADMIN_USERNAME=your_username`
- `ADMIN_PASSWORD=your_strong_password`
- `SECRET_KEY=long_random_secret_value`

Optional:
- `UPLOAD_DIR=/app/uploads`
- `DB_PATH=/app/uploads/inventory.db`

## Persistent volume
Mount your Railway volume to:

`/app/uploads`

## Local run

```bash
pip install -r requirements.txt
export ADMIN_USERNAME=admin
export ADMIN_PASSWORD=change-me
export SECRET_KEY=dev-secret
export APP_BASE_URL=http://127.0.0.1:8000
uvicorn app.main:app --reload
```

Open:
- `/login`
- `/health`
- `/labels`
