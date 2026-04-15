# Electronics Inventory - Private Single User

Working FastAPI project with:
- single-user login
- Railway healthcheck-safe `/health`
- persistent uploads/QR under `/app/uploads`
- duplicate detection
- edit existing item
- quantity adjust
- QR generation/download
- printable labels
- phone-camera QR scan page

## Railway Variables
Set these:
- `APP_BASE_URL=https://electronicsinventory-production.up.railway.app`
- `ADMIN_USERNAME=your_username`
- `ADMIN_PASSWORD=your_strong_password`
- `SECRET_KEY=long_random_secret`

Optional:
- `UPLOAD_DIR=/app/uploads`
- `DB_PATH=/app/uploads/inventory.db`

## Railway Volume
Mount your volume to:
`/app/uploads`

## Local Run
```bash
pip install -r requirements.txt
export APP_BASE_URL=http://127.0.0.1:8000
export ADMIN_USERNAME=admin
export ADMIN_PASSWORD=admin
export SECRET_KEY=dev-secret
uvicorn app.main:app --reload
```
