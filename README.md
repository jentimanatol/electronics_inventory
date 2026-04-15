
# Electronics Inventory + QR Labels

This version upgrades the original visual inventory into an electronics-focused app with:
- structured electronics categories and type dropdowns
- normalized value/model formatting (example: `10k` -> `10kΩ`, `100uF 25V` -> `100µF 25V`)
- QR code generation for every new entry
- per-item QR download
- printable label sheet for later printing
- persistent image + QR storage when Railway volume is mounted to `/app/uploads`

## New routes
- `/items/{id}` - single item page with QR code
- `/labels` - printable QR sheet for all items or filtered items
- `/health` - health check

## Local run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Railway notes
1. Keep your volume mounted to `/app/uploads`
2. Redeploy after pushing this version
3. Optional but recommended: set environment variable `APP_BASE_URL` to your Railway app URL, for example:
   ```
   APP_BASE_URL=https://inventory-system-production-e494.up.railway.app
   ```
   Then each QR code will open the item page directly when scanned.

## Upgrade from current version
This version auto-migrates the SQLite table by adding:
- `qr_path`
- `qr_payload`

Existing items stay in the database. New items get QR codes automatically.
Existing items created before this upgrade will not have QR codes until you recreate them or add a small backfill script later.
