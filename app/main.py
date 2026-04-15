
from __future__ import annotations

import os
import re
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import qrcode
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "inventory.db"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
QRCODE_DIR = UPLOAD_DIR / "qrcodes"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
QRCODE_DIR.mkdir(parents=True, exist_ok=True)
APP_BASE_URL = os.getenv("APP_BASE_URL", "").rstrip("/")

CATEGORY_OPTIONS = [
    "Electronics", "Cable", "Tool", "Hardware", "Other"
]
TYPE_OPTIONS = [
    "Resistor", "Capacitor", "Inductor", "Diode", "Transistor", "IC", "Sensor", "Module",
    "Board", "Connector", "Cable", "Power Supply", "Battery", "Relay", "Switch", "Display",
    "Motor", "Driver", "Adapter", "Tool", "Fastener", "Other"
]
LOCATION_OPTIONS = [
    "Box 1", "Box 2", "Box 3", "Box 4", "Drawer A1", "Drawer A2", "Drawer A3",
    "Drawer B1", "Drawer B2", "Shelf 1", "Shelf 2", "Shelf 3", "Bench", "Field Kit"
]

app = FastAPI(title="Visual Inventory")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                structured_name TEXT NOT NULL,
                category TEXT NOT NULL,
                item_type TEXT NOT NULL,
                value_model TEXT,
                location TEXT NOT NULL,
                quantity INTEGER NOT NULL DEFAULT 1,
                tags TEXT,
                notes TEXT,
                image_path TEXT,
                qr_path TEXT,
                qr_payload TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        if not column_exists(conn, "items", "qr_path"):
            conn.execute("ALTER TABLE items ADD COLUMN qr_path TEXT")
        if not column_exists(conn, "items", "qr_payload"):
            conn.execute("ALTER TABLE items ADD COLUMN qr_payload TEXT")
        conn.commit()


def slugify(value: str) -> str:
    value = value.strip().upper()
    value = re.sub(r"[^A-Z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "NA"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def canonical_category(category: str) -> str:
    category = normalize_text(category)
    for option in CATEGORY_OPTIONS:
        if category.lower() == option.lower():
            return option
    return category.title() if category else "Other"


def canonical_type(item_type: str) -> str:
    item_type = normalize_text(item_type)
    for option in TYPE_OPTIONS:
        if item_type.lower() == option.lower():
            return option
    return item_type.title() if item_type else "Other"


def normalize_location(location: str, location_detail: str = "") -> str:
    base = normalize_text(location)
    detail = normalize_text(location_detail)
    return f"{base} / {detail}" if detail else base


def normalize_value_model(value: str) -> str:
    value = normalize_text(value)
    if not value:
        return ""
    compact = value.replace("Ω", "ohm")
    compact = re.sub(r"\s+", "", compact)
    compact_low = compact.lower()

    resistor = re.fullmatch(r"([0-9]*\.?[0-9]+)([kKmMrR]?)(ohm)?", compact_low)
    if resistor:
        num, suffix, _ = resistor.groups()
        suffix_map = {"": "Ω", "r": "Ω", "k": "kΩ", "m": "MΩ"}
        return f"{num}{suffix_map.get(suffix, '')}"

    cap = re.fullmatch(r"([0-9]*\.?[0-9]+)(uf|nf|pf|mf)([0-9]*\.?[0-9]+v)?", compact_low)
    if cap:
        num, unit, volt = cap.groups()
        unit_map = {"uf": "µF", "nf": "nF", "pf": "pF", "mf": "mF"}
        out = f"{num}{unit_map[unit]}"
        if volt:
            out += f" {volt.upper()}"
        return out

    return value


def build_structured_name(category: str, item_type: str, value_model: str, location: str, quantity: int) -> str:
    parts = [slugify(category), slugify(item_type), slugify(value_model or "GENERIC"), slugify(location), f"{quantity}PCS"]
    return "_".join(parts)


def save_upload(upload: Optional[UploadFile]) -> Optional[str]:
    if not upload or not upload.filename:
        return None

    suffix = Path(upload.filename).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported image type.")

    filename = f"{uuid.uuid4().hex}{suffix}"
    destination = UPLOAD_DIR / filename
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return f"/uploads/{filename}"


def build_qr_payload(item_id: int, structured_name: str, category: str, item_type: str, value_model: str, location: str, quantity: int) -> str:
    if APP_BASE_URL:
        return f"{APP_BASE_URL}/items/{item_id}"
    lines = [
        f"ID: {item_id}",
        f"Name: {structured_name}",
        f"Category: {category}",
        f"Type: {item_type}",
        f"Value/Model: {value_model or '-'}",
        f"Location: {location}",
        f"Quantity: {quantity}",
    ]
    return "\n".join(lines)


def generate_qr_file(payload: str, item_id: int) -> str:
    filename = f"item_{item_id}.png"
    destination = QRCODE_DIR / filename
    img = qrcode.make(payload)
    img.save(destination)
    return f"/uploads/qrcodes/{filename}"


def get_duplicate_candidates(
    conn: sqlite3.Connection,
    category: str,
    item_type: str,
    value_model: str,
    exclude_id: Optional[int] = None,
):
    normalized_value = normalize_value_model(value_model)
    sql = (
        "SELECT * FROM items WHERE lower(category)=lower(?) AND lower(item_type)=lower(?) "
        "AND lower(ifnull(value_model,''))=lower(?)"
    )
    params = [category, item_type, normalized_value]
    if exclude_id is not None:
        sql += " AND id != ?"
        params.append(exclude_id)
    sql += " ORDER BY created_at DESC, id DESC LIMIT 8"
    return conn.execute(sql, params).fetchall()


def serialize_duplicate_rows(rows) -> list[dict]:
    return [
        {
            "id": row["id"],
            "structured_name": row["structured_name"],
            "category": row["category"],
            "item_type": row["item_type"],
            "value_model": row["value_model"] or "-",
            "location": row["location"],
            "quantity": row["quantity"],
            "created_at": row["created_at"],
            "image_path": row["image_path"],
            "qr_path": row["qr_path"],
        }
        for row in rows
    ]




def split_location(location: str) -> tuple[str, str]:
    location = normalize_text(location)
    if " / " in location:
        primary, detail = location.split(" / ", 1)
        return primary, detail
    return location, ""


def refresh_item_identity(conn: sqlite3.Connection, item_id: int) -> None:
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Item not found.")
    structured_name = build_structured_name(
        row["category"], row["item_type"], row["value_model"] or "", row["location"], row["quantity"]
    )
    qr_payload = build_qr_payload(
        item_id, structured_name, row["category"], row["item_type"], row["value_model"] or "", row["location"], row["quantity"]
    )
    qr_path = generate_qr_file(qr_payload, item_id)
    conn.execute(
        "UPDATE items SET structured_name = ?, qr_path = ?, qr_payload = ? WHERE id = ?",
        (structured_name, qr_path, qr_payload, item_id),
    )

def make_form_state(
    category: str = "Electronics",
    item_type: str = "Resistor",
    value_model: str = "",
    location: str = LOCATION_OPTIONS[0],
    location_detail: str = "",
    quantity: int = 1,
    tags: str = "",
    notes: str = "",
) -> dict:
    return {
        "category": category,
        "item_type": item_type,
        "value_model": value_model,
        "location": location,
        "location_detail": location_detail,
        "quantity": quantity,
        "tags": tags,
        "notes": notes,
    }


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request, q: str = ""):
    with get_conn() as conn:
        if q.strip():
            like = f"%{q.strip()}%"
            rows = conn.execute(
                """
                SELECT * FROM items
                WHERE structured_name LIKE ?
                   OR category LIKE ?
                   OR item_type LIKE ?
                   OR value_model LIKE ?
                   OR location LIKE ?
                   OR tags LIKE ?
                   OR notes LIKE ?
                ORDER BY created_at DESC, id DESC
                """,
                (like, like, like, like, like, like, like),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM items ORDER BY created_at DESC, id DESC").fetchall()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "items": rows,
            "q": q,
            "category_options": CATEGORY_OPTIONS,
            "type_options": TYPE_OPTIONS,
            "location_options": LOCATION_OPTIONS,
            "form_data": make_form_state(),
            "duplicate_candidates": [],
            "duplicate_message": "",
        },
    )


@app.get("/api/duplicate-check")
def duplicate_check(
    category: str = Query(...),
    item_type: str = Query(...),
    value_model: str = Query(""),
):
    category = canonical_category(category)
    item_type = canonical_type(item_type)
    value_model = normalize_value_model(value_model)
    if not value_model:
        return JSONResponse({"duplicates": [], "count": 0})
    with get_conn() as conn:
        rows = get_duplicate_candidates(conn, category, item_type, value_model)
    return JSONResponse({"duplicates": serialize_duplicate_rows(rows), "count": len(rows)})


@app.get("/items/{item_id}", response_class=HTMLResponse)
def item_detail(request: Request, item_id: int, updated: int = 0):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Item not found.")
    primary_location, location_detail = split_location(row["location"] or "")
    return templates.TemplateResponse(
        "item_detail.html",
        {
            "request": request,
            "item": row,
            "category_options": CATEGORY_OPTIONS,
            "type_options": TYPE_OPTIONS,
            "location_options": LOCATION_OPTIONS,
            "primary_location": primary_location,
            "location_detail": location_detail,
            "updated": updated,
        },
    )


@app.post("/items")
async def create_item(
    request: Request,
    category: str = Form(...),
    item_type: str = Form(...),
    value_model: str = Form(""),
    location: str = Form(...),
    location_detail: str = Form(""),
    quantity: int = Form(1),
    tags: str = Form(""),
    notes: str = Form(""),
    confirm_duplicate: int = Form(0),
    photo: Optional[UploadFile] = File(None),
):
    if quantity < 1:
        raise HTTPException(status_code=400, detail="Quantity must be at least 1.")

    category = canonical_category(category)
    item_type = canonical_type(item_type)
    value_model = normalize_value_model(value_model)
    location = normalize_location(location, location_detail)
    tags = normalize_text(tags)
    notes = normalize_text(notes)

    with get_conn() as conn:
        duplicates = get_duplicate_candidates(conn, category, item_type, value_model) if value_model else []
        if duplicates and not confirm_duplicate:
            if photo and photo.file:
                try:
                    photo.file.close()
                except Exception:
                    pass
            if request.headers.get("x-requested-with") == "XMLHttpRequest":
                return JSONResponse(
                    {
                        "status": "duplicate_detected",
                        "duplicates": serialize_duplicate_rows(duplicates),
                        "message": f"Found {len(duplicates)} possible duplicate(s) with the same category, type, and value/model.",
                    },
                    status_code=409,
                )
            rows = conn.execute("SELECT * FROM items ORDER BY created_at DESC, id DESC").fetchall()
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "items": rows,
                    "q": "",
                    "category_options": CATEGORY_OPTIONS,
                    "type_options": TYPE_OPTIONS,
                    "location_options": LOCATION_OPTIONS,
                    "form_data": make_form_state(category, item_type, value_model, location.split(" / ")[0], location.split(" / ", 1)[1] if " / " in location else "", quantity, tags, notes),
                    "duplicate_candidates": duplicates,
                    "duplicate_message": f"Found {len(duplicates)} possible duplicate(s). Re-select the photo if you still want to save this new item.",
                },
                status_code=409,
            )

        structured_name = build_structured_name(category, item_type, value_model, location, quantity)
        image_path = save_upload(photo)
        cursor = conn.execute(
            """
            INSERT INTO items (structured_name, category, item_type, value_model, location, quantity, tags, notes, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (structured_name, category, item_type, value_model, location, quantity, tags, notes, image_path),
        )
        item_id = cursor.lastrowid
        qr_payload = build_qr_payload(item_id, structured_name, category, item_type, value_model, location, quantity)
        qr_path = generate_qr_file(qr_payload, item_id)
        conn.execute(
            "UPDATE items SET qr_path = ?, qr_payload = ? WHERE id = ?",
            (qr_path, qr_payload, item_id),
        )
        conn.commit()

    return RedirectResponse(url=f"/?created={item_id}", status_code=303)


@app.post("/items/{item_id}/update")
async def update_item(
    item_id: int,
    category: str = Form(...),
    item_type: str = Form(...),
    value_model: str = Form(""),
    location: str = Form(...),
    location_detail: str = Form(""),
    quantity: int = Form(1),
    tags: str = Form(""),
    notes: str = Form(""),
    photo: Optional[UploadFile] = File(None),
):
    if quantity < 1:
        raise HTTPException(status_code=400, detail="Quantity must be at least 1.")

    category = canonical_category(category)
    item_type = canonical_type(item_type)
    value_model = normalize_value_model(value_model)
    location = normalize_location(location, location_detail)
    tags = normalize_text(tags)
    notes = normalize_text(notes)

    with get_conn() as conn:
        existing = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Item not found.")

        image_path = existing["image_path"]
        if photo and getattr(photo, "filename", None):
            new_image_path = save_upload(photo)
            old_image_path = existing["image_path"]
            image_path = new_image_path
            if old_image_path:
                try:
                    relative = old_image_path.replace("/uploads/", "")
                    local_path = UPLOAD_DIR / relative
                    if local_path.exists():
                        local_path.unlink()
                except OSError:
                    pass

        conn.execute(
            """
            UPDATE items
            SET category = ?, item_type = ?, value_model = ?, location = ?, quantity = ?, tags = ?, notes = ?, image_path = ?
            WHERE id = ?
            """,
            (category, item_type, value_model, location, quantity, tags, notes, image_path, item_id),
        )
        refresh_item_identity(conn, item_id)
        conn.commit()

    return RedirectResponse(url=f"/items/{item_id}?updated=1", status_code=303)


@app.post("/items/{item_id}/add-quantity")
def add_quantity(item_id: int, amount: int = Form(...)):
    if amount < 1:
        raise HTTPException(status_code=400, detail="Added quantity must be at least 1.")

    with get_conn() as conn:
        row = conn.execute("SELECT quantity FROM items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Item not found.")
        new_quantity = int(row["quantity"]) + int(amount)
        conn.execute("UPDATE items SET quantity = ? WHERE id = ?", (new_quantity, item_id))
        refresh_item_identity(conn, item_id)
        conn.commit()

    return RedirectResponse(url=f"/items/{item_id}?updated=1", status_code=303)


@app.post("/items/{item_id}/delete")
def delete_item(item_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT image_path, qr_path FROM items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Item not found.")
        conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        conn.commit()

    for path in (row["image_path"], row["qr_path"]):
        if not path:
            continue
        try:
            relative = path.replace("/uploads/", "")
            local_path = UPLOAD_DIR / relative
            if local_path.exists():
                local_path.unlink()
        except OSError:
            pass

    return RedirectResponse(url="/", status_code=303)


@app.get("/labels", response_class=HTMLResponse)
def labels_page(request: Request, q: str = Query("")):
    with get_conn() as conn:
        if q.strip():
            like = f"%{q.strip()}%"
            rows = conn.execute(
                """
                SELECT * FROM items
                WHERE structured_name LIKE ? OR category LIKE ? OR item_type LIKE ? OR value_model LIKE ? OR location LIKE ?
                ORDER BY created_at DESC, id DESC
                """,
                (like, like, like, like, like),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM items ORDER BY created_at DESC, id DESC").fetchall()
    return templates.TemplateResponse("labels.html", {"request": request, "items": rows, "q": q})
