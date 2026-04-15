import os
import re
import secrets
import shutil
import sqlite3
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import qrcode
from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.middleware.sessions import SessionMiddleware

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change-me")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-not-for-production")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
DB_PATH = Path(os.getenv("DB_PATH", str(UPLOAD_DIR / "inventory.db")))
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Electronics Inventory")

# Session middleware must exist before request.session is accessed anywhere.
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=False,
    same_site="lax",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)

PUBLIC_PATHS = {"/login", "/health"}


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            item_type TEXT NOT NULL,
            value_model TEXT NOT NULL,
            normalized_value TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            location TEXT NOT NULL,
            tags TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            structured_name TEXT NOT NULL,
            photo_filename TEXT DEFAULT '',
            qr_filename TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path

    if path in PUBLIC_PATHS or path.startswith("/static"):
        return await call_next(request)

    if request.session.get("authenticated") is True:
        return await call_next(request)

    next_path = quote(path)
    return RedirectResponse(url=f"/login?next={next_path}", status_code=303)


def render(request: Request, template_name: str, **context):
    template = env.get_template(template_name)
    base_context = {
        "request": request,
        "app_base_url": APP_BASE_URL,
        "logged_in": request.session.get("authenticated", False),
    }
    base_context.update(context)
    return HTMLResponse(template.render(**base_context))


def slugify(text: str) -> str:
    text = (text or "").strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "NA"


def normalize_value(item_type: str, value: str) -> str:
    s = (value or "").strip().lower().replace(" ", "")
    if not s:
        return ""

    if item_type.lower() == "resistor":
        m = re.fullmatch(r"(\d+(?:\.\d+)?)([kmr]?)(?:ohm|Ω)?", s)
        if m:
            num = float(m.group(1))
            suffix = m.group(2)
            mult = {"": 1, "r": 1, "k": 1_000, "m": 1_000_000}.get(suffix, 1)
            return f"{int(num * mult)}OHM"

    if item_type.lower() == "capacitor":
        m = re.fullmatch(r"(\d+(?:\.\d+)?)(pf|nf|uf|mf|f)(?:[-_]?(\d+(?:\.\d+)?v))?", s)
        if m:
            num = float(m.group(1))
            unit = m.group(2).upper()
            volt = (m.group(3) or "").upper()
            return f"{num:g}{unit}{volt}"

    return s.upper()


def build_structured_name(category: str, item_type: str, value_model: str, location: str, quantity: int) -> str:
    return "_".join(
        [
            slugify(category),
            slugify(item_type),
            slugify(value_model),
            slugify(location),
            f"{quantity}PCS",
        ]
    )


def save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix.lower() or ".jpg"
    filename = f"{secrets.token_hex(8)}{suffix}"
    target = UPLOAD_DIR / filename
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return filename


def create_qr(item_id: int) -> str:
    target_url = f"{APP_BASE_URL}/items/{item_id}"
    img = qrcode.make(target_url)
    filename = f"qr_{item_id}.png"
    img.save(UPLOAD_DIR / filename)
    return filename


def get_item(item_id: int):
    conn = db()
    item = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    conn.close()
    return item


def find_duplicates(category: str, item_type: str, normalized_value: str, exclude_id: Optional[int] = None):
    conn = db()
    if exclude_id is None:
        rows = conn.execute(
            """
            SELECT * FROM items
            WHERE category = ? AND item_type = ? AND normalized_value = ?
            ORDER BY updated_at DESC, id DESC
            """,
            (category, item_type, normalized_value),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM items
            WHERE category = ? AND item_type = ? AND normalized_value = ? AND id != ?
            ORDER BY updated_at DESC, id DESC
            """,
            (category, item_type, normalized_value, exclude_id),
        ).fetchall()
    conn.close()
    return rows


TYPE_OPTIONS = [
    "Resistor", "Capacitor", "Inductor", "Diode", "Transistor", "IC",
    "Module", "Sensor", "Board", "Cable", "Connector", "Power Supply", "Tool", "Other"
]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/"):
    if request.session.get("authenticated"):
        return RedirectResponse(url=next or "/", status_code=303)
    return render(request, "login.html", next=next)


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
):
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        request.session["authenticated"] = True
        return RedirectResponse(url=next or "/", status_code=303)
    return render(request, "login.html", next=next, error="Invalid credentials.")


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, q: str = ""):
    conn = db()
    search = q.strip()
    if search:
        pattern = f"%{search}%"
        items = conn.execute(
            """
            SELECT * FROM items
            WHERE structured_name LIKE ?
               OR value_model LIKE ?
               OR location LIKE ?
               OR tags LIKE ?
               OR notes LIKE ?
            ORDER BY updated_at DESC, id DESC
            """,
            (pattern, pattern, pattern, pattern, pattern),
        ).fetchall()
    else:
        items = conn.execute("SELECT * FROM items ORDER BY updated_at DESC, id DESC").fetchall()
    conn.close()

    return render(
        request,
        "index.html",
        items=items,
        q=q,
        type_options=TYPE_OPTIONS,
    )


@app.post("/items")
async def create_item(
    request: Request,
    category: str = Form(...),
    item_type: str = Form(...),
    value_model: str = Form(...),
    quantity: int = Form(...),
    location: str = Form(...),
    tags: str = Form(""),
    notes: str = Form(""),
    confirm_duplicate: str = Form("0"),
    photo: Optional[UploadFile] = File(None),
):
    normalized_value = normalize_value(item_type, value_model)
    duplicates = find_duplicates(category, item_type, normalized_value)

    if duplicates and confirm_duplicate != "1":
        conn = db()
        items = conn.execute("SELECT * FROM items ORDER BY updated_at DESC, id DESC").fetchall()
        conn.close()
        return render(
            request,
            "index.html",
            items=items,
            q="",
            type_options=TYPE_OPTIONS,
            duplicate_warning=True,
            duplicate_items=duplicates,
            pending_form={
                "category": category,
                "item_type": item_type,
                "value_model": value_model,
                "quantity": quantity,
                "location": location,
                "tags": tags,
                "notes": notes,
            },
        )

    structured_name = build_structured_name(category, item_type, value_model, location, quantity)
    photo_filename = save_upload(photo) if photo and photo.filename else ""

    conn = db()
    cur = conn.execute(
        """
        INSERT INTO items (
            category, item_type, value_model, normalized_value,
            quantity, location, tags, notes, structured_name, photo_filename
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            category,
            item_type,
            value_model,
            normalized_value,
            quantity,
            location,
            tags,
            notes,
            structured_name,
            photo_filename,
        ),
    )
    item_id = cur.lastrowid
    qr_filename = create_qr(item_id)
    conn.execute(
        "UPDATE items SET qr_filename = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (qr_filename, item_id),
    )
    conn.commit()
    conn.close()

    return RedirectResponse(url=f"/items/{item_id}", status_code=303)


@app.get("/items/{item_id}", response_class=HTMLResponse)
async def item_detail(request: Request, item_id: int):
    item = get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    duplicates = find_duplicates(
        item["category"],
        item["item_type"],
        item["normalized_value"],
        exclude_id=item_id,
    )
    return render(request, "item.html", item=item, duplicates=duplicates, type_options=TYPE_OPTIONS)


@app.post("/items/{item_id}/edit")
async def edit_item(
    item_id: int,
    category: str = Form(...),
    item_type: str = Form(...),
    value_model: str = Form(...),
    quantity: int = Form(...),
    location: str = Form(...),
    tags: str = Form(""),
    notes: str = Form(""),
    photo: Optional[UploadFile] = File(None),
):
    item = get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    normalized_value = normalize_value(item_type, value_model)
    structured_name = build_structured_name(category, item_type, value_model, location, quantity)
    photo_filename = item["photo_filename"]

    if photo and photo.filename:
        photo_filename = save_upload(photo)

    conn = db()
    conn.execute(
        """
        UPDATE items
        SET category=?, item_type=?, value_model=?, normalized_value=?,
            quantity=?, location=?, tags=?, notes=?, structured_name=?,
            photo_filename=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (
            category,
            item_type,
            value_model,
            normalized_value,
            quantity,
            location,
            tags,
            notes,
            structured_name,
            photo_filename,
            item_id,
        ),
    )
    qr_filename = create_qr(item_id)
    conn.execute("UPDATE items SET qr_filename = ? WHERE id = ?", (qr_filename, item_id))
    conn.commit()
    conn.close()

    return RedirectResponse(url=f"/items/{item_id}", status_code=303)


@app.post("/items/{item_id}/adjust")
async def adjust_quantity(item_id: int, delta: int = Form(...)):
    item = get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    new_qty = max(0, int(item["quantity"]) + int(delta))
    structured_name = build_structured_name(
        item["category"], item["item_type"], item["value_model"], item["location"], new_qty
    )

    conn = db()
    conn.execute(
        "UPDATE items SET quantity=?, structured_name=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (new_qty, structured_name, item_id),
    )
    conn.commit()
    conn.close()
    return RedirectResponse(url=f"/items/{item_id}", status_code=303)


@app.post("/items/{item_id}/delete")
async def delete_item(item_id: int):
    item = get_item(item_id)
    if item:
        conn = db()
        conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        conn.commit()
        conn.close()
    return RedirectResponse(url="/", status_code=303)


@app.get("/media/{filename}")
async def protected_media(filename: str):
    path = UPLOAD_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


@app.get("/items/{item_id}/qr")
async def download_qr(item_id: int):
    item = get_item(item_id)
    if not item or not item["qr_filename"]:
        raise HTTPException(status_code=404, detail="QR not found")
    return FileResponse(
        UPLOAD_DIR / item["qr_filename"],
        media_type="image/png",
        filename=f"item_{item_id}_qr.png",
    )


@app.get("/labels", response_class=HTMLResponse)
async def labels_page(request: Request):
    conn = db()
    items = conn.execute("SELECT * FROM items ORDER BY updated_at DESC, id DESC").fetchall()
    conn.close()
    return render(request, "labels.html", items=items)


@app.get("/scan", response_class=HTMLResponse)
async def scan_page(request: Request):
    return render(request, "scan.html")
