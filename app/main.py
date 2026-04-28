import os
import re
import secrets
import shutil
import sqlite3
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote

import qrcode
from itsdangerous import URLSafeSerializer, BadSignature
from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from fastapi import UploadFile, File
import tempfile
import shutil

try:
    from app.ai_decoder_lab import build_ai_lab_summary
except Exception:
    build_ai_lab_summary = None




APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin1")
SECRET_KEY = os.getenv("SECRET_KEY", "admiin_secret")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
DB_PATH = Path(os.getenv("DB_PATH", str(UPLOAD_DIR / "inventory.db")))
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Electronics Inventory")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)

serializer = URLSafeSerializer(SECRET_KEY, salt="electronics-inventory-auth")
AUTH_COOKIE_NAME = "inventory_auth"
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


def is_logged_in(request: Request) -> bool:
    token = request.cookies.get(AUTH_COOKIE_NAME)
    if not token:
        return False
    try:
        data = serializer.loads(token)
        return data.get("username") == ADMIN_USERNAME
    except BadSignature:
        return False


def make_auth_cookie() -> str:
    return serializer.dumps({"username": ADMIN_USERNAME})


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path

    if path in PUBLIC_PATHS or path.startswith("/static"):
        return await call_next(request)

    if is_logged_in(request):
        return await call_next(request)

    next_path = quote(path)
    return RedirectResponse(url=f"/login?next={next_path}", status_code=303)


def render(request: Request, template_name: str, **context):
    template = env.get_template(template_name)
    base_context = {
        "request": request,
        "app_base_url": APP_BASE_URL,
        "logged_in": is_logged_in(request),
    }
    base_context.update(context)
    return HTMLResponse(template.render(**base_context))


def slugify(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "NA"


def normalize_value(item_type: str, value: str) -> str:
    s = value.strip().lower().replace(" ", "")
    if not s:
        return ""

    if item_type.lower() == "resistor":
        m = re.fullmatch(r"(\d+(?:\.\d+)?)([kmr]?)(?:ohm|Ω)?", s)
        if m:
            num = float(m.group(1))
            suffix = m.group(2)
            mult = {"": 1, "r": 1, "k": 1000, "m": 1000000}.get(suffix, 1)
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
    return "_".join([
        slugify(category),
        slugify(item_type),
        slugify(value_model),
        slugify(location),
        f"{quantity}PCS",
    ])


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


# -------------------------
# Local AI / RAG assistant
# -------------------------
# Dependency-free so it works on Railway without a GPU or paid API key.
# Workflow: database rows -> text documents -> TF-IDF/cosine retrieval ->
# grounded inventory recommendations and diagnostics.

AI_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "have", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the",
    "to", "we", "what", "where", "which", "with", "you", "need", "find",
    "show", "give", "list", "all", "any", "inventory", "item", "items",
}

AI_SYNONYMS = {
    "arduino": {"uno", "mega", "nano", "giga", "microcontroller", "board", "mcu"},
    "raspberry": {"pi", "linux", "sbc", "computer"},
    "sensor": {"dht", "bme", "moisture", "hall", "imu", "temperature", "humidity"},
    "motor": {"stepper", "driver", "a4988", "drv8825", "coil", "winder"},
    "power": {"supply", "buck", "boost", "converter", "lm2596", "battery"},
    "wire": {"cable", "connector", "dupont", "jumper", "trrs", "jack"},
    "ai": {"assistant", "recommend", "smart", "rag", "search"},
    "restock": {"low", "missing", "shortage", "quantity", "qty", "buy"},
    "capacitor": {"cap", "caps", "ceramic", "electrolytic", "uf", "nf", "pf"},
    "resistor": {"ohm", "kohm", "mohm", "resistance"},
    "module": {"board", "breakout", "shield"},
}


def tokenize_ai(text: str):
    words = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    tokens = []
    for w in words:
        if len(w) <= 1 or w in AI_STOPWORDS:
            continue
        tokens.append(w)
        # Tiny singular/plural normalization: capacitors -> capacitor, sensors -> sensor.
        if len(w) > 3 and w.endswith("s"):
            tokens.append(w[:-1])
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(AI_SYNONYMS.get(token, set()))
    return expanded


def item_document(item) -> str:
    return " ".join([
        str(item["structured_name"]),
        str(item["category"]),
        str(item["item_type"]),
        str(item["value_model"]),
        str(item["normalized_value"]),
        str(item["location"]),
        str(item["tags"] or ""),
        str(item["notes"] or ""),
    ])


def load_items():
    conn = db()
    rows = conn.execute("SELECT * FROM items ORDER BY updated_at DESC, id DESC").fetchall()
    conn.close()
    return rows


def retrieve_ai_items(question: str, items, limit: int = 8):
    q_terms = Counter(tokenize_ai(question))
    if not q_terms:
        return []

    docs = [Counter(tokenize_ai(item_document(item))) for item in items]
    if not docs:
        return []

    df = defaultdict(int)
    for doc in docs:
        for term in doc:
            df[term] += 1

    n = len(docs)
    ranked = []
    for item, doc in zip(items, docs):
        dot = 0.0
        q_norm = 0.0
        d_norm = 0.0
        for term, q_count in q_terms.items():
            idf = math.log((n + 1) / (df.get(term, 0) + 1)) + 1
            q_weight = q_count * idf
            d_weight = doc.get(term, 0) * idf
            dot += q_weight * d_weight
            q_norm += q_weight * q_weight
        for term, d_count in doc.items():
            idf = math.log((n + 1) / (df.get(term, 0) + 1)) + 1
            d_norm += (d_count * idf) ** 2
        score = dot / ((math.sqrt(q_norm) * math.sqrt(d_norm)) or 1.0)
        if score > 0:
            ranked.append({"item": item, "score": round(score, 3)})

    return sorted(ranked, key=lambda r: r["score"], reverse=True)[:limit]


def direct_type_matches(question: str, items):
    """Return exact item-type/model matches for questions like 'do I have capacitors?'."""
    q_tokens = set(tokenize_ai(question))
    matches = []
    for item in items:
        item_type_tokens = set(tokenize_ai(str(item["item_type"])))
        value_tokens = set(tokenize_ai(str(item["value_model"])))
        tag_tokens = set(tokenize_ai(str(item["tags"] or "")))
        if q_tokens & (item_type_tokens | value_tokens | tag_tokens):
            matches.append({"item": item, "score": 1.0})
    return matches[:8]


def question_targets_known_type(question: str):
    q_tokens = set(tokenize_ai(question))
    known = {
        "capacitor", "resistor", "inductor", "diode", "transistor", "sensor",
        "module", "board", "cable", "connector", "power", "tool", "motor",
        "arduino", "raspberry", "a4988", "drv8825",
    }
    return sorted(q_tokens & known)


def inventory_ai_stats(items):
    total_quantity = sum(int(item["quantity"] or 0) for item in items)
    by_type = Counter(item["item_type"] for item in items)
    low_stock = [item for item in items if int(item["quantity"] or 0) <= 2]

    duplicate_groups = defaultdict(list)
    for item in items:
        key = (item["category"], item["item_type"], item["normalized_value"])
        duplicate_groups[key].append(item)
    duplicate_groups = [group for group in duplicate_groups.values() if len(group) > 1]

    return {
        "unique_items": len(items),
        "total_quantity": total_quantity,
        "top_types": by_type.most_common(6),
        "low_stock": low_stock[:10],
        "duplicate_groups": duplicate_groups[:8],
    }


def generate_ai_answer(question: str, items):
    question_clean = (question or "").strip()
    stats = inventory_ai_stats(items)
    retrieved = retrieve_ai_items(question_clean, items)
    # Add an exact type/model pass so simple questions like
    # "do I have capacitors?" still work even with a very small database.
    direct_matches = direct_type_matches(question_clean, items)
    seen_ids = {r["item"]["id"] for r in retrieved}
    retrieved.extend([r for r in direct_matches if r["item"]["id"] not in seen_ids])
    retrieved = retrieved[:8]
    q_lower = question_clean.lower()

    if not items:
        return {
            "answer": "Your inventory is empty, so the assistant cannot make grounded recommendations yet. Add a few items first, then ask for project parts, shortages, or duplicate checks.",
            "retrieved": [],
            "stats": stats,
            "mode": "empty-inventory",
        }

    if not question_clean:
        return {
            "answer": "Ask me something like: 'What Arduino parts do I have?', 'What should I restock?', 'Find sensors for a smart garden project', or 'Do I have duplicate capacitors?'.",
            "retrieved": [],
            "stats": stats,
            "mode": "help",
        }

    answer_parts = []
    mode = "semantic-rag"

    if any(word in q_lower for word in ["restock", "low", "shortage", "buy", "missing", "quantity", "qty"]):
        mode = "stock-diagnostic"
        if stats["low_stock"]:
            answer_parts.append("Main warning: these parts have low quantity and should be checked before the next build:")
            for item in stats["low_stock"][:6]:
                answer_parts.append(f"- {item['value_model']} ({item['item_type']}) — qty {item['quantity']} — {item['location']}")
        else:
            answer_parts.append("I do not see low-stock items with quantity 2 or less.")

    if any(word in q_lower for word in ["duplicate", "same", "repeated"]):
        mode = "duplicate-diagnostic"
        if stats["duplicate_groups"]:
            answer_parts.append("Possible duplicate groups found:")
            for group in stats["duplicate_groups"][:5]:
                names = "; ".join([f"#{g['id']} {g['value_model']} qty {g['quantity']}" for g in group])
                answer_parts.append(f"- {group[0]['item_type']} {group[0]['normalized_value']}: {names}")
        else:
            answer_parts.append("No strong duplicate groups were found using category + type + normalized value.")

    if retrieved:
        if not answer_parts:
            answer_parts.append("Best matching inventory items based on your question:")
        for r in retrieved[:5]:
            item = r["item"]
            answer_parts.append(
                f"- #{item['id']} {item['value_model']} ({item['item_type']}) — qty {item['quantity']} — {item['location']} — relevance {r['score']}"
            )
        if any(word in q_lower for word in ["project", "build", "kit", "arduino", "sensor", "motor", "coil", "garden"]):
            answer_parts.append("Project suggestion: open the top matching items, verify quantity, then print QR labels for the physical boxes before starting the build.")
    elif not answer_parts:
        targets = question_targets_known_type(question_clean)
        if targets and any(phrase in q_lower for phrase in ["do i have", "have", "in store", "in stock"]):
            current_types = ", ".join([f"{t} × {c}" for t, c in stats["top_types"]]) or "none"
            answer_parts.append(
                f"I do not see a matching record for: {', '.join(targets)}. "
                f"Current top inventory types are: {current_types}. "
                "If the part exists physically, add it with that component name in Type, Value/Model, Tags, or Notes so AI search can find it."
            )
        else:
            answer_parts.append("I could not find a strong match. Try a component name, model number, project keyword, location, or tag such as 'Arduino', 'sensor', 'A4988', 'capacitor', or 'Drawer A2'.")

    return {
        "answer": "\n".join(answer_parts),
        "retrieved": retrieved,
        "stats": stats,
        "mode": mode,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/"):
    if is_logged_in(request):
        return RedirectResponse(url=next or "/", status_code=303)
    return render(request, "login.html", next=next)


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
):
    if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
        return render(request, "login.html", error="Invalid credentials.", next=next)

    response = RedirectResponse(url=unquote(next) if next else "/", status_code=303)
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=make_auth_cookie(),
        httponly=True,
        secure=True,      # set True later when everything works
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
        path="/",
    )
    return response


@app.post("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    return response


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
        category_options=["Electronics"],
        type_options=[
            "Resistor", "Capacitor", "Inductor", "Diode", "Transistor", "IC",
            "Module", "Sensor", "Board", "Cable", "Connector", "Power Supply", "Tool", "Other"
        ],
    )


@app.get("/ai", response_class=HTMLResponse)
async def ai_page(request: Request, q: str = ""):
    items = load_items()
    result = generate_ai_answer(q, items)
    lab = build_ai_lab_summary(q, result, items) if build_ai_lab_summary else None
    return render(request, "ai.html", q=q, result=result, items=items, lab=lab)


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
            category_options=["Electronics"],
            type_options=[
                "Resistor", "Capacitor", "Inductor", "Diode", "Transistor", "IC",
                "Module", "Sensor", "Board", "Cable", "Connector", "Power Supply", "Tool", "Other"
            ],
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
            category, item_type, value_model, normalized_value,
            quantity, location, tags, notes, structured_name, photo_filename
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
    return render(request, "item.html", item=item, duplicates=duplicates)



@app.get("/backup")
async def backup():
    return FileResponse(
        DB_PATH,
        media_type="application/octet-stream",
        filename="inventory_backup.db",
    )


@app.post("/restore")
async def restore_db(file: UploadFile = File(...)):
    if not file.filename.endswith(".db"):
        return {"error": "Only .db files allowed"}

    backup_path = DB_PATH.with_suffix(".backup.db")
    shutil.copy(DB_PATH, backup_path)

    try:
        with open(DB_PATH, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        shutil.copy(backup_path, DB_PATH)
        return {"error": "Restore failed, rollback applied"}

    return RedirectResponse(url="/", status_code=303)







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
            category, item_type, value_model, normalized_value,
            quantity, location, tags, notes, structured_name,
            photo_filename, item_id
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
        item["category"],
        item["item_type"],
        item["value_model"],
        item["location"],
        new_qty,
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
    if not item:
        return RedirectResponse(url="/", status_code=303)

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



@app.get("/admin/backup", response_class=HTMLResponse)
async def backup_page(request: Request):
    return render(request, "backup.html")




@app.get("/api/items/{item_id}")
async def api_item(item_id: int):
    item = get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Not found")
    return dict(item)