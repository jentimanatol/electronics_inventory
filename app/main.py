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

try:
    from app.ai_intent_model import classify_inventory_question
except Exception:
    classify_inventory_question = None




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

app = FastAPI(title="General Inventory")
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
            price REAL NOT NULL DEFAULT 0.0,
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
    cols = {row[1] for row in conn.execute("PRAGMA table_info(items)").fetchall()}
    if "price" not in cols:
        conn.execute("ALTER TABLE items ADD COLUMN price REAL NOT NULL DEFAULT 0.0")
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


def singularize_ai_token(word: str) -> str:
    irregular = {
        "categories": "category", "batteries": "battery", "supplies": "supply",
        "boxes": "box", "switches": "switch", "wires": "wire", "tools": "tool",
        "materials": "material", "documents": "document", "books": "book",
        "sensors": "sensor", "resistors": "resistor", "capacitors": "capacitor",
        "modules": "module", "connectors": "connector", "motors": "motor",
        "drivers": "driver", "boards": "board", "items": "item", "types": "type",
    }
    if word in irregular:
        return irregular[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and not word.endswith(("ses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def tokenize_ai(text: str):
    words = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    tokens = []
    for w in words:
        base = singularize_ai_token(w)
        for token in {w, base}:
            if len(token) <= 1 or token in AI_STOPWORDS:
                continue
            tokens.append(token)
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
    total_value = sum(int(item["quantity"] or 0) * float(item["price"] or 0) for item in items)
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
        "total_value": round(total_value, 2),
        "top_types": by_type.most_common(6),
        "low_stock": low_stock[:10],
        "duplicate_groups": duplicate_groups[:8],
    }


def ai_intent(question: str):
    if classify_inventory_question:
        try:
            return classify_inventory_question(question)
        except Exception:
            pass
    q = (question or "").lower()
    if any(p in q for p in ["show all", "list all", "all items", "everything", "full inventory"]):
        return {"intent": "show_all", "confidence": 0.8}
    if any(p in q for p in ["category", "categories", "types", "group by"]):
        return {"intent": "show_categories", "confidence": 0.8}
    if any(p in q for p in ["restock", "low stock", "shortage", "qty", "quantity"]):
        return {"intent": "low_stock", "confidence": 0.8}
    if any(p in q for p in ["duplicate", "same", "repeated"]):
        return {"intent": "duplicates", "confidence": 0.8}
    if any(p in q for p in ["price", "cost", "value", "worth", "money"]):
        return {"intent": "inventory_value", "confidence": 0.8}
    return {"intent": "search", "confidence": 0.5}


def format_ai_item_line(item, include_category: bool = True) -> str:
    category = f"{item['category']} / " if include_category else ""
    return (
        f"- #{item['id']} {category}{item['value_model']} ({item['item_type']}) "
        f"— qty {item['quantity']} — ${float(item['price'] or 0):.2f} each "
        f"— total ${int(item['quantity'] or 0) * float(item['price'] or 0):.2f} — {item['location']}"
    )


def build_category_summary(items):
    by_category = defaultdict(list)
    by_type = defaultdict(list)
    for item in items:
        by_category[item["category"]].append(item)
        by_type[item["item_type"]].append(item)

    lines = ["Inventory categories and groups:"]
    for category, group in sorted(by_category.items(), key=lambda kv: kv[0].lower()):
        qty = sum(int(i["quantity"] or 0) for i in group)
        value = sum(int(i["quantity"] or 0) * float(i["price"] or 0) for i in group)
        type_counts = Counter(i["item_type"] for i in group).most_common(4)
        type_text = ", ".join(f"{t} × {c}" for t, c in type_counts)
        lines.append(f"- {category}: {len(group)} records, qty {qty}, value ${value:.2f}; top types: {type_text or 'none'}")

    lines.append("")
    lines.append("Top item types:")
    for item_type, group in sorted(by_type.items(), key=lambda kv: (-len(kv[1]), kv[0].lower()))[:12]:
        qty = sum(int(i["quantity"] or 0) for i in group)
        lines.append(f"- {item_type}: {len(group)} records, qty {qty}")
    return "\n".join(lines)


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
    intent_info = ai_intent(question_clean)
    intent_name = intent_info.get("intent", "search")

    if not items:
        return {
            "answer": "Your inventory is empty, so the assistant cannot make grounded recommendations yet. Add a few items first, then ask for project parts, shortages, or duplicate checks.",
            "retrieved": [],
            "stats": stats,
            "mode": "empty-inventory",
        }

    if not question_clean:
        return {
            "answer": "Ask me something like: 'Show all items', 'Show categories', 'Do I have resistors?', 'What should I restock?', 'Find sensors for a smart garden project', or 'Do I have duplicate capacitors?'.",
            "retrieved": [],
            "stats": stats,
            "mode": "help",
        }

    if intent_name == "show_all":
        max_lines = 40
        lines = [f"Showing {min(len(items), max_lines)} of {len(items)} inventory records:"]
        for item in items[:max_lines]:
            lines.append(format_ai_item_line(item))
        if len(items) > max_lines:
            lines.append(f"... {len(items) - max_lines} more items not shown. Use a category or type question to narrow the list.")
        return {
            "answer": "\n".join(lines),
            "retrieved": [{"item": item, "score": 1.0} for item in items[:8]],
            "stats": stats,
            "mode": f"intent-show-all confidence={intent_info.get('confidence', 0):.2f}",
        }

    if intent_name == "show_categories":
        return {
            "answer": build_category_summary(items),
            "retrieved": [],
            "stats": stats,
            "mode": f"intent-show-categories confidence={intent_info.get('confidence', 0):.2f}",
        }

    answer_parts = []
    mode = f"semantic-rag intent={intent_name} confidence={intent_info.get('confidence', 0):.2f}"

    if intent_name == "low_stock" or any(word in q_lower for word in ["restock", "low", "shortage", "buy", "missing", "quantity", "qty"]):
        mode = "stock-diagnostic"
        if stats["low_stock"]:
            answer_parts.append("Main warning: these parts have low quantity and should be checked before the next build:")
            for item in stats["low_stock"][:6]:
                answer_parts.append(f"- {item['value_model']} ({item['item_type']}) — qty {item['quantity']} — {item['location']}")
        else:
            answer_parts.append("I do not see low-stock items with quantity 2 or less.")

    if intent_name == "duplicates" or any(word in q_lower for word in ["duplicate", "same", "repeated"]):
        mode = "duplicate-diagnostic"
        if stats["duplicate_groups"]:
            answer_parts.append("Possible duplicate groups found:")
            for group in stats["duplicate_groups"][:5]:
                names = "; ".join([f"#{g['id']} {g['value_model']} qty {g['quantity']} price ${float(g['price'] or 0):.2f}" for g in group])
                answer_parts.append(f"- {group[0]['item_type']} {group[0]['normalized_value']}: {names}")
        else:
            answer_parts.append("No strong duplicate groups were found using category + type + normalized value.")

    if intent_name == "inventory_value" or any(word in q_lower for word in ["price", "cost", "value", "worth", "money", "budget"]):
        mode = "value-diagnostic"
        answer_parts.append(f"Estimated stored inventory value: ${stats['total_value']:.2f} based on unit price × quantity. Items without price are counted as $0.00.")

    if retrieved:
        if not answer_parts:
            answer_parts.append("Best matching inventory items based on your question:")
        for r in retrieved[:5]:
            item = r["item"]
            answer_parts.append(
                f"- #{item['id']} {item['value_model']} ({item['item_type']}) — qty {item['quantity']} — ${float(item['price'] or 0):.2f} each — {item['location']} — relevance {r['score']}"
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
        category_options=["Electronics", "Tools", "Materials", "Office", "Household", "Books", "Documents", "Other"],
        type_options=[
            "Resistor", "Capacitor", "Inductor", "Diode", "Transistor", "IC",
            "Module", "Sensor", "Board", "Cable", "Connector", "Power Supply",
            "Tool", "Material", "Hardware", "Office Supply", "Book", "Document", "Furniture", "Other"
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
    price: float = Form(0.0),
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
                "price": price,
                "location": location,
                "tags": tags,
                "notes": notes,
            },
            category_options=["Electronics", "Tools", "Materials", "Office", "Household", "Books", "Documents", "Other"],
            type_options=[
                "Resistor", "Capacitor", "Inductor", "Diode", "Transistor", "IC",
                "Module", "Sensor", "Board", "Cable", "Connector", "Power Supply",
                "Tool", "Material", "Hardware", "Office Supply", "Book", "Document", "Furniture", "Other"
            ],
        )

    structured_name = build_structured_name(category, item_type, value_model, location, quantity)
    photo_filename = save_upload(photo) if photo and photo.filename else ""

    conn = db()
    cur = conn.execute(
        """
        INSERT INTO items (
            category, item_type, value_model, normalized_value,
            quantity, price, location, tags, notes, structured_name, photo_filename
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            category, item_type, value_model, normalized_value,
            quantity, price, location, tags, notes, structured_name, photo_filename
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
    price: float = Form(0.0),
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
            quantity=?, price=?, location=?, tags=?, notes=?, structured_name=?,
            photo_filename=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (
            category, item_type, value_model, normalized_value,
            quantity, price, location, tags, notes, structured_name,
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