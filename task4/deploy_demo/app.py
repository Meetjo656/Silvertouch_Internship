import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import sqlglot
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlglot import exp

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "demo.sqlite3"
MAX_ROWS = 100

app = FastAPI(title="Text-to-SQL RAG Demo", version="0.1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def get_connection(readonly: bool = False):
    if readonly:
        return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    return sqlite3.connect(DB_PATH)


def initialise_database() -> None:
    """Create a small synthetic retail database. No internship/customer data is used."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY,
                customer_name TEXT NOT NULL,
                city TEXT NOT NULL,
                segment TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                product_name TEXT NOT NULL,
                category TEXT NOT NULL,
                unit_price REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
                order_date TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS order_items (
                order_item_id INTEGER PRIMARY KEY,
                order_id INTEGER NOT NULL REFERENCES orders(order_id),
                product_id INTEGER NOT NULL REFERENCES products(product_id),
                quantity INTEGER NOT NULL,
                unit_price REAL NOT NULL
            );
            """
        )
        count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        if count == 0:
            conn.executemany(
                "INSERT INTO customers VALUES (?, ?, ?, ?)",
                [
                    (1, "Aarav Shah", "Ahmedabad", "Consumer"),
                    (2, "Diya Patel", "Mumbai", "Corporate"),
                    (3, "Kabir Mehta", "Pune", "Consumer"),
                    (4, "Anaya Rao", "Bengaluru", "Small Business"),
                    (5, "Ishaan Das", "Ahmedabad", "Corporate"),
                ],
            )
            conn.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?)",
                [
                    (1, "Laptop Pro 14", "Electronics", 82000),
                    (2, "Wireless Mouse", "Electronics", 1200),
                    (3, "Office Chair", "Furniture", 9500),
                    (4, "Desk Lamp", "Furniture", 1800),
                    (5, "Notebook Pack", "Stationery", 350),
                ],
            )
            conn.executemany(
                "INSERT INTO orders VALUES (?, ?, ?, ?)",
                [
                    (101, 1, "2025-01-12", "delivered"),
                    (102, 2, "2025-01-18", "delivered"),
                    (103, 1, "2025-02-03", "shipped"),
                    (104, 3, "2025-02-11", "cancelled"),
                    (105, 4, "2025-03-02", "delivered"),
                    (106, 5, "2025-03-17", "delivered"),
                    (107, 2, "2025-04-06", "processing"),
                ],
            )
            conn.executemany(
                "INSERT INTO order_items VALUES (?, ?, ?, ?, ?)",
                [
                    (1, 101, 1, 1, 82000),
                    (2, 101, 2, 2, 1200),
                    (3, 102, 3, 2, 9500),
                    (4, 103, 4, 3, 1800),
                    (5, 104, 5, 4, 350),
                    (6, 105, 1, 1, 79000),
                    (7, 105, 3, 1, 9500),
                    (8, 106, 2, 5, 1100),
                    (9, 107, 5, 10, 350),
                ],
            )


KNOWLEDGE = [
    {
        "name": "customers",
        "text": "customers(customer_id, customer_name, city, segment). One row per customer. Segment is Consumer, Corporate, or Small Business.",
    },
    {
        "name": "products",
        "text": "products(product_id, product_name, category, unit_price). One row per product. Categories include Electronics, Furniture, and Stationery.",
    },
    {
        "name": "orders",
        "text": "orders(order_id, customer_id, order_date, status). One row per order. Join customers using customer_id. Status values include delivered, shipped, cancelled, processing.",
    },
    {
        "name": "order_items",
        "text": "order_items(order_item_id, order_id, product_id, quantity, unit_price). One row per product line in an order. Join orders using order_id and products using product_id. Revenue per line is quantity * unit_price.",
    },
    {
        "name": "business_rules",
        "text": "Business rule: order revenue is SUM(order_items.quantity * order_items.unit_price). To report completed sales, filter orders.status = 'delivered'. Join orders to order_items on order_id.",
    },
]


def retrieve_context(question: str, top_k: int = 4) -> list[dict[str, str]]:
    """Lightweight lexical retrieval for a zero-vector-store demo."""
    terms = set(re.findall(r"[a-z0-9_]+", question.lower()))
    scored = []
    for item in KNOWLEDGE:
        words = set(re.findall(r"[a-z0-9_]+", item["text"].lower()))
        score = len(terms & words) / max(1, len(terms))
        if item["name"] in terms:
            score += 0.5
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for score, item in scored[:top_k] if score > 0] or KNOWLEDGE[:3]


def build_prompt(question: str, context: list[dict[str, str]]) -> str:
    schema = "\n".join(f"- {item['text']}" for item in context)
    return f"""You generate SQL for a read-only SQLite retail demo.
Return only one SQL SELECT query. No markdown and no explanation.
Use only these tables and columns from the retrieved context:
{schema}

Safety and correctness rules:
- Only SELECT queries are allowed. Never use INSERT, UPDATE, DELETE, DROP, ALTER, PRAGMA, ATTACH, or multiple statements.
- Use explicit JOIN conditions when combining tables.
- Revenue is quantity * unit_price in order_items.
- Do not invent schema elements. If the question cannot be answered from the context, return a simple SELECT that makes the limitation clear only if possible.
Question: {question}
SQL:"""


def validate_sql(sql: str) -> str:
    if not sql or not sql.strip():
        raise ValueError("The model returned an empty SQL query.")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"SQL syntax error: {exc}") from exc
    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Exactly one SQL statement is required.")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise ValueError("Only read-only SELECT queries are permitted.")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.Command, exp.Pragma)
    if any(tree.find(kind) for kind in forbidden):
        raise ValueError("The query contains a forbidden SQL operation.")
    # Restrict referenced tables to this demo database.
    allowed = {"customers", "products", "orders", "order_items"}
    referenced = {table.name.lower() for table in tree.find_all(exp.Table)}
    unknown = referenced - allowed
    if unknown:
        raise ValueError("Unknown table(s): " + ", ".join(sorted(unknown)))
    return tree.sql(dialect="sqlite")


async def generate_sql(question: str, context: list[dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="No model API key configured. Add OPENAI_API_KEY (or GROQ_API_KEY) to the deployment environment.",
        )
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a careful Text-to-SQL generator. Output SQL only."},
            {"role": "user", "content": build_prompt(question, context)},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Model provider error ({exc.response.status_code}). Check model name and API key.") from exc
    except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Could not obtain a valid response from the model provider.") from exc
    content = re.sub(r"^\s*```(?:sql)?\s*|\s*```\s*$", "", content, flags=re.IGNORECASE)
    return content.strip()


def execute_readonly(sql: str) -> tuple[list[str], list[list[Any]]]:
    # Add a hard result cap if the model omitted LIMIT.
    parsed = sqlglot.parse_one(sql, read="sqlite")
    if not parsed.args.get("limit"):
        parsed = parsed.limit(MAX_ROWS)
    safe_sql = parsed.sql(dialect="sqlite")
    conn = get_connection(readonly=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.execute(safe_sql)
        columns = [item[0] for item in (cursor.description or [])]
        rows = [list(row) for row in cursor.fetchmany(MAX_ROWS)]
        return columns, rows
    except sqlite3.Error as exc:
        raise ValueError(f"SQLite rejected the query: {exc}") from exc
    finally:
        conn.close()


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)


@app.on_event("startup")
def startup():
    initialise_database()


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "database": "synthetic retail sample", "model_configured": bool(os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"))}


@app.get("/api/schema")
def schema():
    return {"tables": KNOWLEDGE[:4]}


@app.post("/api/ask")
async def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    context = retrieve_context(question)
    raw_sql = await generate_sql(question, context)
    try:
        sql = validate_sql(raw_sql)
        columns, rows = execute_readonly(sql)
    except ValueError as exc:
        return {
            "question": question,
            "retrieved_context": context,
            "sql": raw_sql,
            "valid": False,
            "error": str(exc),
            "columns": [],
            "rows": [],
        }
    return {
        "question": question,
        "retrieved_context": context,
        "sql": sql,
        "valid": True,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "max_rows": MAX_ROWS,
    }
