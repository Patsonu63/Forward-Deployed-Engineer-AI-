import anthropic, os, json, math, sqlite3, requests, time
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ══════════════════════════════════════════════════════
# MULTI-TOOL AGENT: Sales Intelligence System
# Tools: search, calculator, database query, email draft
# This is the kind of agent FDEs build for enterprise clients
# ══════════════════════════════════════════════════════


# ════════════════════════════════
# SETUP: Create a mock SQLite database
# (In real FDE work, this would be the client's Salesforce/CRM)
# ════════════════════════════════

def setup_demo_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT, tier TEXT, mrr REAL,
            industry TEXT, country TEXT,
            last_contact_days INTEGER
        );
        CREATE TABLE tickets (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER, status TEXT,
            priority TEXT, category TEXT,
            created_days_ago INTEGER
        );
        INSERT INTO customers VALUES
            (1,'Infosys','enterprise',12000,'Technology','India',45),
            (2,'Tata Motors','enterprise',8500,'Automotive','India',12),
            (3,'Zomato','pro',2200,'Food Tech','India',3),
            (4,'Paytm','pro',1800,'FinTech','India',67),
            (5,'Byju''s','standard',400,'EdTech','India',120);
        INSERT INTO tickets VALUES
            (1,1,'open','critical','api',2),
            (2,1,'open','high','billing',5),
            (3,2,'closed','medium','question',1),
            (4,3,'open','low','feature',10),
            (5,4,'open','critical','auth',1),
            (6,5,'open','medium','bug',15);
    """)
    conn.commit()
    return conn

DB_CONN = setup_demo_db()


# ════════════════════════════════
# TOOL DEFINITIONS
# ════════════════════════════════

AGENT_TOOLS = [
    {
        "name": "query_database",
        "description": "Query the customer database using SQL. Use for: customer revenue, ticket counts, tier info, last contact dates. The database has tables: customers (id, name, tier, mrr, industry, country, last_contact_days) and tickets (id, customer_id, status, priority, category, created_days_ago).",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A SELECT SQL query. Only SELECT allowed — no INSERT/UPDATE/DELETE."
                }
            },
            "required": ["sql"]
        }
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations — revenue totals, percentages, growth rates, averages. Always use this for numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Python math expression"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "draft_email",
        "description": "Draft a professional email to a customer. Use when the analysis suggests customer outreach is needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Customer name"},
                "subject": {"type": "string", "description": "Email subject"},
                "purpose": {"type": "string", "description": "What the email should accomplish"},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key points to include in the email"
                }
            },
            "required": ["to", "subject", "purpose", "key_points"]
        }
    },
    {
        "name": "get_company_info",
        "description": "Get basic public information about a company using web lookup. Use to enrich customer context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Name of the company to look up"}
            },
            "required": ["company_name"]
        }
    }
]


# ════════════════════════════════
# TOOL IMPLEMENTATIONS
# ════════════════════════════════

def query_database(sql: str) -> dict:
    """Execute a SELECT query on the CRM database."""
    sql_clean = sql.strip().upper()
    if not sql_clean.startswith("SELECT"):
        return {"error": "Only SELECT queries allowed", "status": "error"}
    try:
        cur = DB_CONN.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return {"columns": cols, "rows": rows, "count": len(rows), "status": "ok"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


def calculator(expression: str) -> dict:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"result": round(result, 4), "expression": expression}
    except Exception as e:
        return {"error": str(e)}


def draft_email(to: str, subject: str, purpose: str, key_points: list) -> dict:
    """Use Claude to draft a professional email."""
    prompt = f"""Draft a professional, concise email (under 150 words) for a Customer Success Manager.

To: {to}
Subject: {subject}
Purpose: {purpose}
Key points to cover:
{chr(10).join(f'- {p}' for p in key_points)}

Write only the email body (no subject line, no headers)."""

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "to": to,
        "subject": subject,
        "body": resp.content[0].text,
        "status": "ok"
    }


def get_company_info(company_name: str) -> dict:
    """Fetch basic company info from DuckDuckGo."""
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": company_name, "format": "json", "no_html": "1"},
            timeout=5
        ).json()
        abstract = resp.get("Abstract", "")
        return {
            "company": company_name,
            "summary": abstract[:300] if abstract else "No public info found.",
            "status": "ok"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


MULTI_TOOL_FUNCTIONS = {
    "query_database":  query_database,
    "calculator":      calculator,
    "draft_email":     draft_email,
    "get_company_info": get_company_info,
}

def execute_tool(name: str, inp: dict) -> str:
    fn = MULTI_TOOL_FUNCTIONS.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return json.dumps(fn(**inp), indent=2)


# ════════════════════════════════
# THE AGENT
# ════════════════════════════════

SALES_AGENT_SYSTEM = """You are a senior Customer Success AI Agent for a B2B SaaS company.

Your capabilities:
- Query the live CRM database for customer data
- Calculate metrics (MRR, churn risk scores, growth rates)
- Draft outreach emails
- Look up company information

Your job:
1. Gather relevant data using tools
2. Perform analysis with the calculator
3. Identify at-risk or high-value customers
4. Draft actionable recommendations
5. Prepare outreach emails when needed

Always show your reasoning. Be specific with numbers."""


def run_sales_agent(task: str, verbose: bool = True) -> str:
    messages = [{"role": "user", "content": task}]
    steps = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print(f"{'='*60}")

    while steps < 12:
        steps += 1
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=SALES_AGENT_SYSTEM,
            tools=AGENT_TOOLS,
            messages=messages
        )

        if resp.stop_reason == "end_turn":
            answer = next((b.text for b in resp.content if b.type == "text"), "")
            if verbose:
                print(f"\n{'─'*60}")
                print(f"FINAL REPORT:\n{answer}")
            return answer

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []

            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if verbose:
                    print(f"\n[Step {steps}] {block.name}({str(block.input)[:80]})")

                result = execute_tool(block.name, block.input)

                if verbose:
                    print(f"  → {result[:150]}...")

                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

            messages.append({"role": "user", "content": results})

    return "Max steps reached."


# ════════════════════════════════
# RUN THE AGENT
# ════════════════════════════════

if __name__ == "__main__":
    # Task 1: At-risk customer analysis
    run_sales_agent(
        "Analyze our customer base. Find customers who: (1) have open critical tickets AND (2) haven't been contacted in 30+ days. Calculate their combined MRR at risk. Draft a priority outreach email for the highest-value at-risk customer."
    )

    # Task 2: Revenue analysis
    run_sales_agent(
        "What is our total MRR? What percentage comes from enterprise vs pro vs standard? Which customer has the most open tickets relative to their MRR? Show all calculations."
    )
