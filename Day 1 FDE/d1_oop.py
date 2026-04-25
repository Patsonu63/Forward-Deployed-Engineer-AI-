# ══════════════════════════════════════════════════════
# PYTHON OOP FUNDAMENTALS — Everything an FDE needs
# ══════════════════════════════════════════════════════

# ── 1. Classes & __init__ ──────────────────────────────
class Customer:
    # Class variable — shared across ALL instances
    company = "Acme Corp"

    def __init__(self, name: str, tier: str):
        # Instance variables — unique to each object
        self.name = name
        self.tier = tier          # "free" | "pro" | "enterprise"
        self.tickets = []         # each customer has their own list

    def add_ticket(self, issue: str):
        self.tickets.append(issue)
        print(f"Ticket added for {self.name}: {issue}")

    def summary(self) -> str:
        return f"{self.name} ({self.tier}) — {len(self.tickets)} tickets"

    # __repr__ — what prints when you do print(obj)
    def __repr__(self):
        return f"Customer(name={self.name!r}, tier={self.tier!r})"


c1 = Customer("Rahul", "enterprise")
c2 = Customer("Priya", "pro")

c1.add_ticket("SSO login fails")
c1.add_ticket("Data export broken")
c2.add_ticket("API rate limit hit")

print(c1.summary())    # Rahul (enterprise) — 2 tickets
print(c2)              # Customer(name='Priya', tier='pro')
print(Customer.company)# Acme Corp


# ── 2. Inheritance ─────────────────────────────────────
class EnterpriseCustomer(Customer):
    def __init__(self, name: str, sla_hours: int):
        super().__init__(name, tier="enterprise")   # call parent __init__
        self.sla_hours = sla_hours

    # Override parent method
    def summary(self) -> str:
        base = super().summary()
        return f"{base} | SLA: {self.sla_hours}h response"


ec = EnterpriseCustomer("Infosys", sla_hours=4)
ec.add_ticket("Integration failure")
print(ec.summary())   # Infosys (enterprise) — 1 tickets | SLA: 4h response


# ── 3. Error handling — critical for FDE work ──────────
def parse_config(data: dict) -> str:
    try:
        api_key = data["api_key"]
        if not api_key:
            raise ValueError("API key cannot be empty")
        return api_key
    except KeyError:
        print("Error: 'api_key' missing from config")
        return ""
    except ValueError as e:
        print(f"Error: {e}")
        return ""
    finally:
        print("Config parse attempt complete")  # always runs

parse_config({"api_key": "sk-abc123"})   # works
parse_config({})                          # missing key
parse_config({"api_key": ""})             # empty key


# ── 4. List comprehensions — you'll use these constantly ──
tickets = ["SSO broken", "API down", "Login slow", "SSO timeout", "Export fails"]

# Filter only SSO-related tickets
sso_issues = [t for t in tickets if "SSO" in t]
print(sso_issues)   # ['SSO broken', 'SSO timeout']

# Make all uppercase
upper = [t.upper() for t in tickets]
print(upper)

# Dict comprehension — map ticket to its length
lengths = {t: len(t) for t in tickets}
print(lengths)


# ── 5. F-strings & string formatting ──────────────────
name = "Claude"
version = 3.7
tokens_used = 1523876

# F-string (use this always)
print(f"Model: {name} v{version}")
print(f"Tokens used: {tokens_used:,}")   # comma-formatted number


# ── 6. Functions with type hints ──────────────────────
def classify_ticket(ticket: str, keywords: list[str]) -> dict:
    """
    Classify a support ticket based on keywords.
    Returns a dict with category and priority.
    """
    ticket_lower = ticket.lower()
    for kw in keywords:
        if kw in ticket_lower:
            return {"category": kw, "priority": "high", "ticket": ticket}
    return {"category": "general", "priority": "normal", "ticket": ticket}


result = classify_ticket("SSO login is completely broken", ["sso", "auth", "login"])
print(result)
# {'category': 'sso', 'priority': 'high', 'ticket': 'SSO login is completely broken'}
