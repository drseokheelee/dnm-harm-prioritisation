import re, json, sys
from pathlib import Path
from typing import Optional

HARMFUL_THRESHOLD = 0.60  
SAFE_THRESHOLD    = 0.35   

DRUG_FALLBACK_SCORE = 0.60 

OBVIOUS_HARMFUL_TITLE_RE = re.compile(
    r"\b("
    # Hard opioids (high-lethality only)
    r"heroin|fentanyl|carfentanil|dmt|lysergic|lsd|ketamine|ricin|botulinum|sarin|cyanide|arsenic|tetrodotoxin|aconitine"
    # Cocaine / crack
    r"|cocaine|crack(\s+cocaine)?"
    # Methamphetamine
    r"|methamphetamine|crystal\s*meth|\bice\b"
    # MDMA / ecstasy
    r"|mdma|ecstasy|xtc"
    # GHB (date-rape / high harm)
    r"|ghb|gamma.hydroxy"
    # Nembutal / euthanasia drugs
    r"|nembutal|pentobarbital|sodium\s+pentobarbital"
    r")\b",
    re.IGNORECASE
)
OBVIOUS_SAFE_TITLE_RE = re.compile(
    r"\b("
    # ED / sexual health
    r"viagra|sildenafil|cialis|tadalafil|levitra|vardenafil|staxyn"
    # Weight loss (GLP-1 / common prescriptions)
    r"|ozempic|wegovy|semaglutide|saxenda|liraglutide|tirzepatide"
    r"|phentermine|adipex|orlistat|xenical|qsymia"
    # Antidepressants / SSRIs / SNRIs
    r"|prozac|fluoxetine|zoloft|sertraline|lexapro|escitalopram"
    r"|paxil|paroxetine|celexa|citalopram|luvox|fluvoxamine"
    r"|wellbutrin|bupropion|effexor|venlafaxine|cymbalta|duloxetine"
    r"|mirtazapine|remeron|trazodone"
    # OTC pain / fever
    r"|ibuprofen|paracetamol|acetaminophen|aspirin|naproxen"
    # Antibiotics
    r"|amoxicillin|penicillin|azithromycin|doxycycline|ciprofloxacin"
    r"|metronidazole|trimethoprim"
    # Metabolic / chronic disease
    r"|metformin|insulin|lipitor|atorvastatin|rosuvastatin|simvastatin"
    r"|lisinopril|amlodipine|metoprolol|atenolol|losartan"
    # CBD / hemp (non-psychoactive)
    r"|cbd\b|cannabidiol|hemp\s+oil"
    r")\b",
    re.IGNORECASE
)

DRUG_CATEGORIES: set[str] = {
    # Clearly dangerous
    "heroin", "cocaine", "opiods", "opoids",
    "mdma", "ecstasy", "ghb", "lsd", "dmt",
    "hallucinogens", "dissociatives", "research-chemicals",
    "amphetamine-stimulators",
    # Ambiguous / prescription
    "drugs", "drugs-2", "drugs-for-sale",
    "stimulants", "stimulators",
    "cannabis", "hashish",
    "benzodiazepines", "benzos",
    "painkillers", "sedative",
    "anti-anxiety-anti-depressants", "anxiety",
    "muscle-relaxers", "oral-steroids",
    "nootropics-smart-drugs",
    # Low-harm categories
    "weightloss", "sexual-assistance", "creams-cbd-oils",
}

CATEGORY_BASE_SCORES: dict[str, Optional[float]] = {

    # Weapons
    "firearms":                      1.00,
    "firearms-for-sale":             1.00,
    "guns":                          1.00,
    "handguns":                      1.00,
    "pistols":                       1.00,
    "revolvers":                     1.00,
    "rifles":                        1.00,
    "rifle":                         1.00,
    "shotguns":                      1.00,
    "silencers":                     1.00,
    "ammo":                          1.00,
    "ammunition":                    1.00,
    "explosives":                    1.00,
    "rocket-launcher":               1.00,
    "drones-anti-drone-systems-uav": 1.00,
    "colt-44-mag":                   1.00,
    "knives-swords":                 0.80,
    "jammers":                       0.85,

    # Poison
    "poison":                        1.00,

    # Financial crime
    "counterfeit":                   0.85,
    "counterfeit-money":             0.90,
    "counterfeit-usd-bills":         0.90,
    "counterfeit-euro-bills":        0.90,
    "counterfeit-gpb":               0.90,
    "money-counterfeits":            0.90,
    "cloned-credit-cards":           0.88,
    "bin":                           0.85,
    "cvvpackages":                   0.85,
    "carding":                       0.88,
    "cards":                         0.80,
    "money-transfer":                0.75,
    "money-transfers":               0.75,
    "transfers":                     0.70,
    "money":                         0.60,
    "crypto-wallets":                0.70,
    "wallets":                       0.65,

    # Fraud / identity
    "passports":                     0.85,
    "id-cards":                      0.80,
    "documents":                     0.75,
    "drivers-license":               0.75,
    "certificates":                  0.65,
    "hacking":                       0.85,
    "hacking-services":              0.85,

    # Human exploitation
    "human-organs":                  1.00,

    # Gift cards
    "gift-cards":                    0.10,
    "giftcards":                     0.10,
    "virtual":                       0.10,

    # Electronics / consumer
    "electronics":                   0.05,
    "smartphones":                   0.05,
    "laptops":                       0.05,
    "apple":                         0.05,
    "watch-replica":                 0.15,

    # Catch-all
    "other":                         None,

    # ── Drug categories — specific scores per substance ─────────────────
    # Clearly dangerous / high abuse potential
    "heroin":                        1.00,
    "cocaine":                       0.95,
    "opiods":                        0.90,
    "opoids":                        0.90,
    "mdma":                          0.75,
    "ecstasy":                       0.75,
    "ghb":                           0.85,
    "lsd":                           0.70,
    "dmt":                           0.70,
    "hallucinogens":                 0.70,
    "dissociatives":                 0.75,
    "research-chemicals":            0.75,
    "amphetamine-stimulators":       0.80,
    # Prescription controlled — moderate risk
    "benzodiazepines":               0.65,
    "benzos":                        0.65,
    "painkillers":                   0.65,
    "sedative":                      0.60,
    "stimulants":                    0.65,
    "stimulators":                   0.65,
    "oral-steroids":                 0.60,   # illegal without Rx; ~0.60 is right
    # Generic drug buckets — unknown contents, assume moderate-high
    "drugs":                         0.70,
    "drugs-2":                       0.70,
    "drugs-for-sale":                0.70,
    # Cannabis / hashish — jurisdiction-dependent
    "hashish":                       0.55,
    "cannabis":                      0.50,
    # Lower-harm categories
    "anti-anxiety-anti-depressants": 0.40,
    "anxiety":                       0.40,
    "muscle-relaxers":               0.35,
    "nootropics-smart-drugs":        0.25,
    # Essentially benign in isolation
    "sexual-assistance":             0.10,
    "weightloss":                    0.10,
    "creams-cbd-oils":               0.10,
}

CATEGORY_BASE_SCORES.update({
    "buy clone cards":   0.88,
    "buy counterfeit":   0.90,
    "buy documents":     0.78,
    "firearms for sale": 1.00,
    "money transfer":    0.75,
    "poison for sale":   1.00,
    "red mercury":       0.95,
    "acounts":           0.70,
})

TITLE_KEYWORD_RULES: list[tuple[str, float]] = [

    # Lethal / CBRN
    (r"\bfentanyl\b",                              +0.35),
    (r"\bcarfentanil\b",                           +0.40),
    (r"\bheroin\b",                                +0.35),
    (r"\bricin\b",                                 +0.40),
    (r"\bbotulinum\b",                             +0.40),
    (r"\bsarin\b",                                 +0.40),
    (r"\bcyanide\b",                               +0.40),
    (r"\barsenic\b",                               +0.30),
    (r"\bpoison\b",                                +0.25),
    (r"\btoxin\b",                                 +0.20),
    (r"\btetrodotoxin\b",                          +0.40),
    (r"\baconitine\b",                             +0.30),

    # Weapons
    (r"\bexplosive\b",                             +0.30),
    (r"\bc-4\b",                                   +0.35),
    (r"\brpg\b",                                   +0.35),
    (r"\bgrenade\b",                               +0.30),
    (r"\bminigun\b",                               +0.35),
    (r"\bsniper\b",                                +0.20),
    (r"\bkamikaze\b",                              +0.30),
    (r"\bdrone\b",                                 +0.15),
    (r"\b3d.printed.gun\b",                        +0.35),
    (r"\bsilencer\b|\bsuppressor\b",               +0.15),
    (r"\bglock\b|\bberetta\b|\bcolt\b|\bruger\b",  +0.15),
    (r"\bsig\b|\bsmith.wesson\b|\btaurus\b",       +0.15),
    (r"\bak.?47\b|\bar.?15\b|\bm16\b",             +0.20),
    (r"\bpanzerfaust\b|\bstinger\b|\bat-4\b",      +0.35),

    # Financial crime
    (r"\bcounterfeit\b",                           +0.10),
    (r"\bundetectable\b",                          +0.15),
    (r"\bfake\b.*(note|bill|dollar|euro|pound)",   +0.25),
    (r"\bcloned?\b.*(card|credit)",                +0.20),
    (r"\bskimmer\b",                               +0.25),
    (r"\bcvv\b",                                   +0.15),
    (r"\bfullz\b",                                 +0.20),
    (r"\bpaypal\s+transfer\b",                     +0.12),
    (r"\bwestern\s+union\b",                       +0.10),
    (r"\bmoney\s+transfer\b",                      +0.10),
    (r"\bbank\s+(transfer|statement)\b",           +0.10),

    # Identity fraud
    (r"\bpassport\b",                              +0.15),
    (r"\bdriving.licen[sc]e\b",                    +0.12),
    (r"\bid\s+card\b",                             +0.12),
    (r"\bsocial\s+security\b",                     +0.10),

    # Hacking / malware
    (r"\brat\b.*(tool|v\d|\bremote)",              +0.15),
    (r"\bmalware\b|\bexploit\b|\b0day\b",          +0.15),
    (r"\bhack(ing|er)?\b",                         +0.10),

    # Human exploitation
    (r"\borgans?\b|\bkidney\b|\bliver\b",          +0.25),

    # Drug name boosts
    (r"\bcocaine\b|\bcrack\b",                     +0.30),
    (r"\bmeth(amphetamine)?\b|\bcrystal\s*meth\b", +0.35),
    (r"\bmdma\b|\becstasy\b|\bxtc\b",              +0.25),
    (r"\blsd\b|\blysergic\b|\bdmt\b",              +0.20),
    (r"\bketamine\b",                              +0.20),
    (r"\bghb\b|\bgamma.hydroxy\b",                 +0.30),
    (r"\boxycodon(e)?\b|\bpercocet\b",             +0.25),
    (r"\bhydrocodon(e)?\b|\bvicodin\b",            +0.25),
    (r"\bxanax\b|\balprazolam\b",                  +0.15),
    (r"\bvalium\b|\bdiazepam\b",                   +0.15),
    (r"\bamphetamine\b|\bspeed\b",                 +0.25),
    (r"\bheroin\b",                                +0.35),
    (r"\bfentanyl\b",                              +0.35),
    (r"\braw\s+powder\b|\bprecursor\b",            +0.10),
    # Anabolic steroids — illegal, ~0.60 total with base
    (r"\banavar\b|\bwinstrol\b|\bdianabol\b",      +0.00),  # base 0.60 is correct
    (r"\btrenbolone\b|\banadrol\b|\bturinabol\b",  +0.05),
    (r"\btestosterone\b",                          +0.00),  # base 0.60 is correct
    (r"\bsteroids?\b",                             +0.00),

    # Downward modifiers
    (r"\biphone\b|\bipad\b|\bmacbook\b",           -0.50),
    (r"\bapple\s+watch\b",                         -0.50),
    (r"\bdell\b|\blaptop\b",                       -0.50),
    (r"\bnetflix\b|\bspotify\b",                   -0.50),
    (r"\bcbd\b|\bcannabidiol\b",                   -0.40),
    (r"\bvitamin\b|\bsupplement\b",                -0.50),
    # ED meds
    (r"\bviagraa?\b|\bsildenafil\b|\bcialis\b|\btadalafil\b|\blevitra\b", -0.40),
    # Weight loss
    (r"\bozempic\b|\bwegovy\b|\bsemaglutide\b",    -0.45),
    (r"\bphentermine\b|\badipex\b|\borlistat\b",   -0.35),
    # Antidepressants
    (r"\bprozac\b|\bfluoxetine\b|\bzoloft\b|\bsertraline\b", -0.40),
    (r"\blexapro\b|\beschitalopram\b|\bcelexa\b|\bcitalopram\b", -0.40),
    (r"\bwellbutrin\b|\bbupropion\b|\beffexor\b|\bvenlafaxine\b", -0.35),
    # Modafinil
    (r"\bmodafinil\b|\barmodafinil\b",             -0.25),
    # OTC
    (r"\bibuprofen\b|\bparacetamol\b|\bacetaminophen\b|\baspirin\b", -0.45),
    # Antibiotics
    (r"\bamoxicillin\b|\bazithromycin\b|\bdoxycycline\b", -0.45),
    # Watch replica / gift cards
    (r"\bwatch\b.*(replica|factory)",              -0.50),
    (r"\bgift",                                    -0.50),
    (r"\bpaxil",                                    -0.50),
]

TITLE_KEYWORD_RULES = [r for r in TITLE_KEYWORD_RULES if isinstance(r, tuple) and len(r) == 2]

_PRICE_SENSITIVE_CATS = {
    "counterfeit", "counterfeit-money", "counterfeit-usd-bills",
    "counterfeit-euro-bills", "counterfeit-gpb", "money-counterfeits",
    "money-transfer", "money-transfers", "transfers",
    "cloned-credit-cards", "bin", "cvvpackages", "carding", "cards",
}

def price_modifier(category: str, price: Optional[float]) -> float:
    if price is None or category not in _PRICE_SENSITIVE_CATS:
        return 0.0
    return min(0.10, price / 5000 * 0.10)

def score_product(record: dict) -> dict:
    title    = (record.get("title")    or "").strip()
    category = (record.get("category") or "").strip().lower()
    price    = record.get("price")
    reasons: list[str] = []

    # ── Step 1: Base score ────────────────────────────────────────────────
    base = CATEGORY_BASE_SCORES.get(category)
    if base is None:
        if category in DRUG_CATEGORIES:
            # Drug category with no specific score → use fallback
            base = DRUG_FALLBACK_SCORE
            reasons.append(f"drug category '{category}' (no specific score) → fallback {base:.2f}")
        else:
            # Non-drug unknown category → neutral
            base = 0.40
            reasons.append(f"unknown category '{category}' → neutral base 0.40")
    else:
        reasons.append(f"category '{category}' → base {base:.2f}")
    score = base

    # ── Step 2: Title keyword modifiers ───────────────────────────────────
    title_lower = title.lower()
    for pattern, delta in TITLE_KEYWORD_RULES:
        if re.search(pattern, title_lower, re.IGNORECASE):
            score += delta
            reasons.append(f"title match '{pattern}' → {delta:+.2f}")

    # ── Step 3: Price modifier ────────────────────────────────────────────
    pdelta = price_modifier(category, price)
    if pdelta:
        score += pdelta
        reasons.append(f"price modifier (${price}) → {pdelta:+.2f}")

    # ── Step 4: Clamp ─────────────────────────────────────────────────────
    score = max(0.0, min(1.0, score))

    # ── Step 5: Route ─────────────────────────────────────────────────────
    result = record.copy()
    result["harm_score"]   = round(score, 4)
    result["score_reason"] = "; ".join(reasons)

    if category in DRUG_CATEGORIES:
        is_obvious_harmful = bool(OBVIOUS_HARMFUL_TITLE_RE.search(title))
        is_obvious_safe    = bool(OBVIOUS_SAFE_TITLE_RE.search(title))

        if is_obvious_harmful:
            result["label"] = "harmful"
        elif is_obvious_safe:
            result["label"] = "not_harmful"
        else:
            result["label"]                = "manual_review"
            result["manual_review_reason"] = "drug_category"

    elif score > HARMFUL_THRESHOLD:
        result["label"] = "harmful"
    elif score < SAFE_THRESHOLD:
        result["label"] = "not_harmful"
    else:
        result["label"]                = "manual_review"
        result["manual_review_reason"] = "uncertainty_band"

    return result

# ─────────────────────────────────────────────────────────────────────────────
# BATCH
# ─────────────────────────────────────────────────────────────────────────────
def label_dataset(records):
    harmful, safe, review = [], [], []
    for rec in records:
        scored = score_product(rec)
        if   scored["label"] == "harmful":       harmful.append(scored)
        elif scored["label"] == "not_harmful":   safe.append(scored)
        else:                                    review.append(scored)
    return harmful, safe, review

def save_outputs(harmful, safe, review, out_dir="."):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    for fname, data in [
        ("labeled_harmful.json",       harmful),
        ("labeled_safe.json",          safe),
        ("labeled_manual_review.json", review),
    ]:
        path = p / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {len(data):>5} records → {path}")

def print_summary(harmful, safe, review):
    total = len(harmful) + len(safe) + len(review)
    dh = [r for r in harmful if r.get("category") in DRUG_CATEGORIES]
    ds = [r for r in safe    if r.get("category") in DRUG_CATEGORIES]
    dr = [r for r in review  if r.get("manual_review_reason") == "drug_category"]
    br = [r for r in review  if r.get("manual_review_reason") == "uncertainty_band"]
    print(f"\n{'='*64}")
    print(f"  Harm Scoring Summary   (thresholds: harmful>{HARMFUL_THRESHOLD}, safe<{SAFE_THRESHOLD})")
    print(f"{'='*64}")
    print(f"  Total                  : {total}")
    print(f"  ├─ Harmful             : {len(harmful):>5}  ({100*len(harmful)/total:.1f}%)")
    print(f"  │    ├─ drugs          : {len(dh):>5}")
    print(f"  │    └─ non-drug       : {len(harmful)-len(dh):>5}")
    print(f"  ├─ Safe                : {len(safe):>5}  ({100*len(safe)/total:.1f}%)")
    print(f"  │    └─ drugs          : {len(ds):>5}")
    print(f"  └─ Manual review       : {len(review):>5}  ({100*len(review)/total:.1f}%)")
    print(f"       ├─ drug_category  : {len(dr):>5}")
    print(f"       └─ uncertainty    : {len(br):>5}")
    print(f"{'='*64}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python harm_scorer.py products.json [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    out_dir    = sys.argv[2] if len(sys.argv) > 2 else "."
    print(f"\nLoading {input_path}…")
    with open(input_path, encoding="utf-8", errors="replace") as f:
        records = json.load(f)
    if not isinstance(records, list):
        records = [records]
    print(f"Loaded {len(records)} records.")
    harmful, safe, review = label_dataset(records)
    print_summary(harmful, safe, review)
    save_outputs(harmful, safe, review, out_dir)
    print("\nDone.\n")