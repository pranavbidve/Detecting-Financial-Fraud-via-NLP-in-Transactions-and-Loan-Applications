import argparse
import random
from typing import Optional

import numpy as np
import pandas as pd


def normalize_value(value: object) -> Optional[str]:
    """Return a clean string or None for missing-like values."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"?", "na", "n/a", "none", "nan"}:
        return None
    return text


def lower_or_unknown(text: Optional[str]) -> str:
    return (text or "unknown").lower()


def format_money(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    try:
        # Keep cents if present; fallback to raw text
        amount = float(value)
        return f"${amount:,.2f}"
    except Exception:
        return str(value)


def format_hour(hour_text: Optional[str]) -> Optional[str]:
    if hour_text is None:
        return None
    try:
        hour_24 = int(float(hour_text))
    except Exception:
        return None
    if hour_24 < 0 or hour_24 > 23:
        return None
    period = "AM" if hour_24 < 12 else "PM"
    hour_12 = hour_24 % 12
    if hour_12 == 0:
        hour_12 = 12
    return f"around {hour_12} {period}"


def map_incident_type(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    text = raw.strip().lower()
    mapping = {
        "single vehicle collision": "single-vehicle collision",
        "multi-vehicle collision": "multi-vehicle collision",
        "parked car": "parked car incident",
        "vehicle theft": "vehicle theft",
    }
    return mapping.get(text, raw.lower())


def map_collision_phrase(raw: Optional[str]) -> Optional[str]:
    if raw is None or raw.strip() == "?":
        return None
    text = raw.strip().lower()
    mapping = {
        "rear collision": "a rear-end collision",
        "front collision": "a head-on collision",
        "side collision": "a side-impact collision",
    }
    return mapping.get(text, f"a {text}")


def map_severity(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    text = raw.strip().lower()
    mapping = {
        "minor damage": "minor damage",
        "major damage": "major damage",
        "total loss": "a total loss",
        "trivial damage": "trivial damage",
    }
    return mapping.get(text, raw.lower())


def yn_to_bool(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    text = raw.strip().lower()
    if text in {"y", "yes", "true", "1"}:
        return True
    if text in {"n", "no", "false", "0"}:
        return False
    return None


def build_sentence(row: pd.Series) -> str:
    # Extract and normalize values we care about
    months_as_customer = normalize_value(row.get("months_as_customer"))
    age = normalize_value(row.get("age"))
    policy_csl = normalize_value(row.get("policy_csl"))
    policy_deductable = normalize_value(row.get("policy_deductable"))
    policy_annual_premium = normalize_value(row.get("policy_annual_premium"))
    umbrella_limit = normalize_value(row.get("umbrella_limit"))
    insured_zip = normalize_value(row.get("insured_zip"))
    insured_sex = normalize_value(row.get("insured_sex"))
    insured_education_level = normalize_value(row.get("insured_education_level"))
    insured_occupation = normalize_value(row.get("insured_occupation"))
    insured_hobbies = normalize_value(row.get("insured_hobbies"))
    insured_relationship = normalize_value(row.get("insured_relationship"))
    capital_gains = normalize_value(row.get("capital-gains"))
    capital_loss = normalize_value(row.get("capital-loss"))
    incident_date = normalize_value(row.get("incident_date"))
    incident_type = normalize_value(row.get("incident_type"))
    collision_type = normalize_value(row.get("collision_type"))
    incident_severity = normalize_value(row.get("incident_severity"))
    authorities_contacted = normalize_value(row.get("authorities_contacted"))
    incident_state = normalize_value(row.get("incident_state"))
    incident_city = normalize_value(row.get("incident_city"))
    incident_location = normalize_value(row.get("incident_location"))
    incident_hour = normalize_value(row.get("incident_hour_of_the_day"))
    number_of_vehicles = normalize_value(row.get("number_of_vehicles_involved"))
    property_damage = normalize_value(row.get("property_damage"))
    bodily_injuries = normalize_value(row.get("bodily_injuries"))
    witnesses = normalize_value(row.get("witnesses"))
    police_report_available = normalize_value(row.get("police_report_available"))
    total_claim_amount = normalize_value(row.get("total_claim_amount"))
    injury_claim = normalize_value(row.get("injury_claim"))
    property_claim = normalize_value(row.get("property_claim"))
    vehicle_claim = normalize_value(row.get("vehicle_claim"))
    auto_make = normalize_value(row.get("auto_make"))
    auto_model = normalize_value(row.get("auto_model"))
    auto_year = normalize_value(row.get("auto_year"))

    # Build modular phrases used by randomized templates
    incident_type_nice = map_incident_type(incident_type)
    collision_phrase = map_collision_phrase(collision_type)
    severity_nice = map_severity(incident_severity)

    where_bits = []
    if incident_city or incident_state:
        city_state = ", ".join([x for x in [incident_city, incident_state] if x])
        where_bits.append(f"in {city_state}")
    if incident_location:
        where_bits.append(f"at {incident_location}")
    date_phrase = f"On {incident_date}" if incident_date else None
    where_phrase = " ".join(where_bits) if where_bits else None

    happened_phrase_parts = []
    if incident_type_nice:
        happened_phrase_parts.append(incident_type_nice)
    if collision_phrase:
        happened_phrase_parts.append(f"involving {collision_phrase}")
    if severity_nice:
        happened_phrase_parts.append(f"that resulted in {severity_nice}")
    happened_phrase = " ".join(happened_phrase_parts) if happened_phrase_parts else None

    # Profile phrases
    profile_bits = []
    if age:
        age_phrase = f"{age}-year-old"
        if insured_sex:
            age_phrase += f" {insured_sex.lower()}"
        profile_bits.append(age_phrase)
    elif insured_sex:
        profile_bits.append(insured_sex.lower())
    if insured_relationship:
        profile_bits.append(insured_relationship.lower())
    if insured_occupation:
        profile_bits.append(f"working as {insured_occupation}")
    if insured_education_level:
        profile_bits.append(f"with {insured_education_level} education")
    extra_profile_bits = []
    if months_as_customer:
        extra_profile_bits.append(f"a customer for {months_as_customer} months")
    if insured_zip:
        extra_profile_bits.append(f"residing in ZIP {insured_zip}")
    if insured_hobbies:
        extra_profile_bits.append(f"hobbies include {insured_hobbies}")
    profile_clause = None
    if profile_bits:
        profile_clause = "The insured is " + ", ".join(profile_bits)
        if extra_profile_bits:
            # Randomly join extras with ';' or ', and'
            joiner = "; " if random.random() < 0.5 else ", "
            profile_clause += ". " + joiner.join(extra_profile_bits)

    # Context clauses
    time_phrase = format_hour(incident_hour)
    vehicles_phrase = None
    if number_of_vehicles:
        try:
            num_v = int(float(number_of_vehicles))
            vehicles_phrase = "one vehicle was involved" if num_v == 1 else f"{num_v} vehicles were involved"
        except Exception:
            vehicles_phrase = f"{number_of_vehicles} vehicles were involved"

    witnesses_phrase = None
    if witnesses is not None:
        try:
            w = int(float(witnesses))
            witnesses_phrase = "no witnesses were present" if w == 0 else f"there were {w} witness(es)"
        except Exception:
            witnesses_phrase = f"witnesses: {witnesses}"

    injuries_phrase = None
    if bodily_injuries is not None:
        try:
            b = int(float(bodily_injuries))
            injuries_phrase = "no injuries were reported" if b == 0 else f"{b} injury(ies) were reported"
        except Exception:
            injuries_phrase = f"bodily injuries {bodily_injuries}"

    police_bool = yn_to_bool(police_report_available)
    police_phrase = "a police report was available" if police_bool is True else ("no police report was available" if police_bool is False else None)

    authorities_phrase = None
    if authorities_contacted:
        ac = authorities_contacted.strip().lower()
        if ac == "police":
            authorities_phrase = "the police were contacted"
        elif ac == "fire":
            authorities_phrase = "the fire department was contacted"
        elif ac == "ambulance":
            authorities_phrase = "an ambulance was called"
        elif ac in {"other", "?"}:
            authorities_phrase = "authorities were contacted"
        else:
            authorities_phrase = f"{authorities_contacted} were contacted"

    property_damage_phrase = None
    pd_bool = yn_to_bool(property_damage)
    if pd_bool is True:
        property_damage_phrase = "property damage was reported"
    elif pd_bool is False:
        property_damage_phrase = "no property damage was reported"

    context_clauses_all = [p for p in [time_phrase, vehicles_phrase, witnesses_phrase, injuries_phrase, police_phrase, authorities_phrase, property_damage_phrase] if p]
    random.shuffle(context_clauses_all)
    # Randomly drop one clause to vary length
    if len(context_clauses_all) > 4 and random.random() < 0.6:
        context_clauses_all = context_clauses_all[:-1]
    context_sentence = None
    if context_clauses_all:
        context_sentence = (context_clauses_all[0].capitalize() + ", " + ", ".join(context_clauses_all[1:]) + ".") if len(context_clauses_all) > 1 else (context_clauses_all[0].capitalize() + ".")

    # Vehicle / Policy / Claims
    vehicle_sentence = None
    if auto_year or auto_make or auto_model:
        vehicle_sentence = "The vehicle was a " + " ".join([x for x in [auto_year, auto_make, auto_model] if x]) + "."

    policy_phrase_parts = []
    if policy_csl:
        policy_phrase_parts.append(f"coverage {policy_csl}")
    if policy_deductable:
        policy_phrase_parts.append(f"a {format_money(policy_deductable)} deductible")
    if policy_annual_premium:
        policy_phrase_parts.append(f"an annual premium of {format_money(policy_annual_premium)}")
    if umbrella_limit:
        policy_phrase_parts.append(f"an umbrella limit of {format_money(umbrella_limit)}")
    policy_sentence = None
    if policy_phrase_parts:
        opener = random.choice(["The policy provides", "Policy details include", "Coverage terms include"])
        policy_sentence = opener + " " + ", ".join(policy_phrase_parts) + "."

    claims_phrase_parts = []
    if total_claim_amount:
        claims_phrase_parts.append(f"totaling {format_money(total_claim_amount)}")
    detail_parts = []
    if injury_claim:
        detail_parts.append(f"injury {format_money(injury_claim)}")
    if property_claim:
        detail_parts.append(f"property {format_money(property_claim)}")
    if vehicle_claim:
        detail_parts.append(f"vehicle {format_money(vehicle_claim)}")
    if detail_parts:
        claims_phrase_parts.append("including " + ", ".join(detail_parts))
    claims_sentence = None
    if claims_phrase_parts:
        starter = random.choice(["Claims were filed", "The claim was recorded", "Reported claims were"])
        claims_sentence = starter + " " + " and ".join(claims_phrase_parts) + "."

    # Build candidate sentences and randomize order ensuring varied starts
    candidates = []
    # Two lead templates: collision-first or date-first
    if happened_phrase:
        lead_collision = None
        if where_phrase and random.random() < 0.7:
            lead_collision = f"A {happened_phrase} {where_phrase}."
        else:
            lead_collision = f"A {happened_phrase}."
        candidates.append(("lead_collision", lead_collision))
    if date_phrase or where_phrase:
        ww_bits = [b for b in [date_phrase, where_phrase] if b]
        lead_when = ", ".join(ww_bits) + (", " if ww_bits else "")
        if happened_phrase:
            lead_when += happened_phrase
        else:
            lead_when += "an incident occurred"
        candidates.append(("lead_when", lead_when + "."))

    # Support sentences
    if profile_clause:
        candidates.append(("profile", profile_clause.strip().rstrip(".") + "."))
    if context_sentence:
        candidates.append(("context", context_sentence))
    if vehicle_sentence:
        candidates.append(("vehicle", vehicle_sentence))
    if policy_sentence:
        candidates.append(("policy", policy_sentence))
    if claims_sentence:
        candidates.append(("claims", claims_sentence))

    # Random ordering: ensure we pick 3-4 sentences and avoid always starting with date
    random.shuffle(candidates)
    # Prefer a collision-first start sometimes
    starts = {k for k, _ in candidates[:2]}
    if "lead_when" in starts and random.random() < 0.7:
        # try to swap to a collision start if available
        idx_collision = next((i for i, (k, _) in enumerate(candidates) if k == "lead_collision"), None)
        if idx_collision is not None:
            candidates.insert(0, candidates.pop(idx_collision))

    chosen = candidates[: random.choice([3, 4])]
    sentences = [s for _, s in chosen]
    return " ".join(sentences) if sentences else "No details available."


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert insurance claims tabular data to sentences for NLP.")
    parser.add_argument("--input", required=True, help="Path to input insurance_claims.csv")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the output CSV with columns: id, text, fraud_reported",
    )
    parser.add_argument(
        "--id_column",
        default="policy_number",
        help="Column to use as the example id (default: policy_number)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sentence variation",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    df = pd.read_csv(args.input)

    # Drop obviously empty helper column if present
    if "_c39" in df.columns:
        df = df.drop(columns=["_c39"])  # type: ignore[assignment]

    # Build text column
    text_series = df.apply(build_sentence, axis=1)

    # Select id, text, and label
    id_column = args.id_column if args.id_column in df.columns else None
    output = pd.DataFrame({"text": text_series})
    if id_column:
        output.insert(0, "id", df[id_column])
    if "fraud_reported" in df.columns:
        output["fraud_reported"] = df["fraud_reported"]

    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()


