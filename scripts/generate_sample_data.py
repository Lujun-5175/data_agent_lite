from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "frontend" / "public" / "sample_data"
RANDOM_SEED = 5175


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_sales_data() -> None:
    random.seed(RANDOM_SEED + 1)
    regions = ["North", "South", "East", "West"]
    categories = {
        "Electronics": [("Wireless Mouse", 28), ("Noise Cancelling Headphones", 145), ("USB-C Hub", 52)],
        "Home Office": [("Standing Desk", 420), ("Ergonomic Chair", 280), ("Desk Lamp", 38)],
        "Kitchen": [("Air Fryer", 110), ("Coffee Maker", 95), ("Blender", 72)],
        "Outdoor": [("Camping Tent", 210), ("Trail Backpack", 88), ("Water Bottle", 24)],
        "Beauty": [("Skin Care Set", 64), ("Hair Dryer", 58), ("Makeup Kit", 46)],
        "Fitness": [("Yoga Mat", 32), ("Adjustable Dumbbells", 190), ("Resistance Bands", 22)],
    }
    segments = ["Consumer", "Corporate", "Small Business"]
    channels = ["Online", "Offline"]
    start_date = date(2025, 1, 1)
    rows: list[dict[str, object]] = []

    for index in range(800):
        order_date = start_date + timedelta(days=random.randint(0, 364))
        month_seasonality = 1 + 0.18 * math.sin((order_date.month - 1) / 12 * 2 * math.pi)
        category = random.choices(
            list(categories.keys()),
            weights=[0.22, 0.18, 0.17, 0.12, 0.14, 0.17],
            k=1,
        )[0]
        product_name, base_price = random.choice(categories[category])
        region = random.choice(regions)
        segment = random.choices(segments, weights=[0.54, 0.28, 0.18], k=1)[0]
        channel = random.choices(channels, weights=[0.58, 0.42], k=1)[0]
        quantity = random.randint(1, 6)
        if segment == "Corporate":
            quantity += random.randint(0, 4)
        price_multiplier = month_seasonality * random.uniform(0.88, 1.15)
        if region == "West":
            price_multiplier *= 1.06
        if channel == "Online":
            price_multiplier *= 0.97
        unit_price = round(base_price * price_multiplier, 2)
        total_amount = round(quantity * unit_price * random.uniform(0.96, 1.04), 2)
        rows.append(
            {
                "order_id": f"ORD-{index + 10001}",
                "order_date": order_date.isoformat(),
                "region": region,
                "product_category": category,
                "product_name": product_name,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_amount": total_amount,
                "customer_segment": segment,
                "channel": channel,
            }
        )

    write_csv(
        OUTPUT_DIR / "sales_data.csv",
        [
            "order_id",
            "order_date",
            "region",
            "product_category",
            "product_name",
            "quantity",
            "unit_price",
            "total_amount",
            "customer_segment",
            "channel",
        ],
        rows,
    )


def score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def generate_student_scores() -> None:
    random.seed(RANDOM_SEED + 2)
    rows: list[dict[str, object]] = []
    for index in range(300):
        gender = random.choice(["Female", "Male"])
        age = random.choices([18, 19, 20, 21, 22, 23], weights=[16, 24, 25, 18, 11, 6], k=1)[0]
        study_hours = clamp(random.gauss(11.5, 4.2), 2, 24)
        attendance_rate = clamp(random.gauss(0.83, 0.1) + (study_hours - 11.5) * 0.008, 0.52, 0.99)
        ability = random.gauss(0, 6)
        midterm_score = clamp(49 + study_hours * 1.65 + attendance_rate * 19 + ability + random.gauss(0, 5), 35, 100)
        final_score = clamp(45 + study_hours * 1.9 + attendance_rate * 23 + ability * 0.8 + random.gauss(0, 5.5), 35, 100)
        total_score = round(midterm_score * 0.4 + final_score * 0.6, 2)
        rows.append(
            {
                "student_id": f"STU-{index + 2001}",
                "gender": gender,
                "age": age,
                "study_hours": round(study_hours, 1),
                "attendance_rate": round(attendance_rate, 3),
                "midterm_score": round(midterm_score, 1),
                "final_score": round(final_score, 1),
                "total_score": total_score,
                "grade": score_to_grade(total_score),
            }
        )

    write_csv(
        OUTPUT_DIR / "student_scores.csv",
        [
            "student_id",
            "gender",
            "age",
            "study_hours",
            "attendance_rate",
            "midterm_score",
            "final_score",
            "total_score",
            "grade",
        ],
        rows,
    )


def generate_user_behavior() -> None:
    random.seed(RANDOM_SEED + 3)
    channels = ["Paid Search", "Organic Search", "Referral", "Email", "Social", "Direct"]
    devices = ["Desktop", "Mobile", "Tablet"]
    start_date = date(2025, 7, 1)
    rows: list[dict[str, object]] = []

    for index in range(1000):
        signup_date = start_date + timedelta(days=random.randint(0, 179))
        active_lag = random.randint(0, min(120, (date(2026, 1, 15) - signup_date).days))
        last_active_date = signup_date + timedelta(days=active_lag)
        ab_group = random.choice(["A", "B"])
        channel = random.choices(channels, weights=[0.22, 0.24, 0.12, 0.15, 0.17, 0.10], k=1)[0]
        device = random.choices(devices, weights=[0.48, 0.43, 0.09], k=1)[0]
        session_base = {"Paid Search": 6, "Organic Search": 8, "Referral": 7, "Email": 9, "Social": 5, "Direct": 6}[channel]
        session_count = max(1, int(random.lognormvariate(math.log(session_base), 0.65)))
        page_views = max(session_count, int(session_count * random.uniform(3.0, 8.0) + random.gauss(0, 5)))
        conversion_score = -2.15
        conversion_score += 0.42 if ab_group == "B" else 0.0
        conversion_score += 0.055 * session_count
        conversion_score += 0.009 * page_views
        conversion_score += {"Email": 0.34, "Organic Search": 0.18, "Referral": 0.08, "Paid Search": 0.02, "Direct": -0.06, "Social": -0.12}[channel]
        conversion_score += {"Desktop": 0.08, "Mobile": -0.05, "Tablet": -0.02}[device]
        probability = 1 / (1 + math.exp(-conversion_score))
        conversion_flag = 1 if random.random() < probability else 0
        rows.append(
            {
                "user_id": f"USR-{index + 50001}",
                "signup_date": signup_date.isoformat(),
                "last_active_date": last_active_date.isoformat(),
                "session_count": session_count,
                "page_views": page_views,
                "conversion_flag": conversion_flag,
                "ab_group": ab_group,
                "channel_source": channel,
                "device_type": device,
            }
        )

    write_csv(
        OUTPUT_DIR / "user_behavior.csv",
        [
            "user_id",
            "signup_date",
            "last_active_date",
            "session_count",
            "page_views",
            "conversion_flag",
            "ab_group",
            "channel_source",
            "device_type",
        ],
        rows,
    )


def main() -> None:
    generate_sales_data()
    generate_student_scores()
    generate_user_behavior()
    print(f"Sample CSV files generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
