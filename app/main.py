import os
from typing import Optional

from flask import Flask, render_template, request
from dotenv import load_dotenv

from scraper.auto_ru import AutoRuScraper


load_dotenv()

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"), static_folder=os.path.join(os.getcwd(), "static"))


def get_scraper() -> AutoRuScraper:
    delay = float(os.getenv("AUTO_RU_REQUEST_DELAY_SECONDS", "3.0"))
    timeout = int(os.getenv("AUTO_RU_TIMEOUT_SECONDS", "20"))
    respect_robots_env = os.getenv("AUTO_RU_RESPECT_ROBOTS", "1")
    respect_robots = respect_robots_env not in {"0", "false", "False"}
    return AutoRuScraper(delay_seconds=delay, timeout_seconds=timeout, respect_robots=respect_robots)


@app.get("/")
def index():
    return render_template("search.html")


@app.get("/search")
def search():
    city: Optional[str] = request.args.get("city") or None
    make: Optional[str] = request.args.get("make") or None
    model: Optional[str] = request.args.get("model") or None

    def parse_int(name: str) -> Optional[int]:
        val = request.args.get(name)
        if not val:
            return None
        try:
            return int(val)
        except ValueError:
            return None

    price_from = parse_int("price_from")
    price_to = parse_int("price_to")
    year_from = parse_int("year_from")
    year_to = parse_int("year_to")
    pages = parse_int("pages") or 1

    scraper = get_scraper()
    try:
        listings = scraper.search(
            city=city,
            make=make,
            model=model,
            price_from=price_from,
            price_to=price_to,
            year_from=year_from,
            year_to=year_to,
            pages=pages,
        )
        error_message = None
    except Exception as exc:
        listings = []
        error_message = str(exc)

    return render_template(
        "results.html",
        listings=listings,
        city=city,
        make=make,
        model=model,
        price_from=price_from,
        price_to=price_to,
        year_from=year_from,
        year_to=year_to,
        pages=pages,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)