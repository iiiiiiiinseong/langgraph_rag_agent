from flask import Blueprint, render_template, send_from_directory, abort
from utils import get_categories, get_banks_in_category, get_pdfs_by_bank
from config import PDF_ROOT_DIR
import os

bp = Blueprint("main", __name__)

@bp.route("/")
def index():
    categories = get_categories()
    return render_template("index.html", categories=categories)

@bp.route("/browse/<category>")
def browse_category(category):
    banks = get_banks_in_category(category)
    bank_files = {
        bank: get_pdfs_by_bank(category, bank)
        for bank in banks
    }
    return render_template("category.html", category=category, bank_files=bank_files)

@bp.route("/pdf/<category>/<bank>/<filename>")
def serve_pdf(category, bank, filename):
    file_dir = os.path.join(PDF_ROOT_DIR, category, bank)
    if os.path.exists(os.path.join(file_dir, filename)):
        return send_from_directory(file_dir, filename)
    else:
        abort(404)

@bp.route("/search")
def search():
    from flask import request

    query = request.args.get("q", "").strip().lower()
    if not query:
        return render_template("search_results.html", query=query, results={})

    results = {}

    for category in get_categories():
        for bank in get_banks_in_category(category):
            matched_files = [
                filename
                for filename in get_pdfs_by_bank(category, bank)
                if query in filename.lower()
            ]
            if matched_files:
                if category not in results:
                    results[category] = {}
                results[category][bank] = matched_files

    return render_template("search_results.html", query=query, results=results)
