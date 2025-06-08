import os
from config import PDF_ROOT_DIR

def get_categories():
    # 상품별 카테고리 목록
    return [
        category_name
        for category_name in os.listdir(PDF_ROOT_DIR)
        if os.path.isdir(os.path.join(PDF_ROOT_DIR, category_name))
    ]

def get_banks_in_category(category):
    # 주어진 카테고리 내의 은행 폴더 목록
    category_path = os.path.join(PDF_ROOT_DIR, category)
    return [
        bank_name
        for bank_name in os.listdir(category_path)
        if os.path.isdir(os.path.join(category_path, bank_name))
    ]

def get_pdfs_by_bank(category, bank):
    # 특정 카테고리와 은행 폴더 내의 PDF 파일 목록
    bank_path = os.path.join(PDF_ROOT_DIR, category, bank)
    return [
        pdf_filename
        for pdf_filename in os.listdir(bank_path)
        if pdf_filename.endswith(".pdf")
    ]