from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import time
import json
import re

# 공통 정리 함수
def clean(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("\n", " ").replace("\t", " ").replace("\xa0", " ").split())

# 주택담보대출 상세 정규식 패턴
section_patterns = {
    "비교공시일": r"비교공시일\s*:\s*([^\n ]+)",
    "담당부서 및 연락처": r"담당부서 및 연락처\s*([^\n]+?)가입방법",
    "가입방법": r"가입방법\s*([^\n]+)",
    "대출 부대비용": r"대출 부대비용[^\d]*(\d\..*?)중도상환 수수료",
    "중도상환 수수료": r"중도상환 수수료[^\d]*(중도상환원금.*?)(연체 이자율|대출한도)",
    "연체 이자율": r"연체 이자율\s*([^\n]+)",
    "대출한도": r"대출한도\s*([^\n]+)",
    "PDF": r"-"
}

def parse_detail_text(detail_text):
    detail_dict = {}
    fields = [
        "비교공시일", "담당부서 및 연락처", "가입방법", 
        "대출 부대비용", "중도상환 수수료", "연체 이자율", "대출한도"
    ]
    for field in fields:
        pattern = re.search(f"{field}\\s+([^비교공시일담당부서및연락처가입방법대출부대비용중도상환수수료연체이자율대출한도]+)", detail_text)
        if pattern:
            detail_dict[field] = clean(pattern.group(1))
    return detail_dict


def extract_selected_value(raw_text):
    if not raw_text:
        return ""
    return clean(raw_text.split()[-1])


def extract_sections(detail_text):
    detail_text = re.sub(r"금융거래\s*계산기\s*소비자보호정보\s*경영정보", "", detail_text)
    sections = {}

    score_levels = [
        "900점 초과", "801~900점", "701~800점", "601~700점",
        "501~600점", "401~500점", "301~400점", "300점 이하", "평균금리"
    ]

    def parse_rate_block(label, block_text):
        values = block_text.strip().split()
        return {f"{label}_{level}": values[i] for i, level in enumerate(score_levels) if i < len(values)}

    pattern_rate_block = re.compile(r"(기준금리|가산금리|가[·\.]?감조정금리)\s+((?:\d+\.\d+%?\s+)+)")
    for match in pattern_rate_block.finditer(detail_text):
        label = match.group(1).replace("가.감", "가·감").replace("가·감", "가·감조정금리")
        rate_values = match.group(2)
        sections.update(parse_rate_block(label, rate_values))

    for key, pattern in section_patterns.items():
        match = re.search(pattern, detail_text)
        if match:
            sections[key] = match.group(1).strip()

    match = re.search(r"공시대상상품\s*(\(.*?\))", detail_text)
    if match:
        sections["공시대상상품"] = clean(match.group(1))

    return sections

# 개인신용대출 상세정보 후처리
def parse_personal_detail(detail_text):
    sections = {}
    detail_text = clean(detail_text)

    score_levels = [
        "900점 초과", "801~900점", "701~800점", "601~700점",
        "501~600점", "401~500점", "301~400점", "300점 이하", "평균금리"
    ]

    def parse_rate_block(label, block_text):
        values = block_text.strip().split()
        return {f"{label}_{level}": values[i] for i, level in enumerate(score_levels) if i < len(values)}

    pattern_rate_block = re.compile(r"(기준금리|가산금리|가[·\.]?감조정금리)\s+((?:\d+\.\d+%?\s+)+)")
    for match in pattern_rate_block.finditer(detail_text):
        label = match.group(1).replace("가.감", "가·감").replace("가·감", "가·감조정금리")
        rate_values = match.group(2)
        sections.update(parse_rate_block(label, rate_values))

    match = re.search(r"비교공시일\s*[:：]?\s*(\d{4}-\d{2}-\d{2})", detail_text)
    if match:
        sections["비교공시일"] = match.group(1)

    match = re.search(r"담당부서 및 연락처\s*(.*?)가입방법", detail_text)
    if match:
        sections["담당부서 및 연락처"] = clean(match.group(1))

    match = re.search(r"가입방법\s*([^\n]+)", detail_text)
    if match:
        sections["가입방법"] = clean(match.group(1))

    match = re.search(r"공시대상상품\s*(\(.*?\))", detail_text)
    if match:
        sections["공시대상상품"] = clean(match.group(1))

    return sections

driver = webdriver.Chrome()
wait = WebDriverWait(driver, 15)
driver.get("https://finlife.fss.or.kr/finlife/ldng/houseMrtg/list.do?menuNo=700007")

wait.until(EC.presence_of_element_located((By.ID, "pageUnit")))
select = Select(driver.find_element(By.ID, "pageUnit"))
select.select_by_visible_text("50")
time.sleep(0.5)

wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.search.ajaxFormSearch"))).click()
time.sleep(0.5)

results = []

for page_num in range(1, 7):
    print(f"📄 페이지 {page_num} 수집 중...")

    if page_num > 1:
        try:
            page_links = driver.find_elements(By.CSS_SELECTOR, "ul.pagination li a[data-pageindex]")
            for link in page_links:
                if link.get_attribute("data-pageindex") == str(page_num):
                    driver.execute_script("arguments[0].click();", link)
                    time.sleep(0.1)
                    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tbody tr")))
                    break
        except Exception as e:
            print(f"❌ 페이지 {page_num} 이동 실패: {e}")
            break

    rows = driver.find_elements(By.CSS_SELECTOR, "tbody tr")
    for i in range(0, len(rows), 2):
        info = {}
        try:
            tds = rows[i].find_elements(By.TAG_NAME, "td")
            if len(tds) < 15:
                continue

            info = {
                "금융회사": clean(tds[1].text),
                "상품명": clean(tds[2].text),
                "주택종류": clean(tds[3].text),
                "금리방식": extract_selected_value(tds[4].text),
                "상환방식": extract_selected_value(tds[5].text),
                "최저금리": clean(tds[6].text),
                "최고금리": clean(tds[7].text),
                "전월평균금리": clean(tds[8].text),
                "문의전화": clean(tds[13].text)
            }

            detail_link = next(
                a for a in rows[i].find_elements(By.TAG_NAME, "a") if "상세" in a.text
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", detail_link)
            driver.execute_script("arguments[0].click();", detail_link)
            time.sleep(0.05)

            # 상세 클릭 및 후처리
            detail_row = rows[i + 1]
            raw_detail = clean(detail_row.text)
            info["상세정보"] = raw_detail
            info.update(parse_detail_text(raw_detail))

                        # 🔍 후처리 추가
            info.update(extract_sections(raw_detail))

        except Exception as e:
            print(f"❌ 상세 클릭 실패 @ row {i}: {e}")
            info["상세정보"] = ""

        results.append(info)

    print(f"✅ 누적 수집: {len(results)}개")

driver.quit()

documents = []
for idx, item in enumerate(results):
    doc = {
        "id": f"대출_{idx+1:03d}",
        "bank": item.get("금융회사", ""),
        "product_name": item.get("상품명", ""),
        "type": "주택담보대출",
        "content": f"{item.get('금융회사', '')}, {item.get('상품명', '')}, "
                   f"{item.get('최저금리', '')}~{item.get('최고금리', '')}, "
                   f"{item.get('대출한도', item.get('한도', item.get('대출금액', '')))}, "
                   f"{item.get('대출기간', '')}",
        "key_summary": f"{item.get('금융회사', '')} / {item.get('상품명', '')} / "
                       f"{item.get('최저금리', '')}~{item.get('최고금리', '')} / "
                       f"{item.get('대출한도', item.get('한도', item.get('대출금액', '')))} / "
                       f"{item.get('대출기간', '')}",
        "metadata": item
    }
    documents.append(doc)


############################
#######개인신용대출##########
############################

driver = webdriver.Chrome()
wait = WebDriverWait(driver, 15)
driver.get("https://finlife.fss.or.kr/finlife/ldng/indvlCrdt/list.do?menuNo=700009")

wait.until(EC.presence_of_element_located((By.ID, "pageUnit")))
select = Select(driver.find_element(By.ID, "pageUnit"))
select.select_by_visible_text("50")
time.sleep(0.5)

wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.search.ajaxFormSearch"))).click()
time.sleep(0.5)

results = []

rows = driver.find_elements(By.CSS_SELECTOR, "tbody tr")
for i in range(0, len(rows), 2):
    info = {}
    try:
        tds = rows[i].find_elements(By.TAG_NAME, "td")
        if len(tds) < 15:
            continue

        info = {
            "금융회사": clean(tds[0].text),
            "대출종류": clean(tds[1].text),
            "금리구분": clean(tds[2].text),
            "900점 초과": clean(tds[3].text),
            "801~900점": clean(tds[4].text),
            "701~800점": clean(tds[5].text),
            "601~700점": clean(tds[6].text),
            "501~600점": clean(tds[7].text),
            "401~500점": clean(tds[8].text),
            "301~400점": clean(tds[9].text),
            "300점 이하": clean(tds[10].text),
            "평균금리": clean(tds[11].text),
            "CB 회사명": clean(tds[12].text),
            "문의전화": clean(tds[13].text)
        }

        detail_link = next(
            a for a in rows[i].find_elements(By.TAG_NAME, "a") if "상세" in a.text
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", detail_link)
        driver.execute_script("arguments[0].click();", detail_link)
        time.sleep(0.05)

        # 상세 클릭 및 후처리
        detail_row = rows[i + 1]
        raw_detail = clean(detail_row.text)
        info["상세정보"] = raw_detail
        info.update(parse_detail_text(raw_detail))

        # 🔍 후처리 추가
        info.update(parse_personal_detail(raw_detail))

    except Exception as e:
        print(f"❌ 상세 클릭 실패 @ row {i}: {e}")
        info["상세정보"] = ""

    results.append(info)

print(f"✅ 누적 수집: {len(results)}개")

driver.quit()

for idx, item in enumerate(results):
    doc = {
        "id": f"대출_{idx+299:03d}",
        "bank": item.get("금융회사", ""),
        "product_name": item.get("대출종류", ""),
        "type": "개인신용대출",
        "content": f"{item.get('금융회사', '')}, {item.get('대출종류', '')}, "
                   f"{item.get('금리구분', '')}~{item.get('평균금리', '')}, ",
        "key_summary": f"{item.get('금융회사', '')} / {item.get('대출종류', '')} / "
                       f"{item.get('금리구분', '')}~{item.get('평균금리', '')} / ",
        "metadata": item
    }
    documents.append(doc)

with open("loan.json", "w", encoding="utf-8") as f:
    json.dump({"documents": documents}, f, ensure_ascii=False, indent=2)

print(" 저장 완료: 대출_상세포함.json")