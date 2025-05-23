from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import re

# 텍스트 정리 함수
def clean(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("\n", " ").replace("\t", " ").replace("\xa0", " ").split())

# 상세정보 필드 추출
savings_detail_patterns = {
    "비교공시일": r"비교공시일\s*[:：]?\s*([^\n ]+)",
    "담당부서 및 연락처": r"담당부서 및 연락처\s*(.*?)우대조건",
    "우대조건": r"우대조건\s*([^\n]+)",
    "가입대상": r"가입대상\s*([^\n]+)",
    "가입방법": r"가입방법\s*([^\n]+)",
    "만기후 이자율": r"만기후 이자율\s*([^\n]+)",
    "기타 유의사항": r"기타 유의사항\s*([^\n]+)"
}

def parse_savings_detail(text):
    metadata = {}
    text = clean(text)
    for key, pattern in savings_detail_patterns.items():
        match = re.search(pattern, text)
        if match:
            metadata[key] = match.group(1).strip()
    return metadata

# Selenium 스크래핑 시작
url = "https://finlife.fss.or.kr/finlife/svings/fdrmEnty/list.do?menuNo=700003"
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 15)
driver.get(url)

wait.until(EC.presence_of_element_located((By.ID, "pageUnit")))
select = Select(driver.find_element(By.ID, "pageUnit"))
select.select_by_visible_text("50")
time.sleep(1)

wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.search.ajaxFormSearch"))).click()
time.sleep(1.5)

results = []
for page_num in range(1, 10):
    print(f"📄 페이지 {page_num} 수집 중...")

    if page_num > 1:
        try:
            page_links = driver.find_elements(By.CSS_SELECTOR, "ul.pagination li a[data-pageindex]")
            for link in page_links:
                if link.get_attribute("data-pageindex") == str(page_num):
                    driver.execute_script("arguments[0].click();", link)
                    time.sleep(2)
                    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tbody tr")))
                    break
        except Exception as e:
            print(f"❌ 페이지 {page_num} 이동 실패: {e}")
            break

    rows = driver.find_elements(By.CSS_SELECTOR, "tbody tr")

    for i in range(0, len(rows), 2):
        try:
            tds = rows[i].find_elements(By.TAG_NAME, "td")
            if len(tds) < 10:
                continue

            info = {
                "금융회사": clean(tds[1].text),
                "상품명": clean(tds[2].text),
                "적립방식":clean(tds[3].text),
                "세전이자율":clean(tds[4].text),
                "세후이자율":clean(tds[5].text),
                "세후이자(예시)":clean(tds[6].text),
                "최고우대금리":clean(tds[7].text),
                "가입대상":clean(tds[8].text),
                "이자계산방식":clean(tds[9].text),
                "금융상품문의":clean(tds[10].text)
            }

            detail_link = next(a for a in rows[i].find_elements(By.TAG_NAME, "a") if "상세" in a.text)
            driver.execute_script("arguments[0].scrollIntoView(true);", detail_link)
            driver.execute_script("arguments[0].click();", detail_link)
            time.sleep(0.05)

            detail_row = rows[i + 1]
            raw_detail = clean(detail_row.text)
            info["상세정보"] = raw_detail
            info.update(parse_savings_detail(raw_detail))

            results.append(info)

        except Exception as e:
            print(f"❌ 오류 @ row {i}: {e}")

print(f"✅ 총 수집: {len(results)}건")
driver.quit()

# JSON 저장
documents = []
for idx, item in enumerate(results):
    bank = item.get("금융회사", "")
    product = item.get("상품명", "")
    base_rate = item.get("세전이자율", "")
    max_rate = item.get("최고우대금리", "")
    interest_type = item.get("이율구분", "")
    pub_date = item.get("비교공시일", item.get("기준일", ""))
    join_method = item.get("가입방법", "")
    pref_cond = item.get("우대조건", "")
    join_target = item.get("가입대상", "")
    caution = item.get("기타 유의사항", "")
    maturity_rate = item.get("만기후 이자율", "")

    content = (
        f"은행: {bank}\n"
        f"상품명: {product}\n"
        f"기본금리( %): {base_rate}\n"
        f"최고금리(우대금리포함,  %): {max_rate}\n"
        f"이자지급방식: {interest_type}\n"
        f"은행 최종제공일: {pub_date}\n"
        f"가입방법: {join_method}\n"
        f"우대조건: {pref_cond}\n"
        f"가입대상: {join_target}\n"
        f"기타 유의사항: {caution}\n"
        f"만기후 이자율: {maturity_rate}"
    )

    key_summary = (
        f"은행: {bank}, 상품명: {product}, "
        f"기본금리( %): {base_rate}, 최고금리(우대금리포함,  %): {max_rate}"
    )

    item["key_summary"] = key_summary

    doc = {
        "id": f"적금_{idx+1:03d}",
        "bank": bank,
        "product_name": product,
        "type": "정기적금",
        "content": content,
        "key_summary": key_summary,
        "metadata": item
    }
    documents.append(doc)


with open("installment_savings.json", "w", encoding="utf-8") as f:
    json.dump({"documents": documents}, f, ensure_ascii=False, indent=2)

print(" 저장 완료: installment_savings.json")