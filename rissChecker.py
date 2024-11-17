from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, parse_qs

def rissLoad(keyword):
    # Chrome 옵션 설정
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # WebDriver 실행
    driver = webdriver.Chrome(service=Service(), options=chrome_options)

    try:
        # RISS 국내학술논문 페이지
        url = f'https://www.riss.kr/search/Search.do?isDetailSearch=N&searchGubun=true&colName=re_a_kor&query={keyword}'
        driver.get(url)

        # 페이지 로딩 대기
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.srchResultListW'))
        )

        # 검색 결과 아이템들 추출
        results = driver.find_elements(By.CSS_SELECTOR, 'div.srchResultListW ul li')

        # 결과를 저장할 리스트 초기화
        result_list = []

        if results:
            result_count = 0
            for i, result in enumerate(results):
                if result_count >= 5:  # 상위 5개 반환
                    break
                try:
                    title = None
                    url = None

                    # 제목과 url 추출
                    try:
                        title_element = result.find_element(By.CSS_SELECTOR, 'p.title > a')
                        title = title_element.text.strip()
                        long_url = title_element.get_attribute('href')

                        # 링크 길이 줄이기
                        parsed_url = urlparse(long_url)
                        query_params = parse_qs(parsed_url.query)
                        control_no = query_params.get('control_no', [None])[0]
                        if control_no:
                            url = f'https://www.riss.kr/link?id={control_no}'
                        else:
                            url = long_url
                    except:
                        pass

                    # 제목과 URL이 모두 있는 경우에만 리스트에 추가
                    if title and url:
                        result_count += 1
                        result_list.append({'title': title, 'url': url})
                    else:
                        # 제목 또는 URL이 없으면 해당 결과를 제외
                        continue
                except Exception as e:
                    # 에러 발생 시 해당 결과를 제외하고 다음으로 진행
                    continue
        else:
            print("검색 결과가 없습니다.")
            return []

        return result_list

    finally: driver.quit()

#print(rissLoad("인공지능"))
#테스트용 출력문