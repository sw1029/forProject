from flask import Flask, request, jsonify
import joblib
import os
import mysql.connector
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from difflib import SequenceMatcher
import rissChecker
import base64
from safetensors.torch import load_file

'''
flutter client에서 받은 데이터를 통해 요청한 값을 반환하는 것이 주요 골자
db는 local에서 mysql로 구축

작동 test 완료:
dbset 포함 db 접속 기능들
getRiss

미완료:
모델 평가(모델 호출 포함은 테스트 완료, 추가 가중치 구현 부분 미적용)
monthData (얘는 평가함수 바탕으로 통계를 내줘야 해서, 일단은 한 달 단위의 데이터 반환으로 구현)
'''



# Flask 앱 초기화
app = Flask(__name__)
# 모델 로드
model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base")
model_path = os.path.join(os.path.dirname(__file__), "C:/Users/tjdnd/OneDrive/바탕 화면/새 폴더/model.safetensors")
state = load_file(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(state)#로드한 가중치 적용
model.to(device)#gpu에 올림

rootid = 'root'
password = ''#db 비밀번호
def dbSet(UID, PASSWORD):#db호출
    return mysql.connector.connect(
        host="localhost",
        user=UID,
        password=PASSWORD,
        database="emotion"
    )


#wds = ["우울","불안","초조","자살","실패","무의미","절망"]
#해당 단어는 가중치 단어 예시

def findWord(keys, sentence, threshold=0.8):
    """
    특정 단어 목록과 문장 내 유사 단어를 찾는 함수.
    유사한 단어 목록 반환
    임계값 이상의 단어만 선택하게 구현
    """
    sim = []
    wSplit = sentence.split()  # 문장을 단어 단위로 분리

    for word in wSplit:
        for key in keys:
            simNum = SequenceMatcher(None, word.lower(), key.lower()).ratio()
            if simNum >= threshold:  # 유사도가 임계값 이상인 경우
                sim.append((word, key, simNum))

    return sim

def validEmotion(text):#분석값 반환-절반쯤 구현
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
    # 로짓 예측
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits#softmax로 분석할 logit
    probabilities = torch.softmax(logits, dim=-1)#softmax로 분석한 확률
    pred = outputs#모델 분석
    #고구마 케이스와 분류값 반환
    case1 = False
    case2 = False
    case3 = False
    case4 = False
    case5 = False
    '''
    case가 적으니 이부분은 하드코딩으로 구현
    이진분류 값, softmax를 통해 얻은 확률값으로 평가
    특정 키워드가 들어가면 가점
    키워드 확인은 findWord를 통해 수행
    '''

    if case1: return 1,pred,probabilities
    if case2: return 2,pred,probabilities
    if case3: return 3,pred,probabilities
    if case4: return 4,pred,probabilities
    if case5: return 5,pred,probabilities#케이스가 적으니 하드코딩
    #팝업해야할 고구마의 종류와 모델의 분석값, softmax를 통한 확률정보 return
    return -1, None,-1 #error

@app.route('/riss', methods=['GET'])
def getRiss():
    keyword = request.args.get('keyword')
    if not keyword:
        return jsonify({'error': 'keyword 없음'}), 400
    riss = rissChecker.rissLoad(keyword)
    # 제목과 url 상위 5개 리스트 반환
    return jsonify({'data': riss})

def monthValid(uid,year,month):
    '''
    db명 emotion
    table 명 user_data
    data_id | int - primary key
| user_id | int
| year    | int
| month   | int
| day     | int
| value   | float
    유저별로 날짜에 따라 분석값 저장
    uid, year, month를 전달받으면 한 달치 분석값을 list로 반환
    '''
    # 데이터베이스 연결
    conn = dbSet(rootid, password)
    cursor = conn.cursor()

    # 쿼리 작성 및 실행
    query = """
        SELECT day, value FROM user_data
        WHERE user_id = %s AND year = %s AND month = %s
        ORDER BY day ASC
        """
    cursor.execute(query, (uid, year, month))

    # 결과 가져오기
    results = cursor.fetchall()

    # 연결 종료
    cursor.close()
    conn.close()

    # 결과를 리스트로 반환
    data_list = [{'day': day, 'value': value} for day, value in results]
    return data_list

@app.route('/emotion', methods=['POST'])
def emotionAnalysis():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': '텍스트 없음'}), 400

    case, pred, probabilities = validEmotion(text)
    if case == -1:
        return jsonify({'error': '오류 발생'}), 500

    # 결과 반환
    return jsonify({
        'case': case,
        'probabilities': probabilities.tolist(),  # 텐서를 리스트로 변환
    })
def getsweetpotatoImg(case):
    '''db명 emotion, table명 sweetp
    table 구성
    id : int primary key
    title : varchar(255)
    image: longblob
    요청해온 case(int)값에 따라 그에 맞는 이미지와 텍스트를 client 앱에 반환 '''
    # 데이터베이스 연결
    conn = dbSet(rootid, password)
    cursor = conn.cursor()

    # 쿼리 작성 및 실행
    query = """
        SELECT title, image FROM sweetp
        WHERE id = %s
        """
    cursor.execute(query, (case,))

    # 결과 가져오기
    result = cursor.fetchone()

    # 연결 종료
    cursor.close()
    conn.close()
    if result:
        title, image_data = result
        # 이미지 데이터를 base64로 인코딩
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        # 결과 반환
        return {'title': title, 'image': image_base64}
    else: return None

def sentence(id):
    '''
    db명 emotion, table명 sentence
    id : int primary key
    content : varchar(255)
    지정한 id값의 content 문장 반환
    '''
    try:
        # 데이터베이스 연결
        conn = dbSet(rootid, password)
        cursor = conn.cursor()

        # 쿼리 작성 및 실행
        query = "SELECT content FROM sentences WHERE id = %s"
        cursor.execute(query, (id,))

        # 결과 가져오기
        result = cursor.fetchone()

        # 결과가 존재할 경우 문장 반환, 없을 경우 None 반환
        if result:
            return result[0]  # content 반환
        else:
            return None

    except mysql.connector.Error as err:
        print(f"DB: {err}")
        return None

    finally:
        # 연결 종료
        cursor.close()
        conn.close()

@app.route('/sentence', methods=['GET'])
def getSentence():
    id = request.args.get('id')
    if not id:
        return jsonify({'error': 'id 없음'}), 400
    id = int(id)
    sen = sentence(id)
    #문장 반환
    return jsonify({'data': sen})

@app.route('/monthdata', methods=['GET'])
def monthData():
    '''
    지정된 사용자 ID와 연도, 월에 해당하는 한 달치 분석값을 반환
    요청 URL 예시:
    /monthdata?uid=1&year=2023&month=10
    '''
    uid = request.args.get('uid')
    year = request.args.get('year')
    month = request.args.get('month')

    if not all([uid, year, month]):
        return jsonify({'error': 'uid, year, month를 모두 제공해주세요.'}), 400

    data = monthValid(uid, year, month)
    return jsonify({'data': data})

@app.route('/sweetpotato', methods=['GET'])
def getSP():
    '''
    요청된 case 값에 따라 해당하는 이미지와 텍스트를 반환
    '''
    case = request.args.get('case')
    if not case: return jsonify({'error': 'case 값이 없음'}), 400
    result = getsweetpotatoImg(case)
    if result: return jsonify(result)
    else: return jsonify({'error': '올바르지 않은 case'}), 404

@app.route('/spAll', methods=['GET'])
def getAllSP():
    '''
    db명 emotion, table명 sweetp
    테이블의 모든 데이터를 반환
    초기 이미지 설정용
    '''
    conn = dbSet(rootid, password)
    cursor = conn.cursor()
    query = "SELECT id, title, image FROM sweetp"
    cursor.execute(query)

    # 결과 가져오기
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    data_list = [
        {
            'id': row[0],
            'title': row[1],
            'image': base64.b64encode(row[2]).decode('utf-8')  # 이미지 데이터를 base64로 인코딩
        } for row in results
    ]
    return jsonify({'data': data_list})

if __name__ == '__main__':
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=5000, debug=True)

