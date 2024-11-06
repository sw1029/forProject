from flask import Flask, request, jsonify
import joblib
import os
import mysql.connector

# Flask 앱 초기화
app = Flask(__name__)
UID = ""
PASSWORD = ""
# 모델 로드
model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
model = joblib.load(model_path)

def dbSet(UID, PASSWORD):#db호출
    return mysql.connector.connect(
        host="localhost",
        user=UID,
        password=PASSWORD,
        database="emotion"
    )
def idpwSet(user_id, password):
    db = dbSet(user_id, password)
    cursor = db.cursor()
    try:
        # ID와 비밀번호가 일치하는 사용자가 있는지 확인
        sql = "SELECT * FROM users WHERE user_id = %s AND password = %s"
        cursor.execute(sql, (user_id, password))
        user = cursor.fetchone()
        return user is not None  # 사용자가 있으면 True, 없으면 False 반환
    except mysql.connector.Error as err:
        print("can not find user:", err)
        return False
    finally:
        cursor.close()
        db.close()

def mPred(text):#모델 학습 결과값 반환
    prediction = model.predict([text])
    return prediction


def valueInsert(uid, pw, value):# 통계값 삽입
    if idpwSet(uid, pw):
        db = dbSet(uid,pw)
        cursor = db.cursor()

        try:
            # 현재 날짜를 기준으로 year, month, day 가져오기
            import datetime
            today = datetime.date.today()
            year, month, day = today.year, today.month, today.day

            # user_data 테이블에 데이터 삽입
            sql = "INSERT INTO user_data (user_id, year, month, day, value) VALUES (%s, %s, %s, %s, %s)"
            values = (uid, year, month, day, value)
            cursor.execute(sql, values)
            db.commit()
            print("분석값 저장 완료.")
            return True
        except mysql.connector.Error as err:
            print("Database error:", err)
            return False
        finally:
            cursor.close()
            db.close()
    else:
        print("insert failed.")
        return False


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user_id = data.get('user_id')
    password = data.get('password')

    if not user_id or not password:
        return jsonify({'error': '사용자 ID 또는 비밀번호가 없음.'}), 400

    # 사용자 인증
    if idpwSet(user_id, password):
        UID = user_id
        PASSWORD = password
        return jsonify({'message': '로그인 성공'})
    else:
        return jsonify({'error': '로그인 실패. 잘못된 사용자 ID 또는 비밀번호'}), 401

@app.route('/analyze', methods=['POST'])
def analyze():
    # 요청에서 텍스트 데이터 추출
    data = request.get_json()
    text = data.get('text')

    db = dbSet(UID,PASSWORD)
    cursor = db.cursor()

    if not text:
        return jsonify({'error': '텍스트 없음.'}), 400

    # 예측값 생성
    prd = mPred(text)

    # 예측 결과 반환
    response = {
        'prediction': prd
    }

    cursor.close()
    db.close()
    return jsonify(response)


if __name__ == '__main__':
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=5000, debug=True)

