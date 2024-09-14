import sqlite3

# SQLite 데이터베이스 초기화
def init_db():
    conn = sqlite3.connect('click_data.db')
    c = conn.cursor()
    # 클릭 데이터 테이블 생성 (URL당 클릭 수 저장)
    c.execute('''
        CREATE TABLE IF NOT EXISTS clicks (
            url TEXT PRIMARY KEY,
            click_count INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

# 클릭 데이터 증가 함수
def increment_click_count(url):
    conn = sqlite3.connect('click_data.db')
    c = conn.cursor()
    # URL의 클릭 수를 증가시키거나 처음 클릭 시 삽입
    c.execute('''
        INSERT INTO clicks (url, click_count) 
        VALUES (?, 1) 
        ON CONFLICT(url) DO UPDATE SET click_count = click_count + 1
    ''', (url,))
    conn.commit()
    conn.close()

# 클릭 수 조회 함수
def get_click_count(url):
    conn = sqlite3.connect('click_data.db')
    c = conn.cursor()
    c.execute('SELECT click_count FROM clicks WHERE url = ?', (url,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 0
