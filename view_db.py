import sqlite3

def view_database(db_path):
    try:
        # 데이터베이스에 연결
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 테이블 목록 조회
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # 테이블 및 데이터 조회
        for table in tables:
            table_name = table[0]
            print(f"Table: {table_name}")

            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()

            # 컬럼명 조회
            column_names = [description[0] for description in cursor.description]
            print(f"Columns: {column_names}")

            # 데이터 출력
            for row in rows:
                print(row)

            print("\n")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

    finally:
        # 데이터베이스 연결 종료
        if conn:
            conn.close()

# 데이터베이스 파일 경로 설정
db_path = 'search_data.db'
view_database(db_path)
