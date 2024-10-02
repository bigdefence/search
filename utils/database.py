import sqlite3

def init_db():
    with sqlite3.connect('click_data.db') as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS clicks (
                url TEXT PRIMARY KEY,  
                click_count INTEGER DEFAULT 0  
            )
        ''')
        conn.commit()

def increment_click_count(url):
    with sqlite3.connect('click_data.db') as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO clicks (url, click_count) 
            VALUES (?, 1) 
            ON CONFLICT(url) DO UPDATE SET click_count = click_count + 1
        ''', (url,))
        conn.commit()

def get_click_count(url):
    with sqlite3.connect('click_data.db') as conn:
        c = conn.cursor()
        c.execute('SELECT click_count FROM clicks WHERE url = ?', (url,))
        result = c.fetchone()
    return result[0] if result else 0

def get_click_weight(url):
    click_count = get_click_count(url)
    return 1 + (click_count * 0.1)