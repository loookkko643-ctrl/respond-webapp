#!/usr/bin/env python3
"""respond server — AI 연동 (Render용)"""
import json, os, re, urllib.request, urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))
ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.environ.get('USERPROFILE','/tmp'), 'AppData', 'Local', 'hermes', 'state', 'chat-logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'chatlog.jsonl')
MIME = {'.html':'text/html;charset=utf-8','.css':'text/css','.js':'application/javascript','.png':'image/png','.jpg':'image/jpeg','.svg':'image/svg+xml','.json':'application/json'}

# DeepSeek API 설정 (Render environment variable)
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_MODEL = os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-flash')
API_URL = 'https://api.deepseek.com/v1/chat/completions'

# 세대 감지 로직
AGE_KEYWORDS = {
    '20대초반':{'words':['あたしー','っす','やば','それな','ぴえん','エモ','しか勝'], 'role':'20대 초반'},
    '20대후반':{'words':['まじ','確かに','って感じ','わかるー'], 'role':'20대 후반'},
    '30대초반':{'words':['そうですね','お世話','恐れ入り'], 'role':'30대 초반'},
    '30대후반':{'words':['させていただき','関しまして','の件'], 'role':'30대 후반'},
    '40대초반':{'words':['でございます','いらっしゃい','不明な点'], 'role':'40대 초반'},
    '40대후반':{'words':['誠に','でいらっしゃい','申し上げ'], 'role':'40대 후반'},
    '50대초반':{'words':['の折','の節','よろしいでしょうか'], 'role':'50대 초반'},
    '50대후반':{'words':['足労','お手数','いただきたく'], 'role':'50대 후반'},
    '60대초반':{'words':['ですわ','ますわ','どちらさま'], 'role':'60대 초반'},
    '60대후반':{'words':['そうですわね','ございますわ'], 'role':'60대 후반'},
    '70대초반':{'words':['ますの','なさいまし','しなさいませ'], 'role':'70대 초반'},
    '70대후반':{'words':['でのう','だて','かねえ'], 'role':'70대 후반'},
    '80대초반':{'words':['かしら','だな','かね'], 'role':'80대 초반'},
    '80대후반':{'words':['あそばせ','ことよ','わい'], 'role':'80대 후반'},
}

def detect_generation(msg):
    scores = {}
    for gen, data in AGE_KEYWORDS.items():
        score = sum(1 for w in data['words'] if w in msg)
        if score > 0: scores[gen] = score
    if scores:
        best = max(scores, key=scores.get)
        return AGE_KEYWORDS[best]['role']
    return None

# 세대별 시스템 프롬프트
GEN_PROMPTS = {
    '20대 초반': 'あなたは20代前半のアルバイト店員です。お客様には「です・ます」調で明るく元気に応対。簡潔に。',
    '20대 후반': 'あなたは20代後半の社会人2〜3年の店員です。「です・ます」調でテキパキと。',
    '30대 초반': 'あなたは30代前半の中堅店員です。正しい敬語を心がけて。丁寧に。',
    '30대 후반': 'あなたは30代後半のベテラン店員です。正確な敬語で落ち着いた対応。',
    '40대 초반': 'あなたは40代前半のキャリアのある店員です。落ち着いた対応。',
    '40대 후반': 'あなたは40代後半の店長クラスです。「でございます」を基本に。',
    '50대 초반': 'あなたは50代前半のベテランスタッフです。品格のある敬語で。',
    '50대 후반': 'あなたは50代後半のベテランスタッフです。非常に丁寧で慎ましく。',
    '60대 초반': 'あなたは60代前半のスタッフです。女性なら「ですわ」を自然に。温かみのある敬語。',
    '60대 후반': 'あなたは60代後半のスタッフです。伝統的な丁寧な敬語。ゆっくり。',
    '70대 초반': 'あなたは70代前半のスタッフです。非常に丁寧な言葉遣い。ゆったり。',
    '70대 후반': 'あなたは70代後半のスタッフです。昔ながらの丁寧な言葉。ゆっくり。',
    '80대 초반': 'あなたは80代前半のスタッフです。格式高い丁寧な言葉。ゆっくり。',
    '80대 후반': 'あなたは80代後半のスタッフです。格式高い昔ながらの言葉。非常にゆっくり。',
}

DEFAULT_PROMPT = 'あなたは丁寧な日本語で応対する店員です。お客様には常に敬語を使ってください。短く簡潔に。'

def call_llm(system_prompt, user_msg):
    """직접 DeepSeek API 호출"""
    payload = json.dumps({
        'model': DEEPSEEK_MODEL,
        'messages': [
            {'role':'system','content':system_prompt},
            {'role':'user','content':user_msg}
        ],
        'max_tokens': 300,
        'temperature': 0.7
    }).encode('utf-8')
    
    req = urllib.request.Request(API_URL, data=payload,
        headers={'Content-Type':'application/json',
                 'Authorization':f'Bearer {DEEPSEEK_API_KEY}'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data['choices'][0]['message']['content'].strip()
    except urllib.error.HTTPError as e:
        return f'申し訳ございません。ただいま応答できません。（{e.code}）'
    except Exception as e:
        return f'申し訳ございません。少々お待ちください。（{str(e)[:50]}）'

class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Access-Control-Allow-Methods','POST,GET,OPTIONS')
        self.send_header('Access-Control-Allow-Headers','Content-Type')
        self.end_headers()
    
    def do_GET(self):
        p = self.path.split('?')[0]
        if p == '/api/logs':
            self.send_response(200)
            self.send_header('Content-Type','application/json;charset=utf-8')
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            logs = []
            if os.path.isfile(LOG_FILE):
                with open(LOG_FILE,'r',encoding='utf-8') as f:
                    for line in f:
                        line=line.strip()
                        if line:
                            try: logs.append(json.loads(line))
                            except: pass
            self.wfile.write(json.dumps(logs,ensure_ascii=False,indent=2).encode())
            return
        if p == '/': p = '/index.html'
        fp = os.path.abspath(os.path.join(ROOT, p.lstrip('/')))
        if fp.startswith(ROOT) and os.path.isfile(fp):
            self.send_response(200)
            ext = os.path.splitext(fp)[1].lower()
            self.send_header('Content-Type', MIME.get(ext, 'application/octet-stream'))
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            with open(fp,'rb') as f: self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def do_POST(self):
        if self.path != '/api/chat':
            self.send_response(404); self.end_headers(); return
        try:
            length = int(self.headers.get('Content-Length',0))
            raw = self.rfile.read(length)
            try: body = json.loads(raw.decode('utf-8'))
            except: body = json.loads(raw.decode('utf-8', errors='replace'))
            msg = body.get('message','').strip()
            
            gen = detect_generation(msg)
            sp = GEN_PROMPTS.get(gen, DEFAULT_PROMPT) if gen else DEFAULT_PROMPT
            reply = call_llm(sp, msg)
            
            now = datetime.now(JST).strftime('%H:%M')
            ts = datetime.now(JST).isoformat()
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'ts':ts,'msg':msg,'reply':reply}, ensure_ascii=False)+'\n')
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            self.wfile.write(json.dumps({'reply':reply,'time':now,'auto':True}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type','application/json')
            self.send_header('Access-Control-Allow-Origin','*')
            self.end_headers()
            self.wfile.write(json.dumps({'error':str(e)}).encode())

PORT = int(os.environ.get('PORT', 8000))
print(f'respond server running on port {PORT}')
HTTPServer(('0.0.0.0',PORT), Handler).serve_forever()
