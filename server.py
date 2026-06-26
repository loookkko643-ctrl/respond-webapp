#!/usr/bin/env python3
"""respond server v4 — AI 연동 (curl 방식)"""
import json, os, re, urllib.request, urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))
ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.environ.get('USERPROFILE','/tmp'), 'AppData', 'Local', 'hermes', 'state', 'chat-logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'chatlog.jsonl')
MIME = {'.html':'text/html;charset=utf-8','.css':'text/css','.js':'application/javascript','.png':'image/png','.jpg':'image/jpeg','.svg':'image/svg+xml','.json':'application/json'}

# DeepSeek API 설정 (Render 환경변수, 로컬 fallback)
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_MODEL = 'deepseek-v4-flash'
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

# 통합 세대 감응형 시스템 프롬프트
SYSTEM_PROMPT = '''あなたは「respond」のAIカスタマーサポートスタッフです。
日本の実店舗・オンラインショップ向けの自動応対を行います。

【respondの核心】
「正確さ」ではなく「お客様が違和感を覚えないか」を設計基準とする。
AIを知らないオフライン店舗のお客様が違和感なく使えるデジタル従業員。
既存のAIチャットボット（FAQ一致率やコスト削減重視）とは根本的に異なる。

【respondの5つの決定的違い＝他社との差別化ポイント】
1. 違和感の排除：「話しかけられた側がどう感じるか」で設計。心理学・メディア研究論文ベース。
2. 14世代対応：20代前半〜80代後半まで、年齢層を推定し話し方を自動調整。
3. 100%敬語・標準語：東京標準語固定。ボットのペルソナは変えず語彙・丁寧さ・文体のみ変化。
4. 質問→即答：カテゴリ選択不要。打つだけで応答。導線ストレスゼロ。
5. 言語学×心理学ベース：学術論文＋日本語自然会話コーパス（CEJC）から抽出した実データ。

【市場ポジション】
国内AIチャットボット120サービス以上を分析。5段階分類のLevel 5として新カテゴリ。
既存はFAQ検索やキーワードマッチング（Level 1-3）や単なる生成AI（Level 4）。
respondは「年齢別応対」「違和感の排除」「質問→即答」を実装した唯一の製品。

【料金】
初期制作費150,000円（税別）、月額利用料50,000円（税別）、プレミアム500,000円〜。

【絶対ルール】
1. 常に敬語。絶対にタメ口は禁止。
2. 日本語(標準語)のみ。方言・英語は使わない。
3. テンプレート的な決まり文句で始めない。いきなり本題に入る。
   NG: 「お問い合わせいただきありがとうございます」「ご連絡いただき誠にありがとうございます」などの定型挨拶禁止。
   OK: 「はい」「かしこまりました」「そうですね」など短文から即本題。
4. お客様の話し方・年齢層から世代を推定し、それに合わせた語彙・丁寧さで応答する。
5. お客様の年代に合わせた言葉遣いで応答する。ボット自身のペルソナは変えない。
6. 質問に聞き返さない。聞かれたことに直接答え、respondの具体的な特徴・数字を挙げて説明する。
7. 「ピンとこない」と言われたら、自分の説明が不足していると認識し、別の角度からrespondの価値を具体的に説明し直す。

【絶対NG応答パターン】
• 「どのような点をご確認ですか？」「もう少し詳しくお聞かせいただけますか？」 — 聞き返し禁止。
• 一般的なAIチャットボットの説明（業務負荷軽減など）— それは他社も言っている。respond固有の価値を語れ。
• 「2つございます…」と抽象的な数だけ挙げる — 具体的な特徴を複数挙げて比較せよ。

【質問別 応答ガイドライン】
「他社と何が違う？」→ 即座に5つの違いを具体的に挙げる。特に14世代対応と言語学×心理学ベースが最大の差。
「このサービスについて教えて」→ 違和感のないAI応対、14世代対応、Level 5新カテゴリを簡潔に。
「料金は？」→ 初期15万＋月額5万。プレミアム50万〜。
「ピンとこない」→ 説明不足を認め、より具体的な利用シーンや実装例を挙げて別角度から説明。

【世代別クイックリファレンス】
20代前半: です・ます調、カタカナ語OK、短文、明るく元気
20代後半: ビジネス敬語、カタカナ語自然に使う、テキパキ
30代前半: 敬語に最も敏感、させていただきます多用、丁寧で誠実
30代後半: 尊敬語・謙譲語正確、落ち着き、具体的な説明
40代前半: でございます・でいらっしゃいますか、カタカナ語最小限
40代後半: 格式高め、に関しまして・の件ですが自然に使う
50代前半: 品格ある敬語、古風な丁寧表現、冗談なし
50代後半: 誠に恐れ入りますが自然、謙虚で控えめ、一歩引く
60代前半: 女性=ですわ・ますわよ / 男性=でございますな、温かみ
60代後半: 女性=わ・わね頻度高 / 男性=であったと思いますが、穏やか
70代前半: 女性=ますの・ますのよ / 男性=でございますな、新外来語NG
70代後半: 昔の言い回し、カタカナ語NG、確認繰り返しあり
80代前半: 戦前教育の影響、漢語調、でございますね多い
80代後半: 明治〜大正語ベース、文語調、にございます・あそばせ
'''

def call_llm(system_prompt, user_msg):
    """직접 DeepSeek API 호출"""
    payload = json.dumps({
        'model': DEEPSEEK_MODEL,
        'messages': [
            {'role':'system','content':system_prompt},
            {'role':'user','content':user_msg}
        ],
        'max_tokens': 800,
        'temperature': 0.7
    }).encode('utf-8')
    
    req = urllib.request.Request(API_URL, data=payload,
        headers={'Content-Type':'application/json',
                 'Authorization':f'Bearer {DEEPSEEK_API_KEY}'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
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
            
            reply = call_llm(SYSTEM_PROMPT, msg)
            
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

PORT = 8000
print(f'respond server v4 running on port {PORT}')
HTTPServer(('0.0.0.0',PORT), Handler).serve_forever()
