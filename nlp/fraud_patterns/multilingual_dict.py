"""
nlp/fraud_patterns/multilingual_dict.py
Cross-lingual maps of social manipulation lexicon, translating 
malicious and benign framing to intercept deepfakes globally.
"""

# Categorized high-risk manipulative vocabulary mapped across highly circulated languages
RISK_PATTERNS = {
    "fraud": {
        "english": ["crypto", "giveaway", "investment", "double your", "send me", "elon musk", "click link", "finance"],
        "spanish": ["cripto", "sorteo", "inversión", "duplica tu", "envíame", "enlace", "dinero fácil"],
        "hindi": ["क्रिप्टो", "मुफ्त", "निवेश", "दोगुना", "पैसे कमाएं", "लिंक पर क्लिक करें"],
        "tamil": ["கிரிப்டோ", "இலவசம்", "முதலீடு", "பணத்தை இரட்டிப்பாக்கு", "இணைப்பை கிளிக் செய்க"],
        "telugu": ["క్రిప్టో", "ఉచితం", "పెట్టుబడి", "డబ్బు ద్విగుణంగా", "లింక్ క్లిక్ చేయండి"],
        "bengali": ["ক্রিপ্টো", "বিনিয়োগ", "টাকা দ্বিগুণ", "লিঙ্কে ক্লিক করুন"],
        "arabic": ["كريبتو", "استثمار", "ضاعف مالك", "انقر على الرابط", "أرسل لي"],
        "mandarin": ["加密货币", "投资", "翻倍", "点击链接", "送钱", "免费"]
    },
    "political_disinfo": {
        "english": ["breaking", "leaked", "scandal", "voting", "rigged", "election fraud", "deep state", "dictator"],
        "spanish": ["última hora", "filtrado", "escándalo", "votación", "amañado", "fraude electoral", "estado profundo"],
        "hindi": ["ब्रेकिंग न्यूज़", "लीक", "घोटाला", "चुनाव", "धांधली", "ईवीएम"],
        "tamil": ["முக்கிய செய்தி", "கசிந்த", "ஊழல்", "வாக்களிப்பு", "மோசடி", "தேர்தல்"],
        "telugu": ["తాజా వార్తలు", "లీక్", "స్కామ్", "ఓటింగ్", "రిగ్గింగ్", "ఎన్నికల బూటకం"],
        "bengali": ["ব্রেকিং", "ফাঁস", "কেলেঙ্কারি", "নির্বাচন জালিয়াতি", "ভোট"],
        "arabic": ["عاجل", "مسرب", "فضيحة", "تزوير", "انتخابات"],
        "mandarin": ["突发新闻", "泄露", "丑闻", "投票", "作弊", "选举舞弊", "内幕"]
    },
    "medical_misinfo": {
        "english": ["cure found", "vaccine", "doctors hide", "secret remedy", "pharma scam"],
        "spanish": ["cura encontrada", "vacuna", "los médicos ocultan", "remedio secreto", "estafa farmacéutica"],
        "hindi": ["इलाज मिल गया", "वैक्सीन", "डॉक्टर छिपाते हैं", "गुप्त उपाय"],
        "tamil": ["சிகிச்சை கிடைத்தது", "தடுப்பூசி", "ரகசிய மருந்து"],
        "telugu": ["నివారణ దొరికింది", "టీకా", "రహస్య ఔషధం"],
        "bengali": ["ওষুধ পাওয়া গেছে", "ভ্যাকসিন", "ডাক্তাররা গোপন করে"],
        "arabic": ["تم العثور على علاج", "لقاح", "يخفي الأطباء", "علاج سري"],
        "mandarin": ["找到解药", "疫苗", "医生隐瞒", "秘方"]
    },
    "ncii": {
        "english": ["leaked video", "exposed", "nude", "onlyfans hack", "scandal"],
        "spanish": ["video filtrado", "expuesto", "desnudo", "hackeo", "escándalo"],
        "hindi": ["लीक वीडियो", "नग्न", "स्कैंडल", "वायरल वीडियो"],
        "tamil": ["கசிந்த வீடியோ", "ஆபாச", "பகிரங்கமான"],
        "telugu": ["లీక్ వీడియో", "నగ్న", "బయటపడింది"],
        "bengali": ["ফাঁস হওয়া ভিডিও", "নগ্ন", "ভাইরাল"],
        "arabic": ["فيديو مسرب", "مكشوف", "فضيحة", "فيديو إباحي"],
        "mandarin": ["泄露视频", "曝光", "裸照", "黑客", "艳照"]
    }
}

SAFE_PATTERNS = {
    "satire": {
        "english": ["parody", "satire", "joke", "meme", "fake clip", "funny edited"],
        "spanish": ["parodia", "sátira", "broma", "meme", "falso", "editado"],
        "hindi": ["पैरोडी", "व्यंग्य", "मजाक", "मेम", "फर्जी"],
        "tamil": ["நையாண்டி", "கேலி", "நகைச்சுவை"],
        "telugu": ["వింగ్", "హాస్యం", "నకిలీ"],
        "bengali": ["প্যারোডি", "জোক", "মিম"],
        "arabic": ["محاكاة ساخرة", "نكتة", "ميم", "مضحك"],
        "mandarin": ["恶搞", "讽刺", "玩笑", "梗", "搞笑"]
    },
    "art": {
        "english": ["generated with midjourney", "dalle", "ai art", "concept art", "vfx breakdown", "blender"],
        "spanish": ["generado con", "arte ia", "arte conceptual", "efectos visuales"],
        "hindi": ["एआई द्वारा निर्मित", "कला", "डिजिटल आर्ट"],
        "tamil": ["கலை", "ஏஐ தொழில்நுட்பம்"],
        "telugu": ["ఆర్ట్", "డిజిటల్ కళ"],
        "bengali": ["এআই শিল্প", "তৈরি করা"],
        "arabic": ["مولد بالذكاء الاصطناعي", "فن", "مؤثرات بصرية"],
        "mandarin": ["AI生成", "艺术", "特效分析"]
    },
    "education": {
        "english": ["how this deepfake", "detecting fake", "example of synthetic media", "tutorial"],
        "spanish": ["cómo este deepfake", "detectando falso", "ejemplo", "tutorial"],
        "hindi": ["यह डीपफेक कैसे", "फर्जी का पता लगाना", "उदाहरण", "ट्यूटोरियल"],
        "tamil": ["எப்படி இது போலியானது", "கற்றல்"],
        "telugu": ["ఈ ఫేక్ ఎలా", "గుర్తించడం", "ట్యుటోరియల్"],
        "bengali": ["কীভাবে এই ডিপফেক", "টিউটোরিয়াল"],
        "arabic": ["كيف هذا التزييف", "اكتشاف", "مثالتعليمي"],
        "mandarin": ["深度伪造原理", "检测假视频", "例子", "教程"]
    }
}
