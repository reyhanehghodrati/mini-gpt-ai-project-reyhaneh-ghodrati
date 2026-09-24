# AI Bootcamp Final Project - MiniGPT with PyTorch

این مخزن نسخه‌ی کامل‌شده‌ی پروژه‌ی پایانی بوت‌کمپ هوش مصنوعی است. هر چهار مرحله‌ی خواسته‌شده پیاده‌سازی شده‌اند: توکنایزر مقدماتی، توکنایزر production، پایپ‌لاین داده‌ی پیش‌آموزش و MiniGPT مبتنی بر PyTorch.

## فایل‌ها

- `tokenizer1.py`: توکنایزر کاراکتری و Byte-Pair Encoding از صفر، تحلیل واژگان و نسبت فشرده‌سازی.
- `tokenizer2.py`: نرمال‌سازی Unicode، pre-tokenization مشابه GPT-2، BPE چندزبانه و special tokenها.
- `data_pipeline.py`: پاک‌سازی و فیلتر کیفیت، MinHash + LSH، حذف داده‌های تکراری، توکن‌سازی، sequence packing و data loader.
- `mini_gpt.py`: معماری Decoder-Only Transformer شامل embedding، LayerNorm دستی، causal multi-head attention، FFN، residual connection، loss دستی، آموزش و تولید خودبازگشتی.
- `tests/test_project.py`: تست‌های یکپارچه و رگرسیون برای همه‌ی مراحل.

## معماری MiniGPT

پیاده‌سازی مدل از APIهای آماده‌ی ممنوع‌شده استفاده نمی‌کند. Softmax، Cross Entropy، LayerNorm، causal mask و Multi-Head Attention با عملیات پایه‌ی tensor نوشته شده‌اند. projection خروجی با embedding ورودی weight tying دارد.

اجزای اصلی:

1. Token embedding + learned positional embedding
2. چند بلوک Pre-LN Transformer
3. Causal multi-head self-attention
4. Feed-forward network با ReLU
5. Residual connections
6. Final LayerNorm و projection به فضای واژگان

## نصب

Python 3.10 یا جدیدتر پیشنهاد می‌شود.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## اجرای تست‌ها

```bash
python -m pytest -q
```

## اجرای بخش‌ها

```bash
python tokenizer1.py
python tokenizer2.py
python data_pipeline.py
python mini_gpt.py
```

اجرای مستقیم `mini_gpt.py` آموزش کامل ۱۵۰۰ مرحله‌ای را شروع می‌کند و روی CPU ممکن است حدود یک دقیقه یا بیشتر زمان ببرد. برای یک آزمایش سریع‌تر می‌توانید تابع `train_mini_gpt` را با `num_steps` کمتر فراخوانی کنید.

## نتایج اعتبارسنجی

- تمام تست‌های اولیه‌ی سه فایل اجباری پاس شده‌اند.
- round-trip متن‌های انگلیسی، فارسی، چینی و emoji در توکنایزر production بررسی شده است.
- خروجی MiniGPT از نظر shape و گرادیان بررسی شده و Cross Entropy دستی با مرجع PyTorch تطابق دارد.
- شمارش تحلیلی پارامترها دقیقاً با مجموع پارامترهای واقعی مدل برابر است.
- در تست آموزش کوتاه ۸۰ مرحله‌ای، loss از `5.5116` به `0.3641` رسید.
- پایپ‌لاین نمونه ۱۵ سند را پردازش کرد، داده‌های کم‌کیفیت/تکراری را حذف کرد و ۲٬۶۶۱ توکن را در ۲۱ sequence بسته‌بندی کرد.

## نکته‌ی تحویل

طبق صورت‌مسئله، نام نهایی فایل/مخزن باید شامل نام تمرین و نام و نام خانوادگی دانشجو به انگلیسی باشد. پیش از ارسال لینک، نام مخزن GitHub را با مشخصات دانشجو تنظیم کنید.
