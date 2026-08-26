# 🚀 دليل نشر مشروع Clinical RAG على AWS (EC2 & Docker)

هذا الدليل يوضح خطوات نقل ونشر المشروع على خوادم **AWS** بالكامل مع تشغيل الـ Embedding والـ Docling محلياً (In-Process) داخل السيرفر.

---

## 1. اختيار نوع الـ Instance على AWS EC2

| الخيار | الـ Instance Type | المعالج / كرت الشاشة | الاستخدام الموصى به |
| :--- | :--- | :--- | :--- |
| **GPU (موصى به للأداء الأسرع)** | `g4dn.xlarge` أو `g5.xlarge` | 1x NVIDIA T4 / A10G (16GB VRAM) + 4 vCPU + 16GB RAM | سرعة فائقة في استخراج الـ Embeddings (BGE-M3) وتقطيع الـ PDFs عبر Docling |
| **CPU (خيار اقتصادي)** | `c6i.2xlarge` أو `m6i.2xlarge` | 8 vCPU + 16-32GB RAM | تشغيل اقتصادي بدون كرت شاشة مخصص |

---

## 2. إعداد الـ Instance على AWS (Ubuntu 22.04 / 24.04)

### أ. تحديث النظام وتثبيت Docker و Docker Compose
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git curl wget libgl1 libglib2.0-0

# تثبيت Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# تثبيت Docker Compose Plugin
sudo apt-get install -y docker-compose-plugin
```

### ب. (في حال استخدام GPU) تثبيت NVIDIA Container Toolkit
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 3. تنزيل الكود وضبط الإعدادات (.env)

```bash
git clone <YOUR_REPO_URL>
cd Clinical_Hackathon

# إنشاء ملف .env
cp .env.example .env
nano .env
```

### تأكد من تعيين المتغيرات الأساسية في `.env`:
```env
APP_NAME=clinical-rag
DEBUG=false

# تشغيل الموديلات محلياً داخل السيرفر
EMBEDDING_PROVIDER_TYPE=local
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=32

DOCLING_PROVIDER_TYPE=local
DOCLING_DO_OCR=false

QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_NAME=clinical_documents

# مفتاح Gemini LLM
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-1.5-flash
MIN_SIMILARITY_SCORE_THRESHOLD=80.0
```

---

## 4. التشغيل باستخدام Docker Compose

لتشغيل كامل الخدمات (`Qdrant` + `FastAPI Backend` + `Streamlit UI`):

```bash
docker compose up -d --build
```

### متابعة السجلات (Logs) والتحميل التلقائي للموديل:
عند أول عملية بحث أو رفع ملف PDF، سيقوم السيرفر بتحميل أوزان `BAAI/bge-m3` ومكتبات Docling وتخزينها في الـ Cache (`/app/.cache/huggingface`):

```bash
# متابعة سجلات الـ Backend
docker logs -f clinical_backend

# متابعة سجلات الـ UI
docker logs -f clinical_ui
```

---

## 5. فحص الحالة والـ Health Check

```bash
curl http://localhost:3000/health
```

**الاستجابة المتوقعة:**
```json
{
  "status": "healthy",
  "app": "clinical-rag",
  "embedding_provider": "local",
  "docling_provider": "local",
  "qdrant_url": "http://qdrant:6333"
}
```

---

## 6. المنافذ (Security Groups) على AWS
تأكد من فتح المنافذ التالية في الـ Security Group للـ EC2:
- `8501`: لواجهة الـ Streamlit للمستخدمين.
- `3000`: لواجهة الـ FastAPI Backend / Swagger Docs (`http://<EC2_IP>:3000/docs`).
- `6333`: (اختياري) للوصول للـ Qdrant Dashboard.

