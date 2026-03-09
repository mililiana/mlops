# Builder
FROM python:3.10-slim AS builder

# Встановлення системних залежностей для компіляції
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Створення та активація віртуального середовища
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копіювання файлу залежностей
COPY requirements.txt .

# Встановлення залежностей проєкту, включаючи DVC, який потрібен для роботи з даними
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir dvc

# Фінальний образ
FROM python:3.10-slim

# Встановлення необхідних системних бібліотек
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Копіювання зібраного віртуального середовища з попереднього етапу
COPY --from=builder /opt/venv /opt/venv

# Налаштування шляхів для використання віртуального середовища за замовчуванням
ENV PATH="/opt/venv/bin:$PATH"

# Встановлення робочої директорії
WORKDIR /app

# Копіювання необхідних файлів проєкту
COPY src/ /app/src/
COPY config/ /app/config/
COPY compare_metrics.py requirements.txt dvc.yaml dvc.lock /app/

# Точка входу за замовчуванням (можна використовувати для запуску pipeline)
# CMD ["dvc", "repro"]
 