# Прогноз часа максимума для Свердловской области

Одностраничный сервис на Streamlit с проверенными результатами CatBoost-79 за январь—август 2026 года.

## Возможности

- выбор одного или нескольких месяцев 2026 года;
- метрики точности для выбранного периода: попадание модели при двух попытках и по схеме 2+2;
- матрица «дата × рыночный час» с вероятностью часа максимума;
- синие ячейки — прогноз модели, зелёная ячейка — фактический час, галочка — попадание;
- два прогнозных часа при одном сегменте ЗЧ и по два часа в каждом сегменте при двух сегментах.

`data/forecast_2026.csv` — компактный снимок утверждённого walk-forward прогона. Для каждого месяца модель обучалась только на фактических данных, доступных до начала этого месяца. Исходные наборы и код обучения в этот репозиторий не входят.

## Локальный запуск

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash на Windows
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Публикация на GitHub

1. На GitHub создайте пустой репозиторий без README, `.gitignore` и лицензии.
2. В терминале из этой папки выполните:

```bash
git add .
git status
git commit -m "Initial Streamlit dashboard"
git branch -M main
git remote add origin https://github.com/<ваш-аккаунт>/<имя-репозитория>.git
git push -u origin main
```

Перед `git add .` проверьте, что в списке есть `data/forecast_2026.csv`, но нет `.venv/`, `outputs/` или `.streamlit/secrets.toml`. Если репозиторий уже имеет `origin`, замените команду добавления remote на `git remote set-url origin <URL>`.

> В CSV есть фактические часы максимума и результаты модели. До публикации публичного репозитория убедитесь, что это допускается политикой работы с данными.

## Развёртывание в Streamlit Community Cloud

1. Откройте [share.streamlit.io](https://share.streamlit.io/) и нажмите **Create app**.
2. Выберите GitHub-репозиторий, ветку `main` и файл `streamlit_app.py` в корне.
3. При желании в **Advanced settings** задайте понятный адрес приложения; секреты для текущей версии не нужны.
4. Нажмите **Deploy** и дождитесь запуска. Cloud установит зависимости из `requirements.txt`.
5. После каждого обновления данных или кода: выполните `git add .`, `git commit -m "..."`, `git push`. Community Cloud автоматически развернёт новую версию.

Для Cloud рекомендован Python 3.12. Дополнительные системные зависимости не требуются.

## Структура

```text
chm-forecast-streamlit/
├── .streamlit/config.toml  # тема
├── data/forecast_2026.csv  # проверенный снимок CatBoost-79
├── src/
│   ├── forecast_data.py    # загрузка данных
│   └── presentation.py     # матрица и подсветка
├── streamlit_app.py        # точка входа
├── requirements.txt
└── .gitignore
```
