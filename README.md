# Доказательный RAG по ТФКП

ФИО: Евсеев Григорий Владимирович

Группа: М05-604б

BM25 и multilingual-E5-small находят фрагменты учебника. Система объединяет
выдачи через RRF и добавляет соседние блоки. Qwen3.5-4B выбирает варианты,
обосновывает ответ и указывает источники. Поиск работает на CPU, генерация
через OpenAI-совместимый API. Обучения нет.

## Подготовка

Все команды выполняются из корня репозитория. Нужен Python 3.11.
Корпус строится из [учебника Горяйнова и Половинкина](https://new.mipt.ru/upload/mathematics/study/uchebniki/%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf).
PDF, веса и индекс хранятся локально в workspace и не публикуются.

~~~bash
python3.11 -m venv workspace/.venv
source workspace/.venv/bin/activate
python -m pip install -r requirements.txt
curl -L 'https://new.mipt.ru/upload/mathematics/study/uchebniki/%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf' -o workspace/textbook.pdf
python -m support.prepare_data --pdf workspace/textbook.pdf --qa support/tfkp_15_qa.md --output workspace/data
python -m support.download_models --config config.json
python build_index.py --corpus workspace/data/corpus.jsonl --output workspace/index --config config.json
~~~

Если корпус уже предоставлен, его путь передается напрямую через --corpus.
QA-ответы и эталоны не включаются в индекс. Ревизии моделей зафиксированы
в config.json и runtime.json, версии зависимостей в requirements.txt.
models_dir задается относительно config.json.

## Предсказание

~~~bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MODEL=Qwen/Qwen3.5-4B
export LLM_API_KEY=local-only
export SUBMISSION_ID=tfkp-development
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python predict.py --index workspace/index --questions workspace/data/dev_questions.jsonl --output workspace/answers.jsonl --trace workspace/trace.jsonl --config config.json
~~~

Для проверки подставляются endpoint и ключ преподавателя. Каждый запрос содержит
X-Question-ID и X-Submission-ID. Один вопрос вызывает один генеративный запрос.
Лимиты составляют 6144 входных и 2048 выходных токенов. История между вопросами
не передается. Ответы и фактический расход записываются в JSONL.

## Colab и локальный сервер

Ноутбук находится в support/run_colab.ipynb. После скачивания PDF соберите
архив для личного запуска:

~~~bash
mkdir -p workspace/artifacts
rm -f workspace/artifacts/tfkp_setup.zip
zip workspace/artifacts/tfkp_setup.zip *.py *.json *.txt *.md *.jsonl *.csv support/*.py support/*.txt support/*.md support/*.ipynb workspace/textbook.pdf
~~~

Откройте ноутбук в Colab, включите GPU и загрузите архив. В расширении Colab
для VS Code загрузите архив командой Upload to Colab из меню файла.
Для ручного запуска сервера на CUDA в отдельном окружении:

~~~bash
python -m pip install -r support/requirements-gpu.txt
python -m support.download_models --config config.json --with-llm
python -m support.serve_qwen --config config.json
~~~

Используется NF4 с вычислениями FP16. Профиль рассчитан на GPU с 16 ГиБ памяти.
В измеренном прогоне на T4 пик VRAM составил 5.790 ГиБ, пик RAM сервера 8.156 ГиБ.
Пик RAM сборки индекса 1.801 ГиБ. Поиск и предсказание использовали до 1.425 ГиБ RAM.

## Эксперименты

При работающем endpoint и заданных переменных окружения:

~~~bash
python -m support.run_experiments --index workspace/index --questions workspace/data/dev_questions.jsonl --gold workspace/data/dev_gold.jsonl --config config.json --endpoint-log workspace/artifacts/endpoint_log.jsonl
~~~

Сравниваются no_context без учебника, baseline с BM25, final с BM25, E5 и соседями,
а также абляции no_dense и no_neighbors. Настройки генерации одинаковы.
Команда перезаписывает dev_answers.jsonl, dev_trace.jsonl, experiments.csv и report.md.
Полные журналы остаются в workspace/artifacts. Обоснования и текст отчета
после нового прогона нужно проверить содержательно.

В репозитории сохранены реальные результаты прогона 2 ноября 2026 года:
итоговая система получила 9 верных наборов вариантов из 15, Q=0.57125.
В no_neighbors один запрос завершился клиентским тайм-аутом, его расход
в trace остался неизвестным. Это отражено в таблице и отчете.

## Файлы

В корне находятся код RAG, настройки, зависимости и обязательные файлы для сдачи.
В support находятся подготовка данных, GPU-сервер, оценка, сборка отчета
и [описание проекта](support/PROJECT_GUIDE.md).
Восемь авторских вопросов и эталоны находятся в own_questions.jsonl и own_gold.jsonl.
Использованные библиотеки и помощь ИИ указаны в report.md.
