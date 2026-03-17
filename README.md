# Домашнее задание 4: Fine-tuning MusicGen

## Установка окружения

```bash
git clone https://github.com/dkzmn/HSE-DLA-HW4.git
cd HSE-DLA-HW4

#Python нужен именно 3.10 иначе зависимости ломаются (3.11 тоже вроде норм, но не пробовал)
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r audiocraft/requirements.txt

#Не стал включать в зависимости, так нужны только на этапе подготовки датасета
python -m pip install gdown datasets ollama

#Устанавливается отдельно, так как нужно ставить без зависимостей, иначе ругается
python -m pip install av --no-deps

#Ставим audiocraft как модель, чтобы потом инференсить
python -m pip install -e audiocraft

dvc remote modify storage endpointurl https://storage.yandexcloud.net
dvc remote modify storage region ru-central1
dvc remote modify storage allow_anonymous_login true
dvc pull data/wav
dvc pull data/dataset.csv
```

## Сбор данных MusicCaps
Данные уже в DVC, этот пункт можно пропустить
Но если вдруг решитесь, то должны быть установлены yt-dlp и ffmpeg
```bash
brew install yt-dlp ffmpeg
sudo apt install -y yt-dlp ffmpeg
```

Собственно запуск парсинга аудио:
```bash
python scripts/download_musiccaps.py --output-dir data --skip-existing
```

## Обогащение метаданных с помощью LLM
Данные уже в DVC, этот пункт можно пропустить
```bash
#Модель может быть любая или несколько, json-ы будут складываться в индивидуальные папки, для обучения нужно будет скопировать их в data/wav
ollama pull llama3.1:8b
ollama run llama3.1:8b
python scripts/create_json.py --models llama3.1:8b --skip-existing
```

После этого проверял данные в блокноте [notebooks/check_dataset.ipynb](notebooks/check_dataset.ipynb)

## Настройка конфигов и запуск обучения

```bash
python scripts/prepare_audiocraft_manifests.py --wav-dir data/wav --manifests-dir data/manifests
```

### Запуск fine-tuning (small)
```bash
#Если скрипты запускаются из корня проекта, то обучение нужно запускать из audiocraft
cd audiocraft
python -m audiocraft.train \
  solver=musicgen/musicgen_hw4_32khz_finetune \
  continue_from=//pretrained/facebook/musicgen-small \
  dataset.num_workers=0
```

Для обучения на Yandex DataSphere использовал ноутбук [notebooks/notebook_for_yandex.ipynb](notebooks/notebook_for_yandex.ipynb)


## Оценка качества генерации

WAV файлы, сгенерированные с помощью лучшей дообученной модели находятся тут: [generated_wav](generated_wav)

### Генерация 5 тестовых промптов

Бинарники модели (small 5 эпох, lr=e-4) тут: [https://drive.google.com/drive/folders/10wjpFoKhsLegbCSnr2rlUEza2R_p6nuZ?usp=sharing](https://drive.google.com/drive/folders/10wjpFoKhsLegbCSnr2rlUEza2R_p6nuZ?usp=sharing)

Для скачивания можно выполнить команду:
```bash
cd ..
python scripts/download_model.py
```

```bash
python scripts/generate_wav.py \
  --model-path checkpoints/best_model_small \
  --prompts-dir prompts \
  --output-dir generated_wav \
  --duration 12 \
  --device cuda
```


---

## Формат сдачи
1. Ссылка на GitHub-репозиторий с вашим кодом (скрипты парсинга, модифицированный AudioCraft, скрипты инференса).
https://github.com/dkzmn/HSE-DLA-HW4.git

2. Ссылка на веса обученной модели (HuggingFace / Google Drive).
https://drive.google.com/drive/folders/10wjpFoKhsLegbCSnr2rlUEza2R_p6nuZ?usp=sharing

3. Папка с 5 сгенерированными `.wav` файлами, названными `prompt_1.wav` ... `prompt_5.wav`.
[generated_wav](generated_wav)

4. Краткий отчет (Markdown/PDF) с описанием:
   * С какими трудностями столкнулись?
   Долгое обучение. Почти сразу получил более менее приемлимый результат на small модели, после 5 эпох 
   с LR=0.0001 (эксперимент best_small в CometML) Лучшего результата не смог добиться ни дообученим этого чекпоинта с меньшим LR, ни другими попытками
   Также пытался medium модель на GPU A100, но очень долго, не успел, хотя графики там, кажется, хорошие
   (эксперимент best_medium в CometML)

   * Какую LLM использовали для парсинга и какой системный промпт сработал лучше всего?
   ollama llama3.1:8b - бесплатно и без лимитов, кажется со своей задачей справилась
   промпт тут: [prompts/prompt_for_llm_1.txt](prompts/prompt_for_llm_1.txt) - его составить помогла также LLM

   * Какие гиперпараметры обучения (learning rate, batch size, steps) вы использовали?
   LR экспериментировал от e-6 до e-3, batch size не менял

   * Приложите логи обучения (ссылку на WandB/CometML), чтобы можно было оценить процесс обучения.
   [https://www.comet.com/dkzmn/hw4-musicgen/view/new/panels](https://www.comet.com/dkzmn/hw4-musicgen/view/new/panels) лучшие эксперименты запинены.


