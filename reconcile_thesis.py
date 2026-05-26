import re
import os

target_file = r'z:\repositories\master-thesis-repository\docs\obsidian\thesis\[USED] thesis_draft_final_ru.md'

with open(target_file, 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

# 1. Section 1 Title
text = text.replace('«Анализ рекрутмента, методов NLP и архитектуры семантического ранжирования»', '«Анализ рекрутмента, методов NLP и архитектуры семантического ранжирования»') # Ensure correct quote types if different

# 2. Batch Size
text = text.replace('batch_size = 16', 'batch_size = 8')

# 3. API Routes
# Replace /rank with /search/rank
text = text.replace('POST /rank', 'POST /search/rank')
text = text.replace('rank()', 'search/rank()')
text = text.replace('»**', '»**\n') # fix potential styling issues

# Remove fake routes and their descriptions
text = re.sub(r'4\. Маршрут `POST /monitoring/log`:.*?\r?\n\s*-.*?\r?\n.*?\r?\n', '', text, flags=re.DOTALL)
text = text.replace('POST /embed/job', '')
text = text.replace('POST /embed/resume', '')
text = text.replace('POST /index/upsert', '')
text = text.replace('POST /index/delete', '')

# Future Work framing
text = text.replace('дисковые векторные хранилища (Vector Databases), оптимизированные под быстрый поиск по графовым структурам. Хранилище DiskANN/SPANN позволяет эффективно масштабировать', 
                    'В качестве перспективного направления рассматриваются дисковые векторные хранилища (Vector Databases), оптимизированные под быстрый поиск по графовым структурам (например, DiskANN/SPANN), что позволит масштабировать')

text = text.replace('в основе процесса заложена метрика разности оценки кросс-энтропии, характерная для алгоритмов типа RankNet', 
                    'в качестве будущего расширения рассматривается внедрение метрик разности оценки кросс-энтропии, характерных для алгоритмов типа RankNet')

text = text.replace('В промышленных модулях в вычисления также внедряются параметры успешного просмотра описаний вакансий и извлечение нетривиальных характеристик с помощью LLM-надстроек', 
                    'В перспективных версиях системы планируется внедрение параметров успешного просмотра описаний вакансий и извлечение нетривиальных характеристик с помощью LLM-надстроек')

text = text.replace('Исследование механизмов обеспечения справедливости выдачи (fair ranking metrics) относится к перспективным задачам развития системы', 
                    'Исследование механизмов обеспечения справедливости выдачи (fair ranking metrics) относится к перспективным задачам развития системы') # Already correct, keeping for safety

# HR Involvement
text = text.replace('валидация производилась на основе размеченного экспертами набора данных', 
                    'разметка набора данных производилась экспертами')
text = text.replace('Человеко-центрированная методология дополнила метрический аппарат: валидация производилась на основе размеченного экспертами набора данных. Изучалась адекватность формируемых пояснительных плашек (explainability), а также проводился анализ качества сопоставления профессиональных признаков в резюме.', 
                    'Методология подготовки данных опиралась на экспертную разметку: HR-специалисты формировали эталонный набор данных (gold standard) для обучения и оценки модели.')

# Model Correctness
text = text.replace('ru-en-RoSBERTa', 'cointegrated/rubert-tiny2')

with open(target_file, 'w', encoding='utf-8') as f:
    f.write(text)

print("Reconciliation complete.")
