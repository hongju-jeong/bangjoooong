import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
#from multiprocessing import Pool
import numpy as np
from  wordcloud  import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
train = pd.read_csv('data/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', header=0, delimiter='\t', quoting=3)

"""
example1 = BeautifulSoup(train['review'][0], "html5lib")

letters_only = re.sub('[^a-zA-Z]', ' ', example1.get_text()) #정규표현식

#소문자로 변환
lower_case = letters_only.lower()
# 문자를 나눈다. => 토큰화
words = lower_case.split()

words = [w for w in words if not w in stopwords.words('english')] # 불용어 제거
print(len(words))
stemmer = SnowballStemmer('english')
words = [stemmer(w) for w in words]    #어간 추출

wordnet_lemmatizer = WordNetLemmatizer()       #동사냐 명사냐에 따라 뜻이 달라지는 것들 굴절 된 형태의 단어를 그룹화하는 과정
words = [wordnet_lemmatizer(w) for w in words]
"""


def review_to_words(raw_review):
    # 1. HTML 제거
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. 영문자가 아닌 문자는 공백으로 변환
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. 소문자 변환
    words = letters_only.lower().split()
    # 4. 파이썬에서는 리스트보다 세트로 찾는게 훨씬 빠르다.
    # stopwords 를 세트로 변환한다.
    stops = set(stopwords.words('english'))
    # 5. Stopwords 불용어 제거
    meaningful_words = [w for w in words if not w in stops]
    # 6. 어간추출
    stemmer = SnowballStemmer('english')
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. 공백으로 구분된 문자열로 결합하여 결과를 반환
    return ' '.join(stemming_words)


num_reviews = train['review'].size
"""
def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    # 키워드 항목 중 workers 파라메터를 꺼냄
    workers = kwargs.pop('workers')
    # 위에서 가져온 workers 수로 프로세스 풀을 정의
    pool = Pool(processes=workers)
    # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠 작업
    result = pool.map(_apply_df, [(d, func, kwargs) for d in np.array_split(df, workers)])
    pool.close()
    # 작업 결과를 합쳐서 반환
    return pd.concat(list(result))


clean_train_reviews = apply_by_multiprocessing(train['review'], review_to_words, workers=4)
clean_test_reviews = apply_by_multiprocessing(test['review'], review_to_words, workers=4)
"""

clean_train_reviews = []
for i in range(0, num_reviews):
    if (i + 1)%5000 == 0:
        print('Review {} of {} '.format(i+1, num_reviews))
    clean_train_reviews.append(review_to_words(train['review'][i]))

clean_test_reviews = []
for i in range(0, num_reviews):
    if (i + 1)%5000 == 0:
        print('Review {} of {} '.format(i+1, num_reviews))
    clean_test_reviews.append(review_to_words(train['review'][i]))


def displayWordCloud(data = None, backgroundcolor = 'white', width=800, height=600 ):
    wordcloud = WordCloud(stopwords = STOPWORDS,
                          background_color = backgroundcolor,
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15 , 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


displayWordCloud(' '.join(clean_train_reviews))