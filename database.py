from uuid import uuid4

from datetime import datetime
import time

AbstractDB = {}


def simulate(db: AbstractDB) :

    # 랜덤 이미지 파일 생성

    while True :
        time_now = datetime.now().strftime("%Y-%m-%d-%H%M")
        imgname = f'{time_now}-{str(uuid4())[:3]}.jpg'
        print(f'{time_now}: {imgname} 데이터 추가')
        # commit
        AbstractDB[imgname] = 'IMGDATA'

        time.sleep(10)

if __name__ == '__main__' :
    simulate(AbstractDB)
