import cv2
import numpy as np


def main():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    print(arr.ndim)
    print(arr.shape)
    arr = np.arange(5)
    print(arr)
    arr = np.arange(5, 15, 2)
    print(arr)

    # zeros
    arr = np.zeros(5)
    print(arr)
    arr = np.zeros((2, 2, 2))
    print(arr)
    img = cv2.imread("/home/eohyun_ee/autocar/numpy/자동차.png")
    img2 = np.zeros((100, 200))
    img3 = np.zeros_like(img)   # 0 으로 채우는데 기존 이미지의 해상도와 dtype을 똑같이 복제를 함.
    img4 = np.zeros(img.shape)
    print(type(img))
    # cv2.imshow("img", img)
    # # cv2.imshow("img2", img2)
    # # cv2.imshow("img3", img3)
    # cv2.waitKey()

    # ones  -> 기준이 1인 연산을 수행할때
    arr = np.ones(5)
    print(arr)
    arr = np.ones((2, 2))
    print(arr)
    img5 = np.ones_like(img)
    print(img5.shape)
    # cv2.imshow("img5", img5)
    # cv2.waitKey()

    # full
    arr = np.full(5, 255)
    print(arr)
    arr = np.full((2, 2), 255)
    img6 = np.full_like(img, 255)
    # cv2.imshow("img6", img6)
    # cv2.waitKey()

    # eye 단위 행렬
    arr = np.eye(5)
    print(arr)

    # random
    arr1 = np.random.random((5, 5))
    arr2 = arr1.dot(arr)
    print(arr)
    print(arr1)
    print(arr2)


if __name__ == "__main__":
    main()