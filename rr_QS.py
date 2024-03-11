import numpy as np
import random
import sys

sys.setrecursionlimit(10 ** 6)
cnt = 0
rr_cnt = 0


def partition(A, p, r):
    piv = r  # 마지막 원소를 피벗 으로
    s = p - 1  # 피벗 보다 작은 원소를 보관할 구역의 마지막 인덱스 번호
    global cnt
    for i in range(r - p):  # 받은 배열의 모든 원소 확인
        cnt += 1
        if A[p + i] <= A[piv]:  # 피벗 보다 작은 원소면
            s += 1
            A[p + i], A[s] = A[s], A[p + i]
    s += 1
    A[r], A[s] = A[s], A[r]  # 피벗 원소를 자신 보다 작은 원소가 있는 구역 오른쪽에 두기
    return s  # 피벗 원소 인덱스 번호를 반환

    # 피벗 원소를 제 자리에 두고 비교연산을 몇번 수행하는지 센다


def rr_quick_sort(A, p, r):  # 배열과 시작과 끝의 인덱스 번호를 받아온다
    if p >= r:  # 원소의 갯수가 1개 이하면 종료
        return
    q = r_partition(A, p, r)  # 피벗 원소를 기준 으로 분할
    rr_quick_sort(A, p, q - 1)  # 왼쪽 정렬
    rr_quick_sort(A, q + 1, r)  # 오른쪽 정렬

    # 재귀적으로 왼쪽 구역과 오른쪽 구역으로 나눠서 처리한다


def r_partition(A, p, r):
    pivot = random.randint(p, r)  # 불러온 구역의 인덱스 번호 중에서 골라 피벗 원소의 인덱스 번호로 사용한다.
    A[pivot], A[r] = A[r], A[pivot]  # 랜덤 피벗 원소를 마지막 원소 위치로 옮긴다
    return partition(A, p, r)  # 만들어진 배열로 분할 한다.


N = [10 ** 2, 10 ** 4, 10 ** 6]

for i in N:
    for j in range(30):
        A = np.random.rand(i)
        rr_quick_sort(A, 0, len(A) - 1)
        rr_cnt += cnt
        cnt = 0
    print(i, "에서의 재귀적 랜덤 피벗 퀵정렬 평균 연산 횟수는", rr_cnt / 30, "번")
print("전체 연산 횟수는", rr_cnt, "번")
