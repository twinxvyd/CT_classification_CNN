import matplotlib.pyplot as plt
import numpy as np

# 주어진 값
values = np.load("MC_3DCNN/result/all_accuracy_data.npy")
average_value = np.mean(values)

# 시도 횟수
attempts = list(range(1, len(values) + 1))

# 그래프 초기화
plt.figure()

# 각 시도 횟수에 대해 꺾은선 그래프 그리기
plt.plot(attempts, values, marker='o', linestyle='-', label='Accuracy')  # "Accuracy" 레이블 추가

# X 축 레이블 설정
plt.xlabel("Index")
plt.ylabel("Percentage")
# Y 축 레이블 설정
y_ticks = [70, 80, 90, 100]
plt.yticks(y_ticks)
# 그래프 제목 설정
plt.title("Accuracy")




# 각 데이터 포인트에 숫자 표시
for x, y in zip(attempts, values):
    plt.text(x, y, f"{y:.2f}%", ha='left', va='bottom')

# 평균선 출력
plt.axhline(y=average_value, color='r', linestyle='--', label=f"average_line ({average_value:.2f}%)")  # 평균값 레이블 추가

#plt.text(len(attempts) - 1, average_value - 5, f"{average_value:.2f}%", ha='right', va='top',color='r')  # 위치 아래로 이동

plt.legend()

# 레전드 (범례) 추가
# 그래프 표시

plt.savefig('all_accuracy_data.png')
plt.show()