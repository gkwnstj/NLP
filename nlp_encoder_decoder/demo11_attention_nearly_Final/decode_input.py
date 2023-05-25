import torch

# 주어진 텐서 예시
tensor = torch.rand(128, 20)

# 첫 번째 열에 2999를 추가
tensor = torch.cat((torch.full((128, 1), 2999, dtype=torch.float32), tensor), dim=1)

# 마지막 열 제거
tensor = tensor[:, :-1]

# 행의 마지막 열에 2가 있는 경우 해당 값을 0으로 변경
mask = tensor[:, -1] == 2
tensor[:, -1] = torch.where(mask, torch.tensor(0, dtype=torch.float32), tensor[:, -1])

# 결과 확인
print(tensor.size())
print(tensor)
