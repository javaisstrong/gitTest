import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 데이터 전처리 및 로딩 ---
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=1, shuffle=True)

# --- CNN 구성요소 구현 ---
def conv2d(y, filter):
    HSize, WSize = y.shape
    HF, WF = filter.shape
    HMove = HSize - HF + 1
    WMove = WSize - WF + 1
    output = np.zeros((HMove, WMove))   
    regions = []
    for i in range(HMove):
        for j in range(WMove):
            yRegion = y[i:i+HF, j:j+WF]
            output[i, j] = np.sum(filter * yRegion)
            regions.append(((i, j), yRegion))
    return output, regions

def ReLu(y):
    return np.maximum(0, y)

def maxpooling(y, size):
    HS, WS = y.shape
    HNumber = HS // size
    WNumber = WS // size
    yPart = np.zeros((HNumber, WNumber))
    for i in range(HNumber):
        for j in range(WNumber):
            yPool = y[i*size:i*size+size, j*size:j*size+size]
            yPart[i, j] = np.max(yPool)
    return yPart

def flat(y, W, b):
    flattened = y.flatten()
    return np.dot(W, flattened) + b

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def cross_entropy_loss(pred_probs, true_class_index):
    return -np.log(pred_probs[true_class_index] + 1e-8)

def backPropagation(y, true_class_index, flattened, w, b, conv_regions, filter, lr=0.01):
    onehot = np.zeros_like(y)
    onehot[true_class_index] = 1
    delta = y - onehot
    dW = np.outer(delta, flattened)
    db = delta
    w -= lr * dW
    b -= lr * db
    dFilter = np.zeros_like(filter)
    for (i, j), region in conv_regions:
        for k in range(len(delta)):
            dFilter += delta[k] * np.mean(region) * 0.01
    filter -= lr * dFilter
    return w, b, filter

print("동작 시작")

# 필터들 정의
filter1 = np.array([[2, 2], [2, -4]], dtype=np.float64)
filter2 = np.array([[1, -1], [1, 1]], dtype=np.float64)
filter3 = np.array([[0, 1], [-1, 0]], dtype=np.float64)
filter4 = np.array([[1, 0], [0, -1]], dtype=np.float64)

sample_input, _ = next(iter(train_loader))
sample_input = sample_input.squeeze().numpy() * 255

pr1, regions1 = conv2d(sample_input, filter1)
pr1 = ReLu(pr1)
pr1 = maxpooling(pr1, 2)

pr2, regions2 = conv2d(pr1, filter2)
pr2 = ReLu(pr2)
pr2 = maxpooling(pr2, 2)

pr3, regions3 = conv2d(pr2, filter3)
pr3 = ReLu(pr3)
pr3 = maxpooling(pr3, 2)

pr4, regions4 = conv2d(pr3, filter4)
pr4 = ReLu(pr4)
pr4 = maxpooling(pr4, 1)

flattened = pr4.flatten()
n = len(flattened)
Nneuron = 10

if n == 0:
    raise ValueError("Flatten된 벡터의 크기가 0입니다. 필터 크기 또는 maxpool size를 확인하세요.")

w = np.random.randn(Nneuron, n) * np.sqrt(2/n)
b = np.zeros(Nneuron)

print("훈련시작")
loss_list = []
for epoch in range(100):
    total_loss = 0
    print("for 문 시작")
    for i, (img, label) in enumerate(train_loader):
        if i >= 100:
            print(f"⏹ 이미지 100개 학습 완료 (Epoch {epoch+1})")
            break
        x = img.squeeze().numpy() * 255
        true_class = label.item()

        pr1, _ = conv2d(x, filter1)
        pr1 = ReLu(pr1)
        pr1 = maxpooling(pr1, 2)

        pr2, _ = conv2d(pr1, filter2)
        pr2 = ReLu(pr2)
        pr2 = maxpooling(pr2, 2)

        pr3, _ = conv2d(pr2, filter3)
        pr3 = ReLu(pr3)
        pr3 = maxpooling(pr3, 2)

        pr4, regions4 = conv2d(pr3, filter4)
        pr4 = ReLu(pr4)
        pr4 = maxpooling(pr4, 1)

        flattened = pr4.flatten()
        if len(flattened) != n:
            continue  # flatten 크기가 다르면 skip

        pr_final = flat(pr4, w, b)
        pr_soft = softmax(pr_final)
        loss = cross_entropy_loss(pr_soft, true_class)
        total_loss += loss

        w, b, filter4 = backPropagation(pr_soft, true_class, flattened, w, b, regions4, filter4, lr=0.01)

    avg_loss = total_loss / 100
    loss_list.append(avg_loss)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    if avg_loss < 0.2:
        print(f"🎉 Loss가 0.2 이하로 도달하여 조기 종료 (Epoch {epoch+1})")
        break

plt.plot(loss_list, marker='o')
plt.title("Average Loss per Epoch (on 100 samples)")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.grid(True)
plt.show()

# 테스트 결과 확인
samples = []
for i, (img, label) in enumerate(train_loader):
    if i >= 100:
        break
    x = img.squeeze().numpy() * 255
    true_class = label.item()

    pr1, _ = conv2d(x, filter1)
    pr1 = ReLu(pr1)
    pr1 = maxpooling(pr1, 2)

    pr2, _ = conv2d(pr1, filter2)
    pr2 = ReLu(pr2)
    pr2 = maxpooling(pr2, 2)

    pr3, _ = conv2d(pr2, filter3)
    pr3 = ReLu(pr3)
    pr3 = maxpooling(pr3, 2)

    pr4, _ = conv2d(pr3, filter4)
    pr4 = ReLu(pr4)
    pr4 = maxpooling(pr4, 1)

    pr_final = flat(pr4, w, b)
    pr_soft = softmax(pr_final)
    pred_class = np.argmax(pr_soft)
    samples.append((x, true_class, pred_class))

plt.figure(figsize=(10, 3))
n = len(samples)
for i, (img, true_cls, pred_cls) in enumerate(samples):
    plt.subplot(1, n, i+1)  # <- n으로 수정
    plt.imshow(img, cmap='gray')
    plt.title(f"T:{true_cls}, P:{pred_cls}")
    plt.axis('off')
plt.suptitle("Test Samples - True vs Predicted (4-layer CNN)")
plt.tight_layout()
plt.show()
