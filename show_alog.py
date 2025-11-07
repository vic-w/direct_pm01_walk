import re
import torch
import matplotlib.pyplot as plt

log_path = "alog.txt"
with open(log_path, "r", encoding="utf-8") as f:
    text = f.read()

# 匹配跨多行的 action tensor
pattern = re.compile(r"action:\s*tensor\(\[([\s\S]*?)\],\s*device=", re.MULTILINE)

actions = []
for match in pattern.finditer(text):
    nums_str = match.group(1).replace("\n", " ").replace("  ", " ")
    nums = [float(x.replace("e", "E")) for x in nums_str.split(",") if x.strip()]
    if len(nums) == 24:
        # 跳过连续重复
        if not actions or any(abs(nums[i] - actions[-1][i]) > 1e-6 for i in range(24)):
            actions.append(nums)

if not actions:
    print("未找到任何 action 行，请检查日志格式。")
    exit()

actions = torch.tensor(actions)
print(f"共读取 {len(actions)} 条有效 action")

# 绘图：24个子图
fig, axes = plt.subplots(4, 6, figsize=(16, 9))
axes = axes.flatten()

for i in range(24):
    ax = axes[i]
    ax.plot(actions[:, i].cpu().numpy())
    ax.set_title(f"Joint {i}", fontsize=9)
    ax.grid(True)
    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)

# 去掉多余空白并整体标题
plt.suptitle("Action Trajectories (24 Joints)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
