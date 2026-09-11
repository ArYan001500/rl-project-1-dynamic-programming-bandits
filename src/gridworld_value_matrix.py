import numpy as np

# اندازه شبکه
grid_size = 5
n_states = grid_size * grid_size

# ضریب تنزیل
gamma = 0.9

# تعریف حالت‌های ویژه بر اساس شکل 3-5 کتاب
# A = (0,1) -> A' = (4,1) با پاداش +10
# B = (0,3) -> B' = (2,3) با پاداش +5

A = (0, 1)
A_prime = (4, 1)

B = (0, 3)
B_prime = (2, 3)

actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # بالا، پایین، چپ، راست
action_prob = 1 / 4

# تبدیل مختصات (i,j) به شماره حالت
def state_index(i, j):
    return i * grid_size + j

# ماتریس گذار و بردار پاداش
P = np.zeros((n_states, n_states))
R = np.zeros(n_states)

for i in range(grid_size):
    for j in range(grid_size):
        s = state_index(i, j)

        # حالت ویژه A
        if (i, j) == A:
            s_next = state_index(*A_prime)
            P[s, s_next] = 1.0
            R[s] = 10
            continue

        # حالت ویژه B
        if (i, j) == B:
            s_next = state_index(*B_prime)
            P[s, s_next] = 1.0
            R[s] = 5
            continue

        expected_reward = 0

        for di, dj in actions:
            ni, nj = i + di, j + dj

            # اگر از مرز خارج شود، در همان خانه می‌ماند
            if ni < 0 or ni >= grid_size or nj < 0 or nj >= grid_size:
                ni, nj = i, j
                reward = -1
            else:
                reward = 0

            s_next = state_index(ni, nj)
            P[s, s_next] += action_prob
            expected_reward += action_prob * reward

        R[s] = expected_reward

# حل دستگاه خطی: V = (I - gamma P)^(-1) R
I = np.eye(n_states)
V = np.linalg.solve(I - gamma * P, R)

# تبدیل به ماتریس 5x5 و گرد کردن به یک رقم اعشار
V_grid = V.reshape((grid_size, grid_size))
V_rounded = np.round(V_grid, 1)

print("Function value states")
print(V_rounded)
