import numpy as np

# ------------------------------
# تنظیمات محیط Gridworld 4×4
# ------------------------------

n_rows = 4
n_cols = 4
n_states = n_rows * n_cols

# اعمال را به صورت: 0: up, 1: right, 2: down, 3: left
actions = [0, 1, 2, 3]
action_sign = {
    0: '^',
    1: '>',
    2: 'v',
    3: '<'
}

gamma = 0.9
reward = -1.0

# حالت‌های ترمینال (بر اساس شماره‌گذاری 0..15)
# این‌جا فرض می‌کنیم خانه‌ی بالا-چپ و پایین-راست ترمینال هستند.
terminal_states = [0, 15]


def state_to_pos(s):
    """تبدیل اندیس حالت (۰..۱۵) به مختصات (سطر، ستون)"""
    return divmod(s, n_cols)  # (row, col)


def pos_to_state(row, col):
    """تبدیل (سطر، ستون) به اندیس حالت (۰..۱۵)"""
    return row * n_cols + col


def step(s, a):
    """
    مدل انتقال قطعی:
    ورودی: حالت s و عمل a
    خروجی: حالت بعدی s_next و پاداش
    """
    if s in terminal_states:
        # در حالت ترمینال می‌مانیم، بدون هزینه‌ اضافی
        return s, 0.0

    row, col = state_to_pos(s)

    if a == 0:      # up
        new_row, new_col = max(row - 1, 0), col
    elif a == 1:    # right
        new_row, new_col = row, min(col + 1, n_cols - 1)
    elif a == 2:    # down
        new_row, new_col = min(row + 1, n_rows - 1), col
    elif a == 3:    # left
        new_row, new_col = row, max(col - 1, 0)
    else:
        raise ValueError("invalid action")

    s_next = pos_to_state(new_row, new_col)
    r = reward   # روی همه‌ی گذارها -۱

    return s_next, r


# ------------------------------
# الگوریتم تکرار سیاست
# ------------------------------

# سیاست اولیه: مثلا همه بالا (a=0)
policy = np.zeros(n_states, dtype=int)

# در حالت‌های ترمینال سیاست معنی ندارد، ولی برای سادگی همان ۰ می‌گذاریم
for s in terminal_states:
    policy[s] = -1  # کد خاص برای ترمینال


def policy_evaluation(policy, theta=1e-6):
    """
    ارزیابی سیاست با تکرار متوالی تا زمانی که تغییرات < theta شود.
    خروجی: بردار V با طول n_states
    """
    V = np.zeros(n_states, dtype=float)
    while True:
        delta = 0.0
        for s in range(n_states):
            if s in terminal_states:
                continue  # V(s) = 0 فرض شده است
            a = policy[s]
            s_next, r = step(s, a)
            v_new = r + gamma * V[s_next]
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V


def policy_improvement(V):
    """
    بهبود سیاست: برای هر s عمل a را که Q(s,a) را ماکزیمم می‌کند انتخاب می‌کنیم.
    خروجی: policy_new (آرایه‌ی طول n_states)
    """
    policy_new = np.copy(policy)
    stable = True

    for s in range(n_states):
        if s in terminal_states:
            continue

        old_a = policy[s]

        # محاسبه‌ی Q(s,a) برای هر عمل ممکن
        q_values = []
        for a in actions:
            s_next, r = step(s, a)
            q_values.append(r + gamma * V[s_next])
        q_values = np.array(q_values)

        best_a = actions[np.argmax(q_values)]
        policy_new[s] = best_a

        if best_a != old_a:
            stable = False

    return policy_new, stable


# حلقه‌ی اصلی تکرار سیاست
iteration = 0
while True:
    iteration += 1
    # 1) ارزیابی سیاست فعلی
    V = policy_evaluation(policy)

    # 2) بهبود سیاست
    policy, stable = policy_improvement(V)

    # اگر سیاست تغییر نکرد، بهینه است
    if stable:
        print(f"Policy iteration converged at iteration {iteration}.")
        break


# ------------------------------
# چاپ نتایج نهایی
# ------------------------------

print("\nOptimal state-value function V* (به صورت ماتریس 4×4):")
V_matrix = V.reshape((n_rows, n_cols))
# گرد کردن برای خوانایی
V_rounded = np.round(V_matrix, 1)
print(V_rounded)

print("\nOptimal policy π* (using ^, v, <, >):")
for i in range(n_rows):
    row_symbols = []
    for j in range(n_cols):
        s = pos_to_state(i, j)
        if s in terminal_states:
            row_symbols.append("T")  # ترمینال
        else:
            a = policy[s]
            row_symbols.append(action_sign[a])
    print(" ".join(row_symbols))
