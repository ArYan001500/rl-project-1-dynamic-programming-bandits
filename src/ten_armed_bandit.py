import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# تنظیمات مسئله
# ----------------------------

NUM_ARMS = 10      # تعداد بازوها
NUM_STEPS = 1000     # هر تلاش: 1000 بار انتخاب عمل
NUM_RUNS = 2000       # تعداد تکرار آزمایش (برای میانگین گیری)
EPSILON = 0.1          # برای روش epsilon-greedy
UCB_C = 2.0            # ضریب c برای روش UCB


def generate_bandit():
    """
    ایجاد یک مسئله 10 راهزن مسلح.
    برای هر بازو یک میانگین واقعی q_* به صورت N(0,1) می‌سازیم.
    پاداش هر بار انتخاب بازو a از N(q_*(a), 1) نمونه‌برداری می‌شود.
    """
    true_action_values = np.random.normal(loc=0.0, scale=1.0, size=NUM_ARMS)
    return true_action_values


def get_reward(true_values, action):
    """
    تولید پاداش برای عمل انتخاب شده
    """
    reward = np.random.normal(loc=true_values[action], scale=1.0)
    return reward


def epsilon_greedy_run():
    """
    اجرای یک بار آزمایش (یک حلقه 1000 گامی) با روش epsilon-greedy.
    خروجی: آرایه پاداش های به دست آمده در طول 1000 گام.
    """
    true_values = generate_bandit()

    # Q(a) و تعداد انتخاب های هر بازو را صفر می‌کنیم
    Q = np.zeros(NUM_ARMS)
    N = np.zeros(NUM_ARMS)

    rewards = np.zeros(NUM_STEPS)

    for t in range(NUM_STEPS):
        # انتخاب عمل
        if np.random.rand() < EPSILON:
            # اکتشاف: انتخاب تصادفی
            action = np.random.randint(NUM_ARMS)
        else:
            # بهره برداری: بیشترین Q
            # در صورت وجود چند عمل با Q برابر، argmax اولین را برمی‌گرداند
            action = np.argmax(Q)

        # دریافت پاداش از محیط
        r = get_reward(true_values, action)

        # به‌روزرسانی تعداد انتخاب‌ها
        N[action] += 1

        # به‌روزرسانی Q(a) با میانگین‌گیری افزایشی
        # Q_{new} = Q_{old} + (1/N) * (r - Q_{old})
        Q[action] += (r - Q[action]) / N[action]

        rewards[t] = r

    return rewards


def ucb_run():
    """
    اجرای یک بار آزمایش با روش UCB.
    خروجی: آرایه پاداش های به دست آمده در طول 1000 گام.
    """
    true_values = generate_bandit()

    Q = np.zeros(NUM_ARMS)
    N = np.zeros(NUM_ARMS)

    rewards = np.zeros(NUM_STEPS)

    # برای جلوگیری از تقسیم بر صفر در فرمول UCB،
    # ابتدا هر بازو را حداقل یک بار امتحان می‌کنیم.
    for a in range(NUM_ARMS):
        r = get_reward(true_values, a)
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        rewards[a] = r

    # از اینجا به بعد t از NUM_ARMS تا NUM_STEPS-1 است
    for t in range(NUM_ARMS, NUM_STEPS):
        # محاسبه مقدار UCB برای هر بازو
        # term = c * sqrt( ln(t+1) / N[a] )
        ucb_values = Q + UCB_C * np.sqrt(np.log(t + 1) / N)

        action = np.argmax(ucb_values)

        r = get_reward(true_values, action)

        N[action] += 1
        Q[action] += (r - Q[action]) / N[action]

        rewards[t] = r

    return rewards


def run_experiments():
    """
    اجرای NUM_RUNS آزمایش برای هر کدام از روش های epsilon-greedy و UCB
    و محاسبه پاداش متوسط در هر گام.
    """
    avg_rewards_eps = np.zeros(NUM_STEPS)
    avg_rewards_ucb = np.zeros(NUM_STEPS)

    for i in range(NUM_RUNS):
        if (i + 1) % 100 == 0:
            print(f"Run {i + 1}/{NUM_RUNS} ...")

        rewards_eps = epsilon_greedy_run()
        rewards_ucb = ucb_run()

        # جمع کردن پاداش ها برای میانگین گیری نهایی
        avg_rewards_eps += rewards_eps
        avg_rewards_ucb += rewards_ucb

    # میانگین گیری بر روی تعداد اجراها
    avg_rewards_eps /= NUM_RUNS
    avg_rewards_ucb /= NUM_RUNS

    return avg_rewards_eps, avg_rewards_ucb


def plot_results(avg_rewards_eps, avg_rewards_ucb):
    """
    رسم نمودار پاداش متوسط بر حسب گام.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(avg_rewards_eps, label=f"ε-greedy (ε={EPSILON})")
    plt.plot(avg_rewards_ucb, label=f"UCB (c={UCB_C})")
    plt.xlabel("(t)")
    plt.ylabel("Average Reward")
    plt.title("Comparison Between UCB and Greedy in 10-Armed bandit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    avg_rewards_eps, avg_rewards_ucb = run_experiments()
    plot_results(avg_rewards_eps, avg_rewards_ucb)


if __name__ == "__main__":
    main()
