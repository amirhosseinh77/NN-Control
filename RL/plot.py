import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

# plt.style.use('fivethirtyeight')
# plt.rc('font', size=10)

def plot_return(returns, agent, window=20):
    display.clear_output(wait=True)
    plt.figure(figsize=(18,5))
    plt.title('RL Control Using Actor-Critic Architecture')

    rolling_mean = pd.Series(returns).rolling(window).mean()
    std = pd.Series(returns).rolling(window).std()

    plt.plot(returns)
    plt.plot(rolling_mean)
    plt.fill_between(range(len(returns)),rolling_mean-std, rolling_mean+std, color='violet', alpha=0.4)
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend(['Score', 'Rolling Mean'], loc='lower right')

    # Get current axes
    ax = plt.gca()

    # Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Optionally, remove the ticks as well
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.pause(0.001)