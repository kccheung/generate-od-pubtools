import matplotlib.pyplot as plt

sample_times = [5, 10, 20, 50, 80, 120]
rmse_mean = [88.264757, 80.141179, 75.636760, 72.880789, 72.291245, 71.908854]
cpc_mean = [0.650504, 0.678156, 0.694668, 0.704844, 0.707390, 0.708985]
runtime_mean = [34.117334, 60.354760, 117.736665, 282.704379, 449.417843, 671.924086]

fig, ax1 = plt.subplots(figsize=(6,4))

# Left axis: CPC
ax1.plot(sample_times, cpc_mean, marker='o', label='CPC', color='r')
ax1.set_xlabel('sample_times for the Diffusion Model')
ax1.set_ylabel('CPC')
ax1.set_xticks(sample_times)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# Right axis: RMSE and runtime (scaled)
ax2 = ax1.twinx()
ax2.plot(sample_times, rmse_mean, marker='s', linestyle='-', label='RMSE')
ax2.plot(sample_times, [r/10 for r in runtime_mean],
         marker='^', linestyle='-', label='Runtime (sec / 10)')
ax2.set_ylabel('RMSE  /  Runtime (sec / 10)')

# Mark sweet spot at sample_times=50
sweet = 50
if sweet in sample_times:
    idx = sample_times.index(sweet)
    ax1.axvline(sweet, color='gray', linestyle='--', linewidth=1)
    ax1.scatter([sweet], [cpc_mean[idx]], color='k', zorder=5)
    ax2.scatter([sweet], [rmse_mean[idx]], color='k', zorder=5)
    # ax1.annotate(f'Sweet spot\nsample_times={sweet}', xy=(sweet, cpc_mean[idx]),
    #              xytext=(sweet + 5, max(cpc_mean) + 0.02),
    #              arrowprops=dict(arrowstyle='->', color='black'),
    #              fontsize=9, va='center')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

plt.tight_layout()
plt.show()
