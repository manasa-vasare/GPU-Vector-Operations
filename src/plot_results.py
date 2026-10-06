import pandas as pd
import matplotlib
matplotlib.use("Agg") # save PNG files; no window needed
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

# Correctness must be PASS for every run used in the final comparison.
bad = df[(df["Add_Correct"] != "PASS") | (df["Mul_Correct"] != "PASS")]
if not bad.empty:
    print("WARNING: some runs failed correctness. Remove them and re-run.")
    print(bad)

# Average repeated runs of the same configuration.
avg = df.groupby(["N", "ThreadsPerBlock"]).mean(numeric_only=True).reset_index()
avg.round(6).to_csv("summary.csv", index=False)

# ---- Graph 1 and 2: vector-size scaling (threads per block = 256) ----
size = avg[avg["ThreadsPerBlock"] == 256].sort_values("N")

cpu_total = size["CPU_Add_ms"] + size["CPU_Mul_ms"]
gpu_kernel_total = size["GPU_Add_Kernel_ms"] + size["GPU_Mul_Kernel_ms"]

plt.figure(figsize=(7, 4.5))
plt.loglog(size["N"], cpu_total, "o-", label="CPU (add + multiply)")
plt.loglog(size["N"], gpu_kernel_total, "s-", label="GPU kernels only")
plt.loglog(size["N"], size["Total_GPU_ms"], "^-", label="GPU total phase")
plt.xlabel("Vector size N")
plt.ylabel("Execution time (ms)")
plt.title("CPU vs GPU execution time (256 threads/block)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("graph1_time_vs_size.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4.5))
plt.semilogx(size["N"], size["Add_Speedup"], "o-", label="Addition kernel speedup")
plt.semilogx(size["N"], size["Mul_Speedup"], "s-", label="Multiplication kernel speedup")
plt.semilogx(size["N"], size["EndToEnd_Speedup"], "^-", label="End-to-end speedup")
plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Vector size N")
plt.ylabel("Speedup (CPU time / GPU time)")
plt.title("GPU speedup vs vector size (256 threads/block)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("graph2_speedup_vs_size.png", dpi=150)
plt.close()

# ---- Graph 3: threads-per-block study at the largest N ----
largest = avg["N"].max()
tpb = avg[avg["N"] == largest].sort_values("ThreadsPerBlock")
labels = [str(int(t)) for t in tpb["ThreadsPerBlock"]]
x = range(len(labels))
w = 0.38

plt.figure(figsize=(7, 4.5))
plt.bar([i - w / 2 for i in x], tpb["GPU_Add_Kernel_ms"], w, label="Addition kernel")
plt.bar([i + w / 2 for i in x], tpb["GPU_Mul_Kernel_ms"], w, label="Multiplication kernel")
plt.xticks(list(x), labels)
plt.xlabel("Threads per block")
plt.ylabel("GPU kernel time (ms)")
plt.title("Kernel time vs threads per block (N = %d)" % largest)
plt.legend()
plt.tight_layout()
plt.savefig("graph3_kernel_vs_threads.png", dpi=150)
plt.close()

print("Created: summary.csv, graph1_time_vs_size.png,")
print(" graph2_speedup_vs_size.png, graph3_kernel_vs_threads.png")
