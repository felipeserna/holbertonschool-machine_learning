#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here

# Container of the 5 plots
fig = plt.figure()
plt.suptitle("All in One")

# Task 0

fig.add_subplot(321)
plt.plot(y0, color='red')
plt.xlim(0, 10)
plt.xticks(np.arange(0, 11, 2))
plt.yticks(np.arange(0, 1500, 500))

# Task 1

fig.add_subplot(322)
plt.scatter(x1, y1, color='magenta', marker=".")
plt.xlabel('Height (in)', fontsize='x-small')
plt.ylabel('Weight (lbs)', fontsize='x-small')
plt.title("Men's Height vs Weight", fontsize='x-small')

# Task 2

fig.add_subplot(323)
plt.plot(x2, y2)
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title("Exponential Decay of C-14", fontsize='x-small')
plt.yscale('log')
plt.xlim(0, 28650)

# Task 3

fig.add_subplot(324)
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title("Exponential Decay of Radioactive Elements", fontsize='x-small')
plt.ylim(0, 1)
plt.xlim(0, 20000)
plt.plot(x3, y31, color='red', linestyle='dashed', label="C-14")
plt.plot(x3, y32, color='green', label="Ra-226")
plt.legend(loc='upper right', fontsize='x-small')

# Task 4

fig.add_subplot(313)
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 40, 10))
plt.xlabel('Grades', fontsize='x-small')
plt.ylabel('Number of Students', fontsize='x-small')
plt.xticks(range(0, 101, 10))
plt.title("Project A", fontsize='x-small')
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
plt.xlim(0, 100)
plt.ylim(0, 30)

# Task 5

fig.tight_layout()
plt.show()
