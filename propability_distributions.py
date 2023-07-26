"""
Probability distributions are mathematical functions that describe the likelihood of different outcomes in a random experiment. Each distribution has specific characteristics that make it useful for modeling different types of data. Let's explain and compare three commonly used probability distributions: Normal, Binomial, and Poisson distributions, using examples to remember them easily.

1. **Normal Distribution (Gaussian Distribution):**
   - The normal distribution is a symmetric bell-shaped curve that is fully defined by its mean (μ) and standard deviation (σ).
   - Many natural phenomena, such as height, weight, test scores, etc., tend to follow a normal distribution.
   - The majority of data in a normal distribution falls close to the mean, and it becomes less likely to find data far away from the mean.

   Example: The distribution of heights of adult males in a population. The mean height might be around 175 cm, and the standard deviation might be 6 cm.

2. **Binomial Distribution:**
   - The binomial distribution models the number of successes (X) in a fixed number of independent Bernoulli trials (n), where each trial has the same probability of success (p).
   - It is discrete and often used for counting the number of occurrences of a binary event (success/failure).

   Example: The number of heads obtained when flipping a fair coin 10 times. Here, n = 10 (number of trials) and p = 0.5 (probability of getting a head).

3. **Poisson Distribution:**
   - The Poisson distribution models the number of events occurring in a fixed interval of time or space.
   - It is often used when events are rare and independent of each other.

   Example: The number of customers arriving at a store in an hour, given that the average arrival rate is 3 customers per hour.

Now, let's compare these distributions:

- Shape:
  - Normal: Symmetric and bell-shaped.
  - Binomial: Discrete and can be skewed depending on the value of p and n.
  - Poisson: Discrete and typically right-skewed.

- Parameters:
  - Normal: Mean (μ) and standard deviation (σ).
  - Binomial: Number of trials (n) and probability of success (p).
  - Poisson: Average rate (λ) of events occurring in a fixed interval.

- Types of Data:
  - Normal: Continuous data that can take any real value.
  - Binomial: Count data with a fixed number of trials, resulting in discrete values.
  - Poisson: Count data for events occurring in a fixed interval, also resulting in discrete values.

Remember, these are just brief explanations and examples to give you a general understanding of these probability distributions. Each distribution has specific use cases and mathematical properties, so it's essential to delve deeper into each one for a comprehensive understanding in a data science context."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson

# Parameters for the distributions
mean_normal, std_normal = 175, 6
n_binomial, p_binomial = 10, 0.5
lambda_poisson = 3

# Generate data points for the x-axis (domain)
x_normal = np.linspace(mean_normal - 3 * std_normal, mean_normal + 3 * std_normal, 100)
x_binomial = np.arange(0, n_binomial + 1)
x_poisson = np.arange(0, 10 * lambda_poisson + 1)

# Calculate the probability density function (PDF) for each distribution
pdf_normal = norm.pdf(x_normal, mean_normal, std_normal)
pdf_binomial = binom.pmf(x_binomial, n_binomial, p_binomial)
pdf_poisson = poisson.pmf(x_poisson, lambda_poisson)

# Plot the distributions
plt.figure(figsize=(12, 6))

# Normal distribution
plt.subplot(131)
plt.plot(x_normal, pdf_normal, color='b')
plt.title('Normal Distribution')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.grid(True)
plt.savefig('plots/normal_distribution.png')

# Binomial distribution
plt.subplot(132)
plt.stem(x_binomial, pdf_binomial, basefmt='b')
plt.title('Binomial Distribution')
plt.xlabel('X')
plt.ylabel('Probability')
plt.grid(True)
plt.savefig('plots/binomial_distribution.png')

# Poisson distribution
plt.subplot(133)
plt.stem(x_poisson, pdf_poisson, basefmt='b')
plt.title('Poisson Distribution')
plt.xlabel('X')
plt.ylabel('Probability')
plt.grid(True)
plt.savefig('plots/poisson_distribution.png')

# plt.tight_layout()
# plt.show()
