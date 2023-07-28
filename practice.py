import pandas as pd

# Sample data for demonstration
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 32, 28, 21, 29],
    'Gender': ['F', 'M', 'M', 'M', 'F'],
    'Salary': [50000, 60000, 55000, 48000, 52000],
    'Hire_Date': pd.to_datetime(['2020-01-15', '2019-07-10', '2022-03-05', '2023-02-20', '2021-11-30']),
    'Is_Employee': [True, True, True, False, True],
    'Ratings': [4.5, 3.8, 4.9, None, 4.2],
    'Department': ['HR', 'Finance', 'Engineering', 'Marketing', 'HR']
}

# Creating a DataFrame
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
img = df.plot(kind='line', x='Name', y='Salary')
plt.show()
# plt.imsave(img,"temp.png")
