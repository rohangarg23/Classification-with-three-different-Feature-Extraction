
import matplotlib.pyplot as plt

# Define 5 points (x, y coordinates)
x = [20, 40, 60, 80, 100]
y = [94.8, 96.7, 97.3, 98.3, 98.3]

# Create a figure and plot the points
plt.figure()
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Points')

# Add labels and title
plt.xlabel('%Training Examples')
plt.ylabel('Validation Accuracy')
plt.title('Plot of Validation Accuracy with percent of training example for Dataset 2')

# Display the legend
plt.legend()

# Save the figure as a PNG file
plt.savefig('Validation Accuracy vs. Fraction of Training Examples.png')

# Show the plot (optional)
plt.show()
