file = open('10clusters.csv','w')
import random
file.write(f"X,Y")

for i in range(0,1000):
    x = random.uniform(-20,20)
    y  =random.uniform(-20,20)

    file.write(f"{x},{y}\n")
file.close()
