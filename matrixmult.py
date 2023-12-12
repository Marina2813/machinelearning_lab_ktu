list1 = []
list2 = []
res = []

m = int(input("Enter the number of rows: "))
n = int(input("Enter the number of columns: "))
for i in range(m):
    row1 = []
    for j in range(n):
        f = int(input(f"Enter the element for {i}{j}"))
        row1.append(f)
    list1.append(row1)

for i in range(m):
    row2 = []
    for j in range(n):
        f = int(input(f"Enter the element for {i}{j}"))
        row2.append(f)
    list2.append(row2)

print(list1)
print(list2)

for i in range(m):
    row3 = []
    for j in range(n):
        sum_ = 0
        for k in range(n):
            sum_ += list1[i][k]*list2[k][j]
        row3.append(sum_)
    res.append(row3)

print(res)