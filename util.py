# correct  = [49,-58,-161,-11,108,-2,67,-76,8,136,6,-19,-77,23,71,126,-82,30,5,19,-118,-64,18,261,47,-281,-40,193,172,-98,113,121]
# shuffled = [49,-161,108,67,-58,-11,-2,-76,8,6,-77,71,136,-19,23,126,-82,5,-118,18,30,19,-64,261,47,-40,172,113,-281,193,-98,121]
# res = []
# for num in correct:
#     index = shuffled.index(num)
#     res.append(index*2)
#     res.append(index*2+1)

# print(res)
count = 0
for num in range(0, 200):
    numString = str(num)
    if "11" in numString:
        count+=1
print(count)