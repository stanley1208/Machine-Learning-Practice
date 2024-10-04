import csv

graph={}

with open('graph.txt','r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        c1,c2,cost=row
        cost=int(cost)

        if c1 not in graph:
            graph[c1]=[]
        if c2 not in graph:
            graph[c2]=[]

        graph[c1].append((c2,cost))
        graph[c2].append((c1,cost))

num=len(graph)
edges=sum(len(edges) for edges in graph.values())//2

print(num)
print(edges)
print(graph)
print(graph.keys())
print(graph.values())