arr=[{"name":"tk","age":"34"},{"name":"sk","age":"24"}]

def filterArr(res):
    print(res)
    return {'name':res['name']}


result=list(map(filterArr,arr))

print(result)