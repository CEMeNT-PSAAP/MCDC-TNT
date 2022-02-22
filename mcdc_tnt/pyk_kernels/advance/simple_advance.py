import pykokkos as pk

@pk.workunit
def add(i: int, v: pk.View1D[int], x: int):
    v[i] += x

if __name__ == "__main__":
    n = 10
    v = pk.View([n], int)
    v.fill(0)

    pk.parallel_for(n, add, v=v, x=1)
