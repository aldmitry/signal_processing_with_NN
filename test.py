class A:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def summ(self):
        return self.a + self.b

class B(A):
    def __init__(self, a, b):
        self.a = a
        self.b = b

ex1 = A(10,5)
print(ex1.summ())
ex2 = B(10,20)
print(ex2.summ())