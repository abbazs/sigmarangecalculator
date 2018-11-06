class room():
    def __init__(self):
        self.length = None
        self.width = None
        self.height = None
    
    def size(self):
        return self.length*self.width
    
    def volume(self):
        return self.length*self.width*self.height

class house(room):
    def __init__(self, l, w, h=10):
        self.length = l
        self.width = w
        self.height = h

        self.address = None
        self.phone = None
        self.Members = None
        self.kitchen_p = kitchen(10, 10, 10)
        self.bedroom_p = bedroom(10, 10, 8)

    def print_details(self):
        print(f'{self.address} - {self.phone} - {self.Members}')

class kitchen(room):
    def __init__(self, l, w, h):
        self.length = l
        self.width = w
        self.height = h

class bedroom(room):
    def __init__(self, l, w, h):
        self.length = l
        self.width = w
        self.height = h


list_of_house = []

for x in range(1, 11):
    h = house(100, 100)
    h.address = x
    h.phone = x*x
    h.Members = (x*x) + (x*x)
    list_of_house.append(h)