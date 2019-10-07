class Node:
    def __init__(self, data = None, next = None):
        self.data = data
        self.next = next
    def setNext(self,next):
        self.next=next
class LinkedList: 
    def __init__(self):
        self.head = None


    def add(self, data):
        if self.head==None:
            self.head=Node(data=data,next=None)
        else:
            temp=self.head
            while temp.next!=None:
                temp=temp.next
            nuevo=Node(data=data,next=None)
            temp.setNext(nuevo)

    def delete(self, key):
        temp = self.head
        prev = None
        while temp and temp.data != key:
            prev = temp
            temp = temp.next
        if prev is None:
            self.head = temp.next
        elif temp:
            prev.next = temp.next
            temp.next = None
    def search(self,key):
        temp=self.head
        while temp.next!=None and temp.data!=key:
            temp=temp.next
        return temp

    def print( self ):
        temp = self.head
        while temp != None:
            print(temp.data, end =" => ")
            temp =temp.next


