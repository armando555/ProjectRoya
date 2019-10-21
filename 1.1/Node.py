
# class used to create node objects for the tree
class treeNode():
    def __init__(self, col=-1, colName='', value=None, results=None, rb=None, lb=None):
        self.col = col #column number the node represents
        self.colName = colName; #name of the column the node represents
        self.value = value #value of the node
        self.results = results #results that are stored in the node
        self.rb = rb #the right children of the node
        self.lb= lb #the left children of the node