class Dir:
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3
	IN = 4
	OUT = 5

# Describes relationships between two tiles. Tile1, Tile2, left means t1 is allowed to be on the left of t2
class Rule:
	def __init__(self, t1, t2, dir):
		self.t1 = t1
		self.t2 = t2
		self.dir = dir
	
	def __eq__(self, other):
		return self.t1 == other.t1 and self.t2 == other.t2 and self.dir == other.dir

	def __hash__(self):
		return hash(str(self.t1) + str(self.t2) + str(self.dir))

	def __str__(self):
		return str(self.t1) + " " + str(self.t2) + " " + str(self.dir)

	def __repr__(self):
		return str(self)

	def __lt__(self, other):
		if self.t1 < other.t1:
			return True
		elif self.t1 > other.t1:
			return False
		
		if self.t2 < other.t2:
			return True
		elif self.t2 > other.t2:
			return False
		
		if self.dir < other.dir:
			return True
		return False
