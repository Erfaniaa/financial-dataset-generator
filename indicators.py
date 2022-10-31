def get_average(l):
	s = 0
	for x in l:
		s += x
	return s / len(l)


def get_wma(l):
	s1 = 0
	s2 = 0
	c = len(l)
	for x in l:
		s1 += c * x
		s2 += c
		c -= 1
	return s1 / s2
