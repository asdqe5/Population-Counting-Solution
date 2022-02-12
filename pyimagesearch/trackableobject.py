class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False
		#원흥관 IN/OUT
		self.counted_one = False
		#신공학관 IN/OUT
		self.counted_new = False
		#중앙도서관 및 팔정도 IN/OUT
		self.counted_lib = False
		#흡연구역 IN/OUT
		self.counted_smoke = False
		#region time
		self.frame_in_region = 0